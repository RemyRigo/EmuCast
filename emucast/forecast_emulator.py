#import os
#os.environ["OMP_NUM_THREADS"] = "4"
# avoid CPU usage warning for k-means, remove to have some surprise (死机, crash)
import warnings
from datetime import datetime,time,timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import morph_nrmse,morph_nmae,morph_eof
from utils import nrmse,nmae,eof
from utils import validate_timeseries


class ForecastEmulator:
    def __init__(self,
                 ts_in: pd.Series,  # input time series as a single column array (or list)
                 nb_states: int = 30,  # number of states at every time steps - default
                 nb_forecast_profiles: int = 300 # numbers of forecast scenarios profiles
                 ):

        #Checke input time series
        validate_timeseries(ts_in)

        # Check that the series is at least one month long
        duration = ts_in.index[-1] - ts_in.index[0]
        if duration <= pd.Timedelta(days = 30):
            warnings.warn("Input time series is less than one month long -- this could degrade performance.",
                          UserWarning)

        # Try to infer frequency (time step)
        ts_in.index = pd.to_datetime(ts_in.index).round("s")
        deltaT = pd.infer_freq(ts_in.index)
        assert deltaT is not None, "Could not infer a regular time step"
        if deltaT.isalpha():  # e.g. "H", "T", "S", "D"
            deltaT = "1" + deltaT

        # Assign Core Attributes
        self.ts_in = ts_in
        self.nb_states = nb_states
        self.nb_forecast_profiles = nb_forecast_profiles
        self.deltaT = pd.to_timedelta(deltaT)

        # Generate the transition matrix
        self.state_maps = self.set_states(self.ts_in, self.nb_states)
        ts_disrete = self.rediscretize_ts_in(self.ts_in, self.state_maps)
        self.trans_matrices= self.compute_transition_matrices(ts_disrete, self.state_maps)

        # Transition matrices: dict[time] -> ndarray -- for code optimization purposes
        self.trans_matrices_arr = {t: np.array(mat) for t,mat in self.trans_matrices.items()}

        # State maps: dict[time] -> ndarray of representative values -- for code optimization purposes
        self.state_maps_arr = {
            t: (
                df["value"].to_numpy()
                if "value" in df.columns
                else ((df["lower"] + df["upper"]) / 2).to_numpy()
            )
            for t,df in self.state_maps.items()
        }

        # Clean transition matrix for sum of probability = to 1 and remove NaN
        for t,mat in self.trans_matrices_arr.items():

            # Replace NaN with 0
            mat = np.nan_to_num(mat,nan = 0.0)

            # Compute row sums
            row_sums = mat.sum(axis = 1,keepdims = True)

            # Handle invalid rows (sum = 0)
            invalid_rows = (row_sums.flatten() == 0)
            if np.any(invalid_rows):
                mat[invalid_rows] = 1.0 / mat.shape[1]
                row_sums = mat.sum(axis = 1,keepdims = True)

            # Normalize
            self.trans_matrices_arr[t] = mat / row_sums



    @staticmethod
    def set_states(ts_in: pd.DataFrame,
                   nb_states: dict):

        # Extract all unique times of day (e.g. 15-min intervals)
        unique_times = np.unique(ts_in.index.time)

        state_maps = {}

        for t in unique_times:
            # Extract all values at this time of day across days
            values_at_t = ts_in[ts_in.index.time == t].values

            if len(values_at_t) == 0:
                continue

            vmin,vmax = values_at_t.min(),values_at_t.max()
            if vmax - vmin < 1e-3:
                state_df = pd.DataFrame({
                    "lower": [vmin],
                    "upper": [vmax],
                    "value": [(vmin + vmax) / 2]
                },index = [0]
                )
                state_df.index.name = 'state'
                state_maps[t] = state_df
                continue

            # Discretize into quantile bins (equal frequency)
            bins = pd.cut(values_at_t, bins = nb_states, retbins = True)[1]

            # Build a DataFrame describing states for this time
            state_df = pd.DataFrame({
                "lower": bins[:-1],
                "upper": bins[1:],
                "value": (bins[:-1] + bins[1:]) / 2
            }, index=range(len(bins)-1) )

            state_maps[t] = state_df

        return state_maps

    @staticmethod
    def rediscretize_ts_in(ts_in: pd.DataFrame,
                           state_maps: dict):

        ts_out = pd.DataFrame(index = ts_in.index, columns = ["state","value"])

        # Loop over all timestamps

        for timestamp,val in ts_in.items():

            t = timestamp.time()
            df_states = state_maps[t]
            lowers = df_states["lower"].to_numpy()
            uppers = df_states["upper"].to_numpy()
            reps = df_states.get("value",np.arange(len(df_states))).to_numpy()

            # Find which state the value belongs to
            state_idx = np.searchsorted(uppers,val,side = "right")
            state_idx = min(max(state_idx,0),len(df_states) - 1)  # clip to valid range

            ts_out.at[timestamp,"state"] = state_idx
            ts_out.at[timestamp,"value"] = val

        ts_out["state"] = ts_out["state"].astype(int)
        ts_out["value"] = ts_out["value"].astype(float)

        return ts_out

    @staticmethod
    def compute_transition_matrices(ts_discrete: pd.DataFrame,
                                    state_maps: dict):

        # Extract states array
        if isinstance(ts_discrete,pd.DataFrame):
            if "state" not in ts_discrete.columns:
                raise KeyError("ts_discrete DataFrame must have a column 'state'")
            ts_states = ts_discrete["state"].to_numpy()
        else:
            ts_states = ts_discrete.to_numpy()

        ts_index = ts_discrete.index
        n = len(ts_states)

        # Determine unique time-of-day slots
        times = np.array([t.time() for t in ts_index])
        unique_times = np.unique(times)

        # Initialize empty transition matrices dynamically
        trans_matrices = {}
        for t in unique_times:
            # Max state observed at this time step
            idxs = np.where(times == t)[0]
            n_states = ts_states[idxs].max() + 1
            trans_matrices[t] = np.zeros((n_states,n_states),dtype = float)

        # Fill in counts
        for i in range(n - 1):
            t_curr = ts_index[i].time()
            s_curr = ts_states[i]
            s_next = ts_states[i + 1]

            # Clip indices to matrix size in case of variable states
            mat = trans_matrices[t_curr]
            s_curr = min(s_curr,mat.shape[0] - 1)
            s_next = min(s_next,mat.shape[1] - 1)

            mat[s_curr,s_next] += 1

        # Normalize to probabilities
        for t,mat in trans_matrices.items():
            row_sums = mat.sum(axis = 1,keepdims = True)
            row_sums[row_sums == 0] = 1  # avoid division by zero
            trans_matrices[t] = mat / row_sums

        return trans_matrices


    @staticmethod
    def plot_transition_heatmap(trans_matrices,
                                state_maps,
                                hour=0,minute=0
                                ):

        # Convert to Python time object
        t = time(hour = hour,minute = minute)

        if t not in trans_matrices:
            raise ValueError(f"No transition matrix found for {t}")

        mat = trans_matrices[t]

        # Get representative state values
        if isinstance(state_maps,dict):
            states_df = state_maps[t]
        else:
            states_df = state_maps  # assume single DataFrame

        # Use 'value' column if exists, else midpoint of lower/upper
        if 'value' in states_df.columns:
            state_values = states_df['value'].to_numpy()
        else:
            state_values = (states_df['lower'] + states_df['upper']) / 2

        plt.figure(figsize = (8,6))
        sns.heatmap(mat.T,annot = False,cmap = "Blues")
        plt.title(f"Transition probabilities at {t}")

        plt.xlabel("Current state")
        plt.ylabel("Next state")

        # Set tick labels to representative values
        # Number of ticks you want to keep
        n_ticks = 6

        # Choose evenly spaced indices
        x_idx = np.linspace(0,len(state_values) - 1,n_ticks,dtype = int)
        y_idx = np.linspace(0,len(state_values) - 1,n_ticks,dtype = int)

        # Apply only those ticks
        plt.xticks(x_idx + 0.5,np.round(np.array(state_values)[x_idx],3),rotation = 45,ha = "right")
        plt.yticks(y_idx + 0.5,np.round(np.array(state_values)[y_idx],3),rotation = 0)

        plt.show()

    def generate_profiles(self,
                          n_profiles: int = 1,
                          start_hour: int = 0,
                          start_minute: int = 0,
                          start_value: float = 0.0,
                          n_steps: int = 1,
                          ):


        # Check that (hour, minute) is valid
        start_time = time(start_hour,start_minute)
        if start_time not in self.state_maps.keys():
            raise ValueError(
                f"Start time ({start_hour:02d}:{start_minute:02d}) not in state_maps."
            )

        # Build a dummy datetime for timestamp arithmetic
        start_time = datetime(2025,1,1,start_hour,start_minute)

        # Compute timestamps
        timestamps = [start_time + i * self.deltaT for i in range(n_steps)]

        # Preallocate output
        profiles = np.zeros((n_steps, n_profiles))

        # ---- INITIAL STATE ----
        t0 = start_time.time()
        state_vals = self.state_maps_arr[t0]
        diffs = np.abs(state_vals - start_value)

        # if diffs.min() > 1e-6:  # no exact state
        #     warnings.warn(
        #         f"Start value {start_value} not in feasible states at {t0}, "
        #         "using nearest."
        #     )

        start_state = int(np.argmin(diffs))

        # Initial assignment
        profiles[0, :] = state_vals[start_state]

        # ---- SIMULATION ----
        current_states = np.full(n_profiles, start_state, dtype=int)

        for i in range(1, n_steps):
            t = timestamps[i - 1].time()
            mat = self.trans_matrices_arr[t]
            state_vals = self.state_maps_arr[t]

            n_states = mat.shape[0]

            # If current_states has some values >= n_states, clip
            current_states = np.clip(current_states, 0, n_states - 1)

            # Sample next states for all scenarios
            new_states = [
                np.random.choice(n_states, p=mat[s] / mat[s].sum())
                for s in current_states
            ]
            current_states = np.array(new_states)

            # Assign representative values
            profiles[i, :] = state_vals[current_states]

        # Convert to DataFrame
        # scenarios_df = pd.DataFrame(
        #     profiles,
        #     index=timestamps,
        #     columns=[f"scen_{j}" for j in range(n_scenarios)],
        # )

        return profiles


    def generate_forecast_profiles(self,
                                  n_profiles,
                                  start_time,
                                  duration_minutes,
                                  reference=None,
                                  metric="nrmse"
                                  ):
        """
        Generate forecast scenarios and compute errors against a reference time series.
        """
        # Generate scenarios first
        total_duration = timedelta(minutes=duration_minutes)
        t0 = start_time - self.deltaT
        tend = t0 + pd.Timedelta(minutes = duration_minutes)
        n_steps = int(total_duration / self.deltaT) + 1

        # --- Reference time series ---
        if reference is None:
            if self.ts_in is None:
                raise ValueError("No reference series provided and self.ts_in is None")

            # Redundant check for object attribute
            validate_timeseries(self.ts_in)

            if not (t0 in self.ts_in.index and tend in self.ts_in.index):
                raise ValueError(f"Reference ts_in must cover range {t0} to {tend}")

            # Extract reference horizon (exclude initial known value)
            reference = self.ts_in.loc[t0:tend]

        else:

            #Check that the input reference is a time serie
            validate_timeseries(reference)

            # Check required range
            if not (t0 in reference.index and tend in reference.index):
                raise ValueError(f"Reference must cover range {t0} to {tend}")

            # Extract correct slice
            reference = reference.loc[t0:tend]

        start_value = reference.iloc[0]

        profiles = self.generate_profiles(n_profiles,
                                          t0.hour,
                                          t0.minute,
                                          start_value,
                                          n_steps)

        # --- Retaylor to fit the forecast hoziron ---
        profiles=profiles[1:,:]
        reference = reference.iloc[1:]

        # Convert to pandas objects
        timestamps = reference.index

        # Choose metric
        metric_func = {"nrmse": nrmse,"nmae": nmae, 'eof': eof}[metric.lower()]

        # Compute errors per scenario
        errors = np.array([metric_func(reference.values,profiles[:,i]) for i in range(n_profiles)])

        profiles_df = pd.DataFrame(profiles,index = timestamps,
                                   columns = [f"scen_{i}" for i in range(n_profiles)]
                                   )

        return profiles_df, reference, errors

    def select_profile(self,
                       profiles,
                       errors,
                       target_error = 0,
                       selection = 'closest'):

        if selection not in {'closest','median'}:
            raise ValueError("selection option for forecast profile is not valid -- must be median or closest")

        if selection == 'closest' :
            # select forecast profile closest to the desired error value
            diffs = np.abs(errors - target_error)
            best_idx = np.argmin(diffs)

        elif selection == 'median':
            # select closer to the median value
            median_error = np.median(errors)
            diffs = np.abs(errors - median_error)
            best_idx = np.argmin(diffs)

        best_profile = profiles.iloc[:,best_idx]
        error_profile = errors [best_idx]

        return best_profile, error_profile

    def forecast(self,
                start_time : datetime,
                duration_minutes : int,
                target_error : float,
                reference : pd.Series = None,
                n_profiles : int = 300,
                metric : str = "nrmse",
                selection : str = 'closest'  #selection oprion before forecast tuning
                ):

        if n_profiles is None :
            n_profiles=self.nb_forecast_profiles

        #Generate forecast profiles
        profiles, reference, errors = self.generate_forecast_profiles(n_profiles = n_profiles,
                                                                    start_time = start_time,
                                                                    duration_minutes = duration_minutes,
                                                                    reference = reference,
                                                                    metric = metric)

        # Select profile
        best_profile, _ = self.select_profile(profiles, errors,
                                           target_error = target_error,
                                           selection = selection)

        #select and apply the morphing function to the selected forecast
        morph_func = {"nrmse": morph_nrmse,"nmae": morph_nmae, 'eof': morph_eof}[metric.lower()]
        tuned_profile = morph_func(reference.values,best_profile.values,target_error)
        tuned_profile = pd.Series(tuned_profile,index = reference.index)

        return reference, tuned_profile




