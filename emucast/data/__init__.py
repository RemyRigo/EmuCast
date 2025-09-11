import os
import pandas as pd

# Get the directory of the file
file_path = os.path.abspath(__file__)
folder_path = os.path.dirname(file_path)

# Get sample time series

# Power load profile
load_sample_data = pd.read_excel(folder_path+'/load_uk_norm.xlsx', index_col=0)
load_sample_data = load_sample_data['demand_MW']*6
# Power load profile
pv_sample_data = pd.read_excel(folder_path+'/pv_uk_norm.xlsx', index_col=0)
pv_sample_data = pv_sample_data['PV1']*3
# Energy prices profile
price_sample_data = pd.read_excel(folder_path+'/da_prices_fr.xlsx', index_col=0)
price_sample_data = price_sample_data['price(€/MWh)']