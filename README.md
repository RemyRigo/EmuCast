# EmuCast

**EmuCast** is a Python package for time series forecasting emulation.  
It includes example datasets and notebooks for demonstration, all embedded in the package.

---

## Installation

You can install directly from GitHub:

```bash
pip install git+https://github.com/<YourUsername>/EmuCast.git
```

Or, for development (editable) mode:

git clone https://github.com/<YourUsername>/EmuCast.git
cd EmuCast
pip install -e .

### Import the main class and load example data

```python
# Import the main class and data loader
from emucast import ForecastEmulator
from emucast.data import load_example_data 

# Initialize the emulator with the sample data
emulator = ForecastEmulator(sample_data)

# Generate Forecast profiles
from datetime import datetime
ref, pred = emulator(start_time=atetime(2019,1,9,0,0),
                     duration_minutes=60*24,
                     target_error=20)
