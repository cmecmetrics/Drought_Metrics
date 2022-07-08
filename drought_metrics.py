"""drought-metrics.py

Driver that runs the drought metrics evaluation.

Use:
python drought_metrics.py settings_file.yaml

Parameters:
-----------
    Parameters are stored in a yaml file provided as the single argument
    to this script.

    test_path : str
        GCM file directory. Can contain multiple files.
    obs_path : str
        Path to observations file
    wgt_path : str
        Weightfile path
    hu_name : str
        Name of evaluation region
    shp_path : str
        Path to regions shapefile
    out_path : str
        Path to output directory (default '.')
    interpolation : bool
        True to perform interpolation (default False)
    pfa : str
        Path to principal metrics file

"""
import json
import os
import sys
from evaluation import evaluation

# Set defaults based on demo data
hu_name = "New England Region"
interpolation = True
run_pfa = False
shp_path = "./HU/WBDHU2.shp"
wgt_path = "./data/weightfile/interpolated_pr_Amon_E3SM-1-1_historical_r1i1p1f1_gr_187001-200912.nc"
pfa = "./output_principal_metrics_column_defined"
obs_file_name = "precip.V1.0.mon.mean.nc"

# Get CMEC environment variables
test_path = os.getenv("CMEC_MODEL_DATA")
obs_path = os.getenv("CMEC_OBS_DATA")
out_path = os.getenv("CMEC_WK_DIR")

# Get user settings from cmec interface
user_settings_json = os.path.expandvars('$CMEC_CONFIG_DIR/cmec.json')
try:
    with open(user_settings_json) as config_file:
        user_settings = json.load(config_file).get("Drought_Metrics")
    # Get any environment variables and check settings type
    for setting in user_settings:
        if isinstance(user_settings[setting], str):
            user_settings[setting] = os.path.expandvars(user_settings[setting])
    # User settings to global variables
    globals().update(user_settings)
    obs_path = os.path.join(obs_path, obs_file_name)
except json.decoder.JSONDecodeError:
    print("*** Could not load settings from " + str(user_settings_json) + ". File may not be valid JSON. Using defaults ***\n")
    obs_path = os.path.join(obs_path, obs_file_name)

# Loop over all files under TEST_PATH and conduct data analysis.
x = evaluation()
x.evaluate_multi_model(
    test_path, obs_path, wgt_path, hu_name,
    shp_path, out_path, interpolation=interpolation)

# Conduct the PFA to get Principal Metrics within the region defined.
# The column names of pricipal metrics are saved at 'output_principal_metrics_column_defined'.
if run_pfa:
    print("Running Principal Features Analysis")
    pfa_path = out_path + "/output_principal_metrics_column_defined"
    x.PFA(out_path=out_path, column_name=pfa_path)
else:
    print("Using principal features from " + str(pfa))
    pfa_path = pfa

# Make sure get the name of pricipal metrics defined by PFA firstly.
# (Here I provide a template named 'output_principal_metrics_column_defined').
# Select the principal metrics defined at 'output_principal_metrics_column_defined' and make plots
x.PM_selection(out_path=out_path, column_name=pfa_path)
x.result_analysis(out_path=out_path, column_name=pfa_path, upper_limit=2)
try:
    x.make_taylor_diagram(out_path)
except Exception as e:
    print("Could not generate Taylor Diagram.")
    print(e)

