"""drought-metrics.py

Driver that runs the drought metrics evaluation.
"""
from evaluation import evaluation

#the GCMs files' directory
#there are only 3 template files provided. We can use any outputs from CMIP5/6.
test_path = './data/test/fake_test/'

#the variable name of precipitation in test data
test_pr_name = 'pr'

# the file of observational data
#observe_path = "CPC_monthly_precipitation_to_1_degree.nc"
#observe_path = "./data/weightfile/interpolated_pr_Amon_E3SM-1-1_historical_r1i1p1f1_gr_187001-200912.nc"
observe_path = "./data/obs_precip_2.nc"

#the variable name of precipitation in observed data
observe_pr_name = 'pr'

# the path of weightfile used in interpolation, if data are already interpolated, set the interpolation=False in x.evaluate_multi_model()\
# here I use a GCMs file with 1*1 degree resolution
#weightfile_path = './data/weightfile/interpolated_pr_Amon_E3SM-1-1_historical_r1i1p1f1_gr_187001-200912.nc'
weightfile_path = './data/test/fake_test/test_precip_1.nc'

#the name of evaluation region in the shapefile
hu_name = 'New England Region'

#the path of shapefile
shp_path = 'HU/WBDHU2.shp'

# Loop over all files under test_path and conduct data analysis
x = evaluation()
x.evaluate_multi_model(test_path,test_pr_name,observe_path,observe_pr_name,weightfile_path,hu_name,shp_path,interpolation=False)

#Conduct the PFA to get Principal Metrics within the region defined. The column names of pricipal metrics are saved at 'output_principal_metrics_column_defined'
# Once we get the name of pricipal metrics defined by PFA. It's no need to run this function again.
#x.PFA()

# Make sure get the name of pricipal metrics defined by PFA firstly. (Here I provide a template named 'output_principal_metrics_column_defined') 
#Select the principal metrics defined at 'output_principal_metrics_column_defined' and make plots 
x.PM_selection()
x.result_analysis(x.principal_metrics)
x.make_taylor_diagram()