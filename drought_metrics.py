#!/usr/bin/python
"""drought-metrics.py

Driver that runs the drought metrics evaluation.

Parameters:
-----------
    test_path : str
        GCM file directory. Can contain multiple files.
    test_pr : str
        Precipitation variable name in test files (default 'pr')
    obs_path : str
        Path to observations file
    obs_pr : str
        Precipitation variable name in observations (default 'pr')
    wgt_path : str
        Weightfile path
    hu_name : str
        Name of evaluation region
    shp_path : str
        Path to regions shapefile
    out_path : string
        path to output directory (default '.')

"""
import argparse
import os
from evaluation import evaluation

parser = argparse.ArgumentParser(description='Get parameters for drought metrics')
parser.add_argument('-test_path', help='GCM file directory')
parser.add_argument('-test_pr', default='pr', help='Precipitation variable name')
parser.add_argument('-obs_path', help='Observational file')
parser.add_argument('-obs_pr', default='pr', help='Precipitation  variable name')
parser.add_argument('-wgt_path', help='Weightfile for interpolation')
parser.add_argument('-hu_name', help='Evaluation region in shapefile')
parser.add_argument('-shp_path', help='Shapefile path')
parser.add_argument('-out_path', default='.', help='Output directory')

args = parser.parse_args()

# Loop over all files under TEST_PATH and conduct data analysis.
x = evaluation()
x.evaluate_multi_model(
    args.test_path, args.test_pr, args.obs_path, args.obs_pr, args.wgt_path,
    args.hu_name, args.shp_path, args.out_path, interpolation=False)

# Conduct the PFA to get Principal Metrics within the region defined.
# The column names of pricipal metrics are saved at 'output_principal_metrics_column_defined'.
if not os.path.isfile(args.out_path + '/output_principal_metrics_column_defined'):
    x.PFA(args.out_path)

# Make sure get the name of pricipal metrics defined by PFA firstly.
# (Here I provide a template named 'output_principal_metrics_column_defined').
# Select the principal metrics defined at 'output_principal_metrics_column_defined' and make plots
x.PM_selection(args.out_path)
x.result_analysis(out_path=args.out_path, upper_limit=2)
x.make_taylor_diagram(args.out_path)
