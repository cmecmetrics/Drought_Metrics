"""Create CMEC compliant outputs for Drought Metrics package.

Parameters:
-----------
    test_path : str
        GCM file directory. Can contain multiple files.
    obs_path : str
        Path to observations file
    log_path : string
        Path to log file from drought metrics
    hu_name : str
        Name of evaluation region
    out_path : string
        path to output directory
"""
import argparse
from datetime import datetime, timezone
import json
import pandas as pd
from pathlib import Path

def get_env():
    import affine
    import climate_indices
    import ESMF
    import geopandas
    import matplotlib
    import mpl_toolkits.basemap as Basemap
    import numpy as np
    import pandas as pd
    import rasterio
    import scipy
    import sklearn
    import xarray
    import xesmf

    versions = {}
    versions['affine'] = affine.__version__
    versions['climate_indices'] = climate_indices.__version__
    versions['esmpy'] = ESMF.__version__
    versions['geopandas'] = geopandas.__version__
    versions['matplotlib'] = matplotlib.__version__
    versions['Basemap'] = Basemap.__version__
    versions['numpy'] = np.__version__
    versions['pandas'] = pd.__version__
    versions['rasterio'] = rasterio.__version__
    versions['scipy'] = scipy.__version__
    versions['sklearn'] = sklearn.__version__
    versions['xarray'] = xarray.__version__
    versions['xesmf'] = xesmf.__version__
    return versions

def write_output_json(test_path, obs_path, log_path, hu_name, out_path='.'):
    """Create json that describes contents of output directory."""
    # Set output file names and descriptions
    rgn_str = str(hu_name).replace(' ','_')
    # Metrics paths
    m1 = 'all_metrics_in_' + rgn_str + '.json'
    m1d = 'All drought metrics for all regions and models'
    m2 = 'all_metrics_in_' + rgn_str + '_cmec.json'
    m2d = 'All drought metrics for all regions and models in cmec format'
    m3 = 'principal_metrics_in_' + rgn_str + '.json'
    m3d = 'Top metrics for all regions and models'
    m4 = 'principal_metrics_in_' + rgn_str + '_cmec.json'
    m4d = 'Top metrics for all regions and models in cmec format'
    # Data paths
    d1 = 'output_principal_metrics_column_defined'
    d2 = 'taylor_score_in_' + rgn_str + '.pkl'
    # Plot paths
    p1 = 'heatmap_of_principal_metrics_' + rgn_str + '.pdf'
    p1j = 'heatmap_of_principal_metrics_' + rgn_str + '.jpeg'
    p2 = 'output__PFA_in_' + rgn_str + '.pdf'
    p2j = 'output__PFA_in_' + rgn_str + '.jpeg'
    p3 = 'taylor_diagram_' + rgn_str + '.pdf'
    p3j = 'taylor_diagram_' + rgn_str + '.jpeg'
    p4j = 'PCA_explained_variance_' + rgn_str + '.jpeg'

    # Initialize json
    out_json = {
        'index': 'index', 'provenance': {}, 'data': {}, 'plots': {}, 'html': {}, 'metrics': {}}
    # Set json fields
    out_json['provenance'] = {
        'environment': get_env(),
        'modeldata': test_path,
        'obsdata': obs_path,
        'log': log_path,
        'version': '1'}
    out_json['data'] = {
        'taylor score': {
            'filename': d2,
            'long_name': d2.replace('_',' '),
            'description': 'Taylor score'}}
    if Path(out_path + '/' + d1).exists():
        data = {
        'column definitions': {
            'filename': d1,
            'long_name': d1,
            'description': 'column names saved for future evaluation'}}
        out_json['data'].update(data)
    out_json['plots'] = {
        'heatmap pdf': {
            'filename': p1,
            'long_name': p1.replace('_',' ')[:-4],
            'description': 'Heatmap of metrics'},
        'taylor pdf': {
            'filename': p3,
            'long_name': p3.replace('_',' ')[:-4],
            'description': 'Taylor Diagram of test data'},
        'heatmap jpeg': {
            'filename': p1j,
            'long_name': p1j.replace('_',' ')[:-5],
            'description': 'Heatmap of metrics for html'},
        'taylor jpeg': {
            'filename': p3j,
            'long_name': p3j.replace('_',' ')[:-5],
            'description': 'Taylor Diagram of test data for html'},
        'PCA': {
            'filename': p4j,
            'long_name': p4j.replace('_',' ')[:-5],
            'description': 'Principal Components Analysis results'}}
    if Path(out_path + '/' + p2).exists():
        pfa = {
        'PFA jpeg': {
            'filename': p2j,
            'long_name': p2j.replace('_',' ')[:-5],
            'description': 'Results of Principal Features Analysis for html'},
        'PFA pdf': {
            'filename': p2,
            'long_name': p2.replace('_',' ')[:-4],
            'description': 'Results of Principal Features Analysis'}}
        out_json['plots'].update(pfa)

    out_json['html'] = {
        'index': {
            'filename': 'index.html',
            'long_name': 'Index',
            'description': 'navigation page'}}
    out_json['metrics'] = {
        'all': {
            'filename': m1,
            'long_name': m1.replace('_',' ')[:-5],
            'description': m1d},
        'all cmec': {
            'filename': m2,
            'long_name': m2.replace('_',' ')[:-5],
            'description': m2d},
        'prncpl': {
            'filename': m3,
            'long_name': m3.replace('_',' ')[:-5],
            'description': m3d},
        'prncpl cmec': {
            'filename': m4,
            'long_name': m4.replace('_',' ')[:-5],
            'description': m4d}}

    filepath = out_path + '/output.json'
    with open(filepath,'w') as f:
        json.dump(out_json, f, indent=2)

def write_cmec_json(hu_name,out_path='.'):
    """Write cmec formatted metrics json.

    Loads the metrics output, rearranges the metrics,
    adds other required cmec fields, and writes to file.
    """
    rgn_str = str(hu_name).replace(' ','_')
    f = out_path + '/all_metrics_in_' + rgn_str + '.json'
    multi_model_table = pd.read_json(f).transpose()
    cmec_json = {'Dimensions': {},'Results': {}, 'Provenance': ''}
    json_structure = ['hydrologic region', 'model', 'metric']
    region = {hu_name: {}}
    model = {item: {} for item in multi_model_table.columns.tolist()}
    metric = {'Metric A1': {
                'long_name': 'Mean Precip.',
                'description': 'Monthly mean precipitation'},
             'Metric A2': {
                'long_name': 'Mean SPI6',
                'description': 'Mean standardized precipitation index calculated with 6 months accumulative precip'},
             'Metric A3': {
                'long_name': 'Mean SPI36',
                'description': 'Mean standardized precipitation index calculated with 66 months accumulative precip'},
             'Metric B1': {
                'long_name': 'Season Precip.',
                'description': 'Seasonality of precipitation'},
             'Metric B2': {
                'long_name': 'LTMM',
                'description': 'Long term monthly mean normalized by total annual precipitation'},
             'Metric C1': {
                'long_name': 'Frac. Cover.',
                'description': 'Fractional area coverage of drought'},
             'Metric D1': {
                'long_name': 'Dry Frac.',
                'description': 'Proportion of dry months determined by SPI6'},
             'Metric D2': {
                'long_name': 'Dry Count',
                'description': 'Annual number of dry months'},
             'Metric E1': {
                'long_name': 'Intensity',
                'description': 'Intensity based on monthly SPI6'},
             'Metric F1': {
                'long_name': 'Prob. Init.',
                'description': 'Probability of drought initiation'},
             'Metric F2': {
                'long_name': 'Prob. Term.',
                'description': 'Probability of drought termination'},
             'Total Score': {
                'long_name': 'Sum of scores',
                'description': 'Overall score obtained from sum of all principal metrics'},}
    cmec_json['Provenance'] = 'Metrics generated ' + datetime.now(timezone.utc).isoformat()
    cmec_json['Dimensions']['json_structure'] = json_structure
    cmec_json['Dimensions']['hydrologic region'] = region
    cmec_json['Dimensions']['metric'] = metric
    cmec_json['Dimensions']['model'] = model

    # Arrange results hierarchically
    # index[0] is hydrologic region which is unwanted
    ind1 = multi_model_table.index[1]
    ind2 = multi_model_table.index[-1]
    for rgn in region:
        cmec_json['Results'].update({rgn: multi_model_table[ind1:ind2].to_dict()})

    filepath = out_path + '/all_metrics_in_' + rgn_str + '_cmec.json'
    with open(filepath,'w') as f:
        json.dump(cmec_json, f, indent=2)

    # Do similar thing for principal metrics json, reusing above json
    f = out_path + '/principal_metrics_in_' + rgn_str + '.json'
    principal = pd.read_json(f).transpose()
    cmec_json['Results'] = {}
    prnc_metric = {}
    for pm in principal.index:
        if pm != 'hydrologic region':
            prnc_metric[pm] = metric[pm]
    cmec_json['Dimensions']['metric'] = prnc_metric

    ind1 = principal.index[1]
    ind2 = principal.index[-1]
    for rgn in region:
        cmec_json['Results'].update({rgn: principal[ind1:ind2].to_dict()})

    filepath = out_path + '/principal_metrics_in_' + rgn_str + '_cmec.json'
    with open(filepath,'w') as f:
        json.dump(cmec_json, f, indent=2)

def make_html(hu_name, out_path='.'):
    """Create html navigation page."""
    filepath = out_path + '/index.html'

    #Set file description and path
    rgn_str = str(hu_name).replace(' ','_')
    m1d = 'all_metrics_in_' + rgn_str + '.json'
    m1p = str(Path(out_path + '/' + m1d).absolute())
    m2d = 'all_metrics_in_' + rgn_str + '_cmec.json'
    m2p = str(Path(out_path + '/' + m2d).absolute())
    m3d = 'principal_metrics_in_' + rgn_str + '.json'
    m3p = str(Path(out_path + '/' + m3d).absolute())
    m4d = 'principal_metrics_in_' + rgn_str + '_cmec.json'
    m4p = str(Path(out_path + '/' + m4d).absolute())
    p1alt = 'PCA results'
    p1p = str(Path(out_path + '/PCA_explained_variance_' + rgn_str + '.jpeg').absolute())
    p2alt = 'Heatmap of principal metrics'
    p2p = str(Path(out_path + '/heatmap_of_principal_metrics_' + rgn_str + '.jpeg').absolute())
    p3alt = 'The number of principal components versus cumulative variance'
    p3p = str(Path(out_path + '/output__PFA_in_' + rgn_str + '.jpeg').absolute())
    p4alt = 'Taylor diagram'
    p4p = str(Path(out_path + '/taylor_diagram_' + rgn_str + '.jpeg').absolute())


    html = f'<html>\
    <body>\
    <head><title>Drought Metrics {rgn_str}</title></head>\
    <br><h1>Drought Metrics Output for {hu_name}</h1>\
    <br><h2>Metrics files</h2>\
    <br><a href="file:///{m1p}">{m1d}</a>\
    <br><a href="file:///{m3p}">{m3d}</a>\
    <br>\
    <br><h4>CMEC formatted outputs</h4>\
    <br><a href="file:///{m2p}">{m2d}</a>\
    <br><a href="file:///{m4p}">{m4d}</a>\
    <br>\
    <br><h2>Plots</h2>\
    <br><p><img src="file:///{p1p}" alt="{p1alt}"</p>\
    <br>\
    <br><p><img src="file:///{p2p}" alt="{p2alt}"</p>\
    <br>\
    <br><p><img src="file:///{p3p}" alt="{p3alt}"</p>\
    <br>\
    <br><p><img src="file:///{p4p}" alt="{p4alt}"</p>\
    </body>\
    </html>'

    with open(filepath,'w') as out_file:
        out_file.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get parameters for cmec output files')
    parser.add_argument('-test_path', help='GCM file directory')
    parser.add_argument('-obs_path', help='Observational file')
    parser.add_argument('-log_path', help='Log file')
    parser.add_argument('-hu_name', help='Evaluation region in shapefile')
    parser.add_argument('-out_path', help='Output directory')
    args = parser.parse_args()

    test_path = args.test_path
    obs_path = args.obs_path
    log_path = args.log_path
    hu_name = args.hu_name
    out_path = args.out_path

    write_output_json(test_path, obs_path, log_path, hu_name, out_path)
    write_cmec_json(hu_name, out_path)
    make_html(hu_name, out_path)
