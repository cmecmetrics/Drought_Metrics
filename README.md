# Drought_Metrics
Including codes about reading, interpolating, masking NetCDF data of GCMs and calculating metrics to evaluate models' performance on droughts.

## Environment
If using conda, an environment can be created using drought_metrics.yml:
`conda env create -f drought_metrics.yml`

## How to run this package
Use git to clone this repository. The CMEC driver will run the module from this codebase.

### CMEC driver
Follow the instructions here to install cmec-driver: https://github.com/cmecmetrics/cmec-driver
From the cmec-driver directory:
- Create the following directories: "obs", "model", and "output"
- Copy (or link) your model files to model/
- Copy (or link) your observational file to obs/
- Register the module:
  `python cmec-driver.py register <path to Drought Metrics repository>`
- (Optional) Edit settings in config/cmec.json
- Run the module:
`python cmec-driver.py run -obs obs/ model/ output/ Drought_Metrics`

For testing, users may use the model files from Drought_Metrics/data/test. The default observations can be downloaded from the source described in the **Observations** section below.

Drought Metrics results will be written to cmec-driver/output/Drought_Metrics. Open cmec-driver/output/Drought_Metrics/index.html with your browser to view the generated metrics files and plots.

Results will be overwritten the next time cmec-driver is run with the same output path. To save results, either copy the output/Drought_Metrics folder to a new location or provide a unique output folder when running cmec-driver.

#### Customizations
The user settings can be modified in cmec-driver/config/cmec.json after the package is registered.

The directory names "obs", "model", and "output" are recommended with cmec-driver but not required. The observations file should be located in the observations folder passed to cmec-driver, and the model files should be located in the model folder passed to cmec-driver.

### User settings
The following files and settings can be changed by the user in cmec.json:
- obs_file_name: Observation file name (default: 'precip.V1.0.mon.mean.nc')
- hu_name: Name of evaluation region (default: 'New England Region')
- shp_path: Watershed boundary file path (default: 'HU/WBDHU2.shp')
- wgt_path: netcdf file with grid for interpolation (default 'data/weightfile/interpolated_pr_Amon_E3SM-1-1_historical_r1i1p1f1_gr_187001-200912.nc')
- interpolation: true to interpolate inputs to same grid (default true)
- pfa: Path to existing principal metrics results (default 'output_principal_metrics_column_defined')
- run_pfa: Set to true to generate new PFA analysis (default false)

## Data

### Observations
This analysis was designed to use the CPC Unified Gauge-Based Analysis of Daily Precipitation over CONUS. This dataset is available from [NOAA PSL](https://psl.noaa.gov/data/gridded/data.unified.daily.conus.html) (download the monthly mean file "precip.V1.0.mon.mean.nc"). For best results, at least 50 years of data should be used.

### Models
Monthly precipitation output should be in [CF-compliant](https://cfconventions.org/) netCDF files that conform to the standards for the CMIP6 project. Required dimensions are latitude, longitude, time, and precipitation flux. The Drought Metrics package assumes that there will be a single precipitation variable in each file and that its variable name contains the letters "pr".

### Principal Metrics
The user has the option to reuse the results of a previous Principal Features Analysis or to generate new principal metrics.

To use the results of an old PFA analysis, set the optional `pfa` key to the path for those PFA results (by default named 'output_principal_metrics_column_defined').

By default, cmec_drought_metrics.sh expects the principal metrics file to be placed in the Drought Metrics code folder.

### Watershed Boundaries
This analysis requires a shapefile containing watershed boundaries. The boundary features must contain the fields "Name" and "geometry".

The example boundaries provided in Drought_Metrics/HU are for watersheds in the US at the 2-digit level, based on data obtained from the U.S. Geological Survey and the U.S. Department of Agriculture, Natural Resources Conservation Service. More information [here](https://www.usgs.gov/core-science-systems/ngp/national-hydrography/watershed-boundary-dataset?qt-science_support_page_related_con=4#qt-science_support_page_related_con).

## Contents
### Scripts
evaluation.py: Evaluation class which computes drought metrics.
drought_metrics.py: Driver for evaluation.py
dm_cmec_outputs.py: Functions for creating CMEC outputs
cmec_drought_metrics.sh: CMEC driver script

### Settings
settings.json: CMEC settings file
drought_metrics.yml: Environment file

### Templates
Tempelates of observational data, model data, weightfile used in interpolation, shapefile defining the evaluation regions and plots are provided. More information can be found in settings.json.

### Example outputs
The folder 'example_output' contains the results from running the Drought Metrics code via cmec-driver while reusing existing principal metrics.

## NOTE

#### Auspices
This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. Lawrence Livermore National Laboratory is operated by Lawrence Livermore National Security, LLC, for the U.S. Department of Energy, National Nuclear Security Administration under Contract DE-AC52-07NA27344.

#### Disclaimer
This document was prepared as an account of work sponsored by an agency of the United States government. Neither the United States government nor Lawrence Livermore National Security, LLC, nor any of their employees makes any warranty, expressed or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States government or Lawrence Livermore National Security, LLC. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States government or Lawrence Livermore National Security, LLC, and shall not be used for advertising or product endorsement purposes.
