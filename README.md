# Drought_Metrics  
Including codes about reading, interpolating, masking NetCDF data of GCMs and calculating metrics to evaluate models' performance on droughts.  

## How to run this package  
Use git to clone this repository. There are two ways to run the evaluation.  

### Environment  
If using conda, an environment can be created using drought_metrics.yml:  
`conda env create -f drought_metrics.yml`  
This file can also be referenced if creating an environment in another manner.  

### Run from Drought_Metrics directory  
`cd` to the cloned Drought_Metrics directory. Use the command line to run drought_metrics.py:  
`python drought_metrics.py <-options>`  

To see all options (help):  
`python drought_metrics.py -h`  

### Run with CMEC driver (recommended)  
Follow the instructions here to install cmec-driver: https://github.com/cmecmetrics/cmec-driver  
From the cmec-driver directory:  
- Create the following directories: "obs", "model", and "output"  
- Copy your model data to model/Drought_Metrics  
- Copy your observations and shapefiles to obs/Drought_Metrics  
- Change the filenames in \<path to Drought Metrics\>/cmec_drought_metrics.sh as needed  
- Run the following commands:  
`python src/cmec-driver.py register <path to Drought Metrics>`  
`python src/cmec-driver.py run -obs obs/Drought_Metrics model/Drought_Metrics output Drought_Metrics`  

Your results will be written to cmec-driver/output/Drought_Metrics. Open cmec-driver/output/Drought_Metrics/index.html with your browser to view the generated metrics files and plots.  

Results will be overwritten the next time cmec-driver is run with the same output path. To save results, either copy the output/Drought_Metrics folder to a new location or provide a unique output folder path when running cmec-driver.  

### Required flags
The following flags are always required to run the Drought Metrics package:
- test_path: Model data directory
- obs_path: Observation file path
- hu_name: Name of evaluation region
- shp_path: Watershed boundary file path

### Optional flags
Use these flags to specify additional files and settings:
- test_pr: model precipitation variable name (default "pr")
- obs_pr: observation precipitation variable name (default "pr")
- wgt_path: netcdf file with grid for interpolation (default '')
- out_path: output directory path (default '.')
- interpolation: True to interpolate inputs (default False)
- pfa: Path to existing principal metrics results (default None)

## Data  

### Observations  
This analysis was designed to use the CPC Unified Gauge-Based Analysis of Daily Precipitation over CONUS. This dataset is available from [NOAA PSL](https://psl.noaa.gov/data/gridded/data.unified.daily.conus.html). For best results, at least 50 years of data should be used.  

### Models
Monthly precipitation output should be in [CF-compliant](https://cfconventions.org/) netCDF files that conform to the standards for the CMIP6 project. Required dimensions are latitude, longitude, time, and precipitation flux "pr". The published analysis uses CMIP6 output.  

### Principal Metrics
The user has the option to reuse the results of a previous Principal Metrics analysis or to generate new principal metrics. 

To use the results of an old PFA analysis, set the optional `-pfa` flag to the path for those PFA results (by default named 'output_principal_metrics_column_defined').  
`python drought_metrics.py -pfa path/to/output_principal_metrics_column_defined`  

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

## NOTE  

#### Auspices  
This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. Lawrence Livermore National Laboratory is operated by Lawrence Livermore National Security, LLC, for the U.S. Department of Energy, National Nuclear Security Administration under Contract DE-AC52-07NA27344.  

#### Disclaimer  
This document was prepared as an account of work sponsored by an agency of the United States government. Neither the United States government nor Lawrence Livermore National Security, LLC, nor any of their employees makes any warranty, expressed or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States government or Lawrence Livermore National Security, LLC. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States government or Lawrence Livermore National Security, LLC, and shall not be used for advertising or product endorsement purposes.  
