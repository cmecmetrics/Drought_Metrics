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

To see all options:  
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

## Data  

### Observations  
This analysis was designed to use the CPC Unified Gauge-Based Analysis of Daily Precipitation over CONUS. This dataset is available from [NOAA PSL](https://psl.noaa.gov/data/gridded/data.unified.daily.conus.html). For best results, at least 50 years of data should be used.  

#### Command line options  
The following options are used when running drought_metrics.py. If using the cmec interface, these changes should be made in Drought_Metrics/cmec_drought_metrics.sh.  

**Precipitation variable not called "pr"**
Use the flag `-obs_pr` to set the variable name.  
`python drought_metrics.py -obs_pr "precip variable name" <other options>`.  

**Different grids**  
If the observation grid is different from the model grid, the observations must be interpolated to the model grid. The weightfile is a netCDF file that has the model grid.  
`python drought_metrics.py -interpolation True -wgt_path <path to model file> <other options>`.

### Models
Monthly precipitation output should be in [CF-compliant](https://cfconventions.org/) netCDF files that conform to the standards for the CMIP6 project. Required dimensions are latitude, longitude, time, and precipitation flux "pr". The published analysis uses CMIP6 output.  

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
