# Drought_Metrics
Including codes about reading, interpolating, masking NetCDF data of GCMs and calculating metrics to evaluate models' performance on droughts.

## How to run this package
Use git to clone this repository. There are two ways to run the evaluation.

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
- Change the filenames in <path to Drought Metrics>/cmec_drought_metrics.sh as needed
- Run the following commands:
`bin/cmec-driver register <path to Drought Metrics>`
`bin/cmec-driver run obs/Drought_Metrics model/Drought_Metrics output Drought_Metrics`

## Templates
Tempelates of observational data, model data, weightfile used in interpolation, shapefile defining the evaluation regions and plots are provided.
