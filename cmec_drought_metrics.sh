#!/bin/bash

# Change file names as needed for region, obs, wgt, and shp.
# See settings.json for more information about input data.
# Optional settings are commented out and not used for this example
#interpolation=True
region="New England Region"
obspath=$CMEC_OBS_DATA/precip.V1.0.mon.mean.pr.nc
#wgtpath=$CMEC_MODEL_DATA/interpolated_pr_Amon_E3SM-1-1_historical_r1i1p1f1_gr_187001-200912.nc
shppath=$CMEC_OBS_DATA/HU/WBDHU2.shp
testpath=$CMEC_MODEL_DATA/
outpath=$CMEC_WK_DIR/
logpath=${outpath}/drought_metrics_log.txt

cd $CMEC_WK_DIR
echo "Running drought metrics"
python $CMEC_CODE_DIR/drought_metrics.py -test_path $testpath -obs_path $obspath -hu_name "$region" -shp_path $shppath -out_path $outpath >> $logpath

# Make cmec outputs if drought metrics succeeds
if [[ $? = 0 ]]; then
    echo "Creating CMEC output"
    python $CMEC_CODE_DIR/dm_cmec_outputs.py -test_path $testpath -obs_path $obspath -log_path $logpath -hu_name "$region" -out_path $outpath
else
    echo "Failure in drought_metrics.py"
fi
