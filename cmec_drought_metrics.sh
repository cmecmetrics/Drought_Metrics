#!/bin/bash

# Change file names as needed for region, obs, wgt, and shp
region="Fake region"
obspath=$CMEC_OBS_DATA/obs_precip_2.nc
wgtpath=$CMEC_MODEL_DATA/interpolated_pr_Amon_E3SM-1-1_historical_r1i1p1f1_gr_187001-200912.nc
shppath=$CMEC_OBS_DATA/HU/WBDHU2.shp
testpath=$CMEC_MODEL_DATA/
outpath=$CMEC_WK_DIR/
logpath=${outpath}/drought_metrics_log.txt

cd $CMEC_WK_DIR
echo "Running drought metrics"
python $CMEC_CODE_DIR/drought_metrics.py -test_path $testpath -obs_path $obspath -wgt_path $wgtpath -hu_name "$region" -shp_path $shppath -out_path $outpath >> $logpath

# Make cmec outputs if drought metrics succeeds
if [[ $? = 0 ]]; then
    echo "Creating CMEC output"
    python $CMEC_CODE_DIR/dm_cmec_outputs.py -test_path $testpath -obs_path $obspath -log_path $logpath -hu_name "$region" -out_path $outpath
else
    echo "Failure in drought_metrics.py"
fi
