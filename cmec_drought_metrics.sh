#!/bin/bash

# Change file names as needed for obs, wgt, and shp
testpath=$CMEC_MODEL_DATA/
obspath=$CMEC_OBS_DATA/obs_precip_2.nc
wgtpath=$CMEC_MODEL_DATA/interpolated_pr_Amon_E3SM-1-1_historical_r1i1p1f1_gr_187001-200912.nc
shppath=$CMEC_OBS_DATA/HU/WBDHU2.shp
outpath=$CMEC_WK_DIR/
logpath=${outpath}/drought_metrics_log.txt

cd $CMEC_WK_DIR
echo "Running drought metrics"
# Set region name in this command
python $CMEC_CODE_DIR/drought_metrics.py -test_path $testpath -obs_path $obspath -wgt_path $wgtpath -hu_name "Ohio Region" -shp_path $shppath -out_path $outpath >> $logpath
echo "Creating CMEC output"
# Set region name in this command
python $CMEC_CODE_DIR/dm_cmec.py -test_path $testpath -obs_path $obspath -log_path logpath -hu_name "Ohio Region" -out_path $outpath