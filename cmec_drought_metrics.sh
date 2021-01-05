#!/bin/bash

# Change file names as needed for region, obs, and shp.
# See settings.json for more information about input data.
region="New England Region"
obspath=${CMEC_OBS_DATA}/precip.V1.0.mon.mean.nc
shppath=${CMEC_OBS_DATA}/HU/WBDHU2.shp
testpath=${CMEC_MODEL_DATA}/
outpath=${CMEC_WK_DIR}/
logpath=${outpath}/drought_metrics_log.txt

# Optional settings (Add or delete flags from L22 drought_metrics.py command as needed):
obspr="precip" # -obs_pr
intrp=True # -interpolation
wgtpath=${CMEC_MODEL_DATA}/interpolated_pr_Amon_E3SM-1-1_historical_r1i1p1f1_gr_187001-200912.nc # -wgt_path
pfa=${CMEC_CODE_DIR}/output_principal_metrics_column_defined

cd $CMEC_WK_DIR
echo "Running drought metrics"
echo "region: "$region
echo "obs path: "$obspath
echo "model path: "$testpath
python $CMEC_CODE_DIR/drought_metrics.py -test_path $testpath -obs_path $obspath -hu_name "$region" -shp_path $shppath -out_path $outpath -wgt_path $wgtpath -obs_pr $obspr -interpolation $intrp -pfa $pfa >> $logpath

# Make cmec outputs if drought metrics succeeds
if [[ $? = 0 ]]; then
    echo "Creating CMEC output"
    python $CMEC_CODE_DIR/dm_cmec_outputs.py -test_path $testpath -obs_path $obspath -log_path $logpath -hu_name "$region" -out_path $outpath
    # Remove regridding files
    rm ${outpath}/conservative_*.nc
else
    echo "Failure in drought_metrics.py"
fi
