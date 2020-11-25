#!/bin/bash

testpath=$CMEC_CODE_DIR/data/test/fake_test/
obspath=$CMEC_CODE_DIR/data/obs_precip_2.nc
wgtpath=$CMEC_CODE_DIR/data/test/weightfile/interpolated_pr_Amon_E3SM-1-1_historical_r1i1p1f1_gr_187001-200912.nc
shppath=$CMEC_CODE_DIR/HU/WBDHU2.shp

python $CMEC_CODE_DIR/drought_metrics.py -test_path $testpath -obs_path $obspath -wgt_path $wgtpath -hu_name "New England Region" -shp_path $shppath