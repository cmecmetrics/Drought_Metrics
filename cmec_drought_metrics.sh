#!/bin/bash

# Needed for local MPI issues
export MPICH_INTERFACE_HOSTNAME=localhost

cd ${CMEC_CODE_DIR}
echo "Running drought metrics"
python drought_metrics.py ${CMEC_CONFIG_DIR}/cmec.json >> ${CMEC_WK_DIR}/drought_metrics_log.txt

# Make cmec outputs if drought metrics succeeds
if [[ $? = 0 ]]; then
    echo "Creating CMEC output"
    python dm_cmec_outputs.py ${CMEC_CONFIG_DIR}/cmec.json
    # Remove regridding files
    if test -f "${CMEC_WK_DIR}/conservative_*.nc"; then
        rm ${CMEC_WK_DIR}/conservative_*.nc
    fi
else
    echo "Failure in drought_metrics.py"
fi
