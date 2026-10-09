#!/bin/bash
# Convert private 2024 NANOEDMAODSIM files to flat NANOAODSIM.
# Usage: convert_job.sh <folder> <comma-separated file indices>
# Files already present at the destination are skipped (safe to resubmit).
FOLDER=$1; IDXS=$2
SRC_BASE=root://cmseos.fnal.gov//eos/uscms/store/group/lnujj/VVJJ_Private_Production_2024/${FOLDER}
DST_HOST=root://eosuser.cern.ch
DST_DIR=/eos/cms/store/group/phys_smp/jinw/2024_VBS_sig/${FOLDER}

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
scramv1 project CMSSW CMSSW_15_0_17 >/dev/null || exit 1
pushd CMSSW_15_0_17/src >/dev/null && eval "$(scramv1 runtime -sh)" && popd >/dev/null
xrdfs ${DST_HOST} mkdir -p ${DST_DIR}

nfail=0
for IDX in ${IDXS//,/ }; do
  DST=${DST_DIR}/nanoaod_${IDX}.root
  if xrdfs ${DST_HOST} stat ${DST} >/dev/null 2>&1; then echo "exists: ${DST}"; continue; fi
  rm -f in.root flat.root
  if xrdcp -s -f ${SRC_BASE}/nanoaod_${IDX}.root in.root \
     && cmsRun edm2flat_cfg.py inputFiles=file:in.root outputFile=file:flat.root >cmsrun.log 2>&1 \
     && xrdcp -s -f flat.root ${DST_HOST}/${DST}; then
    echo "done: ${FOLDER} ${IDX}"
  else
    echo "FAILED: ${FOLDER} ${IDX}"; tail -5 cmsrun.log; nfail=$((nfail+1))
    xrdfs ${DST_HOST} rm ${DST} >/dev/null 2>&1
  fi
done
rm -f in.root flat.root
exit ${nfail}
