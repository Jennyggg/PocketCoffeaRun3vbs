#!/bin/bash
# hadd_job.sh <job_XX.json>
# Re-run one missing hadd from the FNAL skim (semilep_vbs_skim_run3_v2):
# reads the inputs over xrootd, merges into the local scratch dir, checks the
# number of Events against hadd.json, then copies the result to CERN EOS.
set -euo pipefail
JOB_JSON="$1"

set +u   # the LCG setup script uses unset variables
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u
voms-proxy-info --exists || { echo "ERROR: no valid grid proxy"; exit 2; }

WORK="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}/hadd_$$"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

read -r OUTPUT NEVENTS < <(python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['output'], j['nevents'])" "$JOB_JSON")
python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['inputs']))" "$JOB_JSON" > "$WORK/inputs.txt"
LOCAL="$WORK/$(basename "$OUTPUT")"
echo "Merging $(wc -l < "$WORK/inputs.txt") files -> $OUTPUT (expect $NEVENTS events)"

# retry the whole merge once in case of a transient xrootd failure
for attempt in 1 2; do
    # no -k: an unreadable input must fail the job, not be silently skipped
    if hadd -f "$LOCAL" @"$WORK/inputs.txt"; then break; fi
    [[ $attempt == 2 ]] && { echo "ERROR: hadd failed twice"; exit 3; }
    echo "hadd failed, retrying in 60 s"; sleep 60
done

NFOUND=$(python3 -c "
import ROOT,sys
f=ROOT.TFile.Open(sys.argv[1]); t=f.Get('Events')
print(int(t.GetEntries()) if t else -1)" "$LOCAL")
if [[ "$NFOUND" != "$NEVENTS" ]]; then
    echo "ERROR: merged file has $NFOUND events, expected $NEVENTS"; exit 4
fi
echo "Event count OK ($NFOUND)"

xrdcp -f -p "$LOCAL" "$OUTPUT"
REMOTE_SIZE=$(xrdfs "$(echo "$OUTPUT" | cut -d/ -f1-3)" stat "/$(echo "$OUTPUT" | cut -d/ -f4-)" | awk '/Size:/{print $2}')
LOCAL_SIZE=$(stat -c %s "$LOCAL")
[[ "$REMOTE_SIZE" == "$LOCAL_SIZE" ]] || { echo "ERROR: size mismatch after copy ($REMOTE_SIZE vs $LOCAL_SIZE)"; exit 5; }
echo "DONE $OUTPUT ($LOCAL_SIZE bytes)"
