#!/bin/bash
# Report which of the 28 re-hadded files are on EOS and their condor status.
cd "$(dirname "$0")"
ok=0
for j in jobs/*.json; do
  out=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['output'])" $j)
  p="/${out#root://*//}"
  if [[ -s "$p" ]]; then ok=$((ok+1)); else echo "not yet: $(basename $j) $(basename $out)"; fi
done
echo "$ok / $(ls jobs/*.json | wc -l) outputs present"
grep -l "^ERROR" logs/*.out 2>/dev/null | sed 's/^/FAILED: /'
