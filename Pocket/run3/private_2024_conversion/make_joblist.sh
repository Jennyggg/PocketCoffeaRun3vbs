#!/bin/bash
# Write joblist.txt: one line per job, "<folder> <idx1,idx2,...>" with up to N files per job.
N=${1:-100}
B=/eos/uscms/store/group/lnujj/VVJJ_Private_Production_2024
: > joblist.txt
for f in WZunpolarized_Wmtolv_Ztojj WZunpolarized_Wmtolv_Ztojj_QCD WZunpolarized_Wptolv_Ztojj WZunpolarized_Wptolv_Ztojj_QCD \
         osWWunpolarized_Wptojj_Wmtolv osWWunpolarized_Wptojj_Wmtolv_QCD osWWunpolarized_Wptolv_Wmtojj osWWunpolarized_Wptolv_Wmtojj_QCD \
         ssWWunpolarized ssWWunpolarized_QCD; do
  xrdfs root://cmseos.fnal.gov ls $B/$f | grep -o 'nanoaod_[0-9]*\.root' | grep -o '[0-9]*' | sort -n \
    | paste -sd, | tr ',' '\n' | awk -v f=$f -v n=$N '{a[NR]=$1} END{for(i=1;i<=NR;i+=n){s=a[i]; for(j=i+1;j<i+n&&j<=NR;j++) s=s","a[j]; print f, s}}' >> joblist.txt
done
wc -l joblist.txt
