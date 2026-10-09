# Point the *_2024 entries of the private VV dataset JSONs to the converted flat files.
import json, glob, os
here = os.path.dirname(os.path.abspath(__file__))
for jf in sorted(glob.glob(f"{here}/../datasets/*unpolarized*_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json")):
    d = json.load(open(jf))
    for k, v in d.items():
        if k.endswith("_2024"):
            v["files"] = [f.replace("root://cmseos.fnal.gov//eos/uscms/store/group/lnujj/VVJJ_Private_Production_2024/", "root://eosuser.cern.ch//eos/cms/store/group/phys_smp/jinw/2024_VBS_sig/") for f in v["files"]]
    with open(jf, "w") as fo:
        json.dump(d, fo, indent=4)
        fo.write("\n")
    print("updated", os.path.basename(jf))
