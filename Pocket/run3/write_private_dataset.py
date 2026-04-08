import os
import json

dic_dataset = {}

dataset_name = "ssWWunpolarized"
#dataset_name = "osWWunpolarized_Wptolv_Wmtojj"
#dataset_name = "osWWunpolarized_Wptojj_Wmtolv"
#dataset_name = "WZunpolarized_Wmtolv_Ztojj"
#dataset_name = "WZunpolarized_Wptolv_Ztojj"
eras = ["2022_postEE"]
xsec = 0.12669373
#xsec = 0.60760498
#xsec = 0.62564087
#xsec =  0.11383438
#xsec = 0.21237946
paths = ["/eos/home-j/jinw/VBS_semilep/22EE/ssWWunpolarized/"]
#paths = ["/eos/home-j/jinw/VBS_semilep/22EE/osWWunpolarized_Wptolv_Wmtojj_new"]
#paths = ["/eos/home-j/jinw/VBS_semilep/22EE/osWWunpolarized_Wptojj_Wmtolv_new"]
#paths = ["/eos/home-j/jinw/VBS_semilep/22EE/WZunpolarized_Wmtolv_Ztojj"]
#paths = ["/eos/home-j/jinw/VBS_semilep/22EE/WZunpolarized_Wptolv_Ztojj"]
event_per_file = 500

output = f"datasets/{dataset_name}_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json"

for era, path in zip(eras, paths):
    size = 0
    filelist = []
    nevents = 0
    for ele in os.scandir(path):
        size += os.stat(ele).st_size
        filelist.append("root://eosuser.cern.ch/" + ele.path)
        nevents += event_per_file
    dic_dataset[f"{dataset_name}_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_{era}"] = {
            "metadata": {
                "das_names": "['private']",
                "sample": f"{dataset_name}_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "year": era,
                "isMC": "True",
                "xsec": str(xsec),
                "nevents": str(nevents),
                "size": str(size)
                },
            "files": filelist
    }
with open(output, 'w') as fp:
    json.dump(dic_dataset,fp,indent=4)
