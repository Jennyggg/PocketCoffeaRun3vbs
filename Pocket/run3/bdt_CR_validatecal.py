#!/usr/bin/env python3
import xgboost as xgb
import pandas as pd
import coffea.util
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
from matplotlib.colors import TwoSlopeNorm
import importlib.util

print(xgb.__version__)
hep.style.use("CMS")


paths_dic = {
    "resolved_mu_WCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "resolved_e_WCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "boosted_mu_WCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "boosted_e_WCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "resolved_mu_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "resolved_e_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "boosted_mu_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "boosted_e_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
}
paths_nonprompt_dic = {
    "resolved_mu_WCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "resolved_e_WCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "boosted_mu_WCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "bosted_e_WCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "resolved_mu_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "resolved_e_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "boosted_mu_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "boosted_e_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
}
normtables_dic = {
    "resolved_mu_WCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_CR",
    "resolved_e_WCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_CR",
    "boosted_mu_WCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_boosted_WCR",
    "boosted_e_WCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_boosted_WCR",
    "resolved_mu_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_TTCR",
    "resolved_e_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_TTCR",
    "boosted_mu_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_boosted_TTCR",
    "boosted_e_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_boosted_TTCR",

}

normtables_nonprompt_dic = {
    "resolved_mu_WCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_CR",
    "resolved_e_WCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_CR",
    "boosted_mu_WCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_boosted_WCR",
    "boosted_e_WCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_boosted_WCR",
    "resolved_mu_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_TTCR",
    "resolved_e_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_TTCR",
    "boosted_mu_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_TTCR",
    "boosted_e_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_TTCR",

}

config = {
    "category": "resolved_mu_TTCR",
    "year": "2022_postEE",
    "output": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/bdt_cal_val_resolved_mu_TTCR",
    "calibration": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/bdt_CR_cal_resolved_mu_WCR/bdt_CR_calibration.py",
}
config["path"] = paths_dic[config["category"]]
config["path_nonprompt"] = paths_nonprompt_dic[config["category"]]
config["normtable"] = normtables_dic[config["category"]]
config["normtable_nonprompt"] = normtables_nonprompt_dic[config["category"]]



LUMI = 26.67  # fb^-1

models_dic = {
    "resolved_mu":[
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_0_iter_138.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_1_iter_172.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_2_iter_184.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_3_iter_194.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_4_iter_196.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_5_iter_197.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_6_iter_130.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_7_iter_186.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_8_iter_182.json"
    ],
    "resolved_e":[
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_0_iter_177.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_1_iter_180.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_2_iter_121.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_3_iter_157.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_4_iter_211.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_5_iter_149.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_6_iter_112.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_7_iter_166.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_8_iter_166.json"
    ],
    "boosted_mu":[
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_0_iter_115.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_1_iter_99.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_2_iter_131.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_3_iter_81.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_4_iter_130.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_5_iter_196.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_6_iter_128.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_7_iter_97.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_8_iter_98.json"
    ],
    "boosted_e":[
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_0_iter_130.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_1_iter_113.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_2_iter_165.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_3_iter_92.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_4_iter_113.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_5_iter_128.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_6_iter_126.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_7_iter_94.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_8_iter_86.json"
    ]
}


models = models_dic[config["category"].replace("_WCR","").replace("_TTCR","")]

colors = {
    "VBS_EWK":   "#bd1f01",
    "TT":        "#832db6",
    "SingleTop": "#F38AA5",
    "WJets":     "#3f90da",
    "DY":        "#ffa90e",
    "QCD-VV":    "#b9ac70",
    "Other":     "#a96b59",
    "nonprompt": "#a96b59",
}

process_groups = {
    "VBS_EWK": [
        "osWWunpolarized_Wptojj_Wmtolv_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "osWWunpolarized_Wptolv_Wmtojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "ssWWunpolarized_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WZunpolarized_Wmtolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WZunpolarized_Wptolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
    ],
    "TT": [
        "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
        "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
    ],
    "SingleTop": [
        "TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
        "TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
        "TbarWplus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8",
        "TWminus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8",
    ],
    "WJets": [
        "WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "WtoLNu-2Jets_PTLNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"
    ],
    "DY": [
        "DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
    ],
    "QCD-VV": [
        "WW_TuneCP5_13p6TeV_pythia8",
        "WZ_TuneCP5_13p6TeV_pythia8",
        "ZZ_TuneCP5_13p6TeV_pythia8",
        "WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8",
    ],
    "Data": [
        "EGamma_2022_EraE",
        "EGamma_2022_EraF",
        "EGamma_2022_EraG",
        "Muon_2022_EraE",
        "Muon_2022_EraF",
        "Muon_2022_EraG",
    ],
}

MC_STACK_ORDER = ["nonprompt", "QCD-VV", "DY", "WJets", "SingleTop", "TT", "VBS_EWK"]

# ─────────────────────────────────────────────────────────────────────────────
# Load calibration from file
# ─────────────────────────────────────────────────────────────────────────────

spec = importlib.util.spec_from_file_location("calibration", config["calibration"])
calib_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib_module)
calibrate_btagPNetQvG = calib_module.calibrate_btagPNetQvG
print(f"Calibration loaded from {config['calibration']}")

# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_MC_process_from_parquet(parquet_dir, parquet_table_dir,
                                 category_name, process, year):
    path = f"{parquet_dir}/{category_name}/{process}_{year}.parquet"
    if not os.path.exists(path):
        return None, None
    df = pd.read_parquet(path)
    df["process"]  = process
    df["year_tag"] = f"{process}_{year}"
    df["category"] = category_name
    ft  = coffea.util.load(f"{parquet_table_dir}/output_{process}_{year}.coffea")
    sgw = ft["sum_genweights"][f"{process}_{year}"]
    return df, sgw


def load_data_from_parquet(parquet_dir, category_name, process):
    path = f"{parquet_dir}/{category_name}/{process}.parquet"
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["process"]  = process
    df["year_tag"] = "Data"
    df["category"] = category_name
    return df


df_list = []

for group in process_groups:
    if group != "Data":
        for process in process_groups[group]:
            df_p, sgw = load_MC_process_from_parquet(
                config["path"], config["normtable"],
                config["category"], process, config["year"])
            if df_p is not None:
                df_p["weight"] /= sgw
                df_p["group"]   = group
                df_list.append(df_p)
    else:
        for process in process_groups[group]:
            df_p = load_data_from_parquet(
                config["path"], config["category"], process)
            if df_p is not None:
                df_p["group"] = "Data"
                df_list.append(df_p)

for group in process_groups:
    if group != "Data":
        for process in process_groups[group]:
            df_p, sgw = load_MC_process_from_parquet(
                config["path_nonprompt"], config["normtable_nonprompt"],
                config["category"], process, config["year"])
            if df_p is not None:
                df_p["weight"]   = -df_p["weight"] / sgw
                df_p["process"]  = "nonprompt"
                df_p["year_tag"] = f"nonprompt_{config['year']}"
                df_p["group"]    = "nonprompt"
                df_list.append(df_p)
    else:
        for process in process_groups[group]:
            df_p = load_data_from_parquet(
                config["path_nonprompt"], config["category"], process)
            if df_p is not None:
                df_p["process"]  = "nonprompt"
                df_p["year_tag"] = f"nonprompt_{config['year']}"
                df_p["group"]    = "nonprompt"
                df_list.append(df_p)

df = pd.concat(df_list, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load BDT models and score all events
# ─────────────────────────────────────────────────────────────────────────────

boosters = []
for path in models:
    bdt = xgb.Booster()
    bdt.load_model(path)
    boosters.append(bdt)

feature_names = boosters[0].feature_names

X              = df[feature_names]
dmat           = xgb.DMatrix(X, feature_names=feature_names)
raw_scores     = np.array([bdt.predict(dmat) for bdt in boosters])
df["bdt_score"] = raw_scores.mean(axis=0)

mask_data = df["year_tag"] == "Data"
mask_mc   = ~mask_data

df_data = df[mask_data].copy()
df_mc   = df[mask_mc].copy()

os.makedirs(config["output"], exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def whist(values, weights, edges):
    h,  _ = np.histogram(values, bins=edges, weights=weights)
    h2, _ = np.histogram(values, bins=edges, weights=weights**2)
    norm   = np.sum(h) * np.diff(edges)
    norm[norm == 0] = 1
    return h / norm, np.sqrt(h2) / norm


def chi2_data_mc(h_data, h_mc, err_data, err_mc):
    mask   = (h_mc > 0) & np.isfinite(h_data) & np.isfinite(h_mc)
    sigma2 = err_data[mask]**2 + err_mc[mask]**2
    return np.sum((h_data[mask] - h_mc[mask])**2 / sigma2), mask.sum()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Apply calibration to all jets in MC
# ─────────────────────────────────────────────────────────────────────────────

jet_tag_cols = ["jet1_btagPNetQvG", "jet2_btagPNetQvG",
                "jet3_btagPNetQvG", "jet4_btagPNetQvG"]
jet_pt_cols  = ["jet1_pt",  "jet2_pt",  "jet3_pt",  "jet4_pt"]
jet_eta_cols = ["jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta"]

df_mc_cal = df_mc.copy()
for pt_col, eta_col, tag_col in zip(jet_pt_cols, jet_eta_cols, jet_tag_cols):
    valid = (df_mc_cal[pt_col].notna() &
             df_mc_cal[eta_col].notna() &
             df_mc_cal[tag_col].notna())
    df_mc_cal.loc[valid, tag_col] = calibrate_btagPNetQvG(
        df_mc_cal.loc[valid, pt_col].values,
        df_mc_cal.loc[valid, eta_col].values,
        df_mc_cal.loc[valid, tag_col].values,
    )


def score_df(sub_df):
    dmat = xgb.DMatrix(sub_df[feature_names], feature_names=feature_names)
    return np.array([bdt.predict(dmat) for bdt in boosters]).mean(axis=0)


score_mc_nom = df_mc["bdt_score"].values
score_mc_cal = score_df(df_mc_cal)

# ─────────────────────────────────────────────────────────────────────────────
# CMS stacked-histogram plot helper
# ─────────────────────────────────────────────────────────────────────────────

def cms_stack_plot(ax_main, ax_ratio,
                   df_mc_plot, df_data_plot,
                   edges, xlabel,
                   var_col=None,
                   mc_scores=None,
                   data_scores=None):
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)

    stack_bottom = np.zeros(len(centres))
    h_mc_total   = np.zeros(len(centres))
    h_mc_total2  = np.zeros(len(centres))

    for group in MC_STACK_ORDER:
        mask_g = df_mc_plot["group"] == group
        if not mask_g.any():
            continue
        vals = (df_mc_plot.loc[mask_g, var_col].values
                if var_col is not None
                else mc_scores[mask_g.values])
        wgts = df_mc_plot.loc[mask_g, "weight"].values

        h_g,  _ = np.histogram(vals, bins=edges, weights=wgts)
        h_g2, _ = np.histogram(vals, bins=edges, weights=wgts**2)

        color = colors.get(group, "#888888")
        ax_main.bar(centres, h_g, width=widths, bottom=stack_bottom,
                    color=color, label=group, align="center")
        stack_bottom += h_g
        h_mc_total   += h_g
        h_mc_total2  += h_g2

    err_mc_total = np.sqrt(h_mc_total2)
    ax_main.bar(centres, 2 * err_mc_total, width=widths,
                bottom=h_mc_total - err_mc_total,
                color="grey", alpha=0.4, label="MC stat. unc.",
                align="center", hatch="///", linewidth=0)

    # data
    if var_col is not None:
        valid_d = df_data_plot[var_col].notna()
        vals_d  = df_data_plot.loc[valid_d, var_col].values
        wgts_d  = df_data_plot.loc[valid_d, "weight"].values
    else:
        vals_d = data_scores
        wgts_d = df_data_plot["weight"].values

    h_data,  _ = np.histogram(vals_d, bins=edges, weights=wgts_d)
    h_data2, _ = np.histogram(vals_d, bins=edges, weights=wgts_d**2)
    err_data   = np.sqrt(h_data2)

    ax_main.errorbar(centres, h_data, yerr=err_data,
                     fmt="o", color="black", markersize=4,
                     linewidth=1.2, label="Data", zorder=10)
    ax_main.set_ylabel("Events / bin")
    ax_main.legend(fontsize=11, ncol=2)

    # ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio      = np.where(h_mc_total > 0, h_data / h_mc_total, np.nan)
        err_ratio  = np.where(h_mc_total > 0,
                              np.sqrt((err_data / h_mc_total)**2 +
                                      (h_data * err_mc_total / h_mc_total**2)**2),
                              np.nan)
        mc_rel_err = np.where(h_mc_total > 0, err_mc_total / h_mc_total, np.nan)

    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax_ratio.fill_between(centres,
                          1 - mc_rel_err, 1 + mc_rel_err,
                          step="mid", alpha=0.3, color="grey")
    ax_ratio.errorbar(centres, ratio, yerr=err_ratio,
                      fmt="o", color="black", markersize=3, linewidth=1.0)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data / MC")
    ax_ratio.set_xlabel(xlabel)

    chi2, ndof = chi2_data_mc(h_data, h_mc_total, err_data, err_mc_total)
    return chi2, ndof, h_data, h_mc_total, err_data, err_mc_total


def make_cms_fig(title):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    hep.cms.label(ax=axes[0], label="Preliminary",
                  data=True,
                  lumi=LUMI, lumi_format="{:.2f}", com=13.6)
    axes[0].set_title(title, fontsize=10, pad=30)
    return fig, axes[0], axes[1]

# ─────────────────────────────────────────────────────────────────────────────
# 3. CMS plots: BDT score nominal and calibrated
# ─────────────────────────────────────────────────────────────────────────────

SCORE_BINS  = (20 if "resolved" in config["category"] else 10)
score_edges = np.linspace(0.0, 1.0, SCORE_BINS + 1)

# nominal
fig, ax_m, ax_r = make_cms_fig("BDT score — nominal MC")
chi2, ndof, *_ = cms_stack_plot(
    ax_m, ax_r, df_mc, df_data,
    edges=score_edges, xlabel="BDT score",
    mc_scores=score_mc_nom,
    data_scores=df_data["bdt_score"].values,
)
print(f"[BDT nominal]    chi²/ndof = {chi2:.1f} / {ndof}")
plt.tight_layout()
fig.savefig(os.path.join(config["output"], "cms_bdt_score_nominal.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# calibrated
fig, ax_m, ax_r = make_cms_fig("BDT score — calibrated MC (btagPNetQvG)")
chi2_cal, ndof_cal, *_ = cms_stack_plot(
    ax_m, ax_r, df_mc_cal, df_data,
    edges=score_edges, xlabel="BDT score",
    mc_scores=score_mc_cal,
    data_scores=df_data["bdt_score"].values,
)
print(f"[BDT calibrated] chi²/ndof = {chi2_cal:.1f} / {ndof_cal}")
plt.tight_layout()
fig.savefig(os.path.join(config["output"], "cms_bdt_score_calibrated.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-jet tag plots: normalised before/after + CMS stacked nominal/calibrated
# ─────────────────────────────────────────────────────────────────────────────

N_TAG_PLOT_BINS  = 25
tag_plot_edges   = np.linspace(0.0, 1.0, N_TAG_PLOT_BINS + 1)
tag_plot_centres = 0.5 * (tag_plot_edges[:-1] + tag_plot_edges[1:])

for tag_col in jet_tag_cols:

    # ── normalised before/after ───────────────────────────────────────────────
    def get_whist(sub_df, col):
        valid = sub_df[col].notna()
        return whist(sub_df.loc[valid, col].values,
                     sub_df.loc[valid, "weight"].values,
                     tag_plot_edges)

    h_d,   err_d   = get_whist(df_data,   tag_col)
    h_nom, err_nom = get_whist(df_mc,     tag_col)
    h_cal, err_cal = get_whist(df_mc_cal, tag_col)

    fig_n, axes_n = plt.subplots(2, 1, figsize=(7, 7),
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 sharex=True)
    ax_mn, ax_rn = axes_n

    ax_mn.errorbar(tag_plot_centres, h_d, yerr=err_d,
                   fmt="o", color="black", markersize=3, linewidth=1.0,
                   label="Data", zorder=5)
    ax_mn.step(tag_plot_centres, h_nom, where="mid",
               color="steelblue", linewidth=1.5, linestyle="--", label="MC (nominal)")
    ax_mn.fill_between(tag_plot_centres,
                       h_nom - err_nom, h_nom + err_nom,
                       step="mid", alpha=0.3, color="steelblue")
    ax_mn.step(tag_plot_centres, h_cal, where="mid",
               color="tomato", linewidth=1.5, label="MC (calibrated)")
    ax_mn.fill_between(tag_plot_centres,
                       h_cal - err_cal, h_cal + err_cal,
                       step="mid", alpha=0.3, color="tomato")
    ax_mn.set_ylabel("Normalised events / bin")
    ax_mn.set_title(f"{tag_col}: before and after pt-η calibration")
    ax_mn.legend(fontsize=9)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_nom_n     = np.where(h_nom > 0, h_d / h_nom, np.nan)
        r_cal_n     = np.where(h_cal > 0, h_d / h_cal, np.nan)
        err_r_nom_n = np.where(h_nom > 0,
                               np.sqrt((err_d / h_nom)**2 +
                                       (h_d * err_nom / h_nom**2)**2), np.nan)
        err_r_cal_n = np.where(h_cal > 0,
                               np.sqrt((err_d / h_cal)**2 +
                                       (h_d * err_cal / h_cal**2)**2), np.nan)

    ax_rn.errorbar(tag_plot_centres, r_nom_n, yerr=err_r_nom_n,
                   fmt="s", color="steelblue", markersize=3,
                   linestyle="--", linewidth=1.0, label="Data/MC nom.")
    ax_rn.errorbar(tag_plot_centres, r_cal_n, yerr=err_r_cal_n,
                   fmt="o", color="tomato", markersize=3,
                   linewidth=1.0, label="Data/MC cal.")
    ax_rn.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax_rn.set_ylim(0.5, 1.5)
    ax_rn.set_ylabel("Data / MC")
    ax_rn.set_xlabel(tag_col)
    ax_rn.legend(fontsize=8)

    plt.tight_layout()
    fig_n.savefig(os.path.join(config["output"], f"tag_norm_{tag_col}.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(fig_n)

    # ── CMS stacked — nominal ─────────────────────────────────────────────────
    fig_cms, ax_m_cms, ax_r_cms = make_cms_fig(f"{tag_col} — nominal MC")
    chi2_t, ndof_t, *_ = cms_stack_plot(
        ax_m_cms, ax_r_cms, df_mc, df_data,
        edges=tag_plot_edges, xlabel=tag_col,
        var_col=tag_col,
    )
    print(f"[{tag_col} nominal]    chi²/ndof = {chi2_t:.1f} / {ndof_t}")
    plt.tight_layout()
    fig_cms.savefig(os.path.join(config["output"], f"cms_{tag_col}_nominal.png"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig_cms)

    # ── CMS stacked — calibrated ──────────────────────────────────────────────
    fig_cms2, ax_m_cms2, ax_r_cms2 = make_cms_fig(f"{tag_col} — calibrated MC")
    chi2_tc, ndof_tc, *_ = cms_stack_plot(
        ax_m_cms2, ax_r_cms2, df_mc_cal, df_data,
        edges=tag_plot_edges, xlabel=tag_col,
        var_col=tag_col,
    )
    print(f"[{tag_col} calibrated] chi²/ndof = {chi2_tc:.1f} / {ndof_tc}")
    plt.tight_layout()
    fig_cms2.savefig(os.path.join(config["output"], f"cms_{tag_col}_calibrated.png"),
                     dpi=150, bbox_inches="tight")
    plt.close(fig_cms2)

    print(f"Saved all plots for {tag_col}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CDF plots: jet btagPNetQvG before/after calibration vs data, per jet
# ─────────────────────────────────────────────────────────────────────────────

N_CDF_BINS  = 100
cdf_edges   = np.linspace(0.0, 1.0, N_CDF_BINS + 1)
cdf_centres = 0.5 * (cdf_edges[:-1] + cdf_edges[1:])

for tag_col in jet_tag_cols:

    fig, axes = plt.subplots(2, 1, figsize=(7, 7),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    ax_cdf, ax_diff = axes

    cdf_data_ref     = None
    cdf_data_err_ref = None

    for label, sub_df, color, ls in [
        ("Data",            df_data,   "black",     "-"),
        ("MC (nominal)",    df_mc,     "steelblue", "--"),
        ("MC (calibrated)", df_mc_cal, "tomato",    "-"),
    ]:
        valid = sub_df[tag_col].notna()
        vals  = sub_df.loc[valid, tag_col].values
        wgts  = sub_df.loc[valid, "weight"].values

        h,  _ = np.histogram(vals, bins=cdf_edges, weights=wgts)
        h2, _ = np.histogram(vals, bins=cdf_edges, weights=wgts**2)
        cumh    = np.cumsum(h)
        total_w = cumh[-1]
        cdf_y   = cumh / total_w if total_w > 0 else np.zeros_like(cumh, dtype=float)

        cumh2   = np.cumsum(h2)
        cdf_err = np.sqrt(cumh2) / total_w if total_w > 0 else np.zeros_like(cumh2)

        if label == "Data":
            ax_cdf.errorbar(cdf_centres, cdf_y, yerr=cdf_err,
                            fmt="o", color=color, markersize=2,
                            linewidth=1.0, label=label, zorder=5)
            cdf_data_ref     = cdf_y
            cdf_data_err_ref = cdf_err
        else:
            ax_cdf.plot(cdf_centres, cdf_y, color=color,
                        linewidth=1.5, linestyle=ls, label=label)
            ax_cdf.fill_between(cdf_centres,
                                cdf_y - cdf_err, cdf_y + cdf_err,
                                alpha=0.2, color=color)

            diff     = cdf_y - cdf_data_ref
            err_diff = np.sqrt(cdf_err**2 + cdf_data_err_ref**2)
            ax_diff.plot(cdf_centres, diff, color=color,
                         linewidth=1.5, linestyle=ls, label=label)
            ax_diff.fill_between(cdf_centres,
                                 diff - err_diff, diff + err_diff,
                                 alpha=0.2, color=color)

    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title(f"{tag_col}: CDF before and after calibration")
    ax_cdf.legend(fontsize=9)
    ax_cdf.set_ylim(0, 1)

    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax_diff.set_ylabel("MC − Data")
    ax_diff.set_xlabel(tag_col)
    ax_diff.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(config["output"], f"cdf_{tag_col}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: cdf_{tag_col}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Normalised CMS-style plots: BDT score and jet btagPNetQvG
#    Same layout as the stacked plots but MC total and data both normalised
#    to unit area so shape comparison is clear.
# ─────────────────────────────────────────────────────────────────────────────

def cms_norm_plot(ax_main, ax_ratio,
                  df_mc_plot, df_data_plot,
                  edges, xlabel,
                  var_col=None,
                  mc_scores=None,
                  data_scores=None):
    """
    CMS-style normalised shape comparison.
    MC groups are drawn as individually normalised steps stacked visually,
    but the filled total MC and data are each normalised to unit area.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)

    # ── build total weighted MC histogram ────────────────────────────────────
    h_mc_total  = np.zeros(len(centres))
    h_mc_total2 = np.zeros(len(centres))

    # also keep per-group histograms for the stacked fill
    group_hists = {}
    for group in MC_STACK_ORDER:
        mask_g = df_mc_plot["group"] == group
        if not mask_g.any():
            continue
        vals = (df_mc_plot.loc[mask_g, var_col].values
                if var_col is not None
                else mc_scores[mask_g.values])
        wgts = df_mc_plot.loc[mask_g, "weight"].values

        h_g,  _ = np.histogram(vals, bins=edges, weights=wgts)
        h_g2, _ = np.histogram(vals, bins=edges, weights=wgts**2)
        group_hists[group] = (h_g, h_g2)
        h_mc_total  += h_g
        h_mc_total2 += h_g2

    err_mc_total = np.sqrt(h_mc_total2)

    # normalise MC total to unit area
    mc_area = (h_mc_total * widths).sum()
    if mc_area > 0:
        h_mc_norm   = h_mc_total  / mc_area
        err_mc_norm = err_mc_total / mc_area
    else:
        h_mc_norm   = h_mc_total
        err_mc_norm = err_mc_total

    # draw stacked normalised MC (each group scaled by same mc_area)
    stack_bottom = np.zeros(len(centres))
    for group in MC_STACK_ORDER:
        if group not in group_hists:
            continue
        h_g, _ = group_hists[group]
        h_g_norm = h_g / mc_area if mc_area > 0 else h_g
        color = colors.get(group, "#888888")
        ax_main.bar(centres, h_g_norm, width=widths, bottom=stack_bottom,
                    color=color, label=group, align="center", alpha=0.85)
        stack_bottom += h_g_norm

    # MC stat uncertainty band
    ax_main.bar(centres, 2 * err_mc_norm, width=widths,
                bottom=h_mc_norm - err_mc_norm,
                color="grey", alpha=0.4, label="MC stat. unc.",
                align="center", hatch="///", linewidth=0)

    # ── data ─────────────────────────────────────────────────────────────────
    if var_col is not None:
        valid_d = df_data_plot[var_col].notna()
        vals_d  = df_data_plot.loc[valid_d, var_col].values
        wgts_d  = df_data_plot.loc[valid_d, "weight"].values
    else:
        vals_d = data_scores
        wgts_d = df_data_plot["weight"].values

    h_data,  _ = np.histogram(vals_d, bins=edges, weights=wgts_d)
    h_data2, _ = np.histogram(vals_d, bins=edges, weights=wgts_d**2)
    err_data   = np.sqrt(h_data2)

    data_area = (h_data * widths).sum()
    if data_area > 0:
        h_data_norm   = h_data  / data_area
        err_data_norm = err_data / data_area
    else:
        h_data_norm   = h_data
        err_data_norm = err_data

    ax_main.errorbar(centres, h_data_norm, yerr=err_data_norm,
                     fmt="o", color="black", markersize=4,
                     linewidth=1.2, label="Data", zorder=10)
    ax_main.set_ylabel("Normalised events / bin")
    ax_main.legend(fontsize=11, ncol=2)

    # ── ratio ─────────────────────────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio      = np.where(h_mc_norm > 0, h_data_norm / h_mc_norm, np.nan)
        err_ratio  = np.where(h_mc_norm > 0,
                              np.sqrt((err_data_norm / h_mc_norm)**2 +
                                      (h_data_norm * err_mc_norm / h_mc_norm**2)**2),
                              np.nan)
        mc_rel_err = np.where(h_mc_norm > 0, err_mc_norm / h_mc_norm, np.nan)

    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax_ratio.fill_between(centres,
                          1 - mc_rel_err, 1 + mc_rel_err,
                          step="mid", alpha=0.3, color="grey")
    ax_ratio.errorbar(centres, ratio, yerr=err_ratio,
                      fmt="o", color="black", markersize=3, linewidth=1.0)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data / MC")
    ax_ratio.set_xlabel(xlabel)

    chi2, ndof = chi2_data_mc(h_data_norm, h_mc_norm, err_data_norm, err_mc_norm)
    return chi2, ndof


# ── BDT score — nominal (normalised) ─────────────────────────────────────────
fig, ax_m, ax_r = make_cms_fig("BDT score — nominal MC (normalised)")
chi2_n, ndof_n = cms_norm_plot(
    ax_m, ax_r, df_mc, df_data,
    edges=score_edges, xlabel="BDT score",
    mc_scores=score_mc_nom,
    data_scores=df_data["bdt_score"].values,
)
print(f"[BDT nominal norm]    chi²/ndof = {chi2_n:.1f} / {ndof_n}")
plt.tight_layout()
fig.savefig(os.path.join(config["output"], "cms_bdt_score_nominal_norm.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ── BDT score — calibrated (normalised) ──────────────────────────────────────
fig, ax_m, ax_r = make_cms_fig("BDT score — calibrated MC (normalised)")
chi2_cn, ndof_cn = cms_norm_plot(
    ax_m, ax_r, df_mc_cal, df_data,
    edges=score_edges, xlabel="BDT score",
    mc_scores=score_mc_cal,
    data_scores=df_data["bdt_score"].values,
)
print(f"[BDT calibrated norm] chi²/ndof = {chi2_cn:.1f} / {ndof_cn}")
plt.tight_layout()
fig.savefig(os.path.join(config["output"], "cms_bdt_score_calibrated_norm.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ── per-jet tag: normalised ───────────────────────────────────────────────────
for tag_col in jet_tag_cols:

    fig_n, ax_mn, ax_rn = make_cms_fig(f"{tag_col} — nominal MC (normalised)")
    chi2_tn, ndof_tn = cms_norm_plot(
        ax_mn, ax_rn, df_mc, df_data,
        edges=tag_plot_edges, xlabel=tag_col,
        var_col=tag_col,
    )
    print(f"[{tag_col} nominal norm]    chi²/ndof = {chi2_tn:.1f} / {ndof_tn}")
    plt.tight_layout()
    fig_n.savefig(os.path.join(config["output"], f"cms_{tag_col}_nominal_norm.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(fig_n)

    fig_c, ax_mc2, ax_rc2 = make_cms_fig(f"{tag_col} — calibrated MC (normalised)")
    chi2_tcn, ndof_tcn = cms_norm_plot(
        ax_mc2, ax_rc2, df_mc_cal, df_data,
        edges=tag_plot_edges, xlabel=tag_col,
        var_col=tag_col,
    )
    print(f"[{tag_col} calibrated norm] chi²/ndof = {chi2_tcn:.1f} / {ndof_tcn}")
    plt.tight_layout()
    fig_c.savefig(os.path.join(config["output"], f"cms_{tag_col}_calibrated_norm.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(fig_c)

    print(f"Saved normalised plots for {tag_col}")

print("Done.")

