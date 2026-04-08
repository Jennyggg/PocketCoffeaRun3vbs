#!/usr/bin/env python3
import xgboost as xgb
import pandas as pd
import coffea.util
import awkward as ak
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import mplhep as hep
import os
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
print(xgb.__version__)
hep.style.use("CMS")
from matplotlib.ticker import StrMethodFormatter
from scipy.interpolate import interp1d

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
    "category": "resolved_e_TTCR",
    "year": "2022_postEE",
    "output": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/bdt_resolved_e_TTCR",
}
config["path"] = paths_dic[config["category"]]
config["path_nonprompt"] = paths_nonprompt_dic[config["category"]]
config["normtable"] = normtables_dic[config["category"]]
config["normtable_nonprompt"] = normtables_nonprompt_dic[config["category"]]

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

LUMI = 26.67  # fb^-1


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
        "WZunpolarized_Wptolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"
        ],

    "TT": [
        "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
        "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8"
        ],
    "SingleTop": [
        "TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
        "TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
        "TbarWplus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8",
        "TWminus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8"
        ],
    "WJets":[
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
        "DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"
        ],
    "QCD-VV": [
        "WW_TuneCP5_13p6TeV_pythia8",
        "WZ_TuneCP5_13p6TeV_pythia8",
        "ZZ_TuneCP5_13p6TeV_pythia8",
        "WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8"
        ],
    "Data": [
        "EGamma_2022_EraE",
        "EGamma_2022_EraF",
        "EGamma_2022_EraG",
        "Muon_2022_EraE",
        "Muon_2022_EraF",
        "Muon_2022_EraG"
    ]
}

def load_MC_process_from_parquet(parquet_dir,parquet_table_dir, category_name, process, year):
    if not os.path.exists(f"{parquet_dir}/{category_name}/{process}_{year}.parquet"):
        return None, None
    df = pd.read_parquet(f"{parquet_dir}/{category_name}/{process}_{year}.parquet")
    df["process"] = process
    df["year_tag"] = f"{process}_{year}"
    df["category"] = category_name
    file_table = coffea.util.load(f"{parquet_table_dir}/output_{process}_{year}.coffea")
    sum_genweight = file_table['sum_genweights'][f"{process}_{year}"]
    return df,sum_genweight

def load_data_from_parquet(parquet_dir,category_name, process):
    if not os.path.exists(f"{parquet_dir}/{category_name}/{process}.parquet"):
        return None
    df = pd.read_parquet(f"{parquet_dir}/{category_name}/{process}.parquet")
    df["process"] = process
    df["year_tag"] = "Data"
    df["category"] = category_name
    return df

df_list = []

for group in process_groups.keys():
    if group != "Data":
        for process in process_groups[group]:
            df, sum_genweight = load_MC_process_from_parquet(
                config["path"],
                config["normtable"],
                config["category"], 
                process,
                config["year"]
                )
            if df is not None and sum_genweight is not None:
                df["weight"] = df["weight"]/sum_genweight
                df["group"]   = group
                df_list.append(df)
    else:
        for process in process_groups[group]:
            df = load_data_from_parquet(
                config["path"],
                config["category"], 
                process
                )
            if df is not None:
                df["group"] = "Data"
                df_list.append(df)
MC_weights = 0
data_weights = 0
for group in process_groups.keys():
    group_weights = 0
    if group != "Data":
        for process in process_groups[group]:
            df, sum_genweight = load_MC_process_from_parquet(
                config["path_nonprompt"],
                config["normtable_nonprompt"],
                config["category"], 
                process,
                config["year"]
                )
            if df is not None and sum_genweight is not None:
                df["weight"] = -df["weight"]/sum_genweight
                print("nonprompt ", process, sum(df["weight"]))
                group_weights += sum(df["weight"])
                MC_weights += sum(df["weight"])
                df["process"] = "nonprompt"
                df["year_tag"] = f"nonprompt_{config['year']}"
                df["group"]    = "nonprompt"
                df_list.append(df)
    else:
        for process in process_groups[group]:
            df = load_data_from_parquet(
                config["path_nonprompt"],
                config["category"], 
                process
                )
            if df is not None:
                df["process"] = "nonprompt"
                print("nonprompt ", process, sum(df["weight"]))
                group_weights += sum(df["weight"])
                data_weights += sum(df["weight"])
                df["year_tag"] = f"nonprompt_{config['year']}"
                df["group"]    = "nonprompt"
                df_list.append(df)
print("nonprompt MC weights", MC_weights, ", nonprompt data weights",data_weights, ", total", MC_weights+data_weights)
            
df=pd.concat(df_list,ignore_index=True)


# ── 1. Load models and get feature names ──────────────────────────────────────
boosters = []
for path in models:
    bdt = xgb.Booster()
    bdt.load_model(path)
    boosters.append(bdt)

feature_names = boosters[0].feature_names

# ── 2. Score every event (average over all 9 models) ─────────────────────────
X = df[feature_names]
dmat = xgb.DMatrix(X, feature_names=feature_names)

raw_scores = np.array([bdt.predict(dmat) for bdt in boosters])
avg_score  = raw_scores.mean(axis=0)
df["bdt_score"] = avg_score

# ── 3. Split data / simulation ────────────────────────────────────────────────
mask_data = df["year_tag"] == "Data"
mask_mc   = ~mask_data

df_data = df[mask_data].copy()
df_mc   = df[mask_mc].copy()

os.makedirs(config["output"], exist_ok=True)

N_BINS       = 30
SCORE_BINS   = 20   # finer binning for the 1D score distributions

score_edges_1d = np.linspace(df["bdt_score"].min(), df["bdt_score"].max(), SCORE_BINS + 1)
score_centres_1d = 0.5 * (score_edges_1d[:-1] + score_edges_1d[1:])

# ── helper: weighted histogram (normalised) + stat uncertainty ───────────────
def whist(values, weights, edges):
    h,  _ = np.histogram(values, bins=edges, weights=weights)
    h2, _ = np.histogram(values, bins=edges, weights=weights**2)  # sum of w² per bin
    norm   = np.sum(h) * np.diff(edges)
    norm[norm == 0] = 1
    return h / norm, np.sqrt(h2) / norm   # (heights, uncertainties)

# ── baseline score distributions ─────────────────────────────────────────────
h_data_base, err_data_base = whist(df_data["bdt_score"], df_data["weight"], score_edges_1d)
h_mc_base,   err_mc_base   = whist(df_mc  ["bdt_score"], df_mc  ["weight"], score_edges_1d)

# ─────────────────────────────────────────────────────────────────────────────
for feat in feature_names:

    combined       = pd.concat([df_data[feat], df_mc[feat]]).dropna()
    score_combined = pd.concat([df_data["bdt_score"], df_mc["bdt_score"]])

    feat_edges  = np.linspace(combined.quantile(0.01),
                              combined.quantile(0.99), N_BINS + 1)
    score_edges = np.linspace(score_combined.min(),
                              score_combined.max(),  N_BINS + 1)

    def make_hist2d(sub_df):
        h, _, _ = np.histogram2d(
            sub_df["bdt_score"], sub_df[feat],
            bins   = [score_edges, feat_edges],
            weights= sub_df["weight"]
        )
        return h

    h_data_2d = make_hist2d(df_data)
    h_mc_2d   = make_hist2d(df_mc)

    def norm_cols(h):
        col_sum = h.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1
        return h / col_sum

    h_data_n = norm_cols(h_data_2d)
    h_mc_n   = norm_cols(h_mc_2d)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_2d = np.where(h_mc_n > 0, h_data_n / h_mc_n, np.nan)

    # ── 1D Data/MC ratio in feature → per-event reweight for MC ──────────────
    h_data_1d, _ = np.histogram(df_data[feat], bins=feat_edges,
                                weights=df_data["weight"])
    h_mc_1d,   _ = np.histogram(df_mc  [feat], bins=feat_edges,
                                weights=df_mc  ["weight"])

    # normalise to same integral before taking ratio
    scale = h_data_1d.sum() / h_mc_1d.sum() if h_mc_1d.sum() > 0 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rw_factors = np.where(h_mc_1d > 0, h_data_1d / (h_mc_1d * scale), 1.0)

    # assign per-event reweight factor
    mc_feat_bin = np.digitize(df_mc[feat], feat_edges) - 1
    mc_feat_bin = np.clip(mc_feat_bin, 0, N_BINS - 1)
    rw_per_event = rw_factors[mc_feat_bin]
    mc_weight_rw = df_mc["weight"].values * rw_per_event

    # ── reweighted score distribution ────────────────────────────────────────
    h_mc_rw, err_mc_rw = whist(df_mc["bdt_score"], mc_weight_rw, score_edges_1d)

    # ══ FIGURE 1: 2D ratio colormap ══════════════════════════════════════════
    finite_vals = ratio_2d[np.isfinite(ratio_2d)]
    vmin = np.nanpercentile(finite_vals, 2)
    vmax = np.nanpercentile(finite_vals, 98)
    vmin = min(vmin, 2.0 - vmax)
    vmax = max(vmax, 2.0 - vmin)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgrey")

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    extent = [feat_edges[0], feat_edges[-1], score_edges[0], score_edges[-1]]
    im = ax1.imshow(ratio_2d, aspect="auto", origin="lower",
                    extent=extent, cmap=cmap, norm=norm)
    plt.colorbar(im, ax=ax1, label="Data / MC")

    feat_centres  = 0.5 * (feat_edges[:-1] + feat_edges[1:])
    score_centres = 0.5 * (score_edges[:-1] + score_edges[1:])
    for i, sc in enumerate(score_centres):
        for j, fc in enumerate(feat_centres):
            val = ratio_2d[i, j]
            if np.isfinite(val):
                text_color = "black" if 0.7 < val < 1.3 else "white"
                ax1.text(fc, sc, f"{val:.2f}", ha="center", va="center",
                         fontsize=7, color=text_color)

    ax1.set_title(f"Data / MC  —  {feat}", fontsize=11)
    ax1.set_xlabel(feat)
    ax1.set_ylabel("BDT score")
    plt.tight_layout()
    fig1.savefig(os.path.join(config["output"], f"ratio_2d_{feat}.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)

 # ══ FIGURE 2: BDT score before/after reweighting on this feature ═════════
    fig2, axes2 = plt.subplots(2, 1, figsize=(7, 7),
                               gridspec_kw={"height_ratios": [3, 1]},
                               sharex=True)
    ax_main, ax_ratio = axes2

    # Data as points with error bars
    ax_main.errorbar(score_centres_1d, h_data_base, yerr=err_data_base,
                     fmt="o", color="black", markersize=3, linewidth=1.0,
                     label="Data", zorder=5)

    # MC nominal as step + shaded uncertainty band
    ax_main.step(score_centres_1d, h_mc_base, where="mid",
                 color="steelblue", linewidth=1.5, linestyle="--", label="MC (nominal)")
    ax_main.fill_between(score_centres_1d,
                         h_mc_base - err_mc_base,
                         h_mc_base + err_mc_base,
                         step="mid", alpha=0.3, color="steelblue")

    # MC reweighted as step + shaded uncertainty band
    ax_main.step(score_centres_1d, h_mc_rw, where="mid",
                 color="tomato", linewidth=1.5, label=f"MC (rw {feat})")
    ax_main.fill_between(score_centres_1d,
                         h_mc_rw - err_mc_rw,
                         h_mc_rw + err_mc_rw,
                         step="mid", alpha=0.3, color="tomato")

    ax_main.set_ylabel("Normalised events / bin")
    ax_main.set_title(f"BDT score after reweighting on  {feat}")
    ax_main.legend(fontsize=9)

    # ── ratio panel ───────────────────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        r_nom = np.where(h_mc_base > 0, h_data_base / h_mc_base, np.nan)
        r_rw  = np.where(h_mc_rw   > 0, h_data_base / h_mc_rw,   np.nan)

        # uncertainty on ratio: propagate data and MC errors
        err_r_nom = np.where(h_mc_base > 0,
                             np.sqrt((err_data_base / h_mc_base)**2 +
                                     (h_data_base * err_mc_base / h_mc_base**2)**2),
                             np.nan)
        err_r_rw  = np.where(h_mc_rw > 0,
                             np.sqrt((err_data_base / h_mc_rw)**2 +
                                     (h_data_base * err_mc_rw / h_mc_rw**2)**2),
                             np.nan)

    ax_ratio.errorbar(score_centres_1d, r_nom, yerr=err_r_nom,
                      fmt="s", color="steelblue", markersize=3,
                      linewidth=1.0, linestyle="--", label="Data/MC nom.")
    ax_ratio.errorbar(score_centres_1d, r_rw,  yerr=err_r_rw,
                      fmt="o", color="tomato",    markersize=3,
                      linewidth=1.0, label="Data/MC rw")
    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data / MC")
    ax_ratio.set_xlabel("BDT score")
    ax_ratio.legend(fontsize=8)

    plt.tight_layout()
    fig2.savefig(os.path.join(config["output"], f"score_reweight_{feat}.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ══ FIGURE 3: Profile — mean BDT score vs feature ════════════════════════
    feat_edges_prof = np.linspace(combined.quantile(0.01),
                                  combined.quantile(0.99), N_BINS + 1)
    feat_centres_prof = 0.5 * (feat_edges_prof[:-1] + feat_edges_prof[1:])

    def weighted_profile(sub_df, feat, edges):
        """Weighted mean and std-of-mean of bdt_score in each feature bin."""
        means, errs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (sub_df[feat] >= lo) & (sub_df[feat] < hi)
            w = sub_df.loc[mask, "weight"].values
            s = sub_df.loc[mask, "bdt_score"].values
            if w.sum() > 0:
                mu  = np.average(s, weights=w)
                var = np.average((s - mu)**2, weights=w)
                means.append(mu)
                errs.append(np.sqrt(var / np.count_nonzero(w)))
            else:
                means.append(np.nan)
                errs.append(np.nan)
        return np.array(means), np.array(errs)

    prof_data, err_data = weighted_profile(df_data, feat, feat_edges_prof)
    prof_mc,   err_mc   = weighted_profile(df_mc,   feat, feat_edges_prof)

    fig3, axes3 = plt.subplots(2, 1, figsize=(7, 7),
                                gridspec_kw={"height_ratios": [3, 1]},
                                sharex=True)
    ax_prof, ax_diff = axes3

    ax_prof.errorbar(feat_centres_prof, prof_data, yerr=err_data,
                     fmt="o", color="black",    label="Data",  markersize=4)
    ax_prof.errorbar(feat_centres_prof, prof_mc,   yerr=err_mc,
                     fmt="s", color="steelblue", label="MC",    markersize=4)
    ax_prof.set_ylabel("Mean BDT score")
    ax_prof.set_title(f"Profile: mean BDT score vs  {feat}")
    ax_prof.legend(fontsize=9)

    # difference panel
    diff = prof_data - prof_mc
    err_diff = np.sqrt(err_data**2 + err_mc**2)
    ax_diff.errorbar(feat_centres_prof, diff, yerr=err_diff,
                     fmt="o", color="black", markersize=4)
    ax_diff.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax_diff.set_ylabel("Data − MC")
    ax_diff.set_xlabel(feat)

    plt.tight_layout()
    fig3.savefig(os.path.join(config["output"], f"profile_{feat}.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved plots for feature: {feat}")



def chi2_data_mc(h_data, h_mc, err_data, err_mc):
    mask   = (h_mc > 0) & np.isfinite(h_data) & np.isfinite(h_mc)
    sigma2 = err_data[mask]**2 + err_mc[mask]**2
    return np.sum((h_data[mask] - h_mc[mask])**2 / sigma2), mask.sum()

MC_STACK_ORDER = ["nonprompt", "QCD-VV", "DY", "WJets", "SingleTop", "TT", "VBS_EWK"]

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

        print("group", group, "weight", sum(wgts))
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
        print("data","weight", sum(wgts_d))
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
    ax_main.set_yscale('log')
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

SCORE_BINS  = 10
score_edges = np.linspace(0.0, 1.0, SCORE_BINS + 1)
score_mc_nom = df_mc["bdt_score"].values

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
    ax_main.set_yscale('log')
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
