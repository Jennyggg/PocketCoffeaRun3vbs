#!/usr/bin/env python3
import xgboost as xgb
import pandas as pd
import coffea.util
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d

print(xgb.__version__)
hep.style.use("CMS")

config = {
    "categories": ["resolved_mu_WCR","resolved_e_WCR"],
    "year": "2022_postEE",
    "paths": ["/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/","/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/"],
    "paths_nonprompt": ["/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/","/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/"],
    "normtables": ["/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_CR","/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_CR"],
    "normtables_nonprompt": ["/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_CR","/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_CR"],
    "output": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/bdt_CR_reweight_resolved_lep_WCR",
    "calibration": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/bdt_CR_reweight_resolved_lep_WCR/bdt_CR_reweight.py",
}

LUMI = 26.67  # fb^-1

models = [
    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_0_iter_138.json",
    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_1_iter_172.json",
    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_2_iter_184.json",
    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_3_iter_194.json",
    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_4_iter_196.json",
    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_5_iter_197.json",
    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_6_iter_130.json",
    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_7_iter_186.json",
    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_8_iter_182.json",
]

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
            for cat, path, table in zip(config["categories"],config["paths"],config["normtables"]):
                df_p, sgw = load_MC_process_from_parquet(
                    path, table,
                    cat, process, config["year"])
                if df_p is not None:
                    df_p["weight"] /= sgw
                    df_p["group"]   = group
                    df_list.append(df_p)
    else:
        for process in process_groups[group]:
            for cat, path in zip(config["categories"], config["paths"]):
                df_p = load_data_from_parquet(
                    path, cat, process)
                if df_p is not None:
                    df_p["group"] = "Data"
                    df_list.append(df_p)

for group in process_groups:
    if group != "Data":
        for process in process_groups[group]:
            for cat, path, table in zip(config["categories"], config["paths_nonprompt"], config["normtables_nonprompt"]):
                df_p, sgw = load_MC_process_from_parquet(
                    path, table,
                    cat, process, config["year"])
                if df_p is not None:
                    df_p["weight"]   = -df_p["weight"] / sgw
                    df_p["process"]  = "nonprompt"
                    df_p["year_tag"] = f"nonprompt_{config['year']}"
                    df_p["group"]    = "nonprompt"
                    df_list.append(df_p)
    else:
        for process in process_groups[group]:
            for cat, path in zip(config["categories"], config["paths_nonprompt"]):
                df_p = load_data_from_parquet(
                    path, cat, process)
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

X             = df[feature_names]
dmat          = xgb.DMatrix(X, feature_names=feature_names)
raw_scores    = np.array([bdt.predict(dmat) for bdt in boosters])
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
    """Normalised weighted histogram + stat uncertainty."""
    h,  _ = np.histogram(values, bins=edges, weights=weights)
    h2, _ = np.histogram(values, bins=edges, weights=weights**2)
    norm   = np.sum(h) * np.diff(edges)
    norm[norm == 0] = 1
    return h / norm, np.sqrt(h2) / norm


def chi2_data_mc(h_data, h_mc, err_data, err_mc):
    mask   = (h_mc > 0) & np.isfinite(h_data) & np.isfinite(h_mc)
    sigma2 = err_data[mask]**2 + err_mc[mask]**2
    return np.sum((h_data[mask] - h_mc[mask])**2 / sigma2), mask.sum()


def weighted_quantile_edges(values, weights, n_bins, vmin, vmax):
    in_range = (values >= vmin) & (values <= vmax)
    if in_range.sum() < n_bins:
        return np.linspace(vmin, vmax, n_bins + 1)
    v_sorted = np.sort(values[in_range])
    w_sorted = weights[in_range][np.argsort(values[in_range])]
    cumw     = np.cumsum(w_sorted); cumw /= cumw[-1]
    levels   = np.linspace(0.0, 1.0, n_bins + 1)
    edges    = np.interp(levels, cumw, v_sorted)
    edges[0] = vmin; edges[-1] = vmax
    return edges

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build reweighting table: Data/MC ratio in (pt, eta, tag) bins
# ─────────────────────────────────────────────────────────────────────────────

jet_tag_cols  = ["jet1_btagPNetQvG", "jet2_btagPNetQvG",
                 "jet3_btagPNetQvG", "jet4_btagPNetQvG"]
jet_pt_cols   = ["jet1_pt",  "jet2_pt",  "jet3_pt",  "jet4_pt"]
jet_eta_cols  = ["jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta"]

calib_jet_pt_cols  = ["jet1_pt",  "jet2_pt",  "jet3_pt",  "jet4_pt"]
calib_jet_eta_cols = ["jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta"]
calib_jet_tag_cols = ["jet1_btagPNetQvG", "jet2_btagPNetQvG",
                      "jet3_btagPNetQvG", "jet4_btagPNetQvG"]

# ── tag bins: finer at high values where calibration matters most ─────────────
N_TAG_CAL_BINS = 20
# use quantile-based tag edges from data so bins are equally populated
def weighted_quantile_edges_tag(values, weights, n_bins):
    v_sorted = np.sort(values)
    w_sorted = weights[np.argsort(values)]
    cumw     = np.cumsum(w_sorted); cumw /= cumw[-1]
    levels   = np.linspace(0.0, 1.0, n_bins + 1)
    edges    = np.interp(levels, cumw, v_sorted)
    edges[0] = 0.0; edges[-1] = 1.0
    return edges

N_PT_BINS_CENTRAL = 8
N_PT_BINS_FORWARD = 4
N_ETA_BINS_BARREL  = 8
N_ETA_BINS_TRANS   = 2
N_ETA_BINS_FORWARD = 3

def collect_jets_inclusive(sub_df, pt_cols, eta_cols, tag_cols):
    pt_list, eta_list, tag_list, w_list = [], [], [], []
    for pt_col, eta_col, tag_col in zip(pt_cols, eta_cols, tag_cols):
        valid = (sub_df[pt_col].notna() &
                 sub_df[eta_col].notna() &
                 sub_df[tag_col].notna())
        pt_list .append(sub_df.loc[valid, pt_col ].values)
        eta_list.append(sub_df.loc[valid, eta_col].values)
        tag_list.append(sub_df.loc[valid, tag_col].values)
        w_list  .append(sub_df.loc[valid, "weight"].values)
    return (np.concatenate(pt_list),  np.concatenate(eta_list),
            np.concatenate(tag_list), np.concatenate(w_list))

pt_mc,   eta_mc,   tag_mc,   wgt_mc   = collect_jets_inclusive(
    df_mc,   calib_jet_pt_cols, calib_jet_eta_cols, calib_jet_tag_cols)
pt_data, eta_data, tag_data, wgt_data = collect_jets_inclusive(
    df_data, calib_jet_pt_cols, calib_jet_eta_cols, calib_jet_tag_cols)

abs_eta_data = np.abs(eta_data)
abs_eta_mc   = np.abs(eta_mc)

# ── quantile-based tag edges from inclusive data ──────────────────────────────
tag_cal_edges   = weighted_quantile_edges_tag(tag_data, wgt_data, N_TAG_CAL_BINS)
tag_cal_centres = 0.5 * (tag_cal_edges[:-1] + tag_cal_edges[1:])

# ── pt and eta edges (same logic as before) ───────────────────────────────────
pt_edges_central = weighted_quantile_edges(
    pt_data[abs_eta_data <= 2.5], wgt_data[abs_eta_data <= 2.5],
    N_PT_BINS_CENTRAL, 30.0, 450.0)
pt_edges_forward = weighted_quantile_edges(
    pt_data[abs_eta_data > 2.5],  wgt_data[abs_eta_data > 2.5],
    N_PT_BINS_FORWARD, 50.0, 450.0)

eta_edges_barrel  = weighted_quantile_edges(abs_eta_data, wgt_data, N_ETA_BINS_BARREL,  0.0, 2.5)
eta_edges_trans   = weighted_quantile_edges(abs_eta_data, wgt_data, N_ETA_BINS_TRANS,   2.5, 3.0)
eta_edges_fwd_reg = weighted_quantile_edges(abs_eta_data, wgt_data, N_ETA_BINS_FORWARD, 3.0, 4.7)
eta_edges = np.concatenate([eta_edges_barrel, eta_edges_trans[1:], eta_edges_fwd_reg[1:]])
eta_edges_central = eta_edges[eta_edges <= 2.5 + 1e-9]
eta_edges_forward = eta_edges[eta_edges >= 2.5 - 1e-9]

# ── build 3D reweight tables: shape (N_PT, N_ETA, N_TAG) ─────────────────────
def make_rw_table(pt_v, eta_v, tag_v, wgt_v, pt_edges_r, eta_edges_r,
                  tag_edges_r, clip_min=0.1, clip_max=10.0):
    """
    Compute per-(pt, eta, tag) Data/MC weight table.
    Each (pt, eta) cell is normalised independently before taking ratio,
    so the weights correct shape only, not normalisation.
    Sparse bins (MC sum-of-weights < threshold) are set to 1.0.
    """
    h_data, _ = np.histogramdd(
        np.column_stack([pt_v, eta_v, tag_v]),
        bins=[pt_edges_r, eta_edges_r, tag_edges_r],
        weights=wgt_v)
    return h_data   # (N_PT, N_ETA, N_TAG)

def compute_rw_table(h_data, h_mc, clip_min=0.1, clip_max=10.0):
    """
    For each (pt, eta) cell normalise data and MC to same integral
    over tag axis, then take ratio. Bins with insufficient MC are set to 1.
    """
    rw = np.ones_like(h_data)
    for i in range(h_data.shape[0]):
        for j in range(h_data.shape[1]):
            d = h_data[i, j]
            m = h_mc  [i, j]
            sum_d = d.sum(); sum_m = m.sum()
            if sum_m < 1e-9 or sum_d < 1e-9:
                continue   # leave as 1.0
            d_norm = d / sum_d
            m_norm = m / sum_m
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(m_norm > 0, d_norm / m_norm, 1.0)
            rw[i, j] = np.clip(ratio, clip_min, clip_max)
    return rw   # (N_PT, N_ETA, N_TAG)

def get_region(pt_v, eta_v, tag_v, wgt_v, eta_lo, eta_hi, pt_lo, pt_hi):
    m = ((np.abs(eta_v) > eta_lo) & (np.abs(eta_v) <= eta_hi) &
         (pt_v >= pt_lo) & (pt_v <= 450.0))
    return pt_v[m], np.abs(eta_v[m]), tag_v[m], wgt_v[m]

# central
h_data_cen = make_rw_table(
    *get_region(pt_data, eta_data, tag_data, wgt_data, 0.0, 2.5, 30.0, 450.0),
    pt_edges_central, eta_edges_central, tag_cal_edges)
h_mc_cen   = make_rw_table(
    *get_region(pt_mc,   eta_mc,   tag_mc,   wgt_mc,   0.0, 2.5, 30.0, 450.0),
    pt_edges_central, eta_edges_central, tag_cal_edges)
rw_table_cen = compute_rw_table(h_data_cen, h_mc_cen)

# forward
h_data_fwd = make_rw_table(
    *get_region(pt_data, eta_data, tag_data, wgt_data, 2.5, 4.7, 50.0, 450.0),
    pt_edges_forward, eta_edges_forward, tag_cal_edges)
h_mc_fwd   = make_rw_table(
    *get_region(pt_mc,   eta_mc,   tag_mc,   wgt_mc,   2.5, 4.7, 50.0, 450.0),
    pt_edges_forward, eta_edges_forward, tag_cal_edges)
rw_table_fwd = compute_rw_table(h_data_fwd, h_mc_fwd)


def get_jet_weights(pt_vals, eta_vals, tag_vals):
    """
    Look up the per-jet reweighting factor from the 3D table.
    Jets outside the calibration range get weight 1.0.
    """
    pt_vals  = np.asarray(pt_vals,  dtype=float)
    abs_eta  = np.abs(np.asarray(eta_vals, dtype=float))
    tag_vals = np.asarray(tag_vals, dtype=float)
    jet_wgts = np.ones(len(pt_vals))

    # central
    in_cen = (abs_eta <= 2.5) & (pt_vals >= 30.0) & (pt_vals <= 450.0)
    if in_cen.any():
        pt_idx  = np.clip(np.digitize(pt_vals[in_cen], pt_edges_central)  - 1,
                          0, len(pt_edges_central)  - 2)
        eta_idx = np.clip(np.digitize(abs_eta[in_cen], eta_edges_central) - 1,
                          0, len(eta_edges_central) - 2)
        tag_idx = np.clip(np.digitize(tag_vals[in_cen], tag_cal_edges)    - 1,
                          0, len(tag_cal_edges)     - 2)
        jet_wgts[in_cen] = rw_table_cen[pt_idx, eta_idx, tag_idx]

    # forward
    in_fwd = (abs_eta > 2.5) & (abs_eta <= 4.7) & (pt_vals >= 50.0) & (pt_vals <= 450.0)
    if in_fwd.any():
        pt_idx  = np.clip(np.digitize(pt_vals[in_fwd], pt_edges_forward)  - 1,
                          0, len(pt_edges_forward)  - 2)
        eta_idx = np.clip(np.digitize(abs_eta[in_fwd], eta_edges_forward) - 1,
                          0, len(eta_edges_forward) - 2)
        tag_idx = np.clip(np.digitize(tag_vals[in_fwd], tag_cal_edges)    - 1,
                          0, len(tag_cal_edges)     - 2)
        jet_wgts[in_fwd] = rw_table_fwd[pt_idx, eta_idx, tag_idx]

    return jet_wgts


# ── apply reweighting: multiply MC event weight by product of jet weights ─────
df_mc_cal = df_mc.copy()
for pt_col, eta_col, tag_col in zip(jet_pt_cols, jet_eta_cols, jet_tag_cols):
    valid = (df_mc_cal[pt_col].notna() &
             df_mc_cal[eta_col].notna() &
             df_mc_cal[tag_col].notna())
    jet_w = get_jet_weights(
        df_mc_cal.loc[valid, pt_col].values,
        df_mc_cal.loc[valid, eta_col].values,
        df_mc_cal.loc[valid, tag_col].values,
    )
    df_mc_cal.loc[valid, "weight"] *= jet_w


# ── save reweighting table ────────────────────────────────────────────────────
rw_code = f"""# Auto-generated QvG reweighting table
# Data/MC per-bin weights in (pt, |eta|, QvG) space
# Jets outside calibration range get weight 1.0
# Apply weight as: event_weight *= product of jet weights

import numpy as np

pt_edges_central  = np.array({pt_edges_central.tolist()})
pt_edges_forward  = np.array({pt_edges_forward.tolist()})
eta_edges_central = np.array({eta_edges_central.tolist()})
eta_edges_forward = np.array({eta_edges_forward.tolist()})
tag_cal_edges     = np.array({tag_cal_edges.tolist()})

rw_table_cen = np.array({rw_table_cen.tolist()})
rw_table_fwd = np.array({rw_table_fwd.tolist()})

def get_jet_weights(pt_vals, eta_vals, tag_vals):
    pt_vals  = np.asarray(pt_vals,  dtype=float)
    abs_eta  = np.abs(np.asarray(eta_vals, dtype=float))
    tag_vals = np.asarray(tag_vals, dtype=float)
    jet_wgts = np.ones(len(pt_vals))

    in_cen = (abs_eta <= 2.5) & (pt_vals >= 30.0) & (pt_vals <= 450.0)
    if in_cen.any():
        pt_idx  = np.clip(np.digitize(pt_vals[in_cen], pt_edges_central)  - 1,
                          0, len(pt_edges_central)  - 2)
        eta_idx = np.clip(np.digitize(abs_eta[in_cen], eta_edges_central) - 1,
                          0, len(eta_edges_central) - 2)
        tag_idx = np.clip(np.digitize(tag_vals[in_cen], tag_cal_edges)    - 1,
                          0, len(tag_cal_edges)     - 2)
        jet_wgts[in_cen] = rw_table_cen[pt_idx, eta_idx, tag_idx]

    in_fwd = (abs_eta > 2.5) & (abs_eta <= 4.7) & (pt_vals >= 50.0) & (pt_vals <= 450.0)
    if in_fwd.any():
        pt_idx  = np.clip(np.digitize(pt_vals[in_fwd], pt_edges_forward)  - 1,
                          0, len(pt_edges_forward)  - 2)
        eta_idx = np.clip(np.digitize(abs_eta[in_fwd], eta_edges_forward) - 1,
                          0, len(eta_edges_forward) - 2)
        tag_idx = np.clip(np.digitize(tag_vals[in_fwd], tag_cal_edges)    - 1,
                          0, len(tag_cal_edges)     - 2)
        jet_wgts[in_fwd] = rw_table_fwd[pt_idx, eta_idx, tag_idx]

    return jet_wgts
"""

os.makedirs(os.path.dirname(os.path.abspath(config["calibration"])), exist_ok=True)
with open(config["calibration"], "w") as f:
    f.write(rw_code)
print(f"Reweighting table saved to {config['calibration']}")


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
        ratio     = np.where(h_mc_total > 0, h_data / h_mc_total, np.nan)
        err_ratio = np.where(h_mc_total > 0,
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

SCORE_BINS  = 20
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
        cumh  = np.cumsum(h)
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
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)

    # ── build total weighted MC histogram ────────────────────────────────────
    h_mc_total  = np.zeros(len(centres))
    h_mc_total2 = np.zeros(len(centres))
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
    mc_area = (h_mc_total * widths).sum()
    if mc_area > 0:
        h_mc_norm   = h_mc_total  / mc_area
        err_mc_norm = err_mc_total / mc_area
    else:
        h_mc_norm   = h_mc_total
        err_mc_norm = err_mc_total

    # stacked normalised MC
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
