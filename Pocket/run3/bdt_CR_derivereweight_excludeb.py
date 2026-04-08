#!/usr/bin/env python3
"""
Flavour-blind tag-dependent QvG reweighting from W CR only.

Strategy:
  - SF for b-jets = 1 (fixed, not corrected)
  - For all other jets (quark, gluon, unmatched), derive a single
    flavour-blind tag-dependent SF from W CR only.
    "Flavour-blind" means we do NOT use partonFlavour to split the MC
    templates — we treat all non-b jets as a single inclusive population
    and derive one SF(tag, pt, eta) that corrects the inclusive non-b
    tag distribution to match data.
  - The SF is derived shape-only (normalisation-agnostic).
  - Apply to both W CR and TT CR and compare normalised BDT score
    distributions of MC vs data.

Nonprompt is data-driven (negative weights, no partonFlavour):
  - Excluded from MC templates
  - Added to data with negative weights to form prompt-subtracted data
  - Left unchanged when applying SFs
"""

import xgboost as xgb
import pandas as pd
import coffea.util
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
from matplotlib.colors import TwoSlopeNorm

print(xgb.__version__)
hep.style.use("CMS")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

config = {
    # W control region
    "wcr_categories":            ["resolved_mu_WCR"],
    "wcr_paths":                 ["/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_jetflavor/"],
    "wcr_paths_nonprompt":       ["/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/"],
    "wcr_paths_data":            ["/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/"],
    "wcr_normtables":            ["/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_resolvedCR_jetflavor"],
    "wcr_normtables_nonprompt":  ["/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_CR"],
    # TT control region
    "ttcr_categories":           ["resolved_mu_TTCR"],
    "ttcr_paths":                ["/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_jetflavor/"],
    "ttcr_paths_nonprompt":      ["/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/"],
    "ttcr_paths_data":           ["/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/"],
    "ttcr_normtables":           ["/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_resolvedCR_jetflavor"],
    "ttcr_normtables_nonprompt": ["/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_TTCR"],
    "ttcr_normtables_data":      ["/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_TTCR"],
    # output
    "year":        "2022_postEE",
    "output":      "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/bdt_CR_qvg_blind_cal",
    "calibration": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/bdt_CR_qvg_blind_cal/qvg_calibration.py",
}

LUMI = 26.67  # fb^-1

models_dic = {
    "resolved_mu": [
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_0_iter_138.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_1_iter_172.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_2_iter_184.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_3_iter_194.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_4_iter_196.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_5_iter_197.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_6_iter_130.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_7_iter_186.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_mu_8_iter_182.json",
    ],
    "resolved_e": [
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_0_iter_177.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_1_iter_180.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_2_iter_121.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_3_iter_157.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_4_iter_211.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_5_iter_149.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_6_iter_112.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_7_iter_166.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_resolved_e_8_iter_166.json",
    ],
    "boosted_mu": [
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_0_iter_115.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_1_iter_99.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_2_iter_131.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_3_iter_81.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_4_iter_130.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_5_iter_196.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_6_iter_128.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_7_iter_97.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_mu_8_iter_98.json",
    ],
    "boosted_e": [
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_0_iter_130.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_1_iter_113.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_2_iter_165.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_3_iter_92.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_4_iter_113.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_5_iter_128.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_6_iter_126.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_7_iter_94.json",
        "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/params/bdt_boosted_e_8_iter_86.json",
    ],
}

plot_colors = {
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
        "EGamma_2022_EraE", "EGamma_2022_EraF", "EGamma_2022_EraG",
        "Muon_2022_EraE",   "Muon_2022_EraF",   "Muon_2022_EraG",
    ],
}

MC_STACK_ORDER = ["nonprompt", "QCD-VV", "DY", "WJets", "SingleTop", "TT", "VBS_EWK"]

jet_tag_cols  = ["jet1_btagPNetQvG", "jet2_btagPNetQvG",
                 "jet3_btagPNetQvG", "jet4_btagPNetQvG"]
jet_pt_cols   = ["jet1_pt",  "jet2_pt",  "jet3_pt",  "jet4_pt"]
jet_eta_cols  = ["jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta"]
jet_flav_cols = ["jet1_partonFlavour", "jet2_partonFlavour",
                 "jet3_partonFlavour", "jet4_partonFlavour"]

os.makedirs(config["output"], exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Flavour mask: b-jets (SF fixed to 1), everything else gets the blind SF
# ─────────────────────────────────────────────────────────────────────────────

def is_b(flav): return np.abs(flav) == 5

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


def load_cr(categories, paths, normtables, paths_data,
            paths_nonprompt, normtables_nonprompt, year):
    """
    Load one control region.
    Returns:
      df_mc_prompt    : prompt MC (has partonFlavour, positive weights)
      df_mc_nonprompt : nonprompt estimate (data-driven, negative weights)
      df_data         : real data
    """
    df_list = []

    for group in process_groups:
        if group == "Data":
            continue
        for process in process_groups[group]:
            for cat, path, table in zip(categories, paths, normtables):
                df_p, sgw = load_MC_process_from_parquet(
                    path, table, cat, process, year)
                if df_p is not None:
                    df_p["weight"] /= sgw
                    df_p["group"]   = group
                    df_list.append(df_p)

    for process in process_groups["Data"]:
        for cat, path in zip(categories, paths_data):
            df_p = load_data_from_parquet(path, cat, process)
            if df_p is not None:
                df_p["group"] = "Data"
                df_list.append(df_p)

    for group in process_groups:
        if group == "Data":
            continue
        for process in process_groups[group]:
            for cat, path, table in zip(
                    categories, paths_nonprompt, normtables_nonprompt):
                df_p, sgw = load_MC_process_from_parquet(
                    path, table, cat, process, year)
                if df_p is not None:
                    df_p["weight"]   = -df_p["weight"] / sgw
                    df_p["process"]  = "nonprompt"
                    df_p["year_tag"] = f"nonprompt_{year}"
                    df_p["group"]    = "nonprompt"
                    df_list.append(df_p)

    for process in process_groups["Data"]:
        for cat, path in zip(categories, paths_nonprompt):
            df_p = load_data_from_parquet(path, cat, process)
            if df_p is not None:
                df_p["process"]  = "nonprompt"
                df_p["year_tag"] = f"nonprompt_{year}"
                df_p["group"]    = "nonprompt"
                df_list.append(df_p)

    df = pd.concat(df_list, ignore_index=True)
    mask_data      = df["year_tag"] == "Data"
    mask_nonprompt = df["group"]    == "nonprompt"
    mask_prompt_mc = ~mask_data & ~mask_nonprompt

    return (df[mask_prompt_mc].copy(),
            df[mask_nonprompt].copy(),
            df[mask_data].copy())


print("Loading W CR...")
df_mc_wcr_prompt, df_mc_wcr_nonprompt, df_data_wcr = load_cr(
    config["wcr_categories"], config["wcr_paths"], config["wcr_normtables"],
    config["wcr_paths_data"],
    config["wcr_paths_nonprompt"], config["wcr_normtables_nonprompt"],
    config["year"])

print("Loading TT CR...")
df_mc_ttcr_prompt, df_mc_ttcr_nonprompt, df_data_ttcr = load_cr(
    config["ttcr_categories"], config["ttcr_paths"], config["ttcr_normtables"],
    config["ttcr_paths_data"],
    config["ttcr_paths_nonprompt"], config["ttcr_normtables_nonprompt"],
    config["year"])

# prompt-subtracted effective data for SF derivation (W CR only)
df_eff_data_wcr = pd.concat([df_data_wcr, df_mc_wcr_nonprompt], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
# Jet collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_jets(sub_df, pt_cols, eta_cols, tag_cols, flav_cols=None):
    pt_list, eta_list, tag_list, w_list = [], [], [], []
    flav_list = [] if flav_cols is not None else None

    for idx, (pt_col, eta_col, tag_col) in enumerate(
            zip(pt_cols, eta_cols, tag_cols)):
        flav_col = flav_cols[idx] if flav_cols is not None else None
        valid = (sub_df[pt_col].notna() &
                 sub_df[eta_col].notna() &
                 sub_df[tag_col].notna())
        if flav_col is not None:
            valid = valid & sub_df[flav_col].notna()
        pt_list .append(sub_df.loc[valid, pt_col ].values)
        eta_list.append(sub_df.loc[valid, eta_col].values)
        tag_list.append(sub_df.loc[valid, tag_col].values)
        w_list  .append(sub_df.loc[valid, "weight"].values)
        if flav_list is not None:
            flav_list.append(sub_df.loc[valid, flav_col].values.astype(int))

    result = (np.concatenate(pt_list),  np.concatenate(eta_list),
              np.concatenate(tag_list), np.concatenate(w_list))
    if flav_list is not None:
        result = result + (np.concatenate(flav_list),)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Binning from W CR real data
# ─────────────────────────────────────────────────────────────────────────────

N_TAG_BINS        = 20
N_PT_BINS_CENTRAL = 8
N_PT_BINS_FORWARD = 4
N_ETA_BINS_BARREL = 8
N_ETA_BINS_TRANS  = 2
N_ETA_BINS_FWD    = 3

pt_d, eta_d, tag_d, w_d = collect_jets(
    df_data_wcr, jet_pt_cols, jet_eta_cols, jet_tag_cols)
abs_eta_d = np.abs(eta_d)

tag_edges   = weighted_quantile_edges(tag_d, w_d, N_TAG_BINS, 0.0, 1.0)
tag_centres = 0.5 * (tag_edges[:-1] + tag_edges[1:])

pt_edges_central = weighted_quantile_edges(
    pt_d[abs_eta_d <= 2.5], w_d[abs_eta_d <= 2.5],
    N_PT_BINS_CENTRAL, 30.0, 450.0)
pt_edges_forward = weighted_quantile_edges(
    pt_d[abs_eta_d > 2.5],  w_d[abs_eta_d > 2.5],
    N_PT_BINS_FORWARD, 50.0, 450.0)

eta_edges_barrel  = weighted_quantile_edges(abs_eta_d, w_d, N_ETA_BINS_BARREL, 0.0, 2.5)
eta_edges_trans   = weighted_quantile_edges(abs_eta_d, w_d, N_ETA_BINS_TRANS,  2.5, 3.0)
eta_edges_fwd_reg = weighted_quantile_edges(abs_eta_d, w_d, N_ETA_BINS_FWD,    3.0, 4.7)
eta_edges = np.concatenate([
    eta_edges_barrel, eta_edges_trans[1:], eta_edges_fwd_reg[1:]])
eta_edges_central = eta_edges[eta_edges <= 2.5 + 1e-9]
eta_edges_forward = eta_edges[eta_edges >= 2.5 - 1e-9]

# ─────────────────────────────────────────────────────────────────────────────
# Histogram building
# ─────────────────────────────────────────────────────────────────────────────

def region_jets(pt_v, eta_v, tag_v, w_v, flav_v=None,
                eta_lo=0.0, eta_hi=2.5, pt_lo=30.0):
    abs_eta = np.abs(eta_v)
    m = ((abs_eta > eta_lo) & (abs_eta <= eta_hi) &
         (pt_v >= pt_lo) & (pt_v <= 450.0))
    if flav_v is not None:
        return pt_v[m], abs_eta[m], tag_v[m], w_v[m], flav_v[m]
    return pt_v[m], abs_eta[m], tag_v[m], w_v[m]


def hist3d(pt_v, abs_eta_v, tag_v, w_v, pt_edges_r, eta_edges_r):
    if len(pt_v) == 0:
        return np.zeros((len(pt_edges_r)-1, len(eta_edges_r)-1, len(tag_edges)-1))
    h, _ = np.histogramdd(
        np.column_stack([pt_v, abs_eta_v, tag_v]),
        bins=[pt_edges_r, eta_edges_r, tag_edges],
        weights=w_v)
    return h


def build_wcr_hists(df_mc_prompt, df_eff_data,
                    pt_edges_r, eta_edges_r,
                    eta_lo, eta_hi, pt_lo):
    """
    Build two 3D histograms for the W CR:
      h_non_b : inclusive non-b prompt MC (all jets where partonFlavour != 5)
      h_b     : b-jet prompt MC           (partonFlavour == 5)
      h_data  : prompt-subtracted effective data (all jets, no flavour split)

    The SF is derived from: eff_data - h_b = SF(tag) * h_non_b
    so the b contribution is subtracted from data before solving.
    """
    pt_mc, eta_mc, tag_mc, w_mc, flav_mc = collect_jets(
        df_mc_prompt, jet_pt_cols, jet_eta_cols, jet_tag_cols, jet_flav_cols)
    r_mc = region_jets(pt_mc, eta_mc, tag_mc, w_mc, flav_mc,
                       eta_lo=eta_lo, eta_hi=eta_hi, pt_lo=pt_lo)
    pt_r, abs_eta_r, tag_r, w_r, flav_r = r_mc

    mask_b     = is_b(flav_r)
    mask_non_b = ~mask_b

    h_non_b = hist3d(pt_r[mask_non_b], abs_eta_r[mask_non_b],
                     tag_r[mask_non_b], w_r[mask_non_b],
                     pt_edges_r, eta_edges_r)
    h_b     = hist3d(pt_r[mask_b],     abs_eta_r[mask_b],
                     tag_r[mask_b],     w_r[mask_b],
                     pt_edges_r, eta_edges_r)

    pt_eff, eta_eff, tag_eff, w_eff = collect_jets(
        df_eff_data, jet_pt_cols, jet_eta_cols, jet_tag_cols)
    r_eff = region_jets(pt_eff, eta_eff, tag_eff, w_eff,
                        eta_lo=eta_lo, eta_hi=eta_hi, pt_lo=pt_lo)
    h_data = hist3d(r_eff[0], r_eff[1], r_eff[2], r_eff[3],
                    pt_edges_r, eta_edges_r)

    return h_non_b, h_b, h_data

# ─────────────────────────────────────────────────────────────────────────────
# Build histograms from W CR only
# ─────────────────────────────────────────────────────────────────────────────

print("Building W CR histograms...")

h_non_b_cen, h_b_cen, h_data_cen = build_wcr_hists(
    df_mc_wcr_prompt, df_eff_data_wcr,
    pt_edges_central, eta_edges_central, 0.0, 2.5, 30.0)

h_non_b_fwd, h_b_fwd, h_data_fwd = build_wcr_hists(
    df_mc_wcr_prompt, df_eff_data_wcr,
    pt_edges_forward, eta_edges_forward, 2.5, 4.7, 50.0)

# ─────────────────────────────────────────────────────────────────────────────
# Derive flavour-blind tag-dependent SF from W CR
#
# Per (pt, eta) cell, solve:
#   eff_data(tag) - h_b(tag) = SF(tag) * h_non_b(tag)
#   => SF(tag) = [eff_data(tag) - h_b(tag)] / h_non_b(tag)
#
# Both sides normalised to unit area first (shape-only).
# Regularised with smoothness prior to suppress bin noise.
# ─────────────────────────────────────────────────────────────────────────────

def solve_blind_sf(h_non_b, h_b, h_data,
                   smoothing_strength=1.0,
                   clip_min=0.1, clip_max=10.0):
    """
    Derive a single flavour-blind tag-dependent SF per (pt, eta) cell.

    After subtracting b-jet MC (SF_b=1) from effective data, the residual
    should be explained by SF(tag) * MC_non_b(tag). Both are normalised to
    unit area so the SF corrects shape only.

    Regularised least-squares with smoothness prior.
    Returns sf of shape (N_PT, N_ETA, N_TAG).
    """
    n_pt, n_eta, n_tag = h_non_b.shape
    n_unknowns  = n_tag
    n_equations = n_tag

    # smoothness regularisation: penalise |SF(t+1) - SF(t)|
    n_smooth = n_tag - 1
    R = np.zeros((n_smooth, n_unknowns))
    for t in range(n_smooth):
        R[t, t]   = -1.0
        R[t, t+1] =  1.0

    sf_out = np.ones((n_pt, n_eta, n_tag))

    for i in range(n_pt):
        for j in range(n_eta):
            non_b = h_non_b[i, j]
            b_mc  = h_b    [i, j]
            data  = h_data [i, j]

            # residual data after subtracting b-jet MC (SF_b = 1)
            resid = data - b_mc

            non_b_sum = non_b.sum()
            resid_sum = resid.sum()

            # skip cells with insufficient MC or data
            if non_b_sum < 1e-9 or resid_sum < 1e-9:
                continue

            # normalise to unit area (shape-only)
            non_b_norm = non_b / non_b_sum
            resid_norm = resid / resid_sum

            # equation: SF(t) * non_b_norm(t) = resid_norm(t)
            # => diagonal system A where A[t,t] = non_b_norm[t]
            A     = np.diag(non_b_norm)
            b_vec = resid_norm

            # regularised least squares
            lam   = smoothing_strength / np.sqrt(n_tag)
            A_aug = np.vstack([A,      lam * R])
            b_aug = np.concatenate([b_vec, np.zeros(n_smooth)])

            sol, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            sf_out[i, j, :] = np.clip(sol, clip_min, clip_max)

    return sf_out


print("Solving flavour-blind tag-dependent SF from W CR...")
sf_blind_cen = solve_blind_sf(h_non_b_cen, h_b_cen, h_data_cen,
                               smoothing_strength=1.0)
sf_blind_fwd = solve_blind_sf(h_non_b_fwd, h_b_fwd, h_data_fwd,
                               smoothing_strength=1.0)
print("  Done.")

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plots: SF heatmap and SF vs tag
# ─────────────────────────────────────────────────────────────────────────────

def plot_sf_table(sf_table, name, pt_edges_r, eta_edges_r):
    sf_mean = sf_table.mean(axis=2)
    fig, ax = plt.subplots(figsize=(8, 5))
    extent  = [eta_edges_r[0], eta_edges_r[-1],
               pt_edges_r[0],  pt_edges_r[-1]]
    im = ax.imshow(sf_mean, aspect="auto", origin="lower",
                   extent=extent, cmap="RdBu_r", vmin=0.5, vmax=1.5)
    plt.colorbar(im, ax=ax, label="Mean SF (over QvG bins)")
    ax.set_xlabel("|eta|"); ax.set_ylabel("pt [GeV]")
    ax.set_title(f"QvG blind SF: {name}")
    plt.tight_layout()
    fig.savefig(os.path.join(config["output"], f"sf_{name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sf_vs_tag(sf_table, name, pt_edges_r, eta_edges_r):
    n_pt  = len(pt_edges_r) - 1
    n_eta = len(eta_edges_r) - 1
    pt_idxs  = [n_pt  // 4, n_pt  // 2, 3 * n_pt  // 4]
    eta_idxs = [n_eta // 4, n_eta // 2, 3 * n_eta // 4]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors_plot = plt.cm.viridis(
        np.linspace(0, 1, len(pt_idxs) * len(eta_idxs)))
    c = 0
    for pi in pt_idxs:
        for ei in eta_idxs:
            if pi >= n_pt or ei >= n_eta:
                continue
            lbl = (f"pt [{pt_edges_r[pi]:.0f},{pt_edges_r[pi+1]:.0f}] "
                   f"eta [{eta_edges_r[ei]:.1f},{eta_edges_r[ei+1]:.1f}]")
            ax.plot(tag_centres, sf_table[pi, ei, :],
                    color=colors_plot[c], linewidth=1.5, label=lbl)
            c += 1

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("QvG tag value")
    ax.set_ylabel("SF(tag)")
    ax.set_title(f"Flavour-blind tag-dependent SF: {name}")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0.0, 3.0)
    plt.tight_layout()
    fig.savefig(os.path.join(config["output"], f"sf_vs_tag_{name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


for sf, name in [
    (sf_blind_cen, "blind_central"),
    (sf_blind_fwd, "blind_forward"),
]:
    pt_e  = pt_edges_central if "central" in name else pt_edges_forward
    eta_e = eta_edges_central if "central" in name else eta_edges_forward
    plot_sf_table(sf, name, pt_e, eta_e)
    plot_sf_vs_tag(sf, name, pt_e, eta_e)

print("Saved SF diagnostic plots.")

# ─────────────────────────────────────────────────────────────────────────────
# Per-jet SF lookup
# SF_b = 1; all other jets get the blind SF based on (pt, eta, tag)
# ─────────────────────────────────────────────────────────────────────────────

def get_jet_sf(pt_vals, eta_vals, tag_vals, flav_vals):
    """
    Return per-jet SF.
    b-jets (|partonFlavour| == 5): SF = 1.0
    all other jets: SF from blind table indexed by (pt, eta, tag)
    """
    pt_vals   = np.asarray(pt_vals,  dtype=float)
    abs_eta   = np.abs(np.asarray(eta_vals, dtype=float))
    tag_vals  = np.asarray(tag_vals, dtype=float)
    flav_vals = np.asarray(flav_vals, dtype=int)
    sf_out    = np.ones(len(pt_vals))

    for (pt_lo, eta_lo, eta_hi,
         pt_edges_r, eta_edges_r, sf_table) in [
        (30.0, 0.0, 2.5,
         pt_edges_central, eta_edges_central, sf_blind_cen),
        (50.0, 2.5, 4.7,
         pt_edges_forward, eta_edges_forward, sf_blind_fwd),
    ]:
        in_reg = ((abs_eta > eta_lo) & (abs_eta <= eta_hi) &
                  (pt_vals >= pt_lo) & (pt_vals <= 450.0))
        if not in_reg.any():
            continue

        pt_idx  = np.clip(np.digitize(pt_vals[in_reg],  pt_edges_r)  - 1,
                          0, len(pt_edges_r)  - 2)
        eta_idx = np.clip(np.digitize(abs_eta[in_reg],  eta_edges_r) - 1,
                          0, len(eta_edges_r) - 2)
        tag_idx = np.clip(np.digitize(tag_vals[in_reg], tag_edges)   - 1,
                          0, len(tag_edges)   - 2)

        flav_here = flav_vals[in_reg]
        sf_here   = np.ones(in_reg.sum())

        # apply blind SF to all non-b jets
        mask_non_b = ~is_b(flav_here)
        if mask_non_b.any():
            sf_here[mask_non_b] = sf_table[
                pt_idx[mask_non_b],
                eta_idx[mask_non_b],
                tag_idx[mask_non_b]]
        # b-jets: sf_here stays 1.0

        sf_out[in_reg] = sf_here

    return sf_out


def apply_sf_to_prompt_mc(df_mc_prompt_in, df_mc_nonprompt_in):
    """
    Apply geometric-mean per-jet SF to prompt MC only.
    Nonprompt is returned unchanged.
    """
    df_prompt  = df_mc_prompt_in.copy()
    log_sf_sum = np.zeros(len(df_prompt))
    sf_count   = np.zeros(len(df_prompt))

    for pt_col, eta_col, tag_col, flav_col in zip(
            jet_pt_cols, jet_eta_cols, jet_tag_cols, jet_flav_cols):
        valid = (df_prompt[pt_col].notna()   &
                 df_prompt[eta_col].notna()  &
                 df_prompt[tag_col].notna()  &
                 df_prompt[flav_col].notna())
        if not valid.any():
            continue
        sf = get_jet_sf(
            df_prompt.loc[valid, pt_col  ].values,
            df_prompt.loc[valid, eta_col ].values,
            df_prompt.loc[valid, tag_col ].values,
            df_prompt.loc[valid, flav_col].values,
        )
        log_sf_sum[valid.values] += np.log(np.clip(sf, 1e-6, 1e6))
        sf_count  [valid.values] += 1

    sf_count[sf_count == 0] = 1
    df_prompt["weight"] *= np.exp(log_sf_sum / sf_count)
    return pd.concat([df_prompt, df_mc_nonprompt_in], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Save calibration file
# ─────────────────────────────────────────────────────────────────────────────

calib_code = f"""# Auto-generated flavour-blind QvG calibration
# Derived from W CR only. SF_b = 1 (fixed).
# All other jets get a single tag-dependent SF(pt, eta, tag).
# Shape-only: normalisation is not corrected.
# Apply only to prompt MC; nonprompt is left unchanged.

import numpy as np
import pandas as pd

pt_edges_central  = np.array({pt_edges_central.tolist()})
pt_edges_forward  = np.array({pt_edges_forward.tolist()})
eta_edges_central = np.array({eta_edges_central.tolist()})
eta_edges_forward = np.array({eta_edges_forward.tolist()})
tag_edges         = np.array({tag_edges.tolist()})

sf_blind_cen = np.array({sf_blind_cen.tolist()})
sf_blind_fwd = np.array({sf_blind_fwd.tolist()})

def is_b(flav): return np.abs(flav) == 5

def get_jet_sf(pt_vals, eta_vals, tag_vals, flav_vals):
    pt_vals   = np.asarray(pt_vals,  dtype=float)
    abs_eta   = np.abs(np.asarray(eta_vals, dtype=float))
    tag_vals  = np.asarray(tag_vals, dtype=float)
    flav_vals = np.asarray(flav_vals, dtype=int)
    sf_out    = np.ones(len(pt_vals))

    for (pt_lo, eta_lo, eta_hi, pt_edges_r, eta_edges_r, sf_table) in [
        (30.0, 0.0, 2.5, pt_edges_central, eta_edges_central, sf_blind_cen),
        (50.0, 2.5, 4.7, pt_edges_forward, eta_edges_forward, sf_blind_fwd),
    ]:
        in_reg = ((abs_eta > eta_lo) & (abs_eta <= eta_hi) &
                  (pt_vals >= pt_lo) & (pt_vals <= 450.0))
        if not in_reg.any(): continue
        pt_idx  = np.clip(np.digitize(pt_vals[in_reg],  pt_edges_r)  - 1,
                          0, len(pt_edges_r)  - 2)
        eta_idx = np.clip(np.digitize(abs_eta[in_reg],  eta_edges_r) - 1,
                          0, len(eta_edges_r) - 2)
        tag_idx = np.clip(np.digitize(tag_vals[in_reg], tag_edges)   - 1,
                          0, len(tag_edges)   - 2)
        flav_here  = flav_vals[in_reg]
        sf_here    = np.ones(in_reg.sum())
        mask_non_b = ~is_b(flav_here)
        if mask_non_b.any():
            sf_here[mask_non_b] = sf_table[
                pt_idx[mask_non_b], eta_idx[mask_non_b], tag_idx[mask_non_b]]
        sf_out[in_reg] = sf_here
    return sf_out

def apply_sf_to_prompt_mc(df_mc_prompt, df_mc_nonprompt,
                           jet_pt_cols, jet_eta_cols,
                           jet_tag_cols, jet_flav_cols):
    df_prompt  = df_mc_prompt.copy()
    log_sf_sum = np.zeros(len(df_prompt))
    sf_count   = np.zeros(len(df_prompt))
    for pt_col, eta_col, tag_col, flav_col in zip(
            jet_pt_cols, jet_eta_cols, jet_tag_cols, jet_flav_cols):
        valid = (df_prompt[pt_col].notna()  & df_prompt[eta_col].notna() &
                 df_prompt[tag_col].notna() & df_prompt[flav_col].notna())
        if not valid.any(): continue
        sf = get_jet_sf(df_prompt.loc[valid, pt_col].values,
                        df_prompt.loc[valid, eta_col].values,
                        df_prompt.loc[valid, tag_col].values,
                        df_prompt.loc[valid, flav_col].values)
        log_sf_sum[valid.values] += np.log(np.clip(sf, 1e-6, 1e6))
        sf_count  [valid.values] += 1
    sf_count[sf_count == 0] = 1
    df_prompt["weight"] *= np.exp(log_sf_sum / sf_count)
    return pd.concat([df_prompt, df_mc_nonprompt], ignore_index=True)
"""

os.makedirs(os.path.dirname(os.path.abspath(config["calibration"])), exist_ok=True)
with open(config["calibration"], "w") as f:
    f.write(calib_code)
print(f"Calibration saved to {config['calibration']}")

# ─────────────────────────────────────────────────────────────────────────────
# Apply calibration to W CR and TT CR
# ─────────────────────────────────────────────────────────────────────────────

print("Applying calibration to W CR MC...")
df_mc_wcr_cal  = apply_sf_to_prompt_mc(df_mc_wcr_prompt,  df_mc_wcr_nonprompt)

print("Applying calibration to TT CR MC...")
df_mc_ttcr_cal = apply_sf_to_prompt_mc(df_mc_ttcr_prompt, df_mc_ttcr_nonprompt)

df_mc_wcr_nom  = pd.concat([df_mc_wcr_prompt,  df_mc_wcr_nonprompt],  ignore_index=True)
df_mc_ttcr_nom = pd.concat([df_mc_ttcr_prompt, df_mc_ttcr_nonprompt], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# BDT scoring
# ─────────────────────────────────────────────────────────────────────────────

boosters = {}
feature_names = {}
for cat in models_dic:
    boosters[cat] = []
    for path in models_dic[cat]:
        bdt = xgb.Booster(); bdt.load_model(path)
        boosters[cat].append(bdt)
    feature_names[cat] = boosters[cat][0].feature_names


def score_df(sub_df, category):
    dmat = xgb.DMatrix(sub_df[feature_names[category]],
                       feature_names=feature_names[category])
    return np.array([bdt.predict(dmat)
                     for bdt in boosters[category]]).mean(axis=0)


df_mc_wcr_nom ["bdt_score"] = score_df(df_mc_wcr_nom,  "resolved_mu")
df_data_wcr   ["bdt_score"] = score_df(df_data_wcr,    "resolved_mu")
df_mc_ttcr_nom["bdt_score"] = score_df(df_mc_ttcr_nom, "resolved_mu")
df_data_ttcr  ["bdt_score"] = score_df(df_data_ttcr,   "resolved_mu")

score_wcr_cal  = score_df(df_mc_wcr_cal,  "resolved_mu")
score_ttcr_cal = score_df(df_mc_ttcr_cal, "resolved_mu")

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

SCORE_BINS       = 20
score_edges      = np.linspace(0.0, 1.0, SCORE_BINS + 1)
N_TAG_PLOT       = 25
tag_plot_edges   = np.linspace(0.0, 1.0, N_TAG_PLOT + 1)
tag_plot_centres = 0.5 * (tag_plot_edges[:-1] + tag_plot_edges[1:])
N_CDF_BINS       = 100
cdf_edges        = np.linspace(0.0, 1.0, N_CDF_BINS + 1)
cdf_centres      = 0.5 * (cdf_edges[:-1] + cdf_edges[1:])


def make_cms_fig(title):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    hep.cms.label(ax=axes[0], label="Preliminary", data=True,
                  lumi=LUMI, lumi_format="{:.2f}", com=13.6)
    axes[0].set_title(title, fontsize=10, pad=30)
    return fig, axes[0], axes[1]


def cms_norm_plot(ax_main, ax_ratio,
                  df_mc_plot, df_data_plot,
                  edges, xlabel,
                  var_col=None, mc_scores=None, data_scores=None):
    """
    CMS-style normalised shape comparison.
    Both MC total and data are normalised to unit area independently.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)
    stack_bottom = np.zeros(len(centres))
    h_mc_total   = np.zeros(len(centres))
    h_mc_tot2    = np.zeros(len(centres))
    group_hists  = {}

    for group in MC_STACK_ORDER:
        mask_g = df_mc_plot["group"] == group
        if not mask_g.any(): continue
        vals = (df_mc_plot.loc[mask_g, var_col].values
                if var_col is not None else mc_scores[mask_g.values])
        wgts = df_mc_plot.loc[mask_g, "weight"].values
        h_g,  _ = np.histogram(vals, bins=edges, weights=wgts)
        h_g2, _ = np.histogram(vals, bins=edges, weights=wgts**2)
        group_hists[group] = (h_g, h_g2)
        h_mc_total += h_g; h_mc_tot2 += h_g2

    err_mc  = np.sqrt(h_mc_tot2)
    mc_area = (h_mc_total * widths).sum()
    if mc_area > 0:
        h_mc_norm   = h_mc_total  / mc_area
        err_mc_norm = err_mc      / mc_area
    else:
        h_mc_norm   = h_mc_total
        err_mc_norm = err_mc

    for group in MC_STACK_ORDER:
        if group not in group_hists: continue
        h_g, _ = group_hists[group]
        h_g_n  = h_g / mc_area if mc_area > 0 else h_g
        color  = plot_colors.get(group, "#888888")
        ax_main.bar(centres, h_g_n, width=widths, bottom=stack_bottom,
                    color=color, label=group, align="center", alpha=0.85)
        stack_bottom += h_g_n

    ax_main.bar(centres, 2 * err_mc_norm, width=widths,
                bottom=h_mc_norm - err_mc_norm,
                color="grey", alpha=0.4, label="MC stat. unc.",
                align="center", hatch="///", linewidth=0)

    if var_col is not None:
        valid_d = df_data_plot[var_col].notna()
        vals_d  = df_data_plot.loc[valid_d, var_col].values
        wgts_d  = df_data_plot.loc[valid_d, "weight"].values
    else:
        vals_d = data_scores
        wgts_d = df_data_plot["weight"].values

    h_d,  _ = np.histogram(vals_d, bins=edges, weights=wgts_d)
    h_d2, _ = np.histogram(vals_d, bins=edges, weights=wgts_d**2)
    err_d   = np.sqrt(h_d2)
    d_area  = (h_d * widths).sum()
    if d_area > 0:
        h_d_norm   = h_d  / d_area
        err_d_norm = err_d / d_area
    else:
        h_d_norm   = h_d
        err_d_norm = err_d

    ax_main.errorbar(centres, h_d_norm, yerr=err_d_norm,
                     fmt="o", color="black", markersize=4,
                     linewidth=1.2, label="Data", zorder=10)
    ax_main.set_ylabel("Normalised events / bin")
    ax_main.legend(fontsize=11, ncol=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio   = np.where(h_mc_norm > 0, h_d_norm / h_mc_norm, np.nan)
        err_r   = np.where(h_mc_norm > 0,
                           np.sqrt((err_d_norm / h_mc_norm)**2 +
                                   (h_d_norm * err_mc_norm / h_mc_norm**2)**2),
                           np.nan)
        rel_err = np.where(h_mc_norm > 0, err_mc_norm / h_mc_norm, np.nan)

    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax_ratio.fill_between(centres, 1-rel_err, 1+rel_err,
                          step="mid", alpha=0.3, color="grey")
    ax_ratio.errorbar(centres, ratio, yerr=err_r,
                      fmt="o", color="black", markersize=3, linewidth=1.0)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data / MC")
    ax_ratio.set_xlabel(xlabel)

    chi2, ndof = chi2_data_mc(h_d_norm, h_mc_norm, err_d_norm, err_mc_norm)
    return chi2, ndof


def plot_cdf(tag_col, df_nom, df_cal, df_data_plot, cr_label):
    fig, axes = plt.subplots(2, 1, figsize=(7, 7),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    ax_cdf, ax_diff = axes
    cdf_data_ref = None; cdf_data_err_ref = None

    for lbl, sub_df, color, ls in [
        ("Data",            df_data_plot, "black",     "-"),
        ("MC (nominal)",    df_nom,       "steelblue", "--"),
        ("MC (calibrated)", df_cal,       "tomato",    "-"),
    ]:
        valid = sub_df[tag_col].notna()
        vals  = sub_df.loc[valid, tag_col].values
        wgts  = sub_df.loc[valid, "weight"].values
        h,  _ = np.histogram(vals, bins=cdf_edges, weights=wgts)
        h2, _ = np.histogram(vals, bins=cdf_edges, weights=wgts**2)
        cumh  = np.cumsum(h); total = cumh[-1]
        cdf_y   = cumh / total   if total > 0 else np.zeros_like(cumh, float)
        cdf_err = np.sqrt(np.cumsum(h2)) / total if total > 0 else np.zeros_like(cumh)

        if lbl == "Data":
            ax_cdf.errorbar(cdf_centres, cdf_y, yerr=cdf_err,
                            fmt="o", color=color, markersize=2,
                            linewidth=1.0, label=lbl, zorder=5)
            cdf_data_ref = cdf_y; cdf_data_err_ref = cdf_err
        else:
            ax_cdf.plot(cdf_centres, cdf_y, color=color,
                        linewidth=1.5, linestyle=ls, label=lbl)
            ax_cdf.fill_between(cdf_centres,
                                cdf_y - cdf_err, cdf_y + cdf_err,
                                alpha=0.2, color=color)
            diff     = cdf_y - cdf_data_ref
            err_diff = np.sqrt(cdf_err**2 + cdf_data_err_ref**2)
            ax_diff.plot(cdf_centres, diff, color=color,
                         linewidth=1.5, linestyle=ls, label=lbl)
            ax_diff.fill_between(cdf_centres,
                                 diff - err_diff, diff + err_diff,
                                 alpha=0.2, color=color)

    ax_cdf.set_ylabel("CDF"); ax_cdf.set_ylim(0, 1)
    ax_cdf.set_title(f"{cr_label} {tag_col}: CDF before/after calibration")
    ax_cdf.legend(fontsize=9)
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax_diff.set_ylabel("MC - Data"); ax_diff.set_xlabel(tag_col)
    ax_diff.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(config["output"],
                             f"cdf_{cr_label}_{tag_col}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Validation plots: normalised BDT score for W CR and TT CR
# ─────────────────────────────────────────────────────────────────────────────

for cr_label, df_mc_nom, df_mc_cal, df_data_cr, bdt_nom, bdt_cal in [
    ("WCR",  df_mc_wcr_nom,  df_mc_wcr_cal,  df_data_wcr,
     df_mc_wcr_nom ["bdt_score"].values, score_wcr_cal),
    ("TTCR", df_mc_ttcr_nom, df_mc_ttcr_cal, df_data_ttcr,
     df_mc_ttcr_nom["bdt_score"].values, score_ttcr_cal),
]:
    print(f"Making {cr_label} validation plots...")

    for label, df_mc_plot, bdt_scores, suffix in [
        ("nominal",    df_mc_nom, bdt_nom, "nominal"),
        ("calibrated", df_mc_cal, bdt_cal, "calibrated"),
    ]:
        # BDT score (normalised)
        fig, ax_m, ax_r = make_cms_fig(
            f"{cr_label} BDT score - {label} (normalised)")
        chi2, ndof = cms_norm_plot(
            ax_m, ax_r, df_mc_plot, df_data_cr,
            edges=score_edges, xlabel="BDT score",
            mc_scores=bdt_scores,
            data_scores=df_data_cr["bdt_score"].values)
        print(f"  [{cr_label} BDT {label}] chi2/ndof = {chi2:.1f} / {ndof}")
        plt.tight_layout()
        fig.savefig(os.path.join(config["output"],
                                 f"{cr_label.lower()}_bdt_{suffix}_norm.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # jet tag distributions (normalised)
        for tag_col in jet_tag_cols:
            fig, ax_m, ax_r = make_cms_fig(
                f"{cr_label} {tag_col} - {label} (normalised)")
            chi2, ndof = cms_norm_plot(
                ax_m, ax_r, df_mc_plot, df_data_cr,
                edges=tag_plot_edges, xlabel=tag_col, var_col=tag_col)
            print(f"  [{cr_label} {tag_col} {label}] chi2/ndof = {chi2:.1f} / {ndof}")
            plt.tight_layout()
            fig.savefig(os.path.join(config["output"],
                                     f"{cr_label.lower()}_{tag_col}_{suffix}_norm.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # CDF plots
    for tag_col in jet_tag_cols:
        plot_cdf(tag_col, df_mc_nom, df_mc_cal, df_data_cr, cr_label)
        print(f"  Saved CDF: {cr_label} {tag_col}")
# ─────────────────────────────────────────────────────────────────────────────
# Per-flavour tag score plots: stacked by process, separately for W CR / TT CR
# Each plot shows the inclusive distribution of btagPNetQvG for jets of a
# given truth flavour, stacked over all prompt MC processes + nonprompt,
# compared to data.
# ─────────────────────────────────────────────────────────────────────────────

def cms_norm_plot_flavour(ax_main, ax_ratio,
                          df_mc_plot, df_data_plot,
                          edges, xlabel,
                          flav_mask_fn=None):
    """
    Like cms_norm_plot but collects jet-level values (one entry per jet, not
    per event) filtered by a per-jet flavour mask applied to prompt MC jets.
    For data, all jets are included (no flavour label available).
    For nonprompt, all jets are included (no flavour label).

    flav_mask_fn : callable(flav_array) -> bool array, or None for all jets
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)
    stack_bottom = np.zeros(len(centres))
    h_mc_total   = np.zeros(len(centres))
    h_mc_tot2    = np.zeros(len(centres))
    group_hists  = {}

    for group in MC_STACK_ORDER:
        mask_g = df_mc_plot["group"] == group
        if not mask_g.any():
            continue
        sub = df_mc_plot[mask_g]

        vals_list, wgts_list = [], []
        for tag_col, flav_col in zip(jet_tag_cols, jet_flav_cols):
            if group == "nonprompt":
                # nonprompt has no partonFlavour — include all jets regardless
                valid = sub[tag_col].notna()
                vals_list.append(sub.loc[valid, tag_col].values)
                wgts_list.append(sub.loc[valid, "weight"].values)
            else:
                # prompt MC: apply flavour filter if provided
                valid = sub[tag_col].notna() & sub[flav_col].notna()
                if not valid.any():
                    continue
                flav = sub.loc[valid, flav_col].values.astype(int)
                if flav_mask_fn is not None:
                    sel = flav_mask_fn(flav)
                else:
                    sel = np.ones(len(flav), dtype=bool)
                if not sel.any():
                    continue
                vals_list.append(sub.loc[valid, tag_col].values[sel])
                wgts_list.append(sub.loc[valid, "weight" ].values[sel])

        if not vals_list:
            continue
        vals = np.concatenate(vals_list)
        wgts = np.concatenate(wgts_list)

        h_g,  _ = np.histogram(vals, bins=edges, weights=wgts)
        h_g2, _ = np.histogram(vals, bins=edges, weights=wgts**2)
        group_hists[group] = (h_g, h_g2)
        h_mc_total += h_g
        h_mc_tot2  += h_g2

    err_mc  = np.sqrt(h_mc_tot2)
    mc_area = (h_mc_total * widths).sum()
    if mc_area > 0:
        h_mc_norm   = h_mc_total / mc_area
        err_mc_norm = err_mc     / mc_area
    else:
        h_mc_norm   = h_mc_total
        err_mc_norm = err_mc

    for group in MC_STACK_ORDER:
        if group not in group_hists:
            continue
        h_g, _ = group_hists[group]
        h_g_n  = h_g / mc_area if mc_area > 0 else h_g
        color  = plot_colors.get(group, "#888888")
        ax_main.bar(centres, h_g_n, width=widths, bottom=stack_bottom,
                    color=color, label=group, align="center", alpha=0.85)
        stack_bottom += h_g_n

    ax_main.bar(centres, 2 * err_mc_norm, width=widths,
                bottom=h_mc_norm - err_mc_norm,
                color="grey", alpha=0.4, label="MC stat. unc.",
                align="center", hatch="///", linewidth=0)

    # data: all jets (no flavour label)
    vals_d_list, wgts_d_list = [], []
    for tag_col in jet_tag_cols:
        valid_d = df_data_plot[tag_col].notna()
        vals_d_list.append(df_data_plot.loc[valid_d, tag_col].values)
        wgts_d_list.append(df_data_plot.loc[valid_d, "weight"].values)
    vals_d = np.concatenate(vals_d_list)
    wgts_d = np.concatenate(wgts_d_list)

    h_d,  _ = np.histogram(vals_d, bins=edges, weights=wgts_d)
    h_d2, _ = np.histogram(vals_d, bins=edges, weights=wgts_d**2)
    err_d   = np.sqrt(h_d2)
    d_area  = (h_d * widths).sum()
    if d_area > 0:
        h_d_norm   = h_d  / d_area
        err_d_norm = err_d / d_area
    else:
        h_d_norm   = h_d
        err_d_norm = err_d

    ax_main.errorbar(centres, h_d_norm, yerr=err_d_norm,
                     fmt="o", color="black", markersize=4,
                     linewidth=1.2, label="Data", zorder=10)
    ax_main.set_ylabel("Normalised events / bin")
    ax_main.legend(fontsize=11, ncol=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio   = np.where(h_mc_norm > 0, h_d_norm / h_mc_norm, np.nan)
        err_r   = np.where(h_mc_norm > 0,
                           np.sqrt((err_d_norm / h_mc_norm)**2 +
                                   (h_d_norm * err_mc_norm / h_mc_norm**2)**2),
                           np.nan)
        rel_err = np.where(h_mc_norm > 0, err_mc_norm / h_mc_norm, np.nan)

    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax_ratio.fill_between(centres, 1 - rel_err, 1 + rel_err,
                          step="mid", alpha=0.3, color="grey")
    ax_ratio.errorbar(centres, ratio, yerr=err_r,
                      fmt="o", color="black", markersize=3, linewidth=1.0)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data / MC")
    ax_ratio.set_xlabel(xlabel)

    chi2, ndof = chi2_data_mc(h_d_norm, h_mc_norm, err_d_norm, err_mc_norm)
    return chi2, ndof


# flavour categories to plot
flavour_defs = [
    ("light_quark",
     lambda f: (np.abs(f) >= 1) & (np.abs(f) <= 3)),
    ("c_quark",
     lambda f:  np.abs(f) == 4),
    ("b_quark",
     lambda f:  np.abs(f) == 5),
    ("gluon",
     lambda f:  f == 21),
    ("unmatched",
     lambda f: ~((np.abs(f) >= 1) & (np.abs(f) <= 5)) & (f != 21)),
]

# plot both nominal and calibrated, for W CR and TT CR
for cr_label, df_mc_nom, df_mc_cal, df_data_cr in [
    ("WCR",  df_mc_wcr_nom,  df_mc_wcr_cal,  df_data_wcr),
    ("TTCR", df_mc_ttcr_nom, df_mc_ttcr_cal, df_data_ttcr),
]:
    print(f"Making {cr_label} per-flavour tag plots...")

    for label, df_mc_plot, suffix in [
        ("nominal",    df_mc_nom, "nominal"),
        ("calibrated", df_mc_cal, "calibrated"),
    ]:
        for flav_name, flav_fn in flavour_defs:
            fig, ax_m, ax_r = make_cms_fig(
                f"{cr_label} btagPNetQvG ({flav_name} jets) - {label}")
            chi2, ndof = cms_norm_plot_flavour(
                ax_m, ax_r,
                df_mc_plot, df_data_cr,
                edges=tag_plot_edges,
                xlabel="jet btagPNetQvG",
                flav_mask_fn=flav_fn)
            print(f"  [{cr_label} {flav_name} {label}] "
                  f"chi2/ndof = {chi2:.1f} / {ndof}")
            plt.tight_layout()
            fig.savefig(os.path.join(
                config["output"],
                f"{cr_label.lower()}_tag_{flav_name}_{suffix}.png"),
                dpi=150, bbox_inches="tight")
            plt.close(fig)
print("Done.")