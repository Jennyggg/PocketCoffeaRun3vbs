#!/usr/bin/env python3
"""
Validation script for QvG reweighting using the locally derived
btagPNetQvG_SF scale factors from run3_QvGSF/qvgsf_2022_postEE_merged.json.

Per-jet SF is evaluated as:
    btagPNetQvG_SF.evaluate(systematic, flavor, score, abs_eta, pt)
where:
    flavor  = "quark" for |partonFlavour| in {1,2,3,4,5}
              "gluon" for |partonFlavour| == 21
              SF = 1 for all other flavours (b, undef)
    Valid range: 30 <= pT < 8000 GeV, |eta| < 3.0

Structure mirrors bdt_CR_validatereweight_qgtagging.py.
"""

import correctionlib
import xgboost as xgb
import pandas as pd
import coffea.util
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os

hep.style.use("CMS")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

paths_prompt_dic = {
    "resolved_mu_WCR":  "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_jetflavor/",
    "resolved_e_WCR":   "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_jetflavor/",
    "resolved_mu_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_jetflavor/",
    "resolved_e_TTCR":  "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_jetflavor/",
}

paths_nonprompt_dic = {
    "resolved_mu_WCR":  "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "resolved_e_WCR":   "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "resolved_mu_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
    "resolved_e_TTCR":  "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel_invertlepton/",
}

paths_data_dic = {
    "resolved_mu_WCR":  "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "resolved_e_WCR":   "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "resolved_mu_TTCR": "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
    "resolved_e_TTCR":  "/eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_updatesel/",
}

normtables_prompt_dic = {
    "resolved_mu_WCR":  "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_resolvedCR_jetflavor",
    "resolved_e_WCR":   "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_resolvedCR_jetflavor",
    "resolved_mu_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_resolvedCR_jetflavor",
    "resolved_e_TTCR":  "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_save_resolvedCR_jetflavor",
}

normtables_nonprompt_dic = {
    "resolved_mu_WCR":  "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_CR",
    "resolved_e_WCR":   "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_CR",
    "resolved_mu_TTCR": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_TTCR",
    "resolved_e_TTCR":  "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/output_VBS_data2022postEE_nonprompt_save_TTCR",
}

config = {
    "category":  "resolved_mu_TTCR",
    "year":      "2022_postEE",
    "output":    "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3/bdt_reweight_val_qvgsf_resolved_e_WCR",
    "qvgsf_file": "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3_QvGSF/qvgsf_2022_postEE_dj.json",
    "sf_diag":   "/afs/cern.ch/work/j/jinw/public/semilep_vbs_base/Pocket/run3_QvGSF/qvgsf_2022_postEE_dj_diag.npz",
}
config["path_prompt"]         = paths_prompt_dic[config["category"]]
config["path_nonprompt"]      = paths_nonprompt_dic[config["category"]]
config["path_data"]           = paths_data_dic[config["category"]]
config["normtable_prompt"]    = normtables_prompt_dic[config["category"]]
config["normtable_nonprompt"] = normtables_nonprompt_dic[config["category"]]

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
}

cat_key = config["category"].replace("_WCR", "").replace("_TTCR", "")
models  = models_dic[cat_key]

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
        "WtoLNu-2Jets_PTLNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
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

# ─────────────────────────────────────────────────────────────────────────────
# Load correctionlib scale factors
# ─────────────────────────────────────────────────────────────────────────────

cset    = correctionlib.CorrectionSet.from_file(config["qvgsf_file"])
qvg_sf  = cset["btagPNetQvG_SF"]

print(f"Loaded btagPNetQvG_SF from {config['qvgsf_file']}")


def get_jet_sf(pt_vals, eta_vals, score_vals, flav_vals,
               systematic="central"):
    """
    Per-jet SF from locally derived btagPNetQvG_SF correctionlib.

    Evaluates SF(systematic, flavor, score, |eta|, pt) for each jet.
    flavor is "quark" for |partonFlavour| in {1,2,3,4},
              "gluon" for |partonFlavour| == 21.
    All other flavours (b-jets, undefined) get SF = 1.
    Jets outside [30, 8000) GeV pT or |eta| >= 3.0 get SF = 1.
    If the statistical uncertainty (max(|up-central|, |central-dn|)) > 1,
    the jet SF is set to 1.
    """
    pt_vals    = np.asarray(pt_vals,    dtype=float)
    abs_eta    = np.abs(np.asarray(eta_vals,   dtype=float))
    score_vals = np.asarray(score_vals, dtype=float)
    flav_abs   = np.abs(np.asarray(flav_vals,  dtype=int))

    n      = len(pt_vals)
    sf_out = np.ones(n)

    # kinematic validity: pT in [30, 8000), |eta| < 3.0, score in [0, 1]
    valid_kin = (np.isfinite(pt_vals) & np.isfinite(abs_eta) &
                 np.isfinite(score_vals) &
                 (pt_vals    >= 30.0)  & (pt_vals    < 8000.0) &
                 (abs_eta     <  3.0)  &
                 (score_vals >= 0.0)   & (score_vals <= 1.0))

    # light quarks (uds, c): partonFlavour in {1,2,3,4}
    # b-jets (5): no SF (our correction is for light-quark vs gluon only)
    quark_mask = valid_kin & np.isin(flav_abs, [1, 2, 3, 4])
    gluon_mask = valid_kin & (flav_abs == 21)

    for mask, flavor in [(quark_mask, "quark"), (gluon_mask, "gluon")]:
        if not mask.any():
            continue
        # clip score to [0, 1) — correctionlib uses clamp flow but be safe
        score_clipped = np.clip(score_vals[mask], 0.0, 0.9999)
        ae  = abs_eta[mask]
        pt  = pt_vals[mask]
        sf_c  = qvg_sf.evaluate("central",  flavor, score_clipped, ae, pt)
        sf_up = qvg_sf.evaluate("stat_up",  flavor, score_clipped, ae, pt)
        sf_dn = qvg_sf.evaluate("stat_dn",  flavor, score_clipped, ae, pt)
        # If stat uncertainty > 1, reset to SF = 1
        large_unc = (np.abs(sf_up - sf_c) > 1.0) | (np.abs(sf_c - sf_dn) > 1.0)
        sf_c[large_unc] = 1.0
        sf_up[large_unc] = 1.0
        sf_dn[large_unc] = 1.0
        if systematic == "central":
            sf_out[mask] = sf_c
        elif systematic == "stat_up":
            sf_out[mask] = sf_up
        elif systematic == "stat_dn":
            sf_out[mask] = sf_dn
        else:
            sf_out[mask] = sf_c

    return sf_out


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
    ft  = coffea.util.load(
        f"{parquet_table_dir}/output_{process}_{year}.coffea")
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


# ─────────────────────────────────────────────────────────────────────────────
# Load prompt MC (jetflavor path), nonprompt, and data
# ─────────────────────────────────────────────────────────────────────────────

df_prompt_list    = []
df_nonprompt_list = []
df_data_list      = []

for group, processes in process_groups.items():
    if group == "Data":
        for process in processes:
            df_p = load_data_from_parquet(
                config["path_data"], config["category"], process)
            if df_p is not None:
                df_p["group"] = "Data"
                df_data_list.append(df_p)

        for process in processes:
            df_p = load_data_from_parquet(
                config["path_nonprompt"], config["category"], process)
            if df_p is not None:
                df_p["process"]  = "nonprompt"
                df_p["year_tag"] = f"nonprompt_{config['year']}"
                df_p["group"]    = "nonprompt"
                df_nonprompt_list.append(df_p)
    else:
        for process in processes:
            df_p, sgw = load_MC_process_from_parquet(
                config["path_prompt"], config["normtable_prompt"],
                config["category"], process, config["year"])
            if df_p is not None:
                df_p["weight"] /= sgw
                df_p["group"]   = group
                df_prompt_list.append(df_p)

            df_np, sgw_np = load_MC_process_from_parquet(
                config["path_nonprompt"], config["normtable_nonprompt"],
                config["category"], process, config["year"])
            if df_np is not None:
                df_np["weight"]   = -df_np["weight"] / sgw_np
                df_np["process"]  = "nonprompt"
                df_np["year_tag"] = f"nonprompt_{config['year']}"
                df_np["group"]    = "nonprompt"
                df_nonprompt_list.append(df_np)

df_prompt    = pd.concat(df_prompt_list,    ignore_index=True)
df_nonprompt = pd.concat(df_nonprompt_list, ignore_index=True)
df_data      = pd.concat(df_data_list,      ignore_index=True)

df_mc_nom = pd.concat([df_prompt, df_nonprompt], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# BDT scoring
# ─────────────────────────────────────────────────────────────────────────────

boosters = []
for path in models:
    bdt = xgb.Booster()
    bdt.load_model(path)
    boosters.append(bdt)

feature_names = boosters[0].feature_names


def score_df_sub(sub_df):
    dmat = xgb.DMatrix(sub_df[feature_names], feature_names=feature_names)
    return np.array([bdt.predict(dmat) for bdt in boosters]).mean(axis=0)


df_mc_nom["bdt_score"] = score_df_sub(df_mc_nom)
df_data  ["bdt_score"] = score_df_sub(df_data)

os.makedirs(config["output"], exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Apply btagPNetQvG_SF to prompt MC; nonprompt is unchanged
# ─────────────────────────────────────────────────────────────────────────────

df_prompt_cal = df_prompt.copy()

n_jets_with_sf = 0
for pt_col, eta_col, tag_col, flav_col in zip(
        jet_pt_cols, jet_eta_cols, jet_tag_cols, jet_flav_cols):
    valid = (df_prompt_cal[pt_col].notna()   &
             df_prompt_cal[eta_col].notna()  &
             df_prompt_cal[tag_col].notna()  &
             df_prompt_cal[flav_col].notna())
    if not valid.any():
        continue
    sf = get_jet_sf(
        df_prompt_cal.loc[valid, pt_col  ].values,
        df_prompt_cal.loc[valid, eta_col ].values,
        df_prompt_cal.loc[valid, tag_col ].values,
        df_prompt_cal.loc[valid, flav_col].values,
    )
    df_prompt_cal.loc[valid, "weight"] *= sf
    n_jets_with_sf += valid.sum()

print(f"Applied btagPNetQvG_SF to {n_jets_with_sf} jet-event entries in prompt MC")

df_mc_cal = pd.concat([df_prompt_cal, df_nonprompt], ignore_index=True)
df_mc_cal["bdt_score"] = score_df_sub(df_mc_cal)

# ── Load SF variation arrays and compute varied event weights ─────────────
# Used to propagate SF uncertainties as bands on the calibrated plots.
# Systematic variations come from the diagnostic npz; stat from correctionlib.

_SF_SCORE_EDGES = np.array([0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0])
_SF_PT_EDGES    = np.array([30., 40., 50., 60., 80., 140., 200., 260., 8000.])
_SF_ETA_EDGES   = np.array([0.0, 1.3, 2.5, 3.0, 4.7])


def _eval_sf_from_array(sf_q_arr, sf_g_arr, pt_vals, eta_vals, score_vals, flav_vals):
    """Evaluate per-jet SFs from a (18, 8, 4) numpy array using bin lookup."""
    pt    = np.asarray(pt_vals,    dtype=float)
    ae    = np.abs(np.asarray(eta_vals,   dtype=float))
    score = np.asarray(score_vals, dtype=float)
    flav  = np.abs(np.asarray(flav_vals,  dtype=int))
    n     = len(pt)

    valid = (np.isfinite(pt) & np.isfinite(ae) & np.isfinite(score) &
             (pt >= 30.0) & (pt < 8000.0) & (ae < 3.0) &
             (score >= 0.0) & (score <= 1.0))

    sf_out = np.ones(n)
    i_sc = np.clip(np.digitize(score, _SF_SCORE_EDGES) - 1, 0, 17)
    i_pt = np.clip(np.digitize(pt,    _SF_PT_EDGES)    - 1, 0,  7)
    i_et = np.clip(np.digitize(ae,    _SF_ETA_EDGES)   - 1, 0,  3)

    for mask, arr in [(valid & np.isin(flav, [1, 2, 3, 4]), sf_q_arr),
                      (valid & (flav == 21),                 sf_g_arr)]:
        if mask.any():
            sf_out[mask] = arr[i_sc[mask], i_pt[mask], i_et[mask]]
    return sf_out


def _varied_mc_weights(sf_q_arr=None, sf_g_arr=None, correctionlib_sys=None):
    """
    Apply varied SFs to df_prompt (starting from unweighted copy) and return
    a weight array aligned with df_mc_cal (prompt rows first, then nonprompt).
    Exactly one of (sf_q_arr/sf_g_arr) or correctionlib_sys must be given.
    """
    df_var = df_prompt.copy()
    for pt_col, eta_col, tag_col, flav_col in zip(
            jet_pt_cols, jet_eta_cols, jet_tag_cols, jet_flav_cols):
        valid = (df_var[pt_col].notna() & df_var[eta_col].notna() &
                 df_var[tag_col].notna() & df_var[flav_col].notna())
        if not valid.any():
            continue
        if correctionlib_sys is not None:
            sf = get_jet_sf(
                df_var.loc[valid, pt_col].values,
                df_var.loc[valid, eta_col].values,
                df_var.loc[valid, tag_col].values,
                df_var.loc[valid, flav_col].values,
                systematic=correctionlib_sys,
            )
        else:
            sf = _eval_sf_from_array(
                sf_q_arr, sf_g_arr,
                df_var.loc[valid, pt_col].values,
                df_var.loc[valid, eta_col].values,
                df_var.loc[valid, tag_col].values,
                df_var.loc[valid, flav_col].values.astype(int),
            )
        df_var.loc[valid, "weight"] *= sf
    return np.concatenate([df_var["weight"].values,
                           df_nonprompt["weight"].values])


# Stat variations (correctionlib stat_up / stat_dn)
_ps_stat_weights = [_varied_mc_weights(correctionlib_sys="stat_up"),
                    _varied_mc_weights(correctionlib_sys="stat_dn")]

# Systematic variations from diag.npz (isr, fsr, jes, jer, pu up/dn)
_ps_syst_weights = []
if os.path.exists(config["sf_diag"]):
    _diag = np.load(config["sf_diag"])
    _diag_var_keys = [k[len("SF_q_"):] for k in _diag.files
                      if k.startswith("SF_q_") and not k.endswith("_err")]
    for _vk in _diag_var_keys:
        _ps_syst_weights.append(
            _varied_mc_weights(_diag[f"SF_q_{_vk}"], _diag[f"SF_g_{_vk}"]))
    print(f"Computed syst weight variations: {_diag_var_keys}")
else:
    print(f"sf_diag not found, skipping syst bands: {config['sf_diag']}")

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


def make_cms_fig(title):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    hep.cms.label(ax=axes[0], label="Preliminary", data=True,
                  lumi=LUMI, lumi_format="{:.2f}", com=13.6)
    axes[0].set_title(title, fontsize=10, pad=30)
    return fig, axes[0], axes[1]


# ─────────────────────────────────────────────────────────────────────────────
# CMS stacked-histogram helper
# ─────────────────────────────────────────────────────────────────────────────

def _syst_envelope(h_central, varied_weights_list, df_mc_plot, edges,
                    var_col=None, mc_scores=None):
    """
    Compute per-bin syst envelope: max |h_var - h_central| over all variations.
    varied_weights_list: list of weight arrays aligned with df_mc_plot.
    """
    env = np.zeros(len(h_central))
    for w_var in varied_weights_list:
        h_var = np.zeros_like(h_central)
        for group in MC_STACK_ORDER:
            mask_g = (df_mc_plot["group"] == group).values
            if not mask_g.any():
                continue
            vals = (df_mc_plot.loc[mask_g, var_col].values if var_col is not None
                    else mc_scores[mask_g])
            h_g, _ = np.histogram(vals, bins=edges, weights=w_var[mask_g])
            h_var += h_g
        env = np.maximum(env, np.abs(h_var - h_central))
    return env


def cms_stack_plot(ax_main, ax_ratio,
                   df_mc_plot, df_data_plot,
                   edges, xlabel,
                   var_col=None,
                   mc_scores=None,
                   data_scores=None,
                   syst_weights=None,
                   stat_weights=None):
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

    # SF uncertainty bands (only when variations are provided)
    _syst_env = _stat_env = _total_env = None
    if syst_weights or stat_weights:
        kw_syst = dict(var_col=var_col, mc_scores=mc_scores)
        if syst_weights:
            _syst_env = _syst_envelope(h_mc_total, syst_weights,
                                       df_mc_plot, edges, **kw_syst)
        if stat_weights:
            _stat_env = _syst_envelope(h_mc_total, stat_weights,
                                       df_mc_plot, edges, **kw_syst)
        s = (_syst_env if _syst_env is not None else np.zeros_like(h_mc_total))
        t = (_stat_env if _stat_env is not None else np.zeros_like(h_mc_total))
        _total_env = np.sqrt(s**2 + t**2)
        # Medium gray: SF total unc (stat ⊕ syst)
        ax_main.fill_between(centres, h_mc_total - _total_env, h_mc_total + _total_env,
                             step="mid", alpha=0.35, color="#888888", linewidth=0,
                             label="SF total unc.", zorder=3)
        # Inner dark gray: SF syst unc only
        if _syst_env is not None:
            ax_main.fill_between(centres, h_mc_total - _syst_env, h_mc_total + _syst_env,
                                 step="mid", alpha=0.50, color="#444444", linewidth=0,
                                 label="SF syst. unc.", zorder=4)

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
                          step="mid", alpha=0.3, color="grey", label="MC stat. unc.")
    with np.errstate(divide="ignore", invalid="ignore"):
        _syst_rel = (np.where(h_mc_total > 0, _syst_env / h_mc_total, 0)
                     if _syst_env is not None else None)
        _tot_rel  = (np.where(h_mc_total > 0, _total_env / h_mc_total, 0)
                     if _total_env is not None else None)
        _comb_rel = (np.sqrt(_tot_rel**2 + mc_rel_err**2)
                     if _tot_rel is not None else None)
    # Outer band: SF total unc ⊕ MC stat unc
    if _comb_rel is not None:
        ax_ratio.fill_between(centres, 1 - _comb_rel, 1 + _comb_rel,
                              step="mid", alpha=0.20, color="#aaaaaa", linewidth=0,
                              label="SF tot. ⊕ MC stat")
    # Medium band: SF total unc (stat ⊕ syst)
    if _tot_rel is not None:
        ax_ratio.fill_between(centres, 1 - _tot_rel, 1 + _tot_rel,
                              step="mid", alpha=0.35, color="#888888", linewidth=0,
                              label="SF total unc.")
    # Inner band: SF syst unc only
    if _syst_rel is not None:
        ax_ratio.fill_between(centres, 1 - _syst_rel, 1 + _syst_rel,
                              step="mid", alpha=0.50, color="#444444", linewidth=0,
                              label="SF syst. unc.")
    ax_ratio.errorbar(centres, ratio, yerr=err_ratio,
                      fmt="o", color="black", markersize=3, linewidth=1.0)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data / MC")
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.legend(fontsize=7, loc="upper right", ncol=1)

    chi2, ndof = chi2_data_mc(h_data, h_mc_total, err_data, err_mc_total)
    return chi2, ndof, h_data, h_mc_total, err_data, err_mc_total


# ─────────────────────────────────────────────────────────────────────────────
# CMS normalised-shape plot helper
# ─────────────────────────────────────────────────────────────────────────────

def cms_norm_plot(ax_main, ax_ratio,
                  df_mc_plot, df_data_plot,
                  edges, xlabel,
                  var_col=None,
                  mc_scores=None,
                  data_scores=None,
                  syst_weights=None,
                  stat_weights=None):
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)

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

    # SF uncertainty bands (normalised: each variation normalised by its own area)
    _syst_env_n = _stat_env_n = _total_env_n = None
    if syst_weights or stat_weights:
        def _norm_env(weights_list):
            env = np.zeros_like(h_mc_norm)
            for w_var in weights_list:
                h_var = np.zeros(len(centres))
                for group in MC_STACK_ORDER:
                    mask_g = (df_mc_plot["group"] == group).values
                    if not mask_g.any():
                        continue
                    vals_g = (df_mc_plot.loc[mask_g, var_col].values if var_col is not None
                              else mc_scores[mask_g])
                    h_g, _ = np.histogram(vals_g, bins=edges, weights=w_var[mask_g])
                    h_var += h_g
                area_v = (h_var * widths).sum()
                h_var_n = h_var / area_v if area_v > 0 else h_var
                env = np.maximum(env, np.abs(h_var_n - h_mc_norm))
            return env
        if syst_weights:
            _syst_env_n = _norm_env(syst_weights)
        if stat_weights:
            _stat_env_n = _norm_env(stat_weights)
        s = (_syst_env_n if _syst_env_n is not None else np.zeros_like(h_mc_norm))
        t = (_stat_env_n if _stat_env_n is not None else np.zeros_like(h_mc_norm))
        _total_env_n = np.sqrt(s**2 + t**2)
        ax_main.fill_between(centres, h_mc_norm - _total_env_n, h_mc_norm + _total_env_n,
                             step="mid", alpha=0.35, color="#888888", linewidth=0,
                             label="SF total unc.", zorder=3)
        if _syst_env_n is not None:
            ax_main.fill_between(centres, h_mc_norm - _syst_env_n, h_mc_norm + _syst_env_n,
                                 step="mid", alpha=0.50, color="#444444", linewidth=0,
                                 label="SF syst. unc.", zorder=4)

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
                          step="mid", alpha=0.3, color="grey", label="MC stat. unc.")
    with np.errstate(divide="ignore", invalid="ignore"):
        _syst_rel_n = (np.where(h_mc_norm > 0, _syst_env_n / h_mc_norm, 0)
                       if _syst_env_n is not None else None)
        _tot_rel_n  = (np.where(h_mc_norm > 0, _total_env_n / h_mc_norm, 0)
                       if _total_env_n is not None else None)
        _comb_rel_n = (np.sqrt(_tot_rel_n**2 + mc_rel_err**2)
                       if _tot_rel_n is not None else None)
    # Outer band: SF total unc ⊕ MC stat unc
    if _comb_rel_n is not None:
        ax_ratio.fill_between(centres, 1 - _comb_rel_n, 1 + _comb_rel_n,
                              step="mid", alpha=0.20, color="#aaaaaa", linewidth=0,
                              label="SF tot. ⊕ MC stat")
    # Medium band: SF total unc (stat ⊕ syst)
    if _tot_rel_n is not None:
        ax_ratio.fill_between(centres, 1 - _tot_rel_n, 1 + _tot_rel_n,
                              step="mid", alpha=0.35, color="#888888", linewidth=0,
                              label="SF total unc.")
    # Inner band: SF syst unc only
    if _syst_rel_n is not None:
        ax_ratio.fill_between(centres, 1 - _syst_rel_n, 1 + _syst_rel_n,
                              step="mid", alpha=0.50, color="#444444", linewidth=0,
                              label="SF syst. unc.")
    ax_ratio.errorbar(centres, ratio, yerr=err_ratio,
                      fmt="o", color="black", markersize=3, linewidth=1.0)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data / MC")
    ax_ratio.set_xlabel(xlabel)
    if _syst_rel_n is not None:
        ax_ratio.legend(fontsize=7, loc="upper right", ncol=1)

    chi2, ndof = chi2_data_mc(h_data_norm, h_mc_norm, err_data_norm, err_mc_norm)
    return chi2, ndof


# ─────────────────────────────────────────────────────────────────────────────
# 1. BDT score plots: nominal and calibrated
# ─────────────────────────────────────────────────────────────────────────────

SCORE_BINS  = 20
score_edges = np.linspace(0.0, 1.0, SCORE_BINS + 1)

fig, ax_m, ax_r = make_cms_fig("BDT score — nominal MC")
chi2, ndof, *_ = cms_stack_plot(
    ax_m, ax_r, df_mc_nom, df_data,
    edges=score_edges, xlabel="BDT score",
    mc_scores=df_mc_nom["bdt_score"].values,
    data_scores=df_data["bdt_score"].values,
)
print(f"[BDT nominal]    chi²/ndof = {chi2:.1f} / {ndof}")
plt.tight_layout()
fig.savefig(os.path.join(config["output"], "cms_bdt_score_nominal.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax_m, ax_r = make_cms_fig(
    "BDT score — calibrated MC (btagPNetQvG_SF)")
chi2_cal, ndof_cal, *_ = cms_stack_plot(
    ax_m, ax_r, df_mc_cal, df_data,
    edges=score_edges, xlabel="BDT score",
    mc_scores=df_mc_cal["bdt_score"].values,
    data_scores=df_data["bdt_score"].values,
    syst_weights=_ps_syst_weights, stat_weights=_ps_stat_weights,
)
print(f"[BDT calibrated] chi²/ndof = {chi2_cal:.1f} / {ndof_cal}")
plt.tight_layout()
fig.savefig(os.path.join(config["output"], "cms_bdt_score_calibrated.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# normalised BDT score — nominal
fig, ax_m, ax_r = make_cms_fig("BDT score — nominal MC (normalised)")
chi2_n, ndof_n = cms_norm_plot(
    ax_m, ax_r, df_mc_nom, df_data,
    edges=score_edges, xlabel="BDT score",
    mc_scores=df_mc_nom["bdt_score"].values,
    data_scores=df_data["bdt_score"].values,
)
print(f"[BDT nominal norm]    chi²/ndof = {chi2_n:.1f} / {ndof_n}")
plt.tight_layout()
fig.savefig(os.path.join(config["output"], "cms_bdt_score_nominal_norm.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# normalised BDT score — calibrated
fig, ax_m, ax_r = make_cms_fig(
    "BDT score — calibrated MC (btagPNetQvG_SF, normalised)")
chi2_cn, ndof_cn = cms_norm_plot(
    ax_m, ax_r, df_mc_cal, df_data,
    edges=score_edges, xlabel="BDT score",
    mc_scores=df_mc_cal["bdt_score"].values,
    data_scores=df_data["bdt_score"].values,
    syst_weights=_ps_syst_weights, stat_weights=_ps_stat_weights,
)
print(f"[BDT calibrated norm] chi²/ndof = {chi2_cn:.1f} / {ndof_cn}")
plt.tight_layout()
fig.savefig(os.path.join(config["output"], "cms_bdt_score_calibrated_norm.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-jet QvG tag plots: normalised before/after + CMS stacked
# ─────────────────────────────────────────────────────────────────────────────

N_TAG_PLOT_BINS  = 25
tag_plot_edges   = np.linspace(0.0, 1.0, N_TAG_PLOT_BINS + 1)
tag_plot_centres = 0.5 * (tag_plot_edges[:-1] + tag_plot_edges[1:])

for tag_col in jet_tag_cols:

    def get_whist(sub_df, col):
        valid = sub_df[col].notna()
        return whist(sub_df.loc[valid, col].values,
                     sub_df.loc[valid, "weight"].values,
                     tag_plot_edges)

    h_d,   err_d   = get_whist(df_data,   tag_col)
    h_nom, err_nom = get_whist(df_mc_nom, tag_col)
    h_cal, err_cal = get_whist(df_mc_cal, tag_col)

    # normalised before/after
    fig_n, axes_n = plt.subplots(2, 1, figsize=(7, 7),
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 sharex=True)
    ax_mn, ax_rn = axes_n

    ax_mn.errorbar(tag_plot_centres, h_d, yerr=err_d,
                   fmt="o", color="black", markersize=3, linewidth=1.0,
                   label="Data", zorder=5)
    ax_mn.step(tag_plot_centres, h_nom, where="mid",
               color="steelblue", linewidth=1.5, linestyle="--",
               label="MC (nominal)")
    ax_mn.fill_between(tag_plot_centres,
                       h_nom - err_nom, h_nom + err_nom,
                       step="mid", alpha=0.3, color="steelblue")
    ax_mn.step(tag_plot_centres, h_cal, where="mid",
               color="tomato", linewidth=1.5,
               label="MC (btagPNetQvG_SF)")
    ax_mn.fill_between(tag_plot_centres,
                       h_cal - err_cal, h_cal + err_cal,
                       step="mid", alpha=0.3, color="tomato")
    ax_mn.set_ylabel("Normalised events / bin")
    ax_mn.set_title(f"{tag_col}: before/after btagPNetQvG_SF")
    ax_mn.legend(fontsize=9)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_nom = np.where(h_nom > 0, h_d / h_nom, np.nan)
        r_cal = np.where(h_cal > 0, h_d / h_cal, np.nan)
        err_r_nom = np.where(h_nom > 0,
                             np.sqrt((err_d / h_nom)**2 +
                                     (h_d * err_nom / h_nom**2)**2), np.nan)
        err_r_cal = np.where(h_cal > 0,
                             np.sqrt((err_d / h_cal)**2 +
                                     (h_d * err_cal / h_cal**2)**2), np.nan)

    ax_rn.errorbar(tag_plot_centres, r_nom, yerr=err_r_nom,
                   fmt="s", color="steelblue", markersize=3,
                   linestyle="--", linewidth=1.0, label="Data/MC nom.")
    ax_rn.errorbar(tag_plot_centres, r_cal, yerr=err_r_cal,
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

    # CMS stacked — nominal
    fig_cms, ax_m_cms, ax_r_cms = make_cms_fig(f"{tag_col} — nominal MC")
    chi2_t, ndof_t, *_ = cms_stack_plot(
        ax_m_cms, ax_r_cms, df_mc_nom, df_data,
        edges=tag_plot_edges, xlabel=tag_col, var_col=tag_col)
    print(f"[{tag_col} nominal]    chi²/ndof = {chi2_t:.1f} / {ndof_t}")
    plt.tight_layout()
    fig_cms.savefig(
        os.path.join(config["output"], f"cms_{tag_col}_nominal.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig_cms)

    # CMS stacked — calibrated
    fig_cms2, ax_m_cms2, ax_r_cms2 = make_cms_fig(
        f"{tag_col} — calibrated MC (btagPNetQvG_SF)")
    chi2_tc, ndof_tc, *_ = cms_stack_plot(
        ax_m_cms2, ax_r_cms2, df_mc_cal, df_data,
        edges=tag_plot_edges, xlabel=tag_col, var_col=tag_col,
        syst_weights=_ps_syst_weights, stat_weights=_ps_stat_weights)
    print(f"[{tag_col} calibrated] chi²/ndof = {chi2_tc:.1f} / {ndof_tc}")
    plt.tight_layout()
    fig_cms2.savefig(
        os.path.join(config["output"], f"cms_{tag_col}_calibrated.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig_cms2)

    # CMS stacked normalised — nominal
    fig_cn, ax_mn_cms, ax_rn_cms = make_cms_fig(
        f"{tag_col} — nominal MC (normalised)")
    chi2_tn, ndof_tn = cms_norm_plot(
        ax_mn_cms, ax_rn_cms, df_mc_nom, df_data,
        edges=tag_plot_edges, xlabel=tag_col, var_col=tag_col)
    print(f"[{tag_col} nominal norm]    chi²/ndof = {chi2_tn:.1f} / {ndof_tn}")
    plt.tight_layout()
    fig_cn.savefig(
        os.path.join(config["output"], f"cms_{tag_col}_nominal_norm.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig_cn)

    # CMS stacked normalised — calibrated
    fig_cc, ax_mc_cms, ax_rc_cms = make_cms_fig(
        f"{tag_col} — calibrated MC (btagPNetQvG_SF, normalised)")
    chi2_tcn, ndof_tcn = cms_norm_plot(
        ax_mc_cms, ax_rc_cms, df_mc_cal, df_data,
        edges=tag_plot_edges, xlabel=tag_col, var_col=tag_col,
        syst_weights=_ps_syst_weights, stat_weights=_ps_stat_weights)
    print(f"[{tag_col} calibrated norm] chi²/ndof = {chi2_tcn:.1f} / {ndof_tcn}")
    plt.tight_layout()
    fig_cc.savefig(
        os.path.join(config["output"], f"cms_{tag_col}_calibrated_norm.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig_cc)

    print(f"Saved all plots for {tag_col}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CDF plots: per-jet QvG before/after calibration vs data
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
    all_diff_lo      = []
    all_diff_hi      = []

    for label, sub_df, color, ls in [
        ("Data",               df_data,   "black",     "-"),
        ("MC (nominal)",       df_mc_nom, "steelblue", "--"),
        ("MC (btagPNetQvG_SF)", df_mc_cal, "tomato",   "-"),
    ]:
        valid = sub_df[tag_col].notna()
        vals  = sub_df.loc[valid, tag_col].values
        wgts  = sub_df.loc[valid, "weight"].values

        h,  _ = np.histogram(vals, bins=cdf_edges, weights=wgts)
        h2, _ = np.histogram(vals, bins=cdf_edges, weights=wgts**2)
        cumh    = np.cumsum(h)
        total_w = cumh[-1]
        cdf_y   = cumh / total_w if total_w > 0 else np.zeros_like(cumh, float)
        cumh2   = np.cumsum(h2)
        cdf_err = (np.sqrt(cumh2) / total_w
                   if total_w > 0 else np.zeros_like(cumh2))

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
            all_diff_lo.append((diff - err_diff).min())
            all_diff_hi.append((diff + err_diff).max())

    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title(f"{tag_col}: CDF before/after btagPNetQvG_SF")
    ax_cdf.legend(fontsize=9)
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax_diff.set_ylabel("MC − Data CDF")
    ax_diff.set_xlabel(tag_col)
    if all_diff_lo and all_diff_hi:
        margin = 0.005
        lo = min(all_diff_lo) - margin
        hi = max(all_diff_hi) + margin
        bound = min(max(abs(lo), abs(hi)), 0.05)
        ax_diff.set_ylim(-bound, bound)

    plt.tight_layout()
    fig.savefig(os.path.join(config["output"], f"cdf_{tag_col}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CDF plot for {tag_col}")

print("\nDone.")
