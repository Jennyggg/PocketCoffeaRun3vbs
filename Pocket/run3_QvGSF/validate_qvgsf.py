#!/usr/bin/env python3
"""
Validate btagPNetQvG scale factors derived from the 2×2 tag-and-probe method.

Reads the PocketCoffea coffea output (3D histograms: score × pT × |η|) and
the correctionlib JSON produced by derive_qvgsf.py, then:

  1. Applies SF_q(pT, |η|) and SF_g(pT, |η|) per (pT, |η|) bin to the
     flavour-split MC histograms.
  2. Jets with |η| > ETA_MASK (default 2.5) are left unweighted (SF=1).
  3. Produces CMS-style absolute and normalised shape plots for:
       - Z+jets region : probe QvG score, probe pT, probe |η|, mll, ptz
       - Dijet region  : probe QvG score, probe pT, probe |η|, dphi_jj, balance

Output directory: ./plots_validation/
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import mplhep as hep
from coffea.util import load

hep.style.use("CMS")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--coffea",   default="output/output_all.coffea")
parser.add_argument("--sf-json",  default="qvgsf_2022_postEE.json")
parser.add_argument("--sf-diag",  default=None,
                    help="Path to _diag.npz from derive_qvgsf.py (for SF error bars).")
parser.add_argument("--output",   default="plots_validation")
parser.add_argument("--eta-mask", type=float, default=4.7,
                    help="No SF applied above this |η| (default 4.7 = all calibrated).")
parser.add_argument("--lumi",     type=float, default=26.337,
                    help="Z+jets (Muon) luminosity in fb-1 (default: 26.337 for 2022 postEE).")
parser.add_argument("--dijet-data-group", default="ZeroBias",
                    help="Data group for the ZeroBias dijet category (default: ZeroBias).")
parser.add_argument("--mc-lumi",     type=float, default=26337.27,
                    help="Luminosity used to normalize QCD MC in pb-1 (default: 26337.27 for 2022 postEE).")
# Per-trigger effective luminosities (pb^-1) from brilcalc --normtag normtag_PHYSICS.json
parser.add_argument("--lumi-zb",    type=float, default=163.336)
parser.add_argument("--lumi-dj40",  type=float, default=0.232884136)
parser.add_argument("--lumi-dj60",  type=float, default=0.502283081)
parser.add_argument("--lumi-dj80",  type=float, default=1.777733525)
parser.add_argument("--lumi-dj140", type=float, default=18.691568466)
parser.add_argument("--lumi-dj200", type=float, default=79.643729403)
parser.add_argument("--lumi-dj260", type=float, default=256.884745729)
parser.add_argument("--lumi-dj320", type=float, default=684.768937084)
parser.add_argument("--lumi-dj400", type=float, default=2053.998349420)
parser.add_argument("--lumi-dj500", type=float, default=8215.530704931)
args = parser.parse_args()

_zjets_lumi_fb = args.lumi                           # fb-1, for CMS label
# ZeroBias MC scale used for extra 1D variables (dphi, balance)
_zb_mc_scale   = args.lumi_zb / args.mc_lumi

_source_lumis = {
    "ZB":    args.lumi_zb,
    "DJ40":  args.lumi_dj40,
    "DJ60":  args.lumi_dj60,
    "DJ80":  args.lumi_dj80,
    "DJ140": args.lumi_dj140,
    "DJ200": args.lumi_dj200,
    "DJ260": args.lumi_dj260,
    "DJ320": args.lumi_dj320,
    "DJ400": args.lumi_dj400,
    "DJ500": args.lumi_dj500,
}

print(f"Z+jets lumi  : {_zjets_lumi_fb:.3f} fb-1")
print(f"ZeroBias lumi: {args.lumi_zb:.3f} pb-1")
print(f"DJ40 lumi    : {args.lumi_dj40:.6f} pb-1")
print(f"DJ500 lumi   : {args.lumi_dj500:.3f} pb-1")

os.makedirs(args.output, exist_ok=True)

# ── Load SF uncertainties from diagnostic file (optional) ─────────────────────
_sf_q_err = None
_sf_g_err = None
_ps_sf_q, _ps_sf_g = {}, {}
if args.sf_diag is not None:
    _diag = np.load(args.sf_diag)
    _sf_q_err = _diag["SF_q_err"]   # (n_score, n_pt, n_eta)
    _sf_g_err = _diag["SF_g_err"]
    for _k in _diag.files:
        if _k.startswith("SF_q_") and not _k.endswith("_err"):
            _vk = _k[len("SF_q_"):]
            _ps_sf_q[_vk] = _diag[_k]
            _ps_sf_g[_vk] = _diag[f"SF_g_{_vk}"]
    print(f"Loaded SF errors from {args.sf_diag}")
    if _ps_sf_q:
        print(f"Loaded PS SF variations: {list(_ps_sf_q.keys())}")

# ── Load coffea output ────────────────────────────────────────────────────────
print(f"Loading {args.coffea}")
out       = load(args.coffea)
variables = out["variables"]

# ── Load SFs from correctionlib JSON ─────────────────────────────────────────
print(f"Loading SFs from {args.sf_json}")
with open(args.sf_json) as f:
    sf_data = json.load(f)

# Structure: corrections[0].data -> category(systematic) -> category(flavor)
#            -> multibinning(score, abseta, pt)
_corr   = sf_data["corrections"][0]["data"]
_cent   = next(c["value"] for c in _corr["content"] if c["key"] == "central")
_q_node = next(c["value"] for c in _cent["content"] if c["key"] == "quark")
_g_node = next(c["value"] for c in _cent["content"] if c["key"] == "gluon")

# Axes order in multibinning: score (outer), abseta (middle), pt (inner)
_score_edges = np.array(_q_node["edges"][0])
_eta_edges   = np.array(_q_node["edges"][1])
_pt_edges_sf = np.array(_q_node["edges"][2])
_n_score = len(_score_edges) - 1
_n_eta   = len(_eta_edges)   - 1
_n_pt    = len(_pt_edges_sf) - 1

# Content flattened (n_score, n_eta, n_pt) row-major → reshape then transpose
SF_q = np.array(_q_node["content"]).reshape(_n_score, _n_eta, _n_pt
                ).transpose(0, 2, 1)   # → (n_score, n_pt, n_eta)
SF_g = np.array(_g_node["content"]).reshape(_n_score, _n_eta, _n_pt
                ).transpose(0, 2, 1)

# Mask forward region: force SF = 1 for |η| >= args.eta_mask
eta_centres = 0.5 * (_eta_edges[:-1] + _eta_edges[1:])
fwd_mask    = eta_centres >= args.eta_mask
SF_q[:, :, fwd_mask] = 1.0
SF_g[:, :, fwd_mask] = 1.0

print(f"SF_q shape: {SF_q.shape}  (n_score, n_pt, n_eta)")
print(f"SF_q[score=0, :central_pt, :central_eta]:\n"
      f"{np.round(SF_q[0, :8, :5], 3)}")


# ── Histogram extraction helpers ──────────────────────────────────────────────

def sum_group_3d(var_name, group, cat, var="nominal"):
    """
    Sum per-era/dataset hist.Hist objects for *group*, select category + variation.
    Returns arrays of shape (n_score, n_pt, n_eta): values and variances.
    """
    hdict     = variables[var_name][group]
    total_v   = None
    total_var = None
    for h in hdict.values():
        h_sel = h[{"cat": cat, "variation": var}]
        v  = h_sel.values(flow=False)
        w  = h_sel.variances(flow=False)
        if total_v is None:
            total_v   = v.copy()
            total_var = w.copy()
        else:
            total_v   += v
            total_var += w
    return total_v, total_var


def sum_group_1d(var_name, group, cat, var="nominal"):
    """Same but for 1D histograms (returns shape (n_bins,))."""
    hdict   = variables[var_name][group]
    total_v = None
    total_w = None
    for h in hdict.values():
        h_sel = h[{"cat": cat, "variation": var}]
        v = h_sel.values(flow=False)
        w = h_sel.variances(flow=False)
        if total_v is None:
            total_v = v.copy()
            total_w = w.copy()
        else:
            total_v += v
            total_w += w
    return total_v, total_w


def apply_sf_to_3d(val_q, var_q, val_g, var_g, val_u, var_u,
                   sf_q_3d, sf_g_3d):
    """
    Apply per-(score, pT, |η|) SFs to the flavour-split 3D histograms.
    All inputs shape: (n_score, n_pt, n_eta).
    Returns (val_cal, var_cal) of same shape.
    """
    val_cal = sf_q_3d * val_q + sf_g_3d * val_g + val_u
    var_cal = sf_q_3d**2 * var_q + sf_g_3d**2 * var_g + var_u
    return val_cal, var_cal


def project_score(val_3d, var_3d):
    """Integrate over pT and |η| → 1D score distribution."""
    return val_3d.sum(axis=(1, 2)), var_3d.sum(axis=(1, 2))


def project_pt(val_3d, var_3d):
    """Integrate over score and |η| → 1D pT distribution."""
    return val_3d.sum(axis=(0, 2)), var_3d.sum(axis=(0, 2))


def project_eta(val_3d, var_3d):
    """Integrate over score and pT → 1D |η| distribution."""
    return val_3d.sum(axis=(0, 1)), var_3d.sum(axis=(0, 1))


# ── Bin edges matching derive_qvgsf.py output ─────────────────────────────────
_orig_score   = np.linspace(0.0, 1.0, 21)
# 18 score bins: merge first 2 → [0,0.10) and last 2 → [0.90,1.0)
score_edges_z = score_edges_d = np.array([0.0] + list(_orig_score[2:19]) + [1.0])
# Output pT edges: 8 bins matching derive_qvgsf.py
pt_edges_z = pt_edges_d = np.array([30,40,50,60,80,140,200,260,8000], dtype=float)
# Output eta edges: 4 bins (SF=1 for j≥3, i.e. |η|≥3.0)
eta_edges_z = eta_edges_d = np.array([0.0, 1.3, 2.5, 3.0, 4.7])

# ── Bin-merging helpers (same as derive_qvgsf.py) ─────────────────────────────
def merge_axis(arr, groups, axis):
    parts = [np.take(arr, grp, axis=axis).sum(axis=axis, keepdims=True)
             for grp in groups]
    return np.concatenate(parts, axis=axis)

# 20 coffea score bins → 18 output (merge first 2, last 2)
_score_groups = [[0, 1]] + [[s] for s in range(2, 18)] + [[18, 19]]
# 22 coffea pT bins (Z+jets): central (8 bins) and forward (6 bins, [140,∞) merged)
_ZJ_PT_GROUPS_CENTRAL = [[0],[1],[2],[3,4],[5,6,7,8],[9,10,11],
                           [12,13,14],[15,16,17,18,19,20,21]]
_ZJ_PT_GROUPS_FWD     = [[0],[1],[2],[3,4],[5,6,7,8],
                           [9,10,11,12,13,14,15,16,17,18,19,20,21]]
_ZJ_PT_GROUPS = _ZJ_PT_GROUPS_CENTRAL  # alias
# 5 coffea eta bins → 4 output eta bins (merge [0,0.7)+[0.7,1.3) → [0,1.3))
_eta_groups   = [[0, 1], [2], [3], [4]]


def rebin_zjets(v, w, pt_groups=None):
    """(20, 22, 5) → (18, n_pt, 4) for Z+jets."""
    if pt_groups is None:
        pt_groups = _ZJ_PT_GROUPS_CENTRAL
    v = merge_axis(v, pt_groups,    1);  w = merge_axis(w, pt_groups,    1)
    v = merge_axis(v, _eta_groups,  2);  w = merge_axis(w, _eta_groups,  2)
    v = merge_axis(v, _score_groups, 0); w = merge_axis(w, _score_groups, 0)
    return v, w


def rebin_dijet(v, w):
    """(20, n_pt, 5) → (18, n_pt, 4): pT already at output resolution."""
    v = merge_axis(v, _score_groups, 0);  w = merge_axis(w, _score_groups, 0)
    v = merge_axis(v, _eta_groups,   2);  w = merge_axis(w, _eta_groups,   2)
    return v, w


def expand_fwd_pt(v, w):
    """(n_score, 6, n_eta) → (n_score, 8, n_eta): repeat last pT bin 3×."""
    vv = np.concatenate([v[:, :5, :], np.repeat(v[:, 5:6, :], 3, axis=1)], axis=1)
    ww = np.concatenate([w[:, :5, :], np.repeat(w[:, 5:6, :], 3, axis=1)], axis=1)
    return vv, ww


# ── Per-pT-bin dijet data sources (mirrors derive_qvgsf.py) ──────────────────
_N_SCORE_COFFEA    = 20
_N_ETA_COFFEA      = 5
_N_PT_OUT_CENTRAL  = 8
_N_PT_OUT_FWD      = 6
_N_PT_OUT          = _N_PT_OUT_CENTRAL

# Central pT bins (|η| < 1.3, j=0)
_PT_BIN_SOURCES_CENTRAL = [
    ("ZB",    [0]),                           # [30, 40)
    ("DJ40",  [1]),                           # [40, 50)
    ("DJ40",  [2]),                           # [50, 60)
    ("DJ60",  [3, 4]),                        # [60, 80)
    ("DJ80",  [5, 6, 7, 8]),                  # [80, 140)
    ("DJ140", [9, 10, 11]),                   # [140, 200)
    ("DJ200", [12, 13, 14]),                  # [200, 260)
    ("DJ260", [15, 16, 17, 18, 19, 20, 21]),  # [260, ∞)
]
# Forward pT bins (|η| ≥ 1.3, j=1,2): [140,∞) merged using DJ140
_PT_BIN_SOURCES_FWD = [
    ("ZB",    [0]),
    ("DJ40",  [1]),
    ("DJ40",  [2]),
    ("DJ60",  [3, 4]),
    ("DJ80",  [5, 6, 7, 8]),
    ("DJ140", [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
]
_PT_BIN_SOURCES = _PT_BIN_SOURCES_CENTRAL
_SOURCE_CONF = {
    "ZB":    {"hist_var": "dijet_zb_probe_score_pt_eta", "cat_prefix": "Dijet_ZB",   "data_group": "ZeroBias"},
    "DJ40":  {"hist_var": "dijet_dj_probe_score_pt_eta", "cat_prefix": "Dijet_DJ40",  "data_group": "JetMET"},
    "DJ60":  {"hist_var": "dijet_dj_probe_score_pt_eta", "cat_prefix": "Dijet_DJ60",  "data_group": "JetMET"},
    "DJ80":  {"hist_var": "dijet_dj_probe_score_pt_eta", "cat_prefix": "Dijet_DJ80",  "data_group": "JetMET"},
    "DJ140": {"hist_var": "dijet_dj_probe_score_pt_eta", "cat_prefix": "Dijet_DJ140", "data_group": "JetMET"},
    "DJ200": {"hist_var": "dijet_dj_probe_score_pt_eta", "cat_prefix": "Dijet_DJ200", "data_group": "JetMET"},
    "DJ260": {"hist_var": "dijet_dj_probe_score_pt_eta", "cat_prefix": "Dijet_DJ260", "data_group": "JetMET"},
    "DJ320": {"hist_var": "dijet_dj_probe_score_pt_eta", "cat_prefix": "Dijet_DJ320", "data_group": "JetMET"},
    "DJ400": {"hist_var": "dijet_dj_probe_score_pt_eta", "cat_prefix": "Dijet_DJ400", "data_group": "JetMET"},
    "DJ500": {"hist_var": "dijet_dj_probe_score_pt_eta", "cat_prefix": "Dijet_DJ500", "data_group": "JetMET"},
}


def assemble_dijet(flavor_suffix, use_data=False, mc_var="nominal",
                   pt_bin_sources=None):
    """Build (20, n_pt_out, 5) dijet arrays from per-trigger categories."""
    if pt_bin_sources is None:
        pt_bin_sources = _PT_BIN_SOURCES_CENTRAL
    n_pt_out = len(pt_bin_sources)
    V = np.zeros((_N_SCORE_COFFEA, n_pt_out, _N_ETA_COFFEA))
    W = np.zeros((_N_SCORE_COFFEA, n_pt_out, _N_ETA_COFFEA))
    for b, (src_key, fine_idxs) in enumerate(pt_bin_sources):
        conf  = _SOURCE_CONF[src_key]
        cat   = f"{conf['cat_prefix']}_{flavor_suffix}"
        group = conf["data_group"] if use_data else "QCD"
        var   = "nominal"          if use_data else mc_var
        v_full, w_full = sum_group_3d(conf["hist_var"], group, cat, var)
        V[:, b, :] += v_full[:, fine_idxs, :].sum(axis=1)
        W[:, b, :] += w_full[:, fine_idxs, :].sum(axis=1)
        if not use_data:
            scale = _source_lumis[src_key] / args.mc_lumi
            V[:, b, :] *= scale
            W[:, b, :] *= scale ** 2
    return V, W


def _assemble_dijet_combined(flavor_suffix, use_data=False, mc_var="nominal"):
    """
    Build combined (18, 8, 4) dijet arrays using eta-dependent pT binning:
      j=0 (|η|<1.3):   8-bin central config
      j=1,2 (|η|∈[1.3,3.0)): 6-bin forward config, expanded to 8 pT bins
      j=3 (|η|≥3.0):   zeros
    """
    # Central: j=0 — (20, 8, 5) → rebin score+eta → (18, 8, 4)
    V_c, W_c = assemble_dijet(flavor_suffix, use_data, mc_var, _PT_BIN_SOURCES_CENTRAL)
    V_c_rb, W_c_rb = rebin_dijet(V_c, W_c)  # (18, 8, 4)

    # Forward: j=1,2 — (20, 6, 5) → rebin score+eta → (18, 6, 4) → expand pT → (18, 8, 4)
    V_f, W_f = assemble_dijet(flavor_suffix, use_data, mc_var, _PT_BIN_SOURCES_FWD)
    V_f_rb, W_f_rb = rebin_dijet(V_f, W_f)  # (18, 6, 4)
    V_f_exp, W_f_exp = expand_fwd_pt(V_f_rb, W_f_rb)  # (18, 8, 4)

    # Combine: use central for eta col 0, forward for eta cols 1,2; zeros for col 3
    V_out = np.zeros_like(V_c_rb)
    W_out = np.zeros_like(W_c_rb)
    V_out[:, :, 0] = V_c_rb[:, :, 0]
    W_out[:, :, 0] = W_c_rb[:, :, 0]
    V_out[:, :, 1] = V_f_exp[:, :, 1]
    W_out[:, :, 1] = W_f_exp[:, :, 1]
    V_out[:, :, 2] = V_f_exp[:, :, 2]
    W_out[:, :, 2] = W_f_exp[:, :, 2]
    # col 3 stays zero (SF=1 for |η|≥3.0)
    return V_out, W_out

# ── Build calibrated 3D histograms for both regions ───────────────────────────
# Z+jets: rebin from coffea (20 score, 22 pT, 5 eta) → output (18, 8, 4)
val_q_Z,   var_q_Z   = rebin_zjets(*sum_group_3d("zjets_probe_score_pt_eta", "DYto2L-2Jets", "Zjets_quark"))
val_g_Z,   var_g_Z   = rebin_zjets(*sum_group_3d("zjets_probe_score_pt_eta", "DYto2L-2Jets", "Zjets_gluon"))
val_u_Z,   var_u_Z   = rebin_zjets(*sum_group_3d("zjets_probe_score_pt_eta", "DYto2L-2Jets", "Zjets_undef"))
val_dat_Z, var_dat_Z = rebin_zjets(*sum_group_3d("zjets_probe_score_pt_eta", "Muon",         "Zjets_all"))
val_all_Z = val_q_Z  + val_g_Z  + val_u_Z
var_all_Z = var_q_Z  + var_g_Z  + var_u_Z

val_cal_Z, var_cal_Z = apply_sf_to_3d(
    val_q_Z, var_q_Z, val_g_Z, var_g_Z, val_u_Z, var_u_Z, SF_q, SF_g)

# Dijet: eta-dependent pT binning — central (j=0) uses 8-bin config,
# forward (j=1,2) uses 6-bin config with [140,∞) merged using DJ140.
val_q_D,   var_q_D   = _assemble_dijet_combined("quark", use_data=False)
val_g_D,   var_g_D   = _assemble_dijet_combined("gluon", use_data=False)
val_u_D,   var_u_D   = _assemble_dijet_combined("undef", use_data=False)
val_dat_D, var_dat_D = _assemble_dijet_combined("all",   use_data=True)
val_all_D = val_q_D  + val_g_D  + val_u_D
var_all_D = var_q_D  + var_g_D  + var_u_D

val_cal_D, var_cal_D = apply_sf_to_3d(
    val_q_D, var_q_D, val_g_D, var_g_D, val_u_D, var_u_D, SF_q, SF_g)

# PS-varied calibrated 3D arrays: apply each PS SF to nominal MC yields.
# Used to show ISR/FSR uncertainty bands in ratio panels.
_ps_cal_Z = {k: apply_sf_to_3d(val_q_Z, var_q_Z, val_g_Z, var_g_Z, val_u_Z, var_u_Z,
                                 _ps_sf_q[k], _ps_sf_g[k])[0]
             for k in _ps_sf_q}
_ps_cal_D = {k: apply_sf_to_3d(val_q_D, var_q_D, val_g_D, var_g_D, val_u_D, var_u_D,
                                 _ps_sf_q[k], _ps_sf_g[k])[0]
             for k in _ps_sf_q}

# Add SF statistical uncertainty as an additional "stat" variation
if _sf_q_err is not None:
    _ps_cal_Z["stat_up"] = apply_sf_to_3d(
        val_q_Z, var_q_Z, val_g_Z, var_g_Z, val_u_Z, var_u_Z,
        SF_q + _sf_q_err, SF_g + _sf_g_err)[0]
    _ps_cal_Z["stat_dn"] = apply_sf_to_3d(
        val_q_Z, var_q_Z, val_g_Z, var_g_Z, val_u_Z, var_u_Z,
        SF_q - _sf_q_err, SF_g - _sf_g_err)[0]
    _ps_cal_D["stat_up"] = apply_sf_to_3d(
        val_q_D, var_q_D, val_g_D, var_g_D, val_u_D, var_u_D,
        SF_q + _sf_q_err, SF_g + _sf_g_err)[0]
    _ps_cal_D["stat_dn"] = apply_sf_to_3d(
        val_q_D, var_q_D, val_g_D, var_g_D, val_u_D, var_u_D,
        SF_q - _sf_q_err, SF_g - _sf_g_err)[0]

# Systematic color scheme for bands
_PS_COLORS = {
    "stat": "#e6ab02",  # golden yellow
    "isr":  "#f89c20",  # orange
    "fsr":  "#5d9e5a",  # green
    "jes":  "#e42536",  # red
    "jer":  "#964a8b",  # purple
    "pu":   "#1f77b4",  # blue
}


# ── Plotting helpers ──────────────────────────────────────────────────────────

COLORS = {"MC nominal": "#3f90da", "MC calibrated": "#bd1f01", "Data": "black"}


def make_cms_fig(title="", lumi=None):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    _lumi = lumi if lumi is not None else _zjets_lumi_fb
    hep.cms.label(ax=axes[0], label="Preliminary", data=True,
                  lumi=_lumi, lumi_format="{:.4g}", com=13.6, fontsize=12)
    if title:
        axes[0].set_title(title, fontsize=9, pad=28)
    return fig, axes[0], axes[1]


def _ratio_panel(ax_r, centres, h_data, h_mc, err_data, err_mc, xlabel,
                 sys_bands=None):
    """
    sys_bands: list of (color, h_up, h_dn) where h_up/h_dn are alternative MC
    predictions (same shape as h_mc). The band drawn spans h_dn/h_mc to h_up/h_mc.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(h_mc > 0, h_data / h_mc, np.nan)
        err_r = np.where(h_mc > 0,
                         np.sqrt((err_data / h_mc)**2 +
                                 (h_data * err_mc / h_mc**2)**2),
                         np.nan)
        mc_re = np.where(h_mc > 0, err_mc / h_mc, np.nan)
    ax_r.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax_r.fill_between(centres, 1 - mc_re, 1 + mc_re,
                      step="mid", alpha=0.3, color="grey", label="MC stat.")
    if sys_bands:
        for color, h_up, h_dn in sys_bands:
            with np.errstate(divide="ignore", invalid="ignore"):
                ru = np.where(h_mc > 0, h_up / h_mc, np.nan)
                rd = np.where(h_mc > 0, h_dn / h_mc, np.nan)
            ax_r.fill_between(centres, rd, ru, step="mid", alpha=0.25, color=color)
    ax_r.errorbar(centres, ratio, yerr=err_r,
                  fmt="o", color="black", markersize=3, linewidth=1.0)
    ax_r.set_ylim(0.75, 1.25)
    ax_r.set_ylabel("Data / MC")
    ax_r.set_xlabel(xlabel)


def chi2ndf(h_data, h_mc, err_data, err_mc):
    mask  = (h_mc > 0) & np.isfinite(h_data) & np.isfinite(h_mc)
    s2    = err_data[mask]**2 + err_mc[mask]**2
    s2[s2 == 0] = 1e-30
    return np.sum((h_data[mask] - h_mc[mask])**2 / s2), mask.sum()


def stack_plot(ax_m, ax_r, edges, xlabel,
               h_data, var_data,
               h_nom, var_nom,
               h_cal, var_cal,
               title_tag="", normalised=False,
               sys_bands=None):
    """
    Draw three overlaid histograms: Data, MC nominal, MC calibrated.
    If normalised=True, scale all to unit area.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)

    def norm1(h, v):
        area = (h * widths).sum()
        if area <= 0 or not normalised:
            return h, np.sqrt(v)
        return h / area, np.sqrt(v) / area

    h_d, e_d = norm1(h_data, var_data)
    h_n, e_n = norm1(h_nom,  var_nom)
    h_c, e_c = norm1(h_cal,  var_cal)

    # MC nominal: filled bar
    ax_m.bar(centres, h_n, width=widths, color=COLORS["MC nominal"],
             alpha=0.6, label="MC nominal", align="center")
    ax_m.bar(centres, 2 * e_n, width=widths,
             bottom=h_n - e_n, color="grey", alpha=0.35,
             hatch="///", linewidth=0, label="MC stat. unc.", align="center")

    # MC calibrated: step line + MC-stat error band
    _xe = np.append(edges[:-1], edges[-1])
    ax_m.step(_xe, np.append(h_c, h_c[-1]),
              where="post", color=COLORS["MC calibrated"],
              linewidth=1.8, label="MC calibrated")
    ax_m.fill_between(_xe,
                      np.append(h_c - e_c, (h_c - e_c)[-1]),
                      np.append(h_c + e_c, (h_c + e_c)[-1]),
                      step="post", alpha=0.25,
                      color=COLORS["MC calibrated"], linewidth=0)

    # SF systematic envelope: total max/min across all variations
    if sys_bands:
        _syst_max = h_c.copy()
        _syst_min = h_c.copy()
        for _, h_up_raw, h_dn_raw in sys_bands:
            _syst_max = np.maximum(_syst_max, norm1(h_up_raw, np.zeros_like(h_up_raw))[0])
            _syst_min = np.minimum(_syst_min, norm1(h_dn_raw, np.zeros_like(h_dn_raw))[0])
        ax_m.fill_between(_xe,
                          np.append(_syst_min, _syst_min[-1]),
                          np.append(_syst_max, _syst_max[-1]),
                          step="post", alpha=0.20,
                          color=COLORS["MC calibrated"], linewidth=0,
                          label="MC cal. syst.")

    # Data: points
    ax_m.errorbar(centres, h_d, yerr=e_d,
                  fmt="o", color="black", markersize=4,
                  linewidth=1.2, label="Data", zorder=10)

    ylabel = "Normalised events / bin" if normalised else "Events / bin"
    ax_m.set_ylabel(ylabel)
    ax_m.legend(fontsize=13, ncol=2)

    # Ratio panel uses calibrated MC as reference
    _ratio_panel(ax_r, centres, h_d, h_c, e_d, e_c, xlabel, sys_bands=sys_bands)

    chi2_nom, nd  = chi2ndf(h_d, h_n, e_d, e_n)
    chi2_cal, nd2 = chi2ndf(h_d, h_c, e_d, e_c)
    ax_r.set_title(
        f"χ²/ndf  nom: {chi2_nom:.1f}/{nd}   cal: {chi2_cal:.1f}/{nd2}",
        fontsize=8)
    return chi2_nom, nd, chi2_cal, nd2


def save_plot(fig, name):
    path = os.path.join(args.output, name)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── Plot loop: one region at a time ──────────────────────────────────────────

regions = [
    {
        "label":     "Zjets",
        "lumi_fb":   _zjets_lumi_fb,
        "mc_lumi_scale": 1.0,
        "data_3d":   (val_dat_Z, var_dat_Z),
        "nom_3d":    (val_all_Z, var_all_Z),
        "cal_3d":    (val_cal_Z, var_cal_Z),
        "score_edges": score_edges_z,
        "pt_edges":    pt_edges_z,
        "eta_edges":   eta_edges_z,
        "score_label": "Probe jet btagPNetQvG (Z+jets)",
        "pt_label":    "Probe jet p_{T} [GeV] (Z+jets)",
        "eta_label":   "|η| probe jet (Z+jets)",
        # Extra 1D variables stored in the coffea output
        "ps_cal_3d": _ps_cal_Z,
        "extra_1d": [
            ("mll",  "Dimuon mass m_{ℓℓ} [GeV]",  "Zjets_all", "mll",  "DYto2L-2Jets", "Muon"),
            ("ptz",  "p_{T}(Z) [GeV]",              "Zjets_all", "ptz",  "DYto2L-2Jets", "Muon"),
        ],
    },
    {
        "label":     "Dijet",
        "lumi_fb":   args.lumi_zb / 1000.0,   # ZeroBias lumi in fb-1 (for extra 1D vars)
        "mc_lumi_scale": _zb_mc_scale,          # for extra 1D ZeroBias variables only
        "data_3d":   (val_dat_D, var_dat_D),
        "nom_3d":    (val_all_D, var_all_D),
        "cal_3d":    (val_cal_D, var_cal_D),
        "score_edges": score_edges_d,
        "pt_edges":    pt_edges_d,
        "eta_edges":   eta_edges_d,
        "score_label": "Probe jet btagPNetQvG (dijet)",
        "pt_label":    "Probe jet p_{T} [GeV] (dijet)",
        "eta_label":   "|η| probe jet (dijet)",
        "ps_cal_3d": _ps_cal_D,
        "extra_1d": [
            ("dphi_jj",      "|Δφ(j0, j1)|",    "Dijet_ZB_noMCcut", "dphi_jj",
             "QCD", args.dijet_data_group),
            ("dijet_balance","Dijet balance α",  "Dijet_ZB_noMCcut", "dijet_balance",
             "QCD", args.dijet_data_group),
        ],
    },
]

for reg in regions:
    tag = reg["label"]
    print(f"\n── {tag} ──")

    dv, dw        = reg["data_3d"]
    nv, nw        = reg["nom_3d"]
    cv, cw        = reg["cal_3d"]
    ps_cal        = reg["ps_cal_3d"]
    s_edges       = reg["score_edges"]
    pt_edges      = reg["pt_edges"]
    et_edges      = reg["eta_edges"]
    lumi_fb       = reg["lumi_fb"]
    mc_lumi_scale = reg["mc_lumi_scale"]

    def _ps_bands(proj_fn):
        """Build sys_bands list for a given 3D→1D projection function."""
        if not ps_cal:
            return None
        bands = []
        for vname, color in _PS_COLORS.items():
            up_k = f"{vname}_up"; dn_k = f"{vname}_dn"
            if up_k in ps_cal and dn_k in ps_cal:
                bands.append((color, proj_fn(ps_cal[up_k]), proj_fn(ps_cal[dn_k])))
        return bands or None

    # Eta index helpers for rebinned output (4 bins: [0,1.3), [1.3,2.5), [2.5,3.0), [3.0,4.7))
    eta_centres_local = 0.5 * (et_edges[:-1] + et_edges[1:])
    eta_3_idx   = eta_centres_local < 3.0   # abs(eta) < 3.0 → cols 0,1,2
    central_idx = eta_centres_local < 2.5   # abs(eta) < 2.5 → cols 0,1

    # ── 1. QvG score, abs(eta) < 3.0 ─────────────────────────────────────────
    _sb_score = _ps_bands(lambda a: a[:, :, eta_3_idx].sum(axis=(1, 2)))
    for normalised, suffix in [(False, "abs"), (True, "norm")]:
        fig, ax_m, ax_r = make_cms_fig(
            f"{tag}: probe jet btagPNetQvG |η|<3 {'(normalised)' if normalised else ''}",
            lumi=lumi_fb)
        c2n, nd, c2c, nd2 = stack_plot(
            ax_m, ax_r,
            edges=s_edges, xlabel="btagPNetQvG score",
            h_data=dv[:, :, eta_3_idx].sum(axis=(1,2)), var_data=dw[:, :, eta_3_idx].sum(axis=(1,2)),
            h_nom=nv[:, :, eta_3_idx].sum(axis=(1,2)),  var_nom=nw[:, :, eta_3_idx].sum(axis=(1,2)),
            h_cal=cv[:, :, eta_3_idx].sum(axis=(1,2)),  var_cal=cw[:, :, eta_3_idx].sum(axis=(1,2)),
            normalised=normalised, sys_bands=_sb_score,
        )
        ax_r.set_ylim(0.88, 1.12)
        print(f"  [{tag} score {suffix}] nom χ²/ndf={c2n:.1f}/{nd}  "
              f"cal χ²/ndf={c2c:.1f}/{nd2}")
        save_plot(fig, f"{tag}_score_{suffix}.png")

    # ── 2. QvG score, abs(eta) < 2.5 ─────────────────────────────────────────
    _sb_score_c = _ps_bands(lambda a: a[:, :, central_idx].sum(axis=(1, 2)))
    for normalised, suffix in [(False, "abs"), (True, "norm")]:
        fig, ax_m, ax_r = make_cms_fig(
            f"{tag}: probe btagPNetQvG |η|<2.5 {'(norm)' if normalised else ''}",
            lumi=lumi_fb)
        c2n, nd, c2c, nd2 = stack_plot(
            ax_m, ax_r,
            edges=s_edges, xlabel="btagPNetQvG score",
            h_data=dv[:, :, central_idx].sum(axis=(1,2)),
            var_data=dw[:, :, central_idx].sum(axis=(1,2)),
            h_nom=nv[:, :, central_idx].sum(axis=(1,2)),
            var_nom=nw[:, :, central_idx].sum(axis=(1,2)),
            h_cal=cv[:, :, central_idx].sum(axis=(1,2)),
            var_cal=cw[:, :, central_idx].sum(axis=(1,2)),
            normalised=normalised, sys_bands=_sb_score_c,
        )
        ax_r.set_ylim(0.88, 1.12)
        print(f"  [{tag} score central {suffix}] nom χ²/ndf={c2n:.1f}/{nd}  "
              f"cal χ²/ndf={c2c:.1f}/{nd2}")
        save_plot(fig, f"{tag}_score_central_{suffix}.png")

    # ── 3. Probe pT ───────────────────────────────────────────────────────────
    for normalised, suffix in [(False, "abs"), (True, "norm")]:
        fig, ax_m, ax_r = make_cms_fig(
            f"{tag}: probe jet pT {'(normalised)' if normalised else ''}",
            lumi=lumi_fb)
        c2n, nd, c2c, nd2 = stack_plot(
            ax_m, ax_r,
            edges=pt_edges, xlabel=reg["pt_label"],
            h_data=dv.sum(axis=(0,2)), var_data=dw.sum(axis=(0,2)),
            h_nom=nv.sum(axis=(0,2)),  var_nom=nw.sum(axis=(0,2)),
            h_cal=cv.sum(axis=(0,2)),  var_cal=cw.sum(axis=(0,2)),
            normalised=normalised,
            sys_bands=_ps_bands(lambda a: a.sum(axis=(0, 2))),
        )
        ax_m.set_xlim(30, 600)
        ax_r.set_xlim(30, 600)
        print(f"  [{tag} pT {suffix}] nom χ²/ndf={c2n:.1f}/{nd}  "
              f"cal χ²/ndf={c2c:.1f}/{nd2}")
        save_plot(fig, f"{tag}_probe_pt_{suffix}.png")

    # ── 4. Probe |η| ──────────────────────────────────────────────────────────
    for normalised, suffix in [(False, "abs"), (True, "norm")]:
        fig, ax_m, ax_r = make_cms_fig(
            f"{tag}: probe jet |η| {'(normalised)' if normalised else ''}",
            lumi=lumi_fb)
        c2n, nd, c2c, nd2 = stack_plot(
            ax_m, ax_r,
            edges=et_edges, xlabel=reg["eta_label"],
            h_data=dv.sum(axis=(0,1)), var_data=dw.sum(axis=(0,1)),
            h_nom=nv.sum(axis=(0,1)),  var_nom=nw.sum(axis=(0,1)),
            h_cal=cv.sum(axis=(0,1)),  var_cal=cw.sum(axis=(0,1)),
            normalised=normalised,
            sys_bands=_ps_bands(lambda a: a.sum(axis=(0, 1))),
        )
        ax_m.axvline(args.eta_mask, color="red", linestyle="--",
                     linewidth=1.2, label=f"|η|={args.eta_mask} mask")
        ax_m.legend(fontsize=13, ncol=2)
        print(f"  [{tag} |η| {suffix}] nom χ²/ndf={c2n:.1f}/{nd}  "
              f"cal χ²/ndf={c2c:.1f}/{nd2}")
        save_plot(fig, f"{tag}_probe_abseta_{suffix}.png")

    # ── 5. Extra 1D variables (mll, ptz, dphi_jj, balance) ───────────────────
    for fname, xlabel, cat_name, hist_key, mc_group, data_group in reg["extra_1d"]:
        if hist_key not in variables:
            print(f"  (skipping {hist_key}: not in output)")
            continue
        h_mc_1d, w_mc_1d = sum_group_1d(hist_key, mc_group,   cat_name)
        h_dt_1d, w_dt_1d = sum_group_1d(hist_key, data_group, cat_name)

        # Scale MC to match data luminosity (only non-trivial for dijet / ZeroBias)
        h_mc_1d = h_mc_1d * mc_lumi_scale
        w_mc_1d = w_mc_1d * mc_lumi_scale**2

        # Get bin edges from this 1D histogram
        _h_ref = list(variables[hist_key][mc_group].values())[0]
        _h_sel = _h_ref[{"cat": cat_name, "variation": "nominal"}]
        edges_1d = _h_sel.axes[0].edges

        for normalised, suffix in [(False, "abs"), (True, "norm")]:
            fig, ax_m, ax_r = make_cms_fig(
                f"{tag}: {xlabel} {'(normalised)' if normalised else ''}",
                lumi=lumi_fb)
            # No SF applied for extra 1D vars (no flavour split available here)
            # → calibrated = nominal (shown as reference)
            c2n, nd, c2c, nd2 = stack_plot(
                ax_m, ax_r,
                edges=edges_1d, xlabel=xlabel,
                h_data=h_dt_1d, var_data=w_dt_1d,
                h_nom=h_mc_1d,  var_nom=w_mc_1d,
                h_cal=h_mc_1d,  var_cal=w_mc_1d,   # same as nominal
                normalised=normalised,
            )
            print(f"  [{tag} {fname} {suffix}] nom χ²/ndf={c2n:.1f}/{nd}")
            save_plot(fig, f"{tag}_{fname}_{suffix}.png")

# ── SF summary: score-averaged SF per (pT, |η|) bin ─────────────────────────
# Average SF_q/SF_g over the score axis (simple mean, shape (n_pt, n_eta))
print("\n── SF summary plot ──")
SF_q_mean = SF_q.mean(axis=0)   # (n_pt, n_eta)
SF_g_mean = SF_g.mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
hep.cms.label(ax=axes[0], label="Preliminary", data=False,
              lumi=_zjets_lumi_fb, lumi_format="{:.4g}", com=13.6)

_pt_edges_arr  = np.array(_pt_edges_sf)
_eta_edges_arr = np.array(_eta_edges)
pt_centres  = 0.5 * (_pt_edges_arr[:-1]  + _pt_edges_arr[1:])
eta_centres = 0.5 * (_eta_edges_arr[:-1] + _eta_edges_arr[1:])

for ax, SF_mean, title in [(axes[0], SF_q_mean, "SF_quark (score-avg)"),
                            (axes[1], SF_g_mean, "SF_gluon (score-avg)")]:
    im = ax.imshow(SF_mean, origin="lower", aspect="auto",
                   extent=[eta_centres[0], eta_centres[-1],
                           pt_centres[0],  pt_centres[-1]],
                   vmin=0.5, vmax=2.0, cmap="RdBu_r")
    ax.set_xlabel("|η|", fontsize=12)
    ax.set_ylabel("pT [GeV]", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_yscale("log")
    ax.axvline(args.eta_mask, color="white", linestyle="--",
               linewidth=1.5, label=f"|η|={args.eta_mask} mask")
    ax.legend(fontsize=9)
    plt.colorbar(im, ax=ax, label="SF")
    for i, pt in enumerate(pt_centres):
        for j, eta in enumerate(eta_centres):
            ax.text(eta, pt, f"{SF_mean[i,j]:.2f}",
                    ha="center", va="center", fontsize=5,
                    color="white" if abs(SF_mean[i,j] - 1.0) > 0.3 else "black")

plt.tight_layout()
path = os.path.join(args.output, "SF_summary.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {path}")

# ── SF vs score for a few representative (pT, |η|) bins ──────────────────────
score_centres = 0.5 * (np.array(_score_edges[:-1]) + np.array(_score_edges[1:]))
example_bins  = [(0, 0, "pT[30,40) |eta|[0,1.3)"),
                 (1, 0, "pT[40,50) |eta|[0,1.3)"),
                 (3, 0, "pT[60,80) |eta|[0,1.3)"),
                 (5, 0, "pT[140,200) |eta|[0,1.3)"),
                 (7, 0, "pT[260,inf) |eta|[0,1.3)"),
                 (3, 1, "pT[60,80) |eta|[1.3,2.5)")]

fig, axes2 = plt.subplots(1, 2, figsize=(14, 5))
for i_pt, i_eta, label in example_bins:
    eq = _sf_q_err[:, i_pt, i_eta] if _sf_q_err is not None else None
    eg = _sf_g_err[:, i_pt, i_eta] if _sf_g_err is not None else None
    kw = dict(fmt="o-", ms=3, capsize=2)
    if eq is not None:
        axes2[0].errorbar(score_centres, SF_q[:, i_pt, i_eta], yerr=eq, label=label, **kw)
        axes2[1].errorbar(score_centres, SF_g[:, i_pt, i_eta], yerr=eg, label=label, **kw)
    else:
        axes2[0].plot(score_centres, SF_q[:, i_pt, i_eta], marker="o", ms=3, label=label)
        axes2[1].plot(score_centres, SF_g[:, i_pt, i_eta], marker="o", ms=3, label=label)
for ax, title in [(axes2[0], "SF_quark vs score"), (axes2[1], "SF_gluon vs score")]:
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("btagPNetQvG score")
    ax.set_ylabel("SF")
    ax.set_title(title)
    ax.legend(fontsize=8)
plt.tight_layout()
path2 = os.path.join(args.output, "SF_vs_score.png")
fig.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {path2}")

# ── SF vs score per |η| bin (one figure per eta, lines = pT bins) ─────────────
print("\n── SF vs score per |η| bin ──")
score_centres = 0.5 * (np.array(_score_edges[:-1]) + np.array(_score_edges[1:]))
pt_lo_arr  = np.array(_pt_edges_sf[:-1])
pt_hi_arr  = np.array(_pt_edges_sf[1:])
eta_lo_arr = np.array(_eta_edges[:-1])
eta_hi_arr = np.array(_eta_edges[1:])

# One distinct color per pT bin (10 bins); use tab10 which has exactly 10 colors
_n_pt_sf = len(_pt_edges_sf) - 1
PT_COLORS = [plt.cm.tab10(i) for i in range(_n_pt_sf)]

# Forward [2.5,3.0) eta bin uses 12 coarse score bins and 2 coarse pT bins.
# The 18-bin SF array stores duplicated values; select unique representatives.
_fwd_score_edges_arr = np.array(
    [0.0, 0.20] + [round(x, 4) for x in _orig_score[5:14].tolist()] + [1.0]
)
_fwd_score_centres  = 0.5 * (_fwd_score_edges_arr[:-1] + _fwd_score_edges_arr[1:])
_fwd_score_uniq_idx = [0] + list(range(3, 12)) + [12]  # rows in 18-bin array (11 unique)
# Forward pT: derived with 6 bins ([30,40), [40,50), [50,60), [60,80), [80,140), [140,∞)).
# The correctionlib stores 8 bins, with bins 5/6/7 all identical (expand_fwd_pt repeats
# the merged [140,∞) bin). Show only the 6 unique bins with explicit labels.
_fwd_pt_uniq_idx = [0, 1, 2, 3, 4, 5]
_fwd_pt_labels   = [
    "pT [30, 40)",
    "pT [40, 50)",
    "pT [50, 60)",
    "pT [60, 80)",
    "pT [80, 140)",
    "pT [140, \u221e)",   # merged in derive_qvgsf.py; bins 5/6/7 identical
]

# Outer loop: nominal plot + one plot per systematic type
_ps_plot_variants = [(None, "")]
for _vtype in ["isr", "fsr", "jes", "jer", "pu"]:
    if f"{_vtype}_up" in _ps_sf_q:
        _ps_plot_variants.append((_vtype, f"_{_vtype}"))

for _ps_variant, _ps_suffix in _ps_plot_variants:
    for j in range(len(eta_lo_arr)):
        eta_lo = eta_lo_arr[j]; eta_hi = eta_hi_arr[j]
        is_calibrated = eta_lo < args.eta_mask
        is_fwd = abs(eta_lo - 2.5) < 0.01   # [2.5, 3.0) bin: coarser score+pT

        # Choose score x-axis and which SF rows/pT columns to plot
        if is_fwd:
            sc_x       = _fwd_score_centres
            sc_idx     = _fwd_score_uniq_idx
            pt_indices = _fwd_pt_uniq_idx
            pt_labels  = _fwd_pt_labels
        else:
            sc_x       = score_centres
            sc_idx     = list(range(len(score_centres)))
            pt_indices = list(range(len(pt_lo_arr)))
            pt_labels  = None

        fig, (ax_q, ax_g) = plt.subplots(1, 2, figsize=(14, 5))
        hep.cms.label(ax=ax_q, label="Preliminary", data=True,
                      lumi=None, com=None, fontsize=14)
        ax_g.text(1.0, 1.02,
                  f"{_zjets_lumi_fb:.4g}" + r" fb$^{-1}$ (13.6 TeV)",
                  transform=ax_g.transAxes, ha="right", va="bottom", fontsize=12)

        for ii, i in enumerate(pt_indices):
            if pt_labels is not None:
                pt_label = pt_labels[ii]
            else:
                pt_lo = pt_lo_arr[i]; pt_hi = pt_hi_arr[i]
                pt_label = (f"pT [{pt_lo:.0f}, {pt_hi:.0f})" if pt_hi < 7000
                            else f"pT [{pt_lo:.0f}, \u221e)")
            color = PT_COLORS[ii % len(PT_COLORS)]

            sfq = SF_q[np.ix_(sc_idx, [i], [j])][:, 0, 0]
            sfg = SF_g[np.ix_(sc_idx, [i], [j])][:, 0, 0]
            eq  = _sf_q_err[np.ix_(sc_idx, [i], [j])][:, 0, 0] if _sf_q_err is not None else None
            eg  = _sf_g_err[np.ix_(sc_idx, [i], [j])][:, 0, 0] if _sf_g_err is not None else None

            kw = dict(fmt="o-", ms=4, color=color, linewidth=1.5, capsize=2)
            if eq is not None:
                ax_q.errorbar(sc_x, sfq, yerr=eq, label=pt_label, **kw)
            else:
                ax_q.plot(sc_x, sfq, marker="o", ms=4, color=color, linewidth=1.5, label=pt_label)
            if eg is not None:
                ax_g.errorbar(sc_x, sfg, yerr=eg, label=pt_label, **kw)
            else:
                ax_g.plot(sc_x, sfg, marker="o", ms=4, color=color, linewidth=1.5, label=pt_label)

            # PS uncertainty band (filled area between up/dn SF, same color)
            if _ps_variant is not None:
                _up_k = f"{_ps_variant}_up"
                _dn_k = f"{_ps_variant}_dn"
                if _up_k in _ps_sf_q:
                    ps_up_q = _ps_sf_q[_up_k][np.ix_(sc_idx, [i], [j])][:, 0, 0]
                    ps_dn_q = _ps_sf_q[_dn_k][np.ix_(sc_idx, [i], [j])][:, 0, 0]
                    ps_up_g = _ps_sf_g[_up_k][np.ix_(sc_idx, [i], [j])][:, 0, 0]
                    ps_dn_g = _ps_sf_g[_dn_k][np.ix_(sc_idx, [i], [j])][:, 0, 0]
                    ax_q.fill_between(sc_x, ps_dn_q, ps_up_q,
                                      alpha=0.25, color=color, linewidth=0)
                    ax_g.fill_between(sc_x, ps_dn_g, ps_up_g,
                                      alpha=0.25, color=color, linewidth=0)

        mask_note = "  [SF=1, not calibrated]" if not is_calibrated else ""
        ps_note   = f"  [{_ps_variant.upper()} band]" if _ps_variant else ""
        for ax, flavor in [(ax_q, "quark"), (ax_g, "gluon")]:
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
            ax.set_xlabel("btagPNetQvG score", fontsize=14)
            ax.set_ylabel("SF", fontsize=14)
            ax.text(0.5, 0.97,
                    f"SF_{flavor}  |$\\eta$| $\\in$ [{eta_lo:.1f}, {eta_hi:.1f}){mask_note}{ps_note}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75))
            ax.legend(fontsize=12, loc="upper left")
            ax.set_xlim(0, 1)

        plt.tight_layout()
        eta_tag = f"{eta_lo:.1f}to{eta_hi:.1f}".replace(".", "p")
        path_eta = os.path.join(args.output, f"SF_vs_score_eta{eta_tag}{_ps_suffix}.png")
        fig.savefig(path_eta, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {path_eta}")

# ── Per-trigger dijet score distributions ────────────────────────────────────
# For each trigger source, show data vs QCD MC scaled to that trigger's lumi,
# summing over the pT bins covered by that trigger's plateau region.
print("\n── Dijet score per trigger ──")

# Group coffea fine pT indices by trigger source (preserving order)
from collections import OrderedDict as _OD
_trigger_fine_idxs = _OD()
for _src, _idxs in _PT_BIN_SOURCES:
    _trigger_fine_idxs.setdefault(_src, []).extend(_idxs)

_trigger_pt_labels = {
    "ZB":    r"ZeroBias,  $p_{T,\mathrm{avg}} \in [30, 40)$ GeV",
    "DJ40":  r"DJ40,      $p_{T,\mathrm{avg}} \in [40, 60)$ GeV",
    "DJ60":  r"DJ60,      $p_{T,\mathrm{avg}} \in [60, 80)$ GeV",
    "DJ80":  r"DJ80,      $p_{T,\mathrm{avg}} \in [80, 140)$ GeV",
    "DJ140": r"DJ140,     $p_{T,\mathrm{avg}} \in [140, 200)$ GeV",
    "DJ200": r"DJ200,     $p_{T,\mathrm{avg}} \in [200, 260)$ GeV",
    "DJ260": r"DJ260,     $p_{T,\mathrm{avg}} \geq 260$ GeV",
}

# Map from trigger source key → output pT bin indices (for calibrated MC)
_src_to_out_pt_bins = {}
for _b, (_src_k, _) in enumerate(_PT_BIN_SOURCES):
    _src_to_out_pt_bins.setdefault(_src_k, []).append(_b)

# Rebin helper for 1D score arrays (20 coffea → 18 output bins)
_sc_groups_1d = [[0, 1]] + [[s] for s in range(2, 18)] + [[18, 19]]

def _rebin_score_1d(v, w):
    vv = np.array([v[g].sum() for g in _sc_groups_1d])
    ww = np.array([w[g].sum() for g in _sc_groups_1d])
    return vv, ww


_n_trg  = len(_trigger_fine_idxs)
_ncols  = 3
_nrows  = (_n_trg + _ncols - 1) // _ncols

def _draw_per_trigger_score(fname_stem, eta_coffea_n, eta_cal_n, title_suffix=""):
    """
    Draw per-trigger QvG score plots with ratio panels and SF systematic bands.

    eta_coffea_n : number of coffea eta bins to keep (5 total: :4 → |η|<3, :3 → |η|<2.5)
    eta_cal_n    : number of rebinned cal eta cols to keep (4 total: :3 → |η|<3, :2 → |η|<2.5)
    fname_stem   : output filename stem (e.g. "Dijet_score_per_trigger")
    title_suffix : appended to each panel title (e.g. r", $|\\eta|<2.5$")
    """
    _edges  = np.array(score_edges_d)
    _ctrs   = 0.5 * (_edges[:-1] + _edges[1:])
    _widths = np.diff(_edges)

    for _normalised, _suffix in [(False, "abs"), (True, "norm")]:
        # Nested GridSpec: each cell → main (top 3/4) + ratio (bottom 1/4)
        _fig = plt.figure(figsize=(7 * _ncols, 7 * _nrows))
        _outer = GridSpec(_nrows, _ncols, figure=_fig, hspace=0.55, wspace=0.30)
        _axes_m, _axes_r = [], []
        for _ir in range(_nrows):
            for _ic in range(_ncols):
                _inner = GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=_outer[_ir, _ic],
                    height_ratios=[3, 1], hspace=0.07)
                _am = _fig.add_subplot(_inner[0])
                _ar = _fig.add_subplot(_inner[1], sharex=_am)
                _axes_m.append(_am)
                _axes_r.append(_ar)
                plt.setp(_am.get_xticklabels(), visible=False)

        hep.cms.label(ax=_axes_m[0], label="Preliminary", data=True,
                      lumi=None, com=13.6, fontsize=11)

        for _tidx, (_src, _fidxs) in enumerate(_trigger_fine_idxs.items()):
            _ax  = _axes_m[_tidx]
            _axr = _axes_r[_tidx]
            _conf  = _SOURCE_CONF[_src]
            _cat   = _conf["cat_prefix"] + "_all"
            _lumi  = _source_lumis[_src]
            _scale = _lumi / args.mc_lumi

            # Raw data and uncalibrated MC
            _vd, _wd = sum_group_3d(_conf["hist_var"], _conf["data_group"], _cat)
            _vm, _wm = sum_group_3d(_conf["hist_var"], "QCD", _cat)
            _vd1 = _vd[:, _fidxs, :eta_coffea_n].sum(axis=(1, 2))
            _wd1 = _wd[:, _fidxs, :eta_coffea_n].sum(axis=(1, 2))
            _vm1 = _vm[:, _fidxs, :eta_coffea_n].sum(axis=(1, 2)) * _scale
            _wm1 = _wm[:, _fidxs, :eta_coffea_n].sum(axis=(1, 2)) * _scale**2
            _vd1, _wd1 = _rebin_score_1d(_vd1, _wd1)
            _vm1, _wm1 = _rebin_score_1d(_vm1, _wm1)

            # Calibrated MC (central)
            _out_bins = _src_to_out_pt_bins[_src]
            _vc1 = val_cal_D[:, _out_bins, :eta_cal_n].sum(axis=(1, 2))
            _wc1 = var_cal_D[:, _out_bins, :eta_cal_n].sum(axis=(1, 2))

            # SF systematic envelope: max/min over all PS + stat variations
            _vc1_syst_max = _vc1.copy()
            _vc1_syst_min = _vc1.copy()
            for _vk, _cal_arr in _ps_cal_D.items():
                _vc1_v = _cal_arr[:, _out_bins, :eta_cal_n].sum(axis=(1, 2))
                _vc1_syst_max = np.maximum(_vc1_syst_max, _vc1_v)
                _vc1_syst_min = np.minimum(_vc1_syst_min, _vc1_v)

            # Normalise (raw data/MC normalise to their own area; cal variants too)
            _ec1 = np.sqrt(_wc1)   # MC cal stat error (pre-norm; ratio is invariant)
            if _normalised:
                def _n1(v, w):
                    a = (v * _widths).sum()
                    return (v / a, np.sqrt(w) / a) if a > 0 else (v, np.sqrt(w))
                _vd1, _ed1 = _n1(_vd1, _wd1)
                _vm1, _em1 = _n1(_vm1, _wm1)
                _vc1, _ec1 = _n1(_vc1, _wc1)
                # Normalise syst envelope variants consistently
                for _arr in [_vc1_syst_max, _vc1_syst_min]:
                    _a = (_arr * _widths).sum()
                    if _a > 0:
                        _arr[:] /= _a
            else:
                _ed1 = np.sqrt(_wd1)
                _em1 = np.sqrt(_wm1)

            # ── Main panel ──────────────────────────────────────────────────
            _ax.bar(_ctrs, _vm1, width=_widths, color="#3f90da", alpha=0.6,
                    label=f"QCD MC ({_lumi:.3g} pb$^{{-1}}$)", align="center")
            _ax.bar(_ctrs, 2 * _em1, width=_widths, bottom=_vm1 - _em1,
                    color="grey", alpha=0.35, hatch="///", linewidth=0,
                    label="MC stat.", align="center")
            # Calibrated MC + SF systematic envelope
            _xe = np.append(_edges[:-1], _edges[-1])
            _ax.step(_xe, np.append(_vc1, _vc1[-1]),
                     where="post", color="#bd1f01", linewidth=1.8, label="MC calibrated")
            _ax.fill_between(_xe,
                             np.append(_vc1_syst_min, _vc1_syst_min[-1]),
                             np.append(_vc1_syst_max, _vc1_syst_max[-1]),
                             step="post", alpha=0.30, color="#bd1f01", linewidth=0,
                             label="MC cal. syst.")
            _ax.errorbar(_ctrs, _vd1, yerr=_ed1, fmt="o", color="black",
                         markersize=4, linewidth=1.2, label="Data", zorder=10)
            _ax.set_ylabel("Norm. events / bin" if _normalised
                           else "Events / bin", fontsize=9)
            _ax.set_title(_trigger_pt_labels.get(_src, _src) + title_suffix, fontsize=9)
            _ax.legend(fontsize=7, ncol=2)
            _ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)

            # ── Ratio panel: Data / MC calibrated ───────────────────────────
            with np.errstate(divide="ignore", invalid="ignore"):
                _ratio     = np.where(_vc1 > 0, _vd1 / _vc1, np.nan)
                _err_ratio = np.where(_vc1 > 0, _ed1 / _vc1, np.nan)
                _mc_stat_r = np.where(_vc1 > 0, _ec1 / _vc1, np.nan)
                _syst_up_r = np.where(_vc1 > 0, _vc1_syst_max / _vc1, np.nan)
                _syst_dn_r = np.where(_vc1 > 0, _vc1_syst_min / _vc1, np.nan)
            _axr.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
            # MC stat band (grey)
            _axr.fill_between(_ctrs, 1 - _mc_stat_r, 1 + _mc_stat_r,
                              step="mid", alpha=0.30, color="grey",
                              linewidth=0, label="MC stat.")
            # SF systematic envelope (red)
            _axr.fill_between(_ctrs, _syst_dn_r, _syst_up_r,
                              step="mid", alpha=0.25, color="#bd1f01",
                              linewidth=0, label="SF syst.")
            # Data points
            _axr.errorbar(_ctrs, _ratio, yerr=_err_ratio,
                          fmt="o", color="black", markersize=3, linewidth=1.0)
            _axr.set_ylim(0.85, 1.15)
            _axr.set_ylabel("Data/MC cal.", fontsize=8)
            _axr.set_xlabel("btagPNetQvG score", fontsize=9)
            _axr.yaxis.set_major_locator(
                matplotlib.ticker.MultipleLocator(0.1))
            _axr.legend(fontsize=6, loc="upper right", ncol=2)

        for _i in range(_n_trg, _nrows * _ncols):
            _axes_m[_i].set_visible(False)
            _axes_r[_i].set_visible(False)
        _path_trg = os.path.join(args.output, f"{fname_stem}_{_suffix}.png")
        _fig.savefig(_path_trg, dpi=150, bbox_inches="tight")
        plt.close(_fig)
        print(f"  → {_path_trg}")


_draw_per_trigger_score(
    "Dijet_score_per_trigger",
    eta_coffea_n=4, eta_cal_n=3,
    title_suffix="",
)

# ── Per-trigger dijet score distributions — central probe jets (|η| < 2.5) ──
print("\n── Dijet score central per trigger ──")

_draw_per_trigger_score(
    "Dijet_score_central_per_trigger",
    eta_coffea_n=3, eta_cal_n=2,
    title_suffix=r", $|\eta|<2.5$",
)

# ── Total systematic uncertainty heatmap ─────────────────────────────────────
print("\n── Total systematic uncertainty ──")

_syst_types = [k for k in ["isr", "fsr", "jes", "jer", "pu"]
               if f"{k}_up" in _ps_sf_q]

if _syst_types and _sf_q_err is not None:
    # For each (score, pT, eta) compute total unc = sqrt(stat^2 + sum_sys max_sym^2)
    _tot_q = _sf_q_err**2   # stat^2
    _tot_g = _sf_g_err**2
    for _vt in _syst_types:
        _dq_up = np.abs(_ps_sf_q[f"{_vt}_up"] - SF_q)
        _dq_dn = np.abs(_ps_sf_q[f"{_vt}_dn"] - SF_q)
        _dg_up = np.abs(_ps_sf_g[f"{_vt}_up"] - SF_g)
        _dg_dn = np.abs(_ps_sf_g[f"{_vt}_dn"] - SF_g)
        _tot_q += np.maximum(_dq_up, _dq_dn)**2
        _tot_g += np.maximum(_dg_up, _dg_dn)**2
    _tot_q = np.sqrt(_tot_q)   # (n_score, n_pt, n_eta)
    _tot_g = np.sqrt(_tot_g)

    # Relative total uncertainty (average over score bins)
    _rel_q = (_tot_q / np.where(SF_q > 0, SF_q, 1.0)).mean(axis=0)  # (n_pt, n_eta)
    _rel_g = (_tot_g / np.where(SF_g > 0, SF_g, 1.0)).mean(axis=0)

    fig_ts, axes_ts = plt.subplots(1, 2, figsize=(14, 5))
    hep.cms.label(ax=axes_ts[0], label="Preliminary", data=False,
                  lumi=_zjets_lumi_fb, lumi_format="{:.4g}", com=13.6)
    for ax_ts, rel, title in [
        (axes_ts[0], _rel_q, "SF_quark total rel. unc. (score-avg)"),
        (axes_ts[1], _rel_g, "SF_gluon total rel. unc. (score-avg)"),
    ]:
        im_ts = ax_ts.imshow(rel, origin="lower", aspect="auto",
                             extent=[eta_centres[0], eta_centres[-1],
                                     pt_centres[0],  pt_centres[-1]],
                             vmin=0, vmax=0.5, cmap="YlOrRd")
        ax_ts.set_xlabel("|η|", fontsize=12)
        ax_ts.set_ylabel("pT [GeV]", fontsize=12)
        ax_ts.set_title(title, fontsize=12)
        ax_ts.set_yscale("log")
        plt.colorbar(im_ts, ax=ax_ts, label="Relative uncertainty")
        for i, pt in enumerate(pt_centres):
            for j, eta in enumerate(eta_centres):
                ax_ts.text(eta, pt, f"{rel[i,j]:.2f}",
                           ha="center", va="center", fontsize=5,
                           color="white" if rel[i,j] > 0.25 else "black")
    plt.tight_layout()
    _path_ts = os.path.join(args.output, "SF_total_syst.png")
    fig_ts.savefig(_path_ts, dpi=150, bbox_inches="tight")
    plt.close(fig_ts)
    print(f"  → {_path_ts}")

    # Per-syst breakdown: absolute SF unc (score-avg) per (pT, eta)
    fig_sb, axes_sb = plt.subplots(len(_syst_types) + 1, 2,
                                    figsize=(14, 5 * (len(_syst_types) + 1)))
    hep.cms.label(ax=axes_sb[0, 0], label="Preliminary", data=False,
                  lumi=_zjets_lumi_fb, lumi_format="{:.4g}", com=13.6)
    for _row, (_vt, _label) in enumerate(
            [("stat", "Stat.")] + [(v, v.upper()) for v in _syst_types]):
        if _vt == "stat":
            _dq = _sf_q_err.mean(axis=0)
            _dg = _sf_g_err.mean(axis=0)
        else:
            _dq = (np.maximum(np.abs(_ps_sf_q[f"{_vt}_up"] - SF_q),
                              np.abs(_ps_sf_q[f"{_vt}_dn"] - SF_q))).mean(axis=0)
            _dg = (np.maximum(np.abs(_ps_sf_g[f"{_vt}_up"] - SF_g),
                              np.abs(_ps_sf_g[f"{_vt}_dn"] - SF_g))).mean(axis=0)
        for _col, (_d, _flav) in enumerate([(_dq, "quark"), (_dg, "gluon")]):
            _ax_sb = axes_sb[_row, _col]
            _im_sb = _ax_sb.imshow(_d, origin="lower", aspect="auto",
                                   extent=[eta_centres[0], eta_centres[-1],
                                           pt_centres[0],  pt_centres[-1]],
                                   vmin=0, vmax=0.3, cmap="YlOrRd")
            _ax_sb.set_xlabel("|η|", fontsize=10)
            _ax_sb.set_ylabel("pT [GeV]", fontsize=10)
            _ax_sb.set_title(f"SF_{_flav} unc: {_label}", fontsize=11)
            _ax_sb.set_yscale("log")
            plt.colorbar(_im_sb, ax=_ax_sb, label="SF unc.")
            for _i, _pt in enumerate(pt_centres):
                for _j, _eta in enumerate(eta_centres):
                    _ax_sb.text(_eta, _pt, f"{_d[_i,_j]:.2f}",
                                ha="center", va="center", fontsize=5,
                                color="white" if _d[_i,_j] > 0.15 else "black")
    plt.tight_layout()
    _path_sb = os.path.join(args.output, "SF_syst_breakdown.png")
    fig_sb.savefig(_path_sb, dpi=150, bbox_inches="tight")
    plt.close(fig_sb)
    print(f"  → {_path_sb}")

# ── Spline fit + envelope plots ───────────────────────────────────────────────
print("\n── Spline fit + envelope ──")

try:
    from scipy.interpolate import UnivariateSpline
    _SCIPY_OK = True
except ImportError:
    print("  scipy not available, skipping spline plots")
    _SCIPY_OK = False

if _SCIPY_OK and _sf_q_err is not None and _syst_types:
    _sc_fine = np.linspace(score_centres[0], score_centres[-1], 200)
    _all_var_keys = (["stat_up", "stat_dn"]
                     + [f"{v}_{d}" for v in _syst_types for d in ("up", "dn")])

    # Build SF arrays for all variations in same shape as SF_q
    _var_sfs_q = {"central": SF_q, "stat_up": SF_q + _sf_q_err,
                  "stat_dn": SF_q - _sf_q_err}
    _var_sfs_g = {"central": SF_g, "stat_up": SF_g + _sf_g_err,
                  "stat_dn": SF_g - _sf_g_err}
    for _vt in _syst_types:
        _var_sfs_q[f"{_vt}_up"] = _ps_sf_q[f"{_vt}_up"]
        _var_sfs_q[f"{_vt}_dn"] = _ps_sf_q[f"{_vt}_dn"]
        _var_sfs_g[f"{_vt}_up"] = _ps_sf_g[f"{_vt}_up"]
        _var_sfs_g[f"{_vt}_dn"] = _ps_sf_g[f"{_vt}_dn"]

    def _fit_spline(sc_x, sf_y, err_y, k=3):
        """Weighted cubic spline: w = 1/err^2 (clamped). Returns callable."""
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(err_y > 0, 1.0 / np.maximum(err_y, 1e-6)**2, 0.0)
        w = np.clip(w, 0, None)
        if w.sum() < k + 1:
            return None
        try:
            spl = UnivariateSpline(sc_x, sf_y, w=w, k=k,
                                   s=len(sc_x), ext=3)
            return spl
        except Exception:
            return None

    print("\n── SF spline + envelope per |η| bin ──")
    for _j in range(len(eta_lo_arr)):
        _eta_lo = eta_lo_arr[_j]; _eta_hi = eta_hi_arr[_j]
        _is_cal = _eta_lo < args.eta_mask

        if _eta_lo >= 2.5:   # forward: use unique score indices and merged pT bins
            _sc_x    = _fwd_score_centres
            _sc_i    = _fwd_score_uniq_idx
            _pt_i    = _fwd_pt_uniq_idx
            _pt_lbls = _fwd_pt_labels
        else:
            _sc_x    = score_centres
            _sc_i    = list(range(len(score_centres)))
            _pt_i    = list(range(_n_pt_sf))
            _pt_lbls = None

        fig_sp, (ax_sq, ax_sg) = plt.subplots(1, 2, figsize=(14, 5))
        hep.cms.label(ax=ax_sq, label="Preliminary", data=True,
                      lumi=None, com=None, fontsize=14)
        ax_sg.text(1.0, 1.02,
                   f"{_zjets_lumi_fb:.4g}" + r" fb$^{-1}$ (13.6 TeV)",
                   transform=ax_sg.transAxes, ha="right", va="bottom", fontsize=12)

        for _ii, _i in enumerate(_pt_i):
            _color = PT_COLORS[_ii % len(PT_COLORS)]
            if _pt_lbls is not None:
                _pt_lbl = _pt_lbls[_ii]
            else:
                _pt_lo = pt_lo_arr[_i]; _pt_hi = pt_hi_arr[_i]
                _pt_lbl = (f"pT [{_pt_lo:.0f}, {_pt_hi:.0f})"
                           if _pt_hi < 7000 else f"pT [{_pt_lo:.0f}, \u221e)")

            _sfq = SF_q[np.ix_(_sc_i, [_i], [_j])][:, 0, 0]
            _sfg = SF_g[np.ix_(_sc_i, [_i], [_j])][:, 0, 0]
            _eq  = _sf_q_err[np.ix_(_sc_i, [_i], [_j])][:, 0, 0]
            _eg  = _sf_g_err[np.ix_(_sc_i, [_i], [_j])][:, 0, 0]

            # Fit central spline
            _spl_q = _fit_spline(_sc_x, _sfq, _eq)
            _spl_g = _fit_spline(_sc_x, _sfg, _eg)

            # Draw discrete points
            ax_sq.errorbar(_sc_x, _sfq, yerr=_eq, fmt="o", ms=3,
                           color=_color, alpha=0.4, linewidth=0)
            ax_sg.errorbar(_sc_x, _sfg, yerr=_eg, fmt="o", ms=3,
                           color=_color, alpha=0.4, linewidth=0)

            if _spl_q is not None:
                ax_sq.plot(_sc_fine, _spl_q(_sc_fine), color=_color,
                           linewidth=1.8, label=_pt_lbl)
            if _spl_g is not None:
                ax_sg.plot(_sc_fine, _spl_g(_sc_fine), color=_color,
                           linewidth=1.8, label=_pt_lbl)

            # Envelope: compute all variation splines and fill band
            _env_q_lo = np.full(len(_sc_fine), np.inf)
            _env_q_hi = np.full(len(_sc_fine), -np.inf)
            _env_g_lo = np.full(len(_sc_fine), np.inf)
            _env_g_hi = np.full(len(_sc_fine), -np.inf)
            for _vk in _all_var_keys:
                if _vk not in _var_sfs_q:
                    continue
                _sf_v_q = _var_sfs_q[_vk][np.ix_(_sc_i, [_i], [_j])][:, 0, 0]
                _sf_v_g = _var_sfs_g[_vk][np.ix_(_sc_i, [_i], [_j])][:, 0, 0]
                _spl_vq = _fit_spline(_sc_x, _sf_v_q, _eq)
                _spl_vg = _fit_spline(_sc_x, _sf_v_g, _eg)
                if _spl_vq is not None:
                    _yq = _spl_vq(_sc_fine)
                    _env_q_lo = np.minimum(_env_q_lo, _yq)
                    _env_q_hi = np.maximum(_env_q_hi, _yq)
                if _spl_vg is not None:
                    _yg = _spl_vg(_sc_fine)
                    _env_g_lo = np.minimum(_env_g_lo, _yg)
                    _env_g_hi = np.maximum(_env_g_hi, _yg)
            if np.isfinite(_env_q_lo).all():
                ax_sq.fill_between(_sc_fine, _env_q_lo, _env_q_hi,
                                   alpha=0.15, color=_color, linewidth=0)
            if np.isfinite(_env_g_lo).all():
                ax_sg.fill_between(_sc_fine, _env_g_lo, _env_g_hi,
                                   alpha=0.15, color=_color, linewidth=0)

        _mask_note = "  [SF=1, not calibrated]" if not _is_cal else ""
        for _ax_sp, _flav in [(ax_sq, "quark"), (ax_sg, "gluon")]:
            _ax_sp.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
            _ax_sp.set_xlabel("btagPNetQvG score", fontsize=14)
            _ax_sp.set_ylabel("SF", fontsize=14)
            _ax_sp.text(0.5, 0.97,
                        f"SF_{_flav}  |$\\eta$| $\\in$ [{_eta_lo:.1f}, {_eta_hi:.1f})"
                        f"{_mask_note}  [spline+envelope]",
                        transform=_ax_sp.transAxes, ha="center", va="top",
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="white", alpha=0.75))
            _ax_sp.legend(fontsize=10, loc="upper left")
            _ax_sp.set_xlim(0, 1)

        plt.tight_layout()
        _eta_tag = f"{_eta_lo:.1f}to{_eta_hi:.1f}".replace(".", "p")
        _path_sp = os.path.join(args.output,
                                f"SF_vs_score_eta{_eta_tag}_spline.png")
        fig_sp.savefig(_path_sp, dpi=150, bbox_inches="tight")
        plt.close(fig_sp)
        print(f"  → {_path_sp}")

print("\nDone.")
