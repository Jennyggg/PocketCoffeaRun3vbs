#!/usr/bin/env python3
"""
Derive btagPNetQvG scale factors from PocketCoffea 3D histograms.

Method
------
For each (pT_i, |η|_j) bin, solve the 2×2 linear system *per score bin s*:

  [ N_q_Z[s,i,j]   N_g_Z[s,i,j] ] [ SF_q[s,i,j] ]   [ N_data_Z[s,i,j] ]
  [ N_q_D[s,i,j]   N_g_D[s,i,j] ] [ SF_g[s,i,j] ] = [ N_data_D[s,i,j] ]

Z+jets region (quark-enriched): Muon data / DY MC
Dijet region  (gluon-enriched): per-pT-bin trigger data / QCD MC

Per-pT-bin dijet data sources
------------------------------
  [30, 40)  GeV : ZeroBias  (Dijet_ZB)
  [40, 50)  GeV : HLT_DiPFJetAve40  (Dijet_DJ40)
  [50, 60)  GeV : HLT_DiPFJetAve40  (Dijet_DJ40)
  [60, 80)  GeV : HLT_DiPFJetAve60  (Dijet_DJ60)
  [80, 140) GeV : HLT_DiPFJetAve80  (Dijet_DJ80)
  [140,200) GeV : HLT_DiPFJetAve140 (Dijet_DJ140)
  [200,260) GeV : HLT_DiPFJetAve200 (Dijet_DJ200)
  [260,∞)   GeV : HLT_DiPFJetAve260 (Dijet_DJ260)

QCD MC is rescaled to the trigger-specific data luminosity for each pT bin
before the 2×2 solve.

Output bins
-----------
  pT  : [30, 40, 50, 60, 80, 140, 200, 260, 8000]  (8 bins)
  |η| : [0.0, 1.3, 2.5, 3.0, 4.7]                           (4 bins, last SF=1)
  score: 18 central bins / 11 forward bins (same as before)
"""

import argparse
import json
import numpy as np
from coffea.util import load

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",            default="output_dijet_binnedpT/output_all.coffea")
parser.add_argument("--output",           default="qvgsf_2022_postEE_dj.json")
parser.add_argument("--year",             default="2022_postEE")
parser.add_argument("--min-mc-yield",     type=float, default=1.0)
parser.add_argument("--max-sf-unc",       type=float, default=0.5)
parser.add_argument("--zjets-data-group", default="Muon")
parser.add_argument("--zjets-lumi",       type=float, default=26337.27,
                    help="Effective lumi of Z+jets data [pb^-1].")
parser.add_argument("--mc-lumi",          type=float, default=26337.27,
                    help="Lumi used to normalize MC [pb^-1].")
# Per-trigger effective luminosities (pb^-1) — fill from brilcalc
parser.add_argument("--lumi-zb",    type=float, default=163.336)
parser.add_argument("--lumi-dj40",  type=float, default=0.0)
parser.add_argument("--lumi-dj60",  type=float, default=0.0)
parser.add_argument("--lumi-dj80",  type=float, default=0.0)
parser.add_argument("--lumi-dj140", type=float, default=0.0)
parser.add_argument("--lumi-dj200", type=float, default=0.0)
parser.add_argument("--lumi-dj260", type=float, default=0.0)
parser.add_argument("--lumi-dj320", type=float, default=0.0)
parser.add_argument("--lumi-dj400", type=float, default=0.0)
parser.add_argument("--lumi-dj500", type=float, default=0.0)
args = parser.parse_args()

_zjets_mc_scale = args.zjets_lumi / args.mc_lumi

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

print(f"Z+jets data group : {args.zjets_data_group}")
print(f"Z+jets lumi       : {args.zjets_lumi:.3f} pb-1")
print(f"MC lumi           : {args.mc_lumi:.3f} pb-1")
print(f"ZeroBias lumi     : {args.lumi_zb:.3f} pb-1")
print(f"DJ40 lumi         : {args.lumi_dj40:.3f} pb-1")
print(f"DJ60 lumi         : {args.lumi_dj60:.3f} pb-1")
print(f"DJ80 lumi         : {args.lumi_dj80:.3f} pb-1")
print(f"DJ140 lumi        : {args.lumi_dj140:.3f} pb-1")
print(f"DJ200 lumi        : {args.lumi_dj200:.3f} pb-1")
print(f"DJ260 lumi        : {args.lumi_dj260:.3f} pb-1")
print(f"DJ320 lumi        : {args.lumi_dj320:.3f} pb-1")
print(f"DJ400 lumi        : {args.lumi_dj400:.3f} pb-1")
print(f"DJ500 lumi        : {args.lumi_dj500:.3f} pb-1")

# ── Load output ───────────────────────────────────────────────────────────────
print(f"\nLoading {args.input}")
out       = load(args.input)
variables = out["variables"]

# ── Per output pT bin: data source definition ─────────────────────────────────
# Coffea pT bins (_PT_BINS):
#   [30,40,50,60,70,80,90,100,120,140,160,180,200,220,240,260,290,320,360,400,450,500,8000]
# Index: 0  1  2  3  4  5  6  7   8   9  10  11  12  13  14  15  16  17  18  19  20  21

# Central pT bins (|η| < 1.3, j=0): 8 output bins
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
_N_PT_OUT_CENTRAL = len(_PT_BIN_SOURCES_CENTRAL)  # 8

# Forward pT bins (|η| ≥ 1.3, j=1,2): 6 output bins — [140,∞) merged using DJ140
_PT_BIN_SOURCES_FWD = [
    ("ZB",    [0]),                                           # [30, 40)
    ("DJ40",  [1]),                                           # [40, 50)
    ("DJ40",  [2]),                                           # [50, 60)
    ("DJ60",  [3, 4]),                                        # [60, 80)
    ("DJ80",  [5, 6, 7, 8]),                                  # [80, 140)
    ("DJ140", [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),  # [140, ∞)
]
_N_PT_OUT_FWD = len(_PT_BIN_SOURCES_FWD)  # 6

# For backward-compat keep _PT_BIN_SOURCES pointing to central
_PT_BIN_SOURCES = _PT_BIN_SOURCES_CENTRAL
_N_PT_OUT       = _N_PT_OUT_CENTRAL

# Histogram variable name and category prefix per source.
# ZeroBias: probe (j1) histogram only; lead (j0) histogram lacks flavor subcategories
# in the current coffea output so is not included in the SF derivation.
_SOURCE_CONF = {
    "ZB": {
        "hist_vars":  ["dijet_zb_probe_score_pt_eta"],
        "cat_prefix": "Dijet_ZB",
        "data_group": "ZeroBias",
    },
    "DJ40":  {"hist_vars": ["dijet_dj_probe_score_pt_eta"], "cat_prefix": "Dijet_DJ40",  "data_group": "JetMET"},
    "DJ60":  {"hist_vars": ["dijet_dj_probe_score_pt_eta"], "cat_prefix": "Dijet_DJ60",  "data_group": "JetMET"},
    "DJ80":  {"hist_vars": ["dijet_dj_probe_score_pt_eta"], "cat_prefix": "Dijet_DJ80",  "data_group": "JetMET"},
    "DJ140": {"hist_vars": ["dijet_dj_probe_score_pt_eta"], "cat_prefix": "Dijet_DJ140", "data_group": "JetMET"},
    "DJ200": {"hist_vars": ["dijet_dj_probe_score_pt_eta"], "cat_prefix": "Dijet_DJ200", "data_group": "JetMET"},
    "DJ260": {"hist_vars": ["dijet_dj_probe_score_pt_eta"], "cat_prefix": "Dijet_DJ260", "data_group": "JetMET"},
    "DJ320": {"hist_vars": ["dijet_dj_probe_score_pt_eta"], "cat_prefix": "Dijet_DJ320", "data_group": "JetMET"},
    "DJ400": {"hist_vars": ["dijet_dj_probe_score_pt_eta"], "cat_prefix": "Dijet_DJ400", "data_group": "JetMET"},
    "DJ500": {"hist_vars": ["dijet_dj_probe_score_pt_eta"], "cat_prefix": "Dijet_DJ500", "data_group": "JetMET"},
}

# ── Helper: sum histograms over all datasets in a sample group ────────────────
def sum_group(var_name, group, cat, var="nominal"):
    """Returns (values, variances) of shape (n_score, n_pt_coffea, n_eta_coffea)."""
    hdict   = variables[var_name][group]
    total_v = None
    total_w = None
    for h in hdict.values():
        h_sel = h[{"cat": cat, "variation": var}]
        v = h_sel.values(flow=False)
        w = h_sel.variances(flow=False)
        if total_v is None:
            total_v, total_w = v.copy(), w.copy()
        else:
            total_v += v
            total_w += w
    return total_v, total_w


# ── Assemble per-pT-bin dijet arrays ──────────────────────────────────────────
_N_SCORE     = 20
_N_ETA_COFFEA = 5  # [0,0.7), [0.7,1.3), [1.3,2.5), [2.5,3.0), [3.0,4.7)

def assemble_dijet(flavor_suffix, use_data=False, mc_var="nominal",
                   pt_bin_sources=None):
    """
    Build (n_score=20, n_pt_out, n_eta_coffea=5) dijet arrays.

    pt_bin_sources: list of (src_key, fine_idxs); defaults to _PT_BIN_SOURCES_CENTRAL.
    MC arrays are scaled from mc_lumi to the trigger-specific data luminosity.
    """
    if pt_bin_sources is None:
        pt_bin_sources = _PT_BIN_SOURCES_CENTRAL
    n_pt_out = len(pt_bin_sources)
    V = np.zeros((_N_SCORE, n_pt_out, _N_ETA_COFFEA))
    W = np.zeros((_N_SCORE, n_pt_out, _N_ETA_COFFEA))

    for b, (src_key, fine_idxs) in enumerate(pt_bin_sources):
        conf  = _SOURCE_CONF[src_key]
        cat   = f"{conf['cat_prefix']}_{flavor_suffix}"
        group = conf["data_group"] if use_data else "QCD"
        var   = "nominal"          if use_data else mc_var

        for hvar in conf["hist_vars"]:
            v_full, w_full = sum_group(hvar, group, cat, var)
            V[:, b, :] += v_full[:, fine_idxs, :].sum(axis=1)
            W[:, b, :] += w_full[:, fine_idxs, :].sum(axis=1)

        if not use_data:
            scale = _source_lumis[src_key] / args.mc_lumi
            V[:, b, :] *= scale
            W[:, b, :] *= scale ** 2

    return V, W


# ── Extract Z+jets arrays ──────────────────────────────────────────────────────
V_q_Z_raw, W_q_Z_raw = sum_group("zjets_probe_score_pt_eta", "DYto2L-2Jets", "Zjets_quark")
V_g_Z_raw, W_g_Z_raw = sum_group("zjets_probe_score_pt_eta", "DYto2L-2Jets", "Zjets_gluon")
V_u_Z_raw, W_u_Z_raw = sum_group("zjets_probe_score_pt_eta", "DYto2L-2Jets", "Zjets_undef")
V_d_Z_raw, W_d_Z_raw = sum_group("zjets_probe_score_pt_eta", args.zjets_data_group, "Zjets_all")

# ── Assemble dijet arrays ──────────────────────────────────────────────────────
V_q_D, W_q_D = assemble_dijet("quark", use_data=False)
V_g_D, W_g_D = assemble_dijet("gluon", use_data=False)
V_u_D, W_u_D = assemble_dijet("undef", use_data=False)
V_d_D, W_d_D = assemble_dijet("all",   use_data=True)

# ── Rebinning definitions ─────────────────────────────────────────────────────
# Score: 20 coffea bins → 18 central (merge first 2, last 2) / 11 forward
_score_groups_central = [[0, 1]] + [[s] for s in range(2, 18)] + [[18, 19]]
_score_groups_forward = ([[0, 1, 2, 3]] + [[s] for s in range(4, 13)]
                         + [[13, 14, 15, 16, 17, 18, 19]])

# |η|: 5 coffea bins → 4 output bins
#   merge [0,0.7)+[0.7,1.3) → [0,1.3);  keep [1.3,2.5), [2.5,3.0), [3.0,4.7)
_eta_groups    = [[0, 1], [2], [3], [4]]
_N_ETA_CENTRAL = 2   # j=0: [0,1.3), j=1: [1.3,2.5)

# Z+jets pT groups for central eta (j=0): 8 output bins
_ZJ_PT_GROUPS_CENTRAL = [
    [0], [1], [2], [3, 4], [5, 6, 7, 8], [9, 10, 11],
    [12, 13, 14], [15, 16, 17, 18, 19, 20, 21],
]
# Z+jets pT groups for forward eta (j=1,2): 6 output bins — [140,∞) merged
_ZJ_PT_GROUPS_FWD = [
    [0], [1], [2], [3, 4], [5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
]
# Keep _ZJ_PT_GROUPS alias for backward compat
_ZJ_PT_GROUPS = _ZJ_PT_GROUPS_CENTRAL

# Score expansion: forward 11-bin → central 18-bin grid
_score_fwd_to_central = (
    [[0, 1, 2]]
    + [[i + 3] for i in range(9)]
    + [[12, 13, 14, 15, 16, 17]]
)
_n_score_out = 18


def merge_axis(arr, groups, axis):
    parts = [np.take(arr, grp, axis=axis).sum(axis=axis, keepdims=True)
             for grp in groups]
    return np.concatenate(parts, axis=axis)


def merge_score(v, w, groups):
    return merge_axis(v, groups, 0), merge_axis(w, groups, 0)


def merge_eta(v, w):
    """5 coffea eta bins → 4 output bins."""
    return merge_axis(v, _eta_groups, 2), merge_axis(w, _eta_groups, 2)


def rebin_zjets(v, w, pt_groups=None):
    """(n_score, 22, 5) → (n_score, n_pt_out, 4)."""
    if pt_groups is None:
        pt_groups = _ZJ_PT_GROUPS_CENTRAL
    v = merge_axis(v, pt_groups, 1)
    w = merge_axis(w, pt_groups, 1)
    return merge_eta(v, w)


def expand_fwd_pt(sf, sfe):
    """(n_score, 6, n_eta) → (n_score, 8, n_eta): repeat last bin to fill slots 5,6,7."""
    return (
        np.concatenate([sf[:, :5, :],  np.repeat(sf[:, 5:6, :],  3, axis=1)], axis=1),
        np.concatenate([sfe[:, :5, :], np.repeat(sfe[:, 5:6, :], 3, axis=1)], axis=1),
    )


# ── Assemble dijet: central pT (j=0) and forward pT (j=1,2) separately ───────
V_q_D_c, W_q_D_c = merge_eta(*assemble_dijet("quark"))
V_g_D_c, W_g_D_c = merge_eta(*assemble_dijet("gluon"))
V_u_D_c, W_u_D_c = merge_eta(*assemble_dijet("undef"))
V_d_D_c, W_d_D_c = merge_eta(*assemble_dijet("all", use_data=True))

V_q_D_f, W_q_D_f = merge_eta(*assemble_dijet("quark", pt_bin_sources=_PT_BIN_SOURCES_FWD))
V_g_D_f, W_g_D_f = merge_eta(*assemble_dijet("gluon", pt_bin_sources=_PT_BIN_SOURCES_FWD))
V_u_D_f, W_u_D_f = merge_eta(*assemble_dijet("undef", pt_bin_sources=_PT_BIN_SOURCES_FWD))
V_d_D_f, W_d_D_f = merge_eta(*assemble_dijet("all",   use_data=True,
                                              pt_bin_sources=_PT_BIN_SOURCES_FWD))

# ── Rebin Z+jets: central pT (j=0) and forward pT (j=1,2) ────────────────────
def _rebin_zj_c(v, w): return rebin_zjets(v, w, _ZJ_PT_GROUPS_CENTRAL)
def _rebin_zj_f(v, w): return rebin_zjets(v, w, _ZJ_PT_GROUPS_FWD)

V_q_Z_c, W_q_Z_c = _rebin_zj_c(V_q_Z_raw, W_q_Z_raw)
V_g_Z_c, W_g_Z_c = _rebin_zj_c(V_g_Z_raw, W_g_Z_raw)
V_u_Z_c, W_u_Z_c = _rebin_zj_c(V_u_Z_raw, W_u_Z_raw)
V_d_Z_c, W_d_Z_c = _rebin_zj_c(V_d_Z_raw, W_d_Z_raw)

V_q_Z_f, W_q_Z_f = _rebin_zj_f(V_q_Z_raw, W_q_Z_raw)
V_g_Z_f, W_g_Z_f = _rebin_zj_f(V_g_Z_raw, W_g_Z_raw)
V_u_Z_f, W_u_Z_f = _rebin_zj_f(V_u_Z_raw, W_u_Z_raw)
V_d_Z_f, W_d_Z_f = _rebin_zj_f(V_d_Z_raw, W_d_Z_raw)

for _v in (V_q_Z_c, V_g_Z_c, V_u_Z_c, V_q_Z_f, V_g_Z_f, V_u_Z_f):
    _v *= _zjets_mc_scale
for _w in (W_q_Z_c, W_g_Z_c, W_u_Z_c, W_q_Z_f, W_g_Z_f, W_u_Z_f):
    _w *= _zjets_mc_scale ** 2

# Keep unified views for validate-style callers (central pT only)
V_q_Z, W_q_Z = V_q_Z_c, W_q_Z_c
V_g_Z, W_g_Z = V_g_Z_c, W_g_Z_c
V_u_Z, W_u_Z = V_u_Z_c, W_u_Z_c
V_d_Z, W_d_Z = V_d_Z_c, W_d_Z_c
V_q_D, W_q_D = V_q_D_c, W_q_D_c
V_g_D, W_g_D = V_g_D_c, W_g_D_c
V_u_D, W_u_D = V_u_D_c, W_u_D_c
V_d_D, W_d_D = V_d_D_c, W_d_D_c

print(f"After rebin: Z+jets central shape = {V_q_Z_c.shape}  (n_score=20, n_pt=8, n_eta=4)")
print(f"After rebin: Z+jets forward shape = {V_q_Z_f.shape}  (n_score=20, n_pt=6, n_eta=4)")
print(f"After rebin: dijet  central shape = {V_q_D_c.shape}  (n_score=20, n_pt=8, n_eta=4)")
print(f"After rebin: dijet  forward shape = {V_q_D_f.shape}  (n_score=20, n_pt=6, n_eta=4)")

# ── 2×2 solver ────────────────────────────────────────────────────────────────
def process_region(
    Vq_Z, Wq_Z, Vg_Z, Wg_Z, Vu_Z, Wu_Z, Vd_Z, Wd_Z,
    Vq_D, Wq_D, Vg_D, Wg_D, Vu_D, Wu_D, Vd_D, Wd_D,
    pt_groups,
):
    """
    Optionally merge pT, compute k normalisations, solve 2×2 system.
    Input arrays are already scaled; no internal lumi scaling applied.
    Returns SF_q, SF_g, SF_q_err, SF_g_err of shape (n_score, n_pt, n_eta),
    plus k_Z and k_D of shape (n_pt, n_eta).
    """
    def mpt(v, w):
        return merge_axis(v, pt_groups, 1), merge_axis(w, pt_groups, 1)

    Vq_Z, Wq_Z = mpt(Vq_Z, Wq_Z);  Vg_Z, Wg_Z = mpt(Vg_Z, Wg_Z)
    Vu_Z, Wu_Z = mpt(Vu_Z, Wu_Z);  Vd_Z, Wd_Z = mpt(Vd_Z, Wd_Z)
    Vq_D, Wq_D = mpt(Vq_D, Wq_D);  Vg_D, Wg_D = mpt(Vg_D, Wg_D)
    Vu_D, Wu_D = mpt(Vu_D, Wu_D);  Vd_D, Wd_D = mpt(Vd_D, Wd_D)

    n_score, n_pt, n_eta = Vq_Z.shape

    N_MC_Z_tot  = (Vq_Z + Vg_Z + Vu_Z).sum(axis=0)
    N_dat_Z_tot = Vd_Z.sum(axis=0)
    k_Z = np.where(N_MC_Z_tot > 0, N_dat_Z_tot / N_MC_Z_tot, 1.0)

    N_MC_D_tot  = (Vq_D + Vg_D + Vu_D).sum(axis=0)
    N_dat_D_tot = Vd_D.sum(axis=0)
    k_D = np.where(N_MC_D_tot > 0, N_dat_D_tot / N_MC_D_tot, 1.0)

    kZ = k_Z[np.newaxis, :, :]
    kD = k_D[np.newaxis, :, :]

    C     = kZ * Vq_Z;  D_arr = kZ * Vg_Z
    E     = kD * Vq_D;  F     = kD * Vg_D
    A     = Vd_Z - kZ * Vu_Z
    B     = Vd_D - kD * Vu_D
    WA    = Wd_Z + kZ ** 2 * Wu_Z
    WB    = Wd_D + kD ** 2 * Wu_D

    SF_q     = np.ones((n_score, n_pt, n_eta))
    SF_g     = np.ones((n_score, n_pt, n_eta))
    SF_q_err = np.zeros((n_score, n_pt, n_eta))
    SF_g_err = np.zeros((n_score, n_pt, n_eta))

    for i in range(n_pt):
        for j in range(n_eta):
            for s in range(n_score):
                cq_z = C[s, i, j];   cg_z = D_arr[s, i, j]
                cq_d = E[s, i, j];   cg_d = F[s, i, j]
                a    = A[s, i, j];   b    = B[s, i, j]
                det  = cq_z * cg_d - cq_d * cg_z

                if (min(cq_z + cg_z, cq_d + cg_d) < args.min_mc_yield
                        or abs(det) < 1e-9):
                    continue

                sfq = (a * cg_d - b * cg_z) / det
                sfg = (b * cq_z - a * cq_d) / det
                SF_q[s, i, j] = sfq
                SF_g[s, i, j] = sfg

                wa = WA[s, i, j];  wb = WB[s, i, j]
                wc = kZ[0, i, j] ** 2 * Wq_Z[s, i, j]
                wd = kZ[0, i, j] ** 2 * Wg_Z[s, i, j]
                we = kD[0, i, j] ** 2 * Wq_D[s, i, j]
                wf = kD[0, i, j] ** 2 * Wg_D[s, i, j]
                var_sfq = (
                    (cg_d / det) ** 2 * wa + (cg_z / det) ** 2 * wb
                    + (sfq * cg_d / det) ** 2 * wc
                    + ((sfq * cq_d - b) / det) ** 2 * wd
                    + (sfq * cg_z / det) ** 2 * we
                    + ((a - sfq * cq_z) / det) ** 2 * wf
                )
                var_sfg = (
                    (cq_d / det) ** 2 * wa + (cq_z / det) ** 2 * wb
                    + ((b - sfg * cg_d) / det) ** 2 * wc
                    + (sfg * cq_d / det) ** 2 * wd
                    + ((sfg * cg_z - a) / det) ** 2 * we
                    + (sfg * cq_z / det) ** 2 * wf
                )
                SF_q_err[s, i, j] = np.sqrt(max(var_sfq, 0.0))
                SF_g_err[s, i, j] = np.sqrt(max(var_sfg, 0.0))

    SF_q = np.clip(SF_q, 0.01, 10.0)
    SF_g = np.clip(SF_g, 0.01, 10.0)
    return SF_q, SF_g, SF_q_err, SF_g_err, k_Z, k_D


# ── Solve: per-eta-column with appropriate pT binning ────────────────────────
_pt_id_c = [[i] for i in range(_N_PT_OUT_CENTRAL)]   # 8-bin identity
_pt_id_f = [[i] for i in range(_N_PT_OUT_FWD)]       # 6-bin identity


def expand_fwd_score(sf, sfe):
    """(11, n_pt, n_eta) → (18, n_pt, n_eta) by duplicating forward score bins."""
    n_pt  = sf.shape[1]
    n_eta = sf.shape[2]
    out_sf  = np.ones((_n_score_out, n_pt, n_eta))
    out_sfe = np.zeros((_n_score_out, n_pt, n_eta))
    for jc, fine_list in enumerate(_score_fwd_to_central):
        for fi in fine_list:
            out_sf[fi]  = sf[jc]
            out_sfe[fi] = sfe[jc]
    return out_sf, out_sfe


def _solve_col(zj_arrs, dj_arrs, j_slice, score_groups, pt_id, label):
    """Solve one eta column slice. Returns (sfq, sfg, sfqe, sfge) all (n_score, n_pt, 1)."""
    _arrs = [merge_score(a[:, :, j_slice], w[:, :, j_slice], score_groups)
             for a, w in zip(zj_arrs[0::2], zj_arrs[1::2])
             ] + [
             merge_score(a[:, :, j_slice], w[:, :, j_slice], score_groups)
             for a, w in zip(dj_arrs[0::2], dj_arrs[1::2])]
    sfq, sfg, sfqe, sfge, kZ, kD = process_region(
        *[x for pair in _arrs for x in pair], pt_id)
    print(f"\nSolving {label}")
    print(f"  k_Z = {np.round(kZ[:, 0], 3)}")
    print(f"  k_D = {np.round(kD[:, 0], 4)}")
    return sfq, sfg, sfqe, sfge


_zj_c_arrs = (V_q_Z_c, W_q_Z_c, V_g_Z_c, W_g_Z_c,
               V_u_Z_c, W_u_Z_c, V_d_Z_c, W_d_Z_c)
_zj_f_arrs = (V_q_Z_f, W_q_Z_f, V_g_Z_f, W_g_Z_f,
               V_u_Z_f, W_u_Z_f, V_d_Z_f, W_d_Z_f)
_dj_c_arrs = (V_q_D_c, W_q_D_c, V_g_D_c, W_g_D_c,
               V_u_D_c, W_u_D_c, V_d_D_c, W_d_D_c)
_dj_f_arrs = (V_q_D_f, W_q_D_f, V_g_D_f, W_g_D_f,
               V_u_D_f, W_u_D_f, V_d_D_f, W_d_D_f)

# j=0: η [0,1.3), central score, 8 pT bins
sfq_j0, sfg_j0, sfqe_j0, sfge_j0 = _solve_col(
    _zj_c_arrs, _dj_c_arrs, slice(0, 1), _score_groups_central, _pt_id_c,
    "j=0  |η| [0.0, 1.3)  — 8 pT bins")

# j=1: η [1.3,2.5), central score, 6 pT bins (merged [140,∞))
sfq_j1_6, sfg_j1_6, sfqe_j1_6, sfge_j1_6 = _solve_col(
    _zj_f_arrs, _dj_f_arrs, slice(1, 2), _score_groups_central, _pt_id_f,
    "j=1  |η| [1.3, 2.5) — 6 pT bins (DJ140 for [140,∞))")

# j=2: η [2.5,3.0), forward score, 6 pT bins (merged [140,∞))
sfq_j2_f11, sfg_j2_f11, sfqe_j2_f11, sfge_j2_f11 = _solve_col(
    _zj_f_arrs, _dj_f_arrs, slice(2, 3), _score_groups_forward, _pt_id_f,
    "j=2  |η| [2.5, 3.0) — 6 pT bins (DJ140 for [140,∞)), forward score")

# Expand forward score j=2: (11,6,1) → (18,6,1)
sfq_j2_6, sfqe_j2_6 = expand_fwd_score(sfq_j2_f11, sfqe_j2_f11)
sfg_j2_6, sfge_j2_6 = expand_fwd_score(sfg_j2_f11, sfge_j2_f11)

# Expand forward pT j=1,2: (18,6,1) → (18,8,1)  last bin covers [140,200), [200,260), [260,∞)
sfq_j1, sfqe_j1 = expand_fwd_pt(sfq_j1_6, sfqe_j1_6)
sfg_j1, sfge_j1 = expand_fwd_pt(sfg_j1_6, sfge_j1_6)
sfq_j2, sfqe_j2 = expand_fwd_pt(sfq_j2_6, sfqe_j2_6)
sfg_j2, sfge_j2 = expand_fwd_pt(sfg_j2_6, sfge_j2_6)

# j=3: η [3.0,4.7), SF=1
_eta_mask_idx = 3
sfq_j3  = np.ones((_n_score_out,  _N_PT_OUT_CENTRAL, 1))
sfg_j3  = np.ones((_n_score_out,  _N_PT_OUT_CENTRAL, 1))
sfqe_j3 = np.zeros((_n_score_out, _N_PT_OUT_CENTRAL, 1))
sfge_j3 = np.zeros((_n_score_out, _N_PT_OUT_CENTRAL, 1))

# ── Combine → (18, 8, 4) ──────────────────────────────────────────────────────
SF_q     = np.concatenate([sfq_j0,  sfq_j1,  sfq_j2,  sfq_j3 ], axis=2)
SF_g     = np.concatenate([sfg_j0,  sfg_j1,  sfg_j2,  sfg_j3 ], axis=2)
SF_q_err = np.concatenate([sfqe_j0, sfqe_j1, sfqe_j2, sfqe_j3], axis=2)
SF_g_err = np.concatenate([sfge_j0, sfge_j1, sfge_j2, sfge_j3], axis=2)

print(f"\nCombined SF shape: {SF_q.shape}  (n_score=18, n_pt=8, n_eta=4)")
print(f"SF_q range: [{SF_q.min():.3f}, {SF_q.max():.3f}]")
print(f"SF_g range: [{SF_g.min():.3f}, {SF_g.max():.3f}]")


# ── Parton-shower / JES / JER / PU variation SFs ─────────────────────────────
def _extract_mc_yields_var(var_mc):
    """Re-extract MC arrays for a variation; returns (central_arrs, fwd_arrs)."""
    def _zj_c(group, cat):
        return rebin_zjets(*sum_group("zjets_probe_score_pt_eta", group, cat, var_mc),
                           pt_groups=_ZJ_PT_GROUPS_CENTRAL)
    def _zj_f(group, cat):
        return rebin_zjets(*sum_group("zjets_probe_score_pt_eta", group, cat, var_mc),
                           pt_groups=_ZJ_PT_GROUPS_FWD)

    # MC (varied) — central pT
    vqzc, wqzc = _zj_c("DYto2L-2Jets", "Zjets_quark")
    vgzc, wgzc = _zj_c("DYto2L-2Jets", "Zjets_gluon")
    vuzc, wuzc = _zj_c("DYto2L-2Jets", "Zjets_undef")
    # MC (varied) — forward pT
    vqzf, wqzf = _zj_f("DYto2L-2Jets", "Zjets_quark")
    vgzf, wgzf = _zj_f("DYto2L-2Jets", "Zjets_gluon")
    vuzf, wuzf = _zj_f("DYto2L-2Jets", "Zjets_undef")
    # Data stays nominal
    vdzc, wdzc = rebin_zjets(*sum_group("zjets_probe_score_pt_eta",
                                        args.zjets_data_group, "Zjets_all"),
                              pt_groups=_ZJ_PT_GROUPS_CENTRAL)
    vdzf, wdzf = rebin_zjets(*sum_group("zjets_probe_score_pt_eta",
                                        args.zjets_data_group, "Zjets_all"),
                              pt_groups=_ZJ_PT_GROUPS_FWD)
    for _v in (vqzc, vgzc, vuzc, vqzf, vgzf, vuzf):
        _v *= _zjets_mc_scale
    for _w in (wqzc, wgzc, wuzc, wqzf, wgzf, wuzf):
        _w *= _zjets_mc_scale ** 2

    vqdc, wqdc = merge_eta(*assemble_dijet("quark", mc_var=var_mc))
    vgdc, wgdc = merge_eta(*assemble_dijet("gluon", mc_var=var_mc))
    vudc, wudc = merge_eta(*assemble_dijet("undef", mc_var=var_mc))
    vqdf, wqdf = merge_eta(*assemble_dijet("quark", mc_var=var_mc,
                                            pt_bin_sources=_PT_BIN_SOURCES_FWD))
    vgdf, wgdf = merge_eta(*assemble_dijet("gluon", mc_var=var_mc,
                                            pt_bin_sources=_PT_BIN_SOURCES_FWD))
    vudf, wudf = merge_eta(*assemble_dijet("undef", mc_var=var_mc,
                                            pt_bin_sources=_PT_BIN_SOURCES_FWD))

    central = (vqzc, wqzc, vgzc, wgzc, vuzc, wuzc, vdzc, wdzc,
               vqdc, wqdc, vgdc, wgdc, vudc, wudc, V_d_D_c, W_d_D_c)
    fwd     = (vqzf, wqzf, vgzf, wgzf, vuzf, wuzf, vdzf, wdzf,
               vqdf, wqdf, vgdf, wgdf, vudf, wudf, V_d_D_f, W_d_D_f)
    return central, fwd


def _run_full_solve(central_arrs, fwd_arrs):
    """Per-eta-column solve; returns (SF_q, SF_g) shape (18,8,4)."""
    def _col(zj, dj, j_sl, sc_grp, pt_id):
        _a = [merge_score(a[:, :, j_sl], w[:, :, j_sl], sc_grp)
              for a, w in zip(zj[0::2], zj[1::2])] + [
              merge_score(a[:, :, j_sl], w[:, :, j_sl], sc_grp)
              for a, w in zip(dj[0::2], dj[1::2])]
        sfq, sfg, _, _, _, _ = process_region(*[x for p in _a for x in p], pt_id)
        return sfq, sfg

    sq0, sg0 = _col(central_arrs[:8], central_arrs[8:], slice(0, 1),
                    _score_groups_central, _pt_id_c)
    sq1, sg1 = _col(fwd_arrs[:8], fwd_arrs[8:], slice(1, 2),
                    _score_groups_central, _pt_id_f)
    sq2f, sg2f = _col(fwd_arrs[:8], fwd_arrs[8:], slice(2, 3),
                      _score_groups_forward, _pt_id_f)
    sq2_6, _ = expand_fwd_score(sq2f, np.zeros_like(sq2f))
    sg2_6, _ = expand_fwd_score(sg2f, np.zeros_like(sg2f))
    sq1_8, _ = expand_fwd_pt(sq1, np.zeros_like(sq1))
    sg1_8, _ = expand_fwd_pt(sg1, np.zeros_like(sg1))
    sq2_8, _ = expand_fwd_pt(sq2_6, np.zeros_like(sq2_6))
    sg2_8, _ = expand_fwd_pt(sg2_6, np.zeros_like(sg2_6))
    sq3 = np.ones((_n_score_out, _N_PT_OUT_CENTRAL, 1))
    sg3 = np.ones((_n_score_out, _N_PT_OUT_CENTRAL, 1))
    SFq = np.concatenate([sq0, sq1_8, sq2_8, sq3], axis=2)
    SFg = np.concatenate([sg0, sg1_8, sg2_8, sg3], axis=2)
    SFq[:, :, _eta_mask_idx:] = 1.0
    SFg[:, :, _eta_mask_idx:] = 1.0
    return SFq, SFg


_PS_VARIATIONS = {
    "isr_up": "sf_partonshower_isrUp",
    "isr_dn": "sf_partonshower_isrDown",
    "fsr_up": "sf_partonshower_fsrUp",
    "fsr_dn": "sf_partonshower_fsrDown",
    "jes_up": "AK4PFPuppi_JES_TotalUp",
    "jes_dn": "AK4PFPuppi_JES_TotalDown",
    "jer_up": "AK4PFPuppi_JERUp",
    "jer_dn": "AK4PFPuppi_JERDown",
    "pu_up":  "pileupUp",
    "pu_dn":  "pileupDown",
}
ps_sf_q, ps_sf_g = {}, {}
for _pkey, _pvar in _PS_VARIATIONS.items():
    print(f"\nDeriving SFs for {_pkey} ({_pvar}) ...")
    _cen, _fwd = _extract_mc_yields_var(_pvar)
    ps_sf_q[_pkey], ps_sf_g[_pkey] = _run_full_solve(_cen, _fwd)
    print(f"  SF_q [{ps_sf_q[_pkey].min():.3f}, {ps_sf_q[_pkey].max():.3f}]  "
          f"SF_g [{ps_sf_g[_pkey].min():.3f}, {ps_sf_g[_pkey].max():.3f}]")

# ── Output bin edges ──────────────────────────────────────────────────────────
_orig_score = np.linspace(0.0, 1.0, 21)
score_edges = [0.0] + [round(x, 4) for x in _orig_score[2:19].tolist()] + [1.0]
pt_edges    = [30., 40., 50., 60., 80., 140., 200., 260., 8000.]
eta_edges   = [0.0, 1.3, 2.5, 3.0, 4.7]

# ── Build correctionlib JSON ──────────────────────────────────────────────────
def multibinning_3d(score_edges, eta_edges, pt_edges, values):
    """
    values shape: (n_score, n_pt, n_eta)
    MultiBinning axes order: score (outer), abseta (middle), pt (inner)
    → transpose to (n_score, n_eta, n_pt) then flatten.
    """
    arr = np.transpose(values, (0, 2, 1))
    return {
        "nodetype": "multibinning",
        "inputs":   ["score", "abseta", "pt"],
        "edges":    [score_edges, eta_edges, pt_edges],
        "content":  arr.flatten().tolist(),
        "flow":     "clamp",
    }


def flavor_node(sf_arr):
    return {
        "nodetype": "category",
        "input":    "flavor",
        "content": [
            {"key": "quark",
             "value": multibinning_3d(score_edges, eta_edges, pt_edges, sf_arr[0])},
            {"key": "gluon",
             "value": multibinning_3d(score_edges, eta_edges, pt_edges, sf_arr[1])},
        ],
    }


correction = {
    "name": "btagPNetQvG_SF",
    "description": (
        "ParticleNet QvG shape scale factors for AK4 jets, 2022 postEE (Run 3). "
        "Derived via 2×2 tag-and-probe: Z+jets (quark-enriched) + dijet (gluon-enriched). "
        "Dijet data uses per-pT-bin triggers: ZeroBias [30,40), DJ40 [40,60), "
        "DJ60 [60,80), DJ80 [80,140), DJ140 [140,200), DJ200 [200,260), "
        "DJ260 [260,∞) GeV. "
        "pT bins: [30,40,50,60,80,140,200,260,8000] GeV. "
        "|η| bins: [0,1.3,2.5,3.0,4.7] (SF=1 for |η|≥3.0). "
        "Score bins [0,0.20) and [0.70,1.0) merged in forward region."
    ),
    "version": 1,
    "inputs": [
        {"name": "systematic", "type": "string",
         "description": "Variation: central / stat_up / stat_dn / isr_up / isr_dn / fsr_up / fsr_dn / jes_up / jes_dn / jer_up / jer_dn / pu_up / pu_dn"},
        {"name": "flavor",     "type": "string",
         "description": "quark (|pf|=1-5) or gluon (pf=21)"},
        {"name": "score",  "type": "real", "description": "btagPNetQvG score"},
        {"name": "abseta", "type": "real", "description": "Jet |η|"},
        {"name": "pt",     "type": "real", "description": "Jet pT [GeV]"},
    ],
    "output": {"name": "weight", "type": "real"},
    "data": {
        "nodetype": "category",
        "input":    "systematic",
        "content": (
            [
                {"key": "central",  "value": flavor_node([SF_q,             SF_g            ])},
                {"key": "stat_up",  "value": flavor_node([SF_q + SF_q_err,  SF_g + SF_g_err ])},
                {"key": "stat_dn",  "value": flavor_node([SF_q - SF_q_err,  SF_g - SF_g_err ])},
            ] + [
                {"key": k, "value": flavor_node([ps_sf_q[k], ps_sf_g[k]])}
                for k in ps_sf_q
            ]
        ),
    },
}

clib_json = {"schema_version": 2, "corrections": [correction]}
with open(args.output, "w") as f:
    json.dump(clib_json, f, indent=2)
print(f"\nWrote correctionlib JSON → {args.output}")

# ── Save diagnostic arrays ────────────────────────────────────────────────────
diag_path = args.output.replace(".json", "_diag.npz")
np.savez(
    diag_path,
    SF_q=SF_q, SF_g=SF_g,
    SF_q_err=SF_q_err, SF_g_err=SF_g_err,
    **{f"SF_q_{k}": v for k, v in ps_sf_q.items()},
    **{f"SF_g_{k}": v for k, v in ps_sf_g.items()},
    score_edges=score_edges, pt_edges=pt_edges, eta_edges=eta_edges,
)
print(f"Wrote diagnostic arrays → {diag_path}")

# ── Self-check ────────────────────────────────────────────────────────────────
try:
    import correctionlib
    cset = correctionlib.CorrectionSet.from_file(args.output)
    sf   = cset["btagPNetQvG_SF"]
    sfq  = sf.evaluate("central", "quark", 0.5, 0.25, 50.0)
    sfg  = sf.evaluate("central", "gluon", 0.5, 0.25, 50.0)
    print(f"\nSelf-check: SF_q(score=0.5,η=0.25,pT=50)={sfq:.4f}  "
          f"SF_g(score=0.5,η=0.25,pT=50)={sfg:.4f}")
except Exception as e:
    print(f"Self-check skipped: {e}")
