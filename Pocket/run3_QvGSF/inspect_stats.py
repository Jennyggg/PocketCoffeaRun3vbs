#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_stats.py
Print per-(pT, |η|) and per-(score, pT, |η|) statistics for both the
Z+jets and dijet regions to identify low-yield bins.
"""

import argparse
import numpy as np
from coffea.util import load

parser = argparse.ArgumentParser()
parser.add_argument("--input",        default="output/output_all.coffea")
parser.add_argument("--min-mc-yield", type=float, default=1.0)
parser.add_argument("--max-pt",       type=float, default=100.0)
args = parser.parse_args()

print(f"Loading {args.input} ...")
out       = load(args.input)
variables = out["variables"]


def sum_group(var_name, group, cat, var="nominal"):
    hdict = variables[var_name][group]
    total_v = None
    for h in hdict.values():
        v = h[{"cat": cat, "variation": var}].values(flow=False)
        total_v = v.copy() if total_v is None else total_v + v
    return total_v   # (n_score, n_pt, n_eta)


# ── Load yields ───────────────────────────────────────────────────────────────
V_q_Z = sum_group("zjets_probe_score_pt_eta", "DYto2L-2Jets", "Zjets_quark")
V_g_Z = sum_group("zjets_probe_score_pt_eta", "DYto2L-2Jets", "Zjets_gluon")
V_u_Z = sum_group("zjets_probe_score_pt_eta", "DYto2L-2Jets", "Zjets_undef")
V_d_Z = sum_group("zjets_probe_score_pt_eta", "Muon",         "Zjets_all")

V_q_D = sum_group("dijet_probe_score_pt_eta", "QCD",    "Dijet_quark")
V_g_D = sum_group("dijet_probe_score_pt_eta", "QCD",    "Dijet_gluon")
V_u_D = sum_group("dijet_probe_score_pt_eta", "QCD",    "Dijet_undef")
V_d_D = sum_group("dijet_probe_score_pt_eta", "JetMET", "Dijet_all")

# ── Bin edges ─────────────────────────────────────────────────────────────────
_h_ref = list(variables["zjets_probe_score_pt_eta"]["Muon"].values())[0]
_h_sel = _h_ref[{"cat": "Zjets_all", "variation": "nominal"}]
score_edges = _h_sel.axes["events.zjets_probe_score"].edges
pt_edges    = _h_sel.axes["events.zjets_probe_pt"].edges
eta_edges   = _h_sel.axes["events.zjets_probe_abseta"].edges

pt_lo  = pt_edges[:-1]
pt_hi  = pt_edges[1:]
eta_lo = eta_edges[:-1]
eta_hi = eta_edges[1:]

n_score, n_pt, n_eta = V_q_Z.shape

# ── Per-(pT, |η|) summary (integrated over score) ────────────────────────────
MC_Z_tot  = (V_q_Z + V_g_Z + V_u_Z).sum(axis=0)   # (n_pt, n_eta)
MC_D_tot  = (V_q_D + V_g_D + V_u_D).sum(axis=0)
dat_Z_tot = V_d_Z.sum(axis=0)
dat_D_tot = V_d_D.sum(axis=0)

# quark/gluon fractions
frac_q_Z = np.where(MC_Z_tot > 0, V_q_Z.sum(axis=0) / MC_Z_tot, 0.0)
frac_g_Z = np.where(MC_Z_tot > 0, V_g_Z.sum(axis=0) / MC_Z_tot, 0.0)
frac_q_D = np.where(MC_D_tot > 0, V_q_D.sum(axis=0) / MC_D_tot, 0.0)
frac_g_D = np.where(MC_D_tot > 0, V_g_D.sum(axis=0) / MC_D_tot, 0.0)

# normalisation factors
k_Z = np.where(MC_Z_tot > 0, dat_Z_tot / MC_Z_tot, 1.0)
k_D = np.where(MC_D_tot > 0, dat_D_tot / MC_D_tot, 1.0)

print("\n" + "=" * 110)
print("PER-(pT, |η|) SUMMARY  (yields integrated over score bins)")
print("=" * 110)
hdr = (f"{'pT bin':>20}  {'|η| bin':>12}  "
       f"{'Zjets_MC':>10}  {'Zjets_dat':>10}  {'kZ':>6}  {'fq_Z':>5}  {'fg_Z':>5}  "
       f"{'Dijet_MC':>10}  {'Dijet_dat':>10}  {'kD':>6}  {'fq_D':>5}  {'fg_D':>5}  "
       f"{'flag':>15}")
print(hdr)
print("-" * 110)

for i in range(n_pt):
    skip_pt = (pt_lo[i] >= args.max_pt)
    for j in range(n_eta):
        mz = MC_Z_tot[i, j]
        md = MC_D_tot[i, j]
        dz = dat_Z_tot[i, j]
        dd = dat_D_tot[i, j]

        flags = []
        if skip_pt:
            flags.append("PT_CUT")
        if mz < args.min_mc_yield * n_score:
            flags.append("LOW_Z")
        if md < args.min_mc_yield * n_score:
            flags.append("LOW_D")

        print(f"  [{pt_lo[i]:5.0f},{pt_hi[i]:5.0f})  "
              f"[{eta_lo[j]:3.1f},{eta_hi[j]:3.1f})  "
              f"  {mz:10.1f}  {dz:10.1f}  {k_Z[i,j]:6.3f}  "
              f"{frac_q_Z[i,j]:5.2f}  {frac_g_Z[i,j]:5.2f}  "
              f"  {md:10.1f}  {dd:10.1f}  {k_D[i,j]:6.3f}  "
              f"{frac_q_D[i,j]:5.2f}  {frac_g_D[i,j]:5.2f}  "
              f"  {', '.join(flags) if flags else 'OK':>15}")

# ── Per-score-bin minimum yield (bottleneck for the solve) ────────────────────
# After normalisation: C = kZ*V_q_Z + kZ*V_g_Z, E+F = kD*(V_q_D+V_g_D)
# The solve bottleneck is min(C+D, E+F) per (score, pT, eta)
kZ = k_Z[np.newaxis, :, :]
kD = k_D[np.newaxis, :, :]
CD = kZ * (V_q_Z + V_g_Z)   # normalised total MC in Z+jets  (n_score,n_pt,n_eta)
EF = kD * (V_q_D + V_g_D)   # normalised total MC in dijet

# min over score: worst score bin per (pT,eta)
min_CD = CD.min(axis=0)   # (n_pt, n_eta)
min_EF = EF.min(axis=0)
n_low_CD = (CD < args.min_mc_yield).sum(axis=0)
n_low_EF = (EF < args.min_mc_yield).sum(axis=0)

print("\n" + "=" * 100)
print(f"PER-(pT, |η|) WORST SCORE BIN  (normalised yield; threshold={args.min_mc_yield:.1f})")
print("=" * 100)
hdr2 = (f"{'pT bin':>20}  {'|η| bin':>12}  "
        f"{'minZ/score':>12}  {'#lowZ':>6}  "
        f"{'minD/score':>12}  {'#lowD':>6}  {'flag':>15}")
print(hdr2)
print("-" * 100)

for i in range(n_pt):
    skip_pt = (pt_lo[i] >= args.max_pt)
    for j in range(n_eta):
        mz = min_CD[i, j]
        md = min_EF[i, j]
        nlz = n_low_CD[i, j]
        nld = n_low_EF[i, j]

        flags = []
        if skip_pt:
            flags.append("PT_CUT")
        if nlz > 0:
            flags.append(f"{nlz}Z_bins_low")
        if nld > 0:
            flags.append(f"{nld}D_bins_low")

        print(f"  [{pt_lo[i]:5.0f},{pt_hi[i]:5.0f})  "
              f"[{eta_lo[j]:3.1f},{eta_hi[j]:3.1f})  "
              f"  {mz:12.2f}  {nlz:6d}  "
              f"  {md:12.2f}  {nld:6d}  "
              f"  {', '.join(flags) if flags else 'OK':>25}")

# ── Full score-bin table for the most problematic (pT, eta) cells ─────────────
print("\n" + "=" * 80)
print("SCORE-BIN DETAIL for dijet bins with ≥5 low-yield score bins")
print("=" * 80)
score_lo = score_edges[:-1]
score_hi = score_edges[1:]

for i in range(n_pt):
    if pt_lo[i] >= args.max_pt:
        continue
    for j in range(n_eta):
        if n_low_EF[i, j] < 5:
            continue
        print(f"\n  pT=[{pt_lo[i]:.0f},{pt_hi[i]:.0f})  |η|=[{eta_lo[j]:.1f},{eta_hi[j]:.1f})"
              f"   kZ={k_Z[i,j]:.3f}  kD={k_D[i,j]:.3f}")
        print(f"  {'score bin':>16}  {'kZ*MC_Z(q+g)':>14}  {'kD*MC_D(q+g)':>14}  {'low?':>6}")
        for s in range(n_score):
            cz = CD[s, i, j]
            cd = EF[s, i, j]
            low = ("LOW_D" if cd < args.min_mc_yield else
                   "LOW_Z" if cz < args.min_mc_yield else "")
            print(f"  [{score_lo[s]:.3f},{score_hi[s]:.3f})  {cz:14.2f}  {cd:14.2f}  {low:>6}")
