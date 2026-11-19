#!/usr/bin/env python3
# derive_btag_eff_2024.py
#
# Build the 2024 MC b-tagging efficiency maps (BTV method 1a) from the output of
# config_btag_eff_2024.py (+ config_btag_eff_2024_stitch.py):
#
#   eps_WP(group, flavour, |eta|, pT) = sum_w(jets passing WP) / sum_w(all jets)
#   WP = L, T (UParTAK4B); flavour = hadronFlavour 0 (udsg), 4 (c), 5 (b)
#
# Groups: top, wjets, dy, vv, other (by sample name, see GROUPS) and "all"
# (every MC sample). A group bin with fewer than MIN_NEFF effective jets, or a
# non-physical efficiency (negative weights), falls back to the "all" value.
#
# Output: correctionlib JSON with one correction "btag_eff_2024"
#   inputs: group (str), wp (str: L/T), flavour (int: 0/4/5), abseta, pt
# plus a text summary.
#
# Usage:
#   python derive_btag_eff_2024.py output_btageff_2024/output_all.coffea \
#       [output_btageff_2024_stitch/output_all.coffea] [-o params/btag_eff_2024.json]
import argparse
import numpy as np
from coffea.util import load
from coffea.processor import accumulate
import correctionlib.schemav2 as cs

MIN_NEFF = 50.0
EPS_CLIP = (1e-4, 1 - 1e-4)
FLAVOURS = [0, 4, 5]
FLAV_NAME = {0: "light", 4: "c", 5: "b"}


from btag_fixedwp_2024 import group_of   # same grouping as the b-tag weight


GROUPS = ["top", "wjets", "dy", "vv", "other"]


def fold(h):
    """(flavour, pt, eta, score) hist -> counts/variances [flav, pt, abseta, 3 score classes].

    pT overflow goes to the last bin; eta is folded to |eta|; score underflow
    (tagger not run, score < 0) counts as 'fail L', overflow as 'pass T'."""
    v = h.values(flow=True)
    w2 = h.variances(flow=True)
    # axes with flow: flav (variable [0,4,5,6] -> udsg, c, b), pt, eta, score
    flav_ax, pt_ax, eta_ax, sc_ax = h.axes
    v, w2 = v[1:-1], w2[1:-1]                  # drop flavour under/overflow (not filled)
    def collapse(a):
        # pT: drop underflow (<30, not filled), merge overflow into the last bin
        a = a[:, 1:, ...]
        a[:, -2, ...] += a[:, -1, ...]
        a = a[:, :-1, ...]
        # eta: drop under/overflow (|eta|>2.5 not filled), fold
        a = a[:, :, 1:-1, :]
        neta = a.shape[2]
        a = a[:, :, neta // 2:, :] + a[:, :, :neta // 2, :][:, :, ::-1, :]
        # score: [under, failL, LnotT, T, over] -> [fail, LnotT, T]
        a = np.stack([a[..., 0] + a[..., 1], a[..., 2], a[..., 3] + a[..., 4]], axis=-1)
        return a
    return collapse(v.copy()), collapse(w2.copy()), [int(e) for e in flav_ax.edges[:-1]]


def efficiencies(counts, var):
    tot = counts.sum(-1)
    tot_w2 = var.sum(-1)
    passL = counts[..., 1] + counts[..., 2]
    passT = counts[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        effL = passL / tot
        effT = passT / tot
        neff = np.where(tot_w2 > 0, tot ** 2 / tot_w2, 0.0)
    return effL, effT, neff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", default="params/btag_eff_2024.json")
    args = ap.parse_args()

    out = accumulate([load(f) for f in args.inputs])
    hvar = out["variables"]["btag_eff"]

    def select(h):
        names = [a.name for a in h.axes]
        if "cat" in names:
            h = h[{"cat": "baseline"}]
        if "variation" in names:
            h = h[{"variation": "nominal"}]
        return h   # axes: flavour, pt, eta, score

    per_group = {}
    ref_h = None
    for sample, by_ds in hvar.items():
        for ds, h in by_ds.items():
            h = select(h)
            ref_h = ref_h or h
            c, w2, flavs = fold(h)
            g = group_of(sample)
            for key in (g, "all"):
                if key in per_group:
                    per_group[key][0] += c
                    per_group[key][1] += w2
                else:
                    per_group[key] = [c.copy(), w2.copy()]

    # binning from a selected histogram (axes: flavour, pt, eta, score)
    pt_edges = [float(x) for x in ref_h.axes[1].edges]
    eta_edges = [float(x) for x in ref_h.axes[2].edges]
    abseta_edges = [e for e in eta_edges if e >= 0]

    effs = {g: efficiencies(*per_group[g]) for g in per_group}
    final, nfallback = {}, {}

    def is_bad(eL, eT, neff):
        # a bin is replaced as an (eL, eT) pair, so that eT <= eL stays consistent
        bad = (neff < MIN_NEFF)
        for e in (eL, eT):
            bad = bad | ~np.isfinite(e) | (e < 0) | (e > 1)
        return bad | (eT > eL)

    # "all": a bad bin (empty / low stats / unphysical) takes the (eL, eT) of the
    # nearest good pT bin at the same flavour and |eta|; a |eta| row without any
    # good bin takes the flavour-inclusive efficiency.
    effL, effT, neff = effs["all"]
    bad = is_bad(effL, effT, neff)
    eL_all, eT_all = effL.copy(), effT.copy()
    counts_all = per_group["all"][0]
    for i in range(eL_all.shape[0]):
        tot = counts_all[i].sum(-1)
        if tot.sum() <= 0:
            raise RuntimeError(f"No jets of flavour {flavs[i]} at all -- cannot build the maps")
        incl_L = (counts_all[i][..., 1] + counts_all[i][..., 2]).sum() / tot.sum()
        incl_T = counts_all[i][..., 2].sum() / tot.sum()
        for ie in range(eL_all.shape[2]):
            good = np.where(~bad[i, :, ie])[0]
            if not len(good):
                print(f"WARNING: no usable bin for flavour {flavs[i]}, |eta| bin {ie}: "
                      f"using the flavour-inclusive efficiency (L {incl_L:.4f}, T {incl_T:.4f})")
            for ip in np.where(bad[i, :, ie])[0]:
                if len(good):
                    src = good[np.argmin(np.abs(good - ip - 0.1))]
                    eL_all[i, ip, ie], eT_all[i, ip, ie] = effL[i, src, ie], effT[i, src, ie]
                else:
                    eL_all[i, ip, ie], eT_all[i, ip, ie] = incl_L, incl_T
    final[("all", "L")] = np.clip(eL_all, *EPS_CLIP)
    final[("all", "T")] = np.clip(np.minimum(eT_all, eL_all), *EPS_CLIP)
    nfallback[("all", "L")] = nfallback[("all", "T")] = bad.sum(axis=(1, 2))

    # groups: bad bins fall back (as a pair) to the final "all" map
    for g in GROUPS:
        if g not in effs:
            continue
        effL, effT, neff = effs[g]
        bad = is_bad(effL, effT, neff)
        final[(g, "L")] = np.clip(np.where(bad, final[("all", "L")], effL), *EPS_CLIP)
        final[(g, "T")] = np.clip(np.where(bad, final[("all", "T")], effT), *EPS_CLIP)
        nfallback[(g, "L")] = nfallback[(g, "T")] = bad.sum(axis=(1, 2))

    nbins = (len(pt_edges) - 1) * (len(abseta_edges) - 1)
    print(f"{'group':6s} {'flav':5s} {'wp':2s}  "
          + "  ".join(f"|eta|<{e:.1f}" for e in abseta_edges[1:])
          + f"   (pT-averaged eff; fallback bins out of {nbins})")
    for g in GROUPS + ["all"]:
        if (g, "L") not in final:
            continue
        for wp in ("L", "T"):
            e = final[(g, wp)]
            for i, f in enumerate(flavs):
                tot = per_group[g][0][i].sum(-1)            # [pt, abseta]
                avg = (e[i] * tot).sum(0) / np.where(tot.sum(0) != 0, tot.sum(0), np.nan)
                print(f"{g:6s} {FLAV_NAME[f]:5s} {wp:2s}  " + "  ".join(f"{x:9.4f}" for x in avg)
                      + f"   ({int(nfallback[(g, wp)][i])})")

    # correctionlib: group -> wp -> flavour -> MultiBinning(abseta, pt)
    def multibin(arr2d):  # arr2d[pt, abseta]
        return cs.MultiBinning(
            nodetype="multibinning",
            inputs=["abseta", "pt"],
            edges=[abseta_edges, pt_edges],
            content=[float(arr2d[ip, ie]) for ie in range(len(abseta_edges) - 1)
                     for ip in range(len(pt_edges) - 1)],
            flow="clamp",
        )
    groups_present = [g for g in GROUPS + ["all"] if (g, "L") in final]
    data = cs.Category(
        nodetype="category", input="group",
        content=[cs.CategoryItem(key=g, value=cs.Category(
            nodetype="category", input="wp",
            content=[cs.CategoryItem(key=wp, value=cs.Category(
                nodetype="category", input="flavour",
                content=[cs.CategoryItem(key=f, value=multibin(final[(g, wp)][i]))
                         for i, f in enumerate(flavs)]))
                for wp in ("L", "T")]))
            for g in groups_present],
    )
    corr = cs.Correction(
        name="btag_eff_2024", version=1,
        description=("MC b-tagging efficiency of UParTAK4B (2024 Summer24) at the L/T working "
                     "points, per process group, after the semileptonic VBS preselection "
                     "(no b-tag requirement), jets of JetGoodBtagAcc. Groups: "
                     + ", ".join(groups_present) + ". |eta| and pT clamped to the map range."),
        inputs=[cs.Variable(name="group", type="string", description="process group"),
                cs.Variable(name="wp", type="string", description="L or T"),
                cs.Variable(name="flavour", type="int", description="hadronFlavour 0/4/5"),
                cs.Variable(name="abseta", type="real", description="|eta|"),
                cs.Variable(name="pt", type="real", description="jet pT [GeV]")],
        output=cs.Variable(name="eff", type="real"),
        data=data,
    )
    cset = cs.CorrectionSet(schema_version=2, corrections=[corr],
                            description="2024 MC b-tag efficiencies (derive_btag_eff_2024.py)")
    with open(args.output, "w") as f:
        f.write(cset.model_dump_json(exclude_unset=True, indent=1)
                if hasattr(cset, "model_dump_json") else cset.json(exclude_unset=True, indent=1))
    print(f"\nWritten {args.output}")


if __name__ == "__main__":
    main()
