# tagger_sfs_2022.py
#
# 2022 (postEE) QvG, WvsQCD and tau21 scale factors with their 2022
# uncertainties, applied unchanged to 2024 by workflow_evaluate_2024.py and
# config_semilep_evaluateMC_addscaleweight_2024.py (no extra 2022->2024
# uncertainty, by choice).
#
# Extracted VERBATIM from the 2022 analysis, so that both years use identical code:
#   SF evaluation  <- workflow_evaluate_qvgcal_WvsQCDcal_v10.py
#   weight classes <- config_semilep_evaluateMC_qvgcal_addscaleweight_v10.py
# Do not edit one copy without the other.
#
# Inputs (2022 postEE measurements):
#   ../run3_QvGSF/qvgsf_2022_postEE_dj.json (+ _diag.npz systematics)
#   output_boosted_control/fj_WvsQCD_SF_tau21cut.json
#   output_boosted_control/fj_tau21_SF.json
import numpy as np
import awkward as ak
import correctionlib
import os as _os
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation


# ── QvG SF correctionlib + systematic arrays ─────────────────────────
_QVGSF_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "../run3_QvGSF/qvgsf_2022_postEE_dj.json",
)
_QVGSF_DIAG_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "../run3_QvGSF/qvgsf_2022_postEE_dj_diag.npz",
)

_qvgsf_cache = {}
_qvgsf_diag_cache = {}

# SF bin edges matching derive_qvgsf.py output
_SF_SCORE_EDGES = np.array([0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0])
_SF_PT_EDGES    = np.array([30., 40., 50., 60., 80., 140., 200., 260., 8000.])
_SF_ETA_EDGES   = np.array([0.0, 1.3, 2.5, 3.0, 4.7])

# QvG uncertainty sources: stat (correctionlib) + syst (must match diag.npz)
_QVG_SOURCES = ["stat", "isr", "fsr", "jes", "jer", "pu"]
# Each source is split into two decorrelated uncertainties by jet |eta|:
# only jets in the given region are varied, the others stay at the central SF.
_QVG_ETA_SPLIT = 2.5
_QVG_ETA_REGIONS = ["etalt2p5", "etage2p5"]
# Variation names, e.g. "stat_etalt2p5" -> event fields sf_qvg_stat_etalt2p5_{up,dn}
_QVG_VARIATIONS = [f"{s}_{r}" for s in _QVG_SOURCES for r in _QVG_ETA_REGIONS]
_QVG_SYST_KEYS = [f"{v}_{d}" for v in _QVG_VARIATIONS for d in ["up", "dn"]]


def _get_qvgsf():
    if _QVGSF_PATH not in _qvgsf_cache:
        cset = correctionlib.CorrectionSet.from_file(_QVGSF_PATH)
        _qvgsf_cache[_QVGSF_PATH] = cset["btagPNetQvG_SF"]
    return _qvgsf_cache[_QVGSF_PATH]


def _get_diag_arrays():
    """Load syst SF arrays from diag.npz; returns {var_key: (sf_q_arr, sf_g_arr)}."""
    if _QVGSF_DIAG_PATH not in _qvgsf_diag_cache:
        if not _os.path.exists(_QVGSF_DIAG_PATH):
            _qvgsf_diag_cache[_QVGSF_DIAG_PATH] = {}
        else:
            d = np.load(_QVGSF_DIAG_PATH)
            _qvgsf_diag_cache[_QVGSF_DIAG_PATH] = {
                k[len("SF_q_"):]: (d[k], d[f"SF_g_{k[len('SF_q_'):]}"])
                for k in d.files
                if k.startswith("SF_q_") and not k.endswith("_err")
            }
    return _qvgsf_diag_cache[_QVGSF_DIAG_PATH]


def _eval_sf_flat(sf_q_arr, sf_g_arr, pt_flat, abs_eta, score_flat, flav_abs, valid):
    """Evaluate per-jet SFs from a (18, 8, 4) array using bin lookup."""
    n      = len(pt_flat)
    sf_out = np.ones(n, dtype=np.float64)
    i_sc   = np.clip(np.digitize(score_flat, _SF_SCORE_EDGES) - 1, 0, 17)
    i_pt   = np.clip(np.digitize(pt_flat,    _SF_PT_EDGES)    - 1, 0,  7)
    i_et   = np.clip(np.digitize(abs_eta,    _SF_ETA_EDGES)   - 1, 0,  3)
    for mask, arr in [(valid & np.isin(flav_abs, [1, 2, 3, 4]), sf_q_arr),
                      (valid & (flav_abs == 21),                 sf_g_arr)]:
        if mask.any():
            sf_out[mask] = arr[i_sc[mask], i_pt[mask], i_et[mask]]
    return sf_out


def _compute_qvg_sf_lead4(jets):
    """
    Per-event QvG SF products for the leading 4 jets only.

    Valid jets: 30 <= pT, |eta| < 3.0, score in [0, 1].
    partonFlavour {1,2,3,4} -> "quark"; 21 -> "gluon"; else SF=1.
    If stat uncertainty > 1 for a jet, reset that jet to SF=1 for stat variations.
    Each variation is split by jet |eta| (< or >= _QVG_ETA_SPLIT): only jets in
    that region take the varied SF, all other jets keep the central SF.
    Returns dict: {"central", "stat_etalt2p5_up", "stat_etage2p5_up", ...}
    each of length n_events.
    """
    lead4 = jets[:, :4]

    pt_flat    = ak.to_numpy(ak.flatten(lead4.pt))
    eta_flat   = ak.to_numpy(ak.flatten(lead4.eta))
    score_flat = ak.to_numpy(ak.flatten(ak.fill_none(lead4.btagPNetQvG, -1.0)))
    flav_flat  = ak.to_numpy(ak.flatten(lead4.partonFlavour)).astype(int)
    counts     = ak.to_numpy(ak.num(lead4))

    abs_eta  = np.abs(eta_flat)
    flav_abs = np.abs(flav_flat)
    valid    = ((pt_flat >= 30.0) & (abs_eta < 3.0) &
                (score_flat >= 0.0) & (score_flat <= 1.0))

    # Central + stat via correctionlib
    corr  = _get_qvgsf()
    sf_c  = np.ones(len(pt_flat), dtype=np.float64)
    sf_up = np.ones(len(pt_flat), dtype=np.float64)
    sf_dn = np.ones(len(pt_flat), dtype=np.float64)

    for flavor_str, fmask in [("quark", np.isin(flav_abs, [1, 2, 3, 4])),
                               ("gluon", flav_abs == 21)]:
        m = valid & fmask
        if not m.any():
            continue
        s  = np.clip(score_flat[m], 0.0, 0.9999)
        ae = abs_eta[m]
        pt = pt_flat[m]
        sf_c[m]  = corr.evaluate("central",  flavor_str, s, ae, pt)
        sf_up[m] = corr.evaluate("stat_up",  flavor_str, s, ae, pt)
        sf_dn[m] = corr.evaluate("stat_dn",  flavor_str, s, ae, pt)

    large_unc = (np.abs(sf_up - sf_c) > 1.0) | (np.abs(sf_c - sf_dn) > 1.0)
    sf_c[large_unc]  = 1.0
    sf_up[large_unc] = 1.0
    sf_dn[large_unc] = 1.0

    def _prod(arr):
        return ak.to_numpy(ak.prod(ak.unflatten(arr, counts), axis=1))

    # Per-jet varied SFs: {"stat_up": arr, ..., "isr_up": arr, ...}
    per_jet_var = {"stat_up": sf_up, "stat_dn": sf_dn}
    # Syst variations from diag.npz (array lookup)
    for vk, (sf_q_arr, sf_g_arr) in _get_diag_arrays().items():
        per_jet_var[vk] = _eval_sf_flat(sf_q_arr, sf_g_arr,
                                        pt_flat, abs_eta, score_flat, flav_abs, valid)

    result = {"central": _prod(sf_c)}
    region_masks = {"etalt2p5": abs_eta <  _QVG_ETA_SPLIT,
                    "etage2p5": abs_eta >= _QVG_ETA_SPLIT}
    for vk, sf_var in per_jet_var.items():
        src, d = vk.rsplit("_", 1)
        for reg, rmask in region_masks.items():
            result[f"{src}_{reg}_{d}"] = _prod(np.where(rmask, sf_var, sf_c))

    return result

# ── WvsQCD SF correctionlib ───────────────────────────────────────────
_WVSQCD_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "output_boosted_control/fj_WvsQCD_SF_tau21cut.json",
)
_wvsqcd_cache = {}

_WVSQCD_SYST_KEYS = [
    "nominal",
    "stat_up", "stat_down",
    "pileup_up", "pileup_down",
    "sf_btag_up", "sf_btag_down",
    "sf_partonshower_isr_up", "sf_partonshower_isr_down",
    "sf_partonshower_fsr_up", "sf_partonshower_fsr_down",
    "JES_up", "JES_down",
    "JER_up", "JER_down",
]




def _get_wvsqcd():
    if _WVSQCD_PATH not in _wvsqcd_cache:
        cset = correctionlib.CorrectionSet.from_file(_WVSQCD_PATH)
        _wvsqcd_cache[_WVSQCD_PATH] = cset["fj_WvsQCD_SF_tau21cut"]
    return _wvsqcd_cache[_WVSQCD_PATH]


def _compute_wvsqcd_sf(candidate_boost, gen_parts, isMC):
    """Evaluate WvsQCD SF for the leading fatjet candidate, all syst variations.

    fjtype = 'matched' if any gen W boson (|pdgId|=24) is within dR<0.8 of the
    fatjet, else 'unmatched'.  Events without a fatjet, or with score/msd out of
    the SF range, receive SF=1.
    Returns dict: {syst_key: np.ndarray(n_ev)} for all keys in _WVSQCD_SYST_KEYS.
    """
    n_ev = len(candidate_boost)
    ones = np.ones(n_ev, dtype=np.float64)

    if not isMC:
        return {k: ones.copy() for k in _WVSQCD_SYST_KEYS}

    corr = _get_wvsqcd()

    # Gen-matching on the full collection, then take the leading entry.
    gen_W = gen_parts[(np.abs(gen_parts.pdgId) == 24)]
    dR_fj_w = candidate_boost.metric_table(gen_W)          # (n_ev, n_fj, n_genW)
    is_matched_coll = ak.sum(dR_fj_w < 0.8, axis=2) > 0   # (n_ev, n_fj)
    is_matched = ak.to_numpy(ak.fill_none(ak.firsts(is_matched_coll), False))

    lead_fj  = ak.firsts(candidate_boost)
    has_fj   = ak.to_numpy(~ak.is_none(lead_fj))
    fj_msd   = ak.to_numpy(ak.fill_none(lead_fj.msoftdrop, -1.0))
    fj_score = ak.to_numpy(ak.fill_none(lead_fj.particleNetWithMass_WvsQCD, -1.0))

    fjtype_arr = np.where(is_matched, "matched", "unmatched")
    valid = (has_fj & (fj_score >= 0.0) & (fj_score <= 1.0) &
             (fj_msd >= 40.0) & (fj_msd < 200.0))
    score_cl = np.clip(fj_score, 0.0, 0.9999)
    msd_cl   = np.clip(fj_msd, 40.0, 199.9)

    result = {}
    for syst_key in _WVSQCD_SYST_KEYS:
        sf = ones.copy()
        for ft in ("matched", "unmatched"):
            mask = valid & (fjtype_arr == ft)
            if not mask.any():
                continue
            sf[mask] = corr.evaluate(syst_key, ft, msd_cl[mask], score_cl[mask])
        result[syst_key] = sf

    return result


# ── tau21 SF correctionlib ─────────────────────────────────────────────
_TAU21SF_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "output_boosted_control/fj_tau21_SF.json",
)
_tau21sf_cache = {}

_TAU21_SYST_KEYS = [
    "nominal",
    "stat_up", "stat_down",
    "pileup_up", "pileup_down",
    "sf_btag_up", "sf_btag_down",
    "sf_partonshower_isr_up", "sf_partonshower_isr_down",
    "sf_partonshower_fsr_up", "sf_partonshower_fsr_down",
    "JES_up", "JES_down",
    "JER_up", "JER_down",
]


def _get_tau21sf():
    if _TAU21SF_PATH not in _tau21sf_cache:
        cset = correctionlib.CorrectionSet.from_file(_TAU21SF_PATH)
        _tau21sf_cache[_TAU21SF_PATH] = cset["fj_tau21_SF"]
    return _tau21sf_cache[_TAU21SF_PATH]


def _compute_tau21_sf(candidate_boost_notau21, gen_parts, isMC):
    """Evaluate tau21 SF for the leading fatjet (no tau21 cut), all syst variations.

    fjtype = 'matched' if any gen W (|pdgId|=24) OR Z (|pdgId|=23) is within dR<0.8,
    else 'unmatched'.  Events without a fatjet receive SF=1.
    Returns dict: {syst_key: np.ndarray(n_ev)} for all keys in _TAU21_SYST_KEYS.
    """
    n_ev = len(candidate_boost_notau21)
    ones = np.ones(n_ev, dtype=np.float64)

    if not isMC:
        return {k: ones.copy() for k in _TAU21_SYST_KEYS}

    corr = _get_tau21sf()

    gen_V = gen_parts[(np.abs(gen_parts.pdgId) == 24) | (np.abs(gen_parts.pdgId) == 23)]
    dR_fj_v = candidate_boost_notau21.metric_table(gen_V)
    is_matched_coll = ak.sum(dR_fj_v < 0.8, axis=2) > 0
    is_matched = ak.to_numpy(ak.fill_none(ak.firsts(is_matched_coll), False))

    lead_fj = ak.firsts(candidate_boost_notau21)
    has_fj = ak.to_numpy(~ak.is_none(lead_fj))
    t1 = ak.to_numpy(ak.fill_none(lead_fj.tau1, np.nan))
    t2 = ak.to_numpy(ak.fill_none(lead_fj.tau2, np.nan))
    fj_tau21 = np.where((t1 > 0) & np.isfinite(t1), t2 / t1, np.nan)

    fjtype_arr = np.where(is_matched, "matched", "unmatched")
    valid = has_fj & np.isfinite(fj_tau21)
    tau21_cl = np.clip(np.where(valid, fj_tau21, 0.0), 0.0, 0.9999)

    result = {}
    for syst_key in _TAU21_SYST_KEYS:
        sf = ones.copy()
        for ft in ("matched", "unmatched"):
            mask = valid & (fjtype_arr == ft)
            if not mask.any():
                continue
            sf[mask] = corr.evaluate(syst_key, ft, tau21_cl[mask])
        result[syst_key] = sf

    return result


# ── Weight classes (from the 2022 v10 config) ──────────────────────────

class QvGSFWeight(WeightWrapper):
    """Per-event QvG scale factor: product over leading 4 jets of btagPNetQvG_SF.
    Variations: stat (correctionlib), isr, fsr, jes, jer, pu (from diag.npz),
    each split into |eta|<2.5 and |eta|>=2.5 jet parts, e.g. stat_etalt2p5, stat_etage2p5.
    """
    name = "sf_qvg"
    has_variations = True

    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)
        self._variations = list(_QVG_VARIATIONS)

    def compute(self, events, size, shape_variation):
        if shape_variation == "nominal":
            return WeightDataMultiVariation(
                name=self.name,
                nominal=ak.to_numpy(events.sf_qvg_central),
                variations=list(_QVG_VARIATIONS),
                up=[ak.to_numpy(events[f"sf_qvg_{v}_up"]) for v in _QVG_VARIATIONS],
                down=[ak.to_numpy(events[f"sf_qvg_{v}_dn"]) for v in _QVG_VARIATIONS],
            )
        else:
            return WeightData(name=self.name, nominal=ak.to_numpy(events.sf_qvg_central))


class WvsQCDSFWeight(WeightWrapper):
    """Per-event WvsQCD scale factor.
    Variations:     "stat_up", "stat_down",
    "pileup_up", "pileup_down",
    "sf_btag_up", "sf_btag_down",
    "sf_partonshower_isr_up", "sf_partonshower_isr_down",
    "sf_partonshower_fsr_up", "sf_partonshower_fsr_down",
    "JES_up", "JES_down",
    "JER_up", "JER_down".
    """
    name = "sf_wvsqcd"
    has_variations = True

    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)
        self._variations = ["stat","pu", "btag", "isr", "fsr", "jes", "jer"]

    def compute(self, events, size, shape_variation):
        if shape_variation == "nominal":
            return WeightDataMultiVariation(
                name=self.name,
                nominal=ak.to_numpy(events.sf_wvsqcd_nominal),
                variations=["stat","pu", "btag", "isr", "fsr", "jes", "jer"],
                up=[ak.to_numpy(events.sf_wvsqcd_stat_up),
                    ak.to_numpy(events.sf_wvsqcd_pileup_up),
                    ak.to_numpy(events.sf_wvsqcd_sf_btag_up),
                    ak.to_numpy(events.sf_wvsqcd_sf_partonshower_isr_up),
                    ak.to_numpy(events.sf_wvsqcd_sf_partonshower_fsr_up),
                    ak.to_numpy(events.sf_wvsqcd_JES_up),
                    ak.to_numpy(events.sf_wvsqcd_JER_up)
                    ],
                down=[ak.to_numpy(events.sf_wvsqcd_stat_down),
                    ak.to_numpy(events.sf_wvsqcd_pileup_down),
                    ak.to_numpy(events.sf_wvsqcd_sf_btag_down),
                    ak.to_numpy(events.sf_wvsqcd_sf_partonshower_isr_down),
                    ak.to_numpy(events.sf_wvsqcd_sf_partonshower_fsr_down),
                    ak.to_numpy(events.sf_wvsqcd_JES_down),
                    ak.to_numpy(events.sf_wvsqcd_JER_down)
                    ],
            )
        else:
            return WeightData(name=self.name, nominal=ak.to_numpy(events.sf_wvsqcd_nominal))

class Tau21SFWeight(WeightWrapper):
    """Per-event tau21 scale factor applied to the leading fatjet (no tau21 cut).
    Variations: "stat_up", "stat_down",
    "pileup_up", "pileup_down",
    "sf_btag_up", "sf_btag_down",
    "sf_partonshower_isr_up", "sf_partonshower_isr_down",
    "sf_partonshower_fsr_up", "sf_partonshower_fsr_down",
    "JES_up", "JES_down",
    "JER_up", "JER_down".
    """
    name = "sf_tau21"
    has_variations = True

    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)
        self._variations = ["stat", "pu", "btag", "isr", "fsr", "jes", "jer"]

    def compute(self, events, size, shape_variation):
        if shape_variation == "nominal":
            return WeightDataMultiVariation(
                name=self.name,
                nominal=ak.to_numpy(events.sf_tau21_nominal),
                variations=["stat", "pu", "btag", "isr", "fsr", "jes", "jer"],
                up=[ak.to_numpy(events.sf_tau21_stat_up),
                    ak.to_numpy(events.sf_tau21_pileup_up),
                    ak.to_numpy(events.sf_tau21_sf_btag_up),
                    ak.to_numpy(events.sf_tau21_sf_partonshower_isr_up),
                    ak.to_numpy(events.sf_tau21_sf_partonshower_fsr_up),
                    ak.to_numpy(events.sf_tau21_JES_up),
                    ak.to_numpy(events.sf_tau21_JER_up),
                    ],
                down=[ak.to_numpy(events.sf_tau21_stat_down),
                    ak.to_numpy(events.sf_tau21_pileup_down),
                    ak.to_numpy(events.sf_tau21_sf_btag_down),
                    ak.to_numpy(events.sf_tau21_sf_partonshower_isr_down),
                    ak.to_numpy(events.sf_tau21_sf_partonshower_fsr_down),
                    ak.to_numpy(events.sf_tau21_JES_down),
                    ak.to_numpy(events.sf_tau21_JER_down),
                    ],
            )
        else:
            return WeightData(name=self.name, nominal=ak.to_numpy(events.sf_tau21_nominal))
