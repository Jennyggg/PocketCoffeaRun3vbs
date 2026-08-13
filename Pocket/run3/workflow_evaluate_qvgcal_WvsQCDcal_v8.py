# workflow.py
import awkward as ak
import numpy as np
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.objects import lepton_selection, jet_selection, btagging, soft_lepton_selection
from types import SimpleNamespace
import vector
import math
import xgboost as xgb
vector.register_awkward()

import correctionlib
import os as _os

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

# Syst variation keys (must match diag.npz)
_QVG_SYST_KEYS = ["isr_up", "isr_dn", "fsr_up", "fsr_dn",
                   "jes_up", "jes_dn", "jer_up", "jer_dn",
                   "pu_up",  "pu_dn"]


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
    Returns dict: {"central", "stat_up", "stat_dn", "isr_up", ...} each of length n_events.
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

    result = {
        "central":  _prod(sf_c),
        "stat_up":  _prod(sf_up),
        "stat_dn":  _prod(sf_dn),
    }

    # Syst variations from diag.npz (array lookup)
    for vk, (sf_q_arr, sf_g_arr) in _get_diag_arrays().items():
        sf_var = _eval_sf_flat(sf_q_arr, sf_g_arr,
                               pt_flat, abs_eta, score_flat, flav_abs, valid)
        result[vk] = _prod(sf_var)

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



METDef = "DeepMETResolutionTune"

def veto_jer_forward_unmatched(jets,variations):
    """
    Run 3 AK4 Puppi jets:
    Disable JER smearing for unmatched jets in 2.5 < |eta| < 3.0
    """

    abs_eta = np.abs(jets.eta)

    veto_pt = (
        (abs_eta > 2.5)
        & (abs_eta < 3.0)
        & (jets.genJetIdx < 0)
        & (jets.pt_sf_jer != 0)
    )

    veto_mass = (
        (abs_eta > 2.5)
        & (abs_eta < 3.0)
        & (jets.genJetIdx < 0)
        & (jets.mass_sf_jer != 0)
    )
    # Nominal
    jets["pt"] = ak.where(veto_pt, jets.pt/jets.pt_sf_jer, jets.pt)
    jets["mass"] = ak.where(veto_mass, jets.mass/jets.mass_sf_jer, jets.mass)
    # Systematics (important!)
    if "pt_JER_up" in jets.fields:
        jets["pt_JER_up"] = ak.where(veto_pt, jets.pt, jets.pt_JER_up)
    if "pt_jer_down" in jets.fields:
        jets["pt_JER_down"] = ak.where(veto_pt, jets.pt, jets.pt_JER_down)
    if "mass_JER_up" in jets.fields:
        jets["mass_JER_up"] = ak.where(veto_mass, jets.mass, jets.mass_JER_up)
    if "mass_JER_down" in jets.fields:
        jets["mass_JER_down"] = ak.where(veto_mass, jets.mass, jets.mass_JER_down)
    for var in jets.fields:
        if "pt_JES" in var and ("up" in var or "down" in var):
            jets[var] = ak.where(veto_pt, jets[var] / jets.pt_sf_jer, jets[var])
        if "mass_JES" in var and ("up" in var or "down" in var):
            jets[var] = ak.where(veto_mass, jets[var] / jets.mass_sf_jer, jets[var])

    return jets



def compute_MT(lep, met):
    return np.sqrt( #CHANGED mT DEFINITION TO USE PUPPIMET
            2.0 * lep.pt * met.pt * (1.0 - np.cos(lep.delta_phi(met)))
        )

def delta_phi(phi1, phi2):
    dphi = abs(phi1 - phi2)
    return np.minimum(dphi, 2*np.pi - dphi)

def custom_dR(j1,j2):
    dphi=delta_phi(j1.phi,j2.phi)
    deta= (j1.eta - j2.eta)
    dR = np.sqrt(dphi**2 + deta **2)
    return dR
    
class VBSSemileptonicProcessor(BaseProcessorABC):
    """
        - Build LeptonGood and JetGood (lepton-clean)
        - Identifies VBS tagging jets as the pair with the highest mjj
        - Reconstructs the hadronic W with two non-VBS jets that minimize |m-80.4|
        - Calculates auxiliary variables for histograms (mt, pt/eta, dR, etc.)
    """

    # QvG weight modifier names (PocketCoffea convention: {name}_{var}Up/Down)
    _QVG_MODIFIERS = [
        f"sf_qvg_{v}{d}"
        for v in ["stat", "isr", "fsr", "jes", "jer", "pu"]
        for d in ["Up", "Down"]
    ]

    # Scale / PDF / parton-shower modifier names
    _SCALE_MODIFIERS = (
        # LHE renorm/fact scale: 2 variations × Up/Down = 4 modifiers
        [f"LHEScaleWeight_{v}{d}"
         for v in ["renorm_scale", "fact_scale"]
         for d in ["Up", "Down"]]
        +
        # LHE PDF eigenvectors (pdf_1…pdf_100) + alpha_S: 101 variations × Up/Down = 202 modifiers
        [f"LHEPdfWeight_{v}{d}"
         for v in [f"pdf_{i}" for i in range(1, 101)] + ["alpha_S"]
         for d in ["Up", "Down"]]
        +
        # Parton-shower ISR and FSR: 2 × Up/Down = 4 modifiers
        [f"sf_partonshower_isr{d}" for d in ["Up", "Down"]]
        + [f"sf_partonshower_fsr{d}" for d in ["Up", "Down"]]
    )

    # WvsQCD fatjet tagger SF modifier names
    _WVSQCD_MODIFIERS = [
        f"sf_wvsqcd_{v}{d}"
        for v in ["stat", "pileup", "sf_btag", "sf_partonshower_isr",
                  "sf_partonshower_fsr", "JES", "JER"]
        for d in ["Up", "Down"]
    ]


    # tau21 SF modifier names — must match Tau21SFWeight._variations exactly
    _TAU21_MODIFIERS = [
        f"sf_tau21_{v}{d}"
        for v in ["stat", "pu", "btag", "isr", "fsr", "jes", "jer"]
        for d in ["Up", "Down"]
    ]

    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        # Lazy cache: populated on first apply_object_preselection call for a given year.
        # Keyed by year so the same processor instance can handle multiple years.
        self._classifiers = {}

        # sumw_noqvgcal: Σ(w_total × sf_qvg_central_inv × sf_wvsqcd_central_inv × mask)
        # per category — weight without both QvG and WvsQCD SFs.
        # C_nominal = sumw_noqvgcal[cat] / sumw[cat].
        self.output_format["sumw_noqvgcal"] = {
            cat: {} for cat in self._categories
        }
        # sumw_qvg_variations: same but with each QvG weight variation applied.
        # C_var = sumw_noqvgcal[cat] / sumw_qvg_variations[modifier][cat]
        # allows renormalizing each variation back to the pre-SF yield.
        self.output_format["sumw_qvg_variations"] = {
            mod: {cat: {} for cat in self._categories}
            for mod in self._QVG_MODIFIERS
        }
        # sumw_wvsqcd_variations: Σ(w_with_wvsqcd_variation × mask) per category.
        self.output_format["sumw_wvsqcd_variations"] = {
            mod: {cat: {} for cat in self._categories}
            for mod in self._WVSQCD_MODIFIERS
        }

        # sumw_noqvgcal_nowvsqcdcal / nozvsqcdcal: yields with both QvG and tagger SFs removed.
        self.output_format["sumw_noqvgcal"] = {
            cat: {} for cat in self._categories
        }

        # sumw_tau21_variations: Σ(w_with_tau21_variation × mask) per category.
        self.output_format["sumw_tau21_variations"] = {
            mod: {cat: {} for cat in self._categories}
            for mod in self._TAU21_MODIFIERS
        }
        # sumw_scale_variations: Σ(w_with_variation × mask) for LHEScaleWeight,
        # LHEPdfWeight, and parton-shower ISR/FSR variations.
        self.output_format["sumw_scale_variations"] = {
            mod: {cat: {} for cat in self._categories}
            for mod in self._SCALE_MODIFIERS
        }

    def fill_histograms_extra(self, variation):
        """Save per-category sumw_noqvgcal, sumw_qvg_variations, sumw_wvsqcd_variations.
        Only runs for the nominal shape variation; SF weight variations are only
        meaningful when jet kinematics are at nominal.
        """
        if not self._isMC or variation != "nominal":
            return
        inv_qvg    = ak.to_numpy(self.events.sf_qvg_central_inv)
        # Get the set of installed QvG modifiers (may be a subset if diag.npz absent)
        try:
            installed_qvg_mods = set(
                self.weights_manager.get_available_modifiers_byweight("sf_qvg")
            )
        except Exception:
            installed_qvg_mods = set()

        try:
            installed_wvsqcd_mods = set(
                self.weights_manager.get_available_modifiers_byweight("sf_wvsqcd")
            )
        except Exception:
            installed_wvsqcd_mods = set()


        try:
            installed_tau21_mods = set(
                self.weights_manager.get_available_modifiers_byweight("sf_tau21")
            )
        except Exception:
            installed_tau21_mods = set()

        for category, cat_mask in self._categories.get_masks():
            mask_on_events = cat_mask
            if hasattr(mask_on_events, "ndim") and mask_on_events.ndim > 1:
                mask_on_events = ak.any(mask_on_events, axis=1)
            w_full = self.weights_manager.get_weight(category)
            w_no_qvg  = w_full * inv_qvg
            sumw_no_qvg = float(ak.sum(w_no_qvg * mask_on_events))
            (self.output["sumw_noqvgcal"][category]
             .setdefault(self._dataset, {})
             .setdefault(self._sample, {}))["nominal"] = sumw_no_qvg

            # sumw_qvg_variations: weight with each QvG variation applied
            for mod in self._QVG_MODIFIERS:
                if mod not in installed_qvg_mods:
                    continue
                try:
                    w_var = self.weights_manager.get_weight(category, modifier=mod)
                except ValueError:
                    continue
                sumw_var = float(ak.sum(w_var * mask_on_events))
                (self.output["sumw_qvg_variations"][mod][category]
                 .setdefault(self._dataset, {})
                 .setdefault(self._sample, {}))["nominal"] = sumw_var

            # sumw_wvsqcd_variations: weight with each WvsQCD variation applied
            for mod in self._WVSQCD_MODIFIERS:
                if mod not in installed_wvsqcd_mods:
                    continue
                try:
                    w_var = self.weights_manager.get_weight(category, modifier=mod)
                except ValueError:
                    continue
                sumw_var = float(ak.sum(w_var * mask_on_events))
                (self.output["sumw_wvsqcd_variations"][mod][category]
                 .setdefault(self._dataset, {})
                 .setdefault(self._sample, {}))["nominal"] = sumw_var


            # sumw_tau21_variations: weight with each tau21 SF variation applied
            for mod in self._TAU21_MODIFIERS:
                if mod not in installed_tau21_mods:
                    continue
                try:
                    w_var = self.weights_manager.get_weight(category, modifier=mod)
                except ValueError:
                    continue
                sumw_var = float(ak.sum(w_var * mask_on_events))
                (self.output["sumw_tau21_variations"][mod][category]
                 .setdefault(self._dataset, {})
                 .setdefault(self._sample, {}))["nominal"] = sumw_var

            # sumw_scale_variations: weight with each LHEScale/LHEPdf/PS variation
            for mod in self._SCALE_MODIFIERS:
                try:
                    w_var = self.weights_manager.get_weight(category, modifier=mod)
                except (ValueError, KeyError):
                    continue
                sumw_var = float(ak.sum(w_var * mask_on_events))
                (self.output["sumw_scale_variations"][mod][category]
                 .setdefault(self._dataset, {})
                 .setdefault(self._sample, {}))["nominal"] = sumw_var

    def postprocess(self, accumulator):
        """Extend base postprocess to rescale sumw_noqvgcal and sumw_qvg_variations
        by 1/sum_genweights, matching how sumw is rescaled, so that
        C = sumw_noqvgcal[cat] / sumw_qvg_variations[mod][cat] is dimensionless.
        """
        accumulator = super().postprocess(accumulator)
        if not self.cfg.do_postprocessing or self.workflow_options.get(
            "donotscale_sumgenweights", False
        ):
            return accumulator

        def _rescale_sumw_dict(sumw_dict):
            """Apply 1/sum_genweights rescaling to a sumw-like dict."""
            for cat, catdata in sumw_dict.items():
                for dataset, dataset_data in catdata.items():
                    sample_key = list(dataset_data.keys())[0]
                    if not self.cfg.samples_metadata[sample_key]["isMC"]:
                        continue
                    wei = self.cfg.weights_config[sample_key]["inclusive"]
                    if "genWeight" in wei and "signOf_genWeight" not in wei:
                        sumgenw_dict = accumulator["sum_genweights"]
                    elif "signOf_genWeight" in wei and "genWeight" not in wei:
                        sumgenw_dict = accumulator["sum_signOf_genweights"]
                    else:
                        continue
                    if dataset not in sumgenw_dict:
                        continue
                    scaling = 1.0 / sumgenw_dict[dataset]
                    for sample in dataset_data.keys():
                        if "nominal" in dataset_data[sample]:
                            dataset_data[sample]["nominal"] *= scaling

        _rescale_sumw_dict(accumulator["sumw_noqvgcal"])
        for mod_data in accumulator["sumw_qvg_variations"].values():
            _rescale_sumw_dict(mod_data)
        for mod_data in accumulator["sumw_wvsqcd_variations"].values():
            _rescale_sumw_dict(mod_data)
        for mod_data in accumulator["sumw_scale_variations"].values():
            _rescale_sumw_dict(mod_data)
        return accumulator

    # 1) object-level preselection
    def apply_object_preselection(self, variation):
        ev = self.events
        def _tau21(fj):
            t1 = ak.fill_none(getattr(fj, "tau1", None), np.nan)
            t2 = ak.fill_none(getattr(fj, "tau2", None), np.nan)
            return ak.where((t1 > 0) & np.isfinite(t1), t2 / t1, np.nan)
        ev["Electron", "etaSC"] = ev.Electron.eta + ev.Electron.deltaEtaSC

        # Good Leptons

        tight_criteria = SimpleNamespace(
            object_preselection = {
                "Muon": {
                    "pt": 26.0,
                    "eta": 2.4,
                    "id": "tightId",
                    "iso": 0.15,
                }
            }
        )

        ev["MuonGood_0"]     = lepton_selection(ev, "Muon", tight_criteria)
        mu = ev.MuonGood_0
        mask_muon_ip = (
            (np.abs(mu.dxy) < 0.2) & (np.abs(mu.dz) < 0.5)
        )

        ev["MuonGood"] = mu[mask_muon_ip]

        mask_good_ele_kin = ( ev.Electron.pt > 35 ) & ( np.abs(ev.Electron.eta) < 2.4) & ( ev.Electron.cutBased >= 3 )

        ev["ElectronGood_0"] = ev.Electron[mask_good_ele_kin]
        ele = ev.ElectronGood_0
        mask_ele_ip = (
            (np.abs(ele.dxy) < 0.05) & (np.abs(ele.eta) < 1.5) & (np.abs(ele.dz) < 0.1)
        ) | (
            (np.abs(ele.dxy) < 0.1) & (np.abs(ele.eta) >= 1.5) & (np.abs(ele.eta) < 2.4) & (np.abs(ele.dz) < 0.2)
        )

        ev["ElectronGood"] = ele[mask_ele_ip]
        veto_criteria = SimpleNamespace(
            object_preselection = {
                "Muon": {
                    "pt": 10.0,
                    "eta": 2.4,
                    "id": "looseId",
                    "iso": 500.0,
                }
            }
        )


        loose_criteria = SimpleNamespace(
            object_preselection = {
                "Muon": {
                    "pt": 26.0,
                    "eta": 2.4,
                    "id": "looseId",
                    "iso": 500.0,
                }
            }
        )



        # Good Leptons
        ev["MuonVeto"]     = lepton_selection(ev, "Muon", veto_criteria)

        ev["MuonLoose"]     = lepton_selection(ev, "Muon", loose_criteria)

        mask_ele_loose = (ev.Electron.pt > 35) & (np.abs(ev.Electron.eta) < 2.4) & (ev.Electron.cutBased >=1 )
        ev["ElectronLoose"] = ev.Electron[mask_ele_loose]
        mask_ele_veto = (ev.Electron.pt > 10) & (np.abs(ev.Electron.eta) < 2.4) & (ev.Electron.cutBased >=1 )
        ev["ElectronVeto"]     = ev.Electron[mask_ele_veto]

        # Leptons (mu+e) and ordered in pt
        leptons = ak.with_name(
            ak.concatenate([ev.MuonGood, ev.ElectronGood], axis=1),
            "PtEtaPhiMCandidate",
        )

        loose_lep = ak.with_name(
            ak.concatenate([ev.MuonLoose, ev.ElectronLoose], axis=1),
            "PtEtaPhiMCandidate",
        )

        ev["LeptonLoose"] = loose_lep[ak.argsort(loose_lep.pt, ascending=False)]
        ev["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        lead_lep = ak.firsts(ev.LeptonGood)
        lead_lep_loose = ak.firsts(ev.LeptonLoose)
        ev["lead_lep"] = lead_lep
        ev["JetGood_0"], _ = jet_selection(ev, "Jet", self.params, self._year,"LeptonLoose") #MAYBE THIS SHOULD BE LOOSE LEPTON
        if self._isMC:
            for jet_type, jet_coll_name in self.params.jets_calibration.collection[self._year].items():
                if jet_coll_name == "Jet":
                    JEC_type = jet_type
            ev.JetGood_1 = veto_jer_forward_unmatched(ev.JetGood_0,self.params.jets_calibration.variations[JEC_type][self._year])
        else:
            ev.JetGood_1 = ev.JetGood_0
        mask_jet_cleaning = (ev.JetGood_1.pt>50) | (abs(ev.JetGood_1.eta)<2.5)
        ev["JetGood"] = ev.JetGood_1[mask_jet_cleaning]
        ev["JetGoodCentral"] = ev.JetGood[abs(ev.JetGood.eta)<2.4]

        #TODO: jet_selection_nanoaodv12 only used for 2022, check other versions for other years.

        ev["FatJetGood"] = ev.FatJet[abs(ev.FatJet.eta)<2.4]
        ev["FatJetGood", "idx"] = ak.local_index(ev.FatJetGood, axis=1)
        dR_fatjets_lep = ev.FatJetGood.metric_table(ev.LeptonGood)
        mask_lepjet_cleaning = ak.prod(dR_fatjets_lep > 0.8, axis=2) == 1
        ev["FatJetGood"] = ev.FatJetGood[mask_lepjet_cleaning]
        ev["FatJetGood", "idx"] = ak.local_index(ev.FatJetGood, axis=1)

        ev["candidate_boost_notau21"] = ev.FatJetGood[(ev.FatJetGood.msoftdrop < 250) & (ev.FatJetGood.msoftdrop > 40) & (ev.FatJetGood.pt > 200)]

        ev["candidate_boost"] = ev.FatJetGood[(_tau21(ev.FatJetGood) < 0.45) & (ev.FatJetGood.msoftdrop < 250) & (ev.FatJetGood.msoftdrop > 40) & (ev.FatJetGood.pt > 200)]
        dR_jets_jet = ev.JetGood.metric_table(ev.candidate_boost)
        mask_jet_cleaning = ak.prod(dR_jets_jet > 0.8, axis=2) == 1
        separation = ak.fill_none(ev.JetGood.metric_table(ev.candidate_boost), np.nan)

        ev["JetGood"] = ev.JetGood[mask_jet_cleaning]
        ev["JetGood", "idx"] = ak.local_index(ev.JetGood, axis=1)

        # b-tagging
        ev["BJetTight"] = btagging(
            ev.JetGood[np.abs(ev.JetGood.eta) < 2.5],
            self.params.btagging.working_point[self._year],
            wp="T",
        )
        ev["BJetLoose"] = btagging(
            ev.JetGood[np.abs(ev.JetGood.eta) < 2.5],
            self.params.btagging.working_point[self._year],
            wp=self.params.object_preselection.Jet.btag.wp,
        )
        # ------------- VBS tagging jets -------------
        has4j = ak.num(ev.JetGood) >= 4
        has2j = (ak.num(ev.JetGood) >= 2)
        jj = ak.combinations(ev.JetGood, 2, fields=["jet1", "jet2"])
        jj["mass"] = (jj.jet1 + jj.jet2).mass

        idx_vbs = ak.argmax(jj.mass, axis=1, keepdims=True)

        ev["vbsjets"] = ak.mask(jj[idx_vbs], has2j)
        ev["vbsjet1"] = ak.firsts(ev.vbsjets.jet1)
        ev["vbsjet2"] = ak.firsts(ev.vbsjets.jet2)
        ev["vbsjets", "delta_eta"] = np.abs(ev.vbsjet1.eta - ev.vbsjet2.eta) 
        ev["vbsjets", "delta_phi"] = delta_phi(ev.vbsjet1.phi, ev.vbsjet2.phi)
        # ------ Boosted jet -------------

        ev["candidate_boost1"] = ak.firsts(ev.candidate_boost)
        ev["candidate_boost1" ,"tau21"] = _tau21(ev.candidate_boost1)

        # ------------- W hadronic (resolved) -------------
        vbs1 = ak.fill_none(getattr(ev.vbsjet1, "idx", None), -1)
        vbs2 = ak.fill_none(getattr(ev.vbsjet2, "idx", None), -1)

        nonvbs_mask = (ev.JetGood.idx != vbs1) & (ev.JetGood.idx != vbs2)
        ev["CentralJets"] = ev.JetGood[nonvbs_mask]

        ev['CentralJetsGood']= ev.CentralJets[np.abs(ev.CentralJets.eta) < 2.4]



        pairs_w = ak.combinations(ev.CentralJetsGood, 2, fields=["jet1", "jet2"])
        pairs_w["mass"] = (pairs_w.jet1 + pairs_w.jet2).mass

        target_mw = 85
        best_w_idx = ak.argmin(np.abs(pairs_w.mass - target_mw), axis=1, keepdims=True)

        ev["w_had_jets"] = ak.mask(pairs_w[best_w_idx], has4j)
        ev["w_had_jets", "mass"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).mass
        ev["w_had_jets", "pt"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).pt
        ev["w_had_jets", "eta"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).eta
        ev["w_had_jets", "phi"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).phi
        ev["w_had_jets", "delta_eta"] = np.abs(ev.w_had_jets.jet1.eta - ev.w_had_jets.jet2.eta) 
        ev["w_had_jets", "delta_phi"] = delta_phi(ev.w_had_jets.jet1.phi, ev.w_had_jets.jet2.phi)
        ev["w_had_jets", "dR"] = np.sqrt(ev.w_had_jets.delta_eta**2 + ev.w_had_jets.delta_phi**2)
        ev["w_had_jet1"] = ak.firsts(ev.w_had_jets.jet1)
        ev["w_had_jet2"] = ak.firsts(ev.w_had_jets.jet2)


        # ------------- W Leptonic -------------
        ev["mt_w_leptonic"] = compute_MT(lead_lep, ev.PuppiMET)
        ev["mt_w_leptonic_deepMET_resolutiontune"] = compute_MT(lead_lep, ev.DeepMETResolutionTune)

        ev["mt_w_leptonic_deepMET_resolutiontune_loose"] = compute_MT(lead_lep_loose, ev.DeepMETResolutionTune)

        ev["mt_w_leptonic_deepMET_responsetune"] = compute_MT(lead_lep, ev.DeepMETResponseTune)

        ev["mt_w_leptonic_deepMET_responsetune_loose"] = compute_MT(lead_lep_loose, ev.DeepMETResponseTune)

        # Zeppenfeld variables (basically eta significance of V decay to VBS jet)
        def zeppenfeld(target, vbs_jet1, vbs_jet2):
            mask_valid = (
                ~ak.is_none(target) &
                ~ak.is_none(vbs_jet1) &
                ~ak.is_none(vbs_jet2)
            )
            target_eta = ak.where(mask_valid, target.eta, np.nan)
            vbs1_eta = ak.where(mask_valid, vbs_jet1.eta, np.nan)
            vbs2_eta = ak.where(mask_valid, vbs_jet2.eta, np.nan)
            numerator = target_eta - (vbs1_eta + vbs2_eta)/2
            denominator = np.abs(vbs1_eta - vbs2_eta)
            zep = numerator / denominator
            return zep

        ev['z_lep'] = ak.fill_none(zeppenfeld(lead_lep, ev.vbsjet1,ev.vbsjet2),np.nan)
        ev['z_fat'] = ak.fill_none(zeppenfeld(ev.candidate_boost1, ev.vbsjet1,ev.vbsjet2),np.nan)
        
        nu = ak.zip(
                {
                    "pt": ev[METDef].pt,
                    "eta": ak.zeros_like(ev[METDef].pt),
                    "phi": ev[METDef].phi,
                    "mass": ak.zeros_like(ev[METDef].pt),
                    "charge": ak.zeros_like(ev[METDef].pt),
                    },
                with_name="PtEtaPhiMCandidate",
                )
        w_lep = nu + lead_lep
        whad = ev.w_had_jet1 + ev.w_had_jet2
        def solve_neutrino_pz(lep, nu):
            m_w = 80.36
            A = m_w**2 - lep.mass**2
            delta_phi = lep.phi - nu.phi
            C = 0.5 * A + lep.pt * nu.pt * np.cos(delta_phi)
            D = lep.pz

            a = (lep.mass**2 + lep.pt**2 + lep.pz**2) - D**2
            b = -2 * C * D
            c = (lep.mass**2 + lep.pt**2 + lep.pz**2) * nu.pt**2 - C**2

            discriminant = b**2 - 4 * a * c

            a_zero_mask = abs(a) < 1e-12
            b_nonzero_mask = abs(b) > 1e-12
            disc_neg_mask = discriminant < 0

            pz_a0 = ak.where(b_nonzero_mask, -c / b, 0.0)
            pz_no_real = -b / (2 * a)
            sqrt_disc = ak.where(disc_neg_mask, 0.0, np.sqrt(discriminant))

            pz1 = (-b + sqrt_disc) / (2 * a)
            pz2 = (-b - sqrt_disc) / (2 * a)

            best_pz = ak.where(abs(pz1) < abs(pz2), pz1, pz2)

            result = ak.where(
                a_zero_mask,
                pz_a0,
                ak.where(disc_neg_mask, pz_no_real, best_pz)
            )

            return result



        def centrality(w_lep_eta,v_had,vbs1,vbs2):
            eta_plus= np.maximum(vbs1.eta,vbs2.eta) - np.maximum(w_lep_eta,v_had.eta)
            eta_minus=np.minimum(v_had.eta,w_lep_eta) - np.minimum(vbs1.eta,vbs2.eta)
            C = np.minimum(eta_plus, eta_minus)
            return C

        ev['neutrino_pz'] = ak.fill_none(solve_neutrino_pz(lead_lep, ev[METDef]),np.nan)
        ev['neutrino_eta'] = ak.fill_none(np.arcsinh(ev.neutrino_pz / ev[METDef].pt),np.nan)
        ev['wleptonic_eta'] = ak.fill_none(np.arcsinh((ev.neutrino_pz+lead_lep.pz)/(w_lep.pt)),np.nan)
        ev['wleptonic_pt'] = ak.fill_none(w_lep.pt,np.nan)
        ev['wleptonic_phi'] = ak.fill_none(w_lep.phi,np.nan)
        ev['centrality_resolved'] = ak.fill_none(centrality(ev.wleptonic_eta, whad,ev.vbsjet1,ev.vbsjet2),np.nan)
        ev['centrality_boosted'] = ak.fill_none(centrality(ev.wleptonic_eta,ev.candidate_boost1,ev.vbsjet1,ev.vbsjet2),np.nan)

        if self._isMC:
            dress_lep = ak.firsts(ev.GenDressedLepton)
            gen_met = ev.GenMET
            ev["gen_w_pt_dressed"] = (dress_lep + gen_met).pt
            w_pt_dressed = ak.firsts(ev.gen_w_pt_dressed, axis=-1)
            w_pt_direct = ak.firsts(ev.GenPart[abs(ev.GenPart.pdgId) == 24].pt, axis=-1)
            ev["gen_w_pt_by_pdg"] = ak.fill_none(w_pt_direct, w_pt_dressed)

        jets_sorted = ev.JetGood[ak.argsort(ev.JetGood.pt, ascending=False)]
        lepton_sorted = ev.LeptonGood[ak.argsort(ev.LeptonGood.pt, ascending=False)]
        n_jets = max(4, int(np.max(ak.num(jets_sorted, axis=1))))
        for i in range(n_jets):
            ev[f'jet{i+1}'] = ak.firsts(jets_sorted[:, i:i+1])
        n_leptons = max(1, int(np.max(ak.num(lepton_sorted, axis=1))))
        for i in range(n_leptons):  # at least 1, so ev.lepton1 always exists
            ev[f'lepton{i+1}'] = ak.firsts(lepton_sorted[:, i:i+1])

        names=['jet1','jet2','jet3','jet4', 'lepton1', METDef,'candidate_boost1']
        objects=[ev.jet1, ev.jet2, ev.jet3, ev.jet4, ev.lepton1, nu ,ev.candidate_boost1]


        ev["deta"] = {}
        ev["dphi"] ={}
        ev["dR"] ={}
        ev["mass"] ={}
        for i in range(len(names)):
            a = objects[i]
            for j in range(i+1, len(names)):
                b = objects[j]
                dphi = delta_phi(a.phi, b.phi)
                deta = np.abs(a.eta - b.eta)
                dR   = np.sqrt(dphi**2 + deta**2)
                mass = (a+b).mass


                # Store them
                ev["deta", f"{names[i]}_{names[j]}"] = deta
                ev["dphi", f"{names[i]}_{names[j]}"] = dphi
                ev["dR",   f"{names[i]}_{names[j]}"] = dR
                ev["mass", f"{names[i]}_{names[j]}"] = mass
        # ── QvG scale factor (MC only, leading 4 jets) ───────────────────
        _ones = np.ones(len(ev))
        if self._isMC:
            _qvg_sfs = _compute_qvg_sf_lead4(ev.JetGood)
            for _k, _v in _qvg_sfs.items():
                ev[f"sf_qvg_{_k}"] = _v
            # Fill data-like ones for any syst key absent in diag (safety)
            for _k in _QVG_SYST_KEYS:
                if f"sf_qvg_{_k}" not in ev.fields:
                    ev[f"sf_qvg_{_k}"] = _ones
            # Inverse SF: weight × inv = weight_without_sf_qvg (for _noqvgcal histograms)
            _sf_c = _qvg_sfs["central"]
            ev["sf_qvg_central_inv"] = np.where(np.abs(_sf_c) > 1e-6, 1.0 / _sf_c, _ones)
        else:
            for _k in ["central", "stat_up", "stat_dn"] + _QVG_SYST_KEYS:
                ev[f"sf_qvg_{_k}"] = _ones
            ev["sf_qvg_central_inv"] = _ones

        # ── WvsQCD scale factor (MC only, leading fatjet) ──────────────────
        if self._isMC:
            _wvsqcd_sfs = _compute_wvsqcd_sf(ev.candidate_boost, ev.GenPart, True)
        else:
            _wvsqcd_sfs = {k: _ones.copy() for k in _WVSQCD_SYST_KEYS}

        for _k, _v in _wvsqcd_sfs.items():
            ev[f"sf_wvsqcd_{_k}"] = _v

        _sf_wvsqcd_c = _wvsqcd_sfs["nominal"]
        ev["sf_wvsqcd_central_inv"] = np.where(
            np.abs(_sf_wvsqcd_c) > 1e-6, 1.0 / _sf_wvsqcd_c, _ones
        )

        ev["sf_qvg_wvsqcd_central_inv"] = ev["sf_qvg_central_inv"] * ev["sf_wvsqcd_central_inv"]

        # ── tau21 scale factor (MC only, leading fatjet without tau21 cut) ────
        if self._isMC:
            _tau21_sfs = _compute_tau21_sf(ev.candidate_boost_notau21, ev.GenPart, True)
        else:
            _tau21_sfs = {k: _ones.copy() for k in _TAU21_SYST_KEYS}

        for _k, _v in _tau21_sfs.items():
            ev[f"sf_tau21_{_k}"] = _v
        _sf_tau21_c = _tau21_sfs["nominal"]
        ev["sf_tau21_central_inv"] = np.where(
            np.abs(_sf_tau21_c) > 1e-6, 1.0 / _sf_tau21_c, _ones
        )



        jets_sorted = ev.JetGood[ak.argsort(ev.JetGood.pt, ascending=False)]
        lepton_sorted = ev.LeptonGood[ak.argsort(ev.LeptonGood.pt, ascending=False)]
        n_jets = max(4, int(np.max(ak.num(jets_sorted, axis=1))))
        for i in range(n_jets):
            ev[f'jet{i+1}'] = ak.firsts(jets_sorted[:, i:i+1])
        n_leptons = max(1, int(np.max(ak.num(lepton_sorted, axis=1))))
        for i in range(n_leptons):  # at least 1, so ev.lepton1 always exists
            ev[f'lepton{i+1}'] = ak.firsts(lepton_sorted[:, i:i+1])

        names=['jet1','jet2','jet3','jet4', 'lepton1', METDef,'candidate_boost1']
        objects=[ev.jet1, ev.jet2, ev.jet3, ev.jet4, ev.lepton1, nu ,ev.candidate_boost1]
        ev["deta"] = {}
        ev["dphi"] ={}
        ev["dR"] ={}
        ev["mass"] ={}
        for i in range(len(names)):
            a = objects[i]
            for j in range(i+1, len(names)):
                b = objects[j]
                dphi = delta_phi(a.phi, b.phi)
                deta = np.abs(a.eta - b.eta)
                dR   = np.sqrt(dphi**2 + deta**2)
                mass = (a+b).mass


                # Store them
                ev["deta", f"{names[i]}_{names[j]}"] = deta
                ev["dphi", f"{names[i]}_{names[j]}"] = dphi
                ev["dR",   f"{names[i]}_{names[j]}"] = dR
                ev["mass", f"{names[i]}_{names[j]}"] = mass


        if hasattr(self.params, 'classifiers') and self._year not in self._classifiers:
            self._classifiers[self._year] = {}
            for region in ["boosted_WW_WZ","resolved_WW_WZ",
                           "boosted_mu", "boosted_e", "resolved_mu", "resolved_e"]:
                self._classifiers[self._year][region] = []
                for model_path in self.params.classifiers[self._year][region]:
                    model = xgb.XGBClassifier()
                    model.load_model(model_path)
                    self._classifiers[self._year][region].append(model)

        if self._year in self._classifiers:
            for region, models in self._classifiers[self._year].items():
                arrays_to_stack = []
                y_pred = []
                for imodel, model in enumerate(models):
                    if imodel == 0:
                        features = model.get_booster().feature_names
                        for name in features:
                            if name.startswith("events_n"):
                                field_name = name.replace("events_n", "")
                                val = eval(f"ak.num(ev.{field_name})")
                            elif name.startswith("events_"):
                                field_name = name.replace("events_", "")
                                val = ev[field_name]
                            elif name.startswith("w_had_jets_"):
                                field_name = name.replace("w_had_jets_", "")
                                val = eval(f"ev.w_had_jets.{field_name}")
                            elif name == "candidate_boost1_particleNetWithMass_WvsZ":
                                eps=1e-6
                                w_val = np.maximum(np.minimum(ev.candidate_boost1.particleNetWithMass_WvsQCD, 1-eps), 0)
                                z_val = np.maximum(np.minimum(ev.candidate_boost1.particleNetWithMass_ZvsQCD, 1-eps), 0)
                                num = w_val * (1 - z_val)
                                den = num + z_val * (1 - w_val)
                                val = np.where(den > eps, num / den, 0.5)
                            elif name.startswith("candidate_boost1_"):
                                field_name = name.replace("candidate_boost1_", "")
                                val = eval(f"ev.candidate_boost1.{field_name}")
                            elif "_" in name:
                                path = name.replace("_", ".", 1)
                                val = eval(f"ev.{path}")
                            else:
                                val = ev[name]
                            if val.ndim > 1:
                                val = ak.pad_none(val, 1, axis=1)[:, 0]
                            val = ak.fill_none(val, -999.0)
                            arrays_to_stack.append(ak.to_numpy(val))
                        X_test = np.column_stack(arrays_to_stack)
                    y_pred.append(model.get_booster().inplace_predict(X_test))
                ev[f"bdt_{region}"] = np.mean(np.array(y_pred), axis=0)
                

    def count_objects(self, variation):
        ev = self.events
        ev["nMuonGood"]     = ak.num(ev.MuonGood)
        ev["nElectronGood"] = ak.num(ev.ElectronGood)
        ev["nLeptonGood"]   = ev.nMuonGood + ev.nElectronGood
        ev["nJetGood"]      = ak.num(ev.JetGood)
        ev["nJetGoodCentral"]      = ak.num(ev.JetGoodCentral)
        ev["nBJetTight"]     = ak.num(ev.BJetTight)
        ev["nBJetLoose"]     = ak.num(ev.BJetLoose)
        ev["nCentralJetsGood"] = ak.num(ev.CentralJetsGood)
        ev["nFatJetGood"] = ak.num(ev.FatJetGood)
        ev["nFatJetCandidate"] = ak.num(ev.candidate_boost)

        ev["nMuonLoose"]     = ak.num(ev.MuonLoose)
        ev["nElectronLoose"] = ak.num(ev.ElectronLoose)
        ev["nMuonVeto"]     = ak.num(ev.MuonVeto)
        ev["nElectronVeto"] = ak.num(ev.ElectronVeto)
        ev["nLeptonLoose"]   = ev.nMuonLoose + ev.nElectronLoose
        ev["nLeptonVeto"]   = ev.nMuonVeto + ev.nElectronVeto
