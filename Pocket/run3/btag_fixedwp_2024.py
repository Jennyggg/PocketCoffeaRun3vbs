# btag_fixedwp_2024.py
#
# Fixed-working-point b-tagging event weight for 2024 (BTV method 1a, two WPs).
#
# BTV provides for UParTAK4B in 2024 only fixed-WP scale factors
# (UParTAK4_comb for b/c jets, UParTAK4_light for udsg jets), no shape SFs.
# The analysis cuts on two WPs of the same tagger (jets of JetGoodBtagAcc):
#   nBJetLoose == 0 (b-veto: SR, WCR) and nBJetTight > 0 (TTCR)
# so each jet falls in one of three exclusive classes, with
#   pass T          : P_MC = eT          P_data = SF_T eT
#   pass L, fail T  : P_MC = eL - eT     P_data = SF_L eL - SF_T eT
#   fail L          : P_MC = 1 - eL      P_data = 1 - SF_L eL
#   w = prod_jets P_data / P_MC
# eL, eT = MC efficiencies from derive_btag_eff_2024.py (per process group,
# flavour, |eta|, pT); SFs from the BTV correctionlib file.
#
# Variations (SFs shifted at both WPs together):
#   bc_correlated, bc_uncorrelated       -- b/c SFs, 'up/down_(un)correlated'
#   light_correlated, light_uncorrelated -- udsg SFs, 'up/down_(un)correlated'
# 'correlated' = systematic part (correlated between years), 'uncorrelated' =
# statistical part (decorrelated between years), as defined by BTV.
import numpy as np
import awkward as ak
import correctionlib
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation

# BTV 2024 file pinned to a release date (not "latest") for reproducibility
BTV_FILE = ("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/"
            "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-10-20/btagging.json.gz")
SF_PT_RANGE = (20.0, 1000.0)     # validity of the BTV SFs
SF_ABSETA_MAX = 2.5
P_MC_MIN = 1e-3                  # floor of the per-jet MC class probability

VARIATIONS = ["bc_correlated", "bc_uncorrelated", "light_correlated", "light_uncorrelated"]

_csets = {}


def _cset(path):
    if path not in _csets:
        _csets[path] = correctionlib.CorrectionSet.from_file(path)
    return _csets[path]


def group_of(sample):
    """Process group of a sample for the efficiency maps (shared with
    derive_btag_eff_2024.py)."""
    if sample.startswith(("TT", "Tbar", "TB", "TW")):
        return "top"
    if sample.startswith("Wto"):
        return "wjets"
    if sample.startswith("DYto"):
        return "dy"
    if "unpolarized" in sample:
        return "vv"
    return "other"


def _flat(jets, field, dtype=float):
    return ak.to_numpy(ak.flatten(jets[field])).astype(dtype)


def btag_fixedwp_weight(jets, sample, eff_file, wp_L, wp_T, tagger,
                        btv_file=BTV_FILE, variations=()):
    """Per-event weight for `jets` (jagged). Returns {name: np.array} with
    "nominal" and "<var>Up"/"<var>Down" for each requested variation."""
    counts = ak.to_numpy(ak.num(jets))
    flav = _flat(jets, "hadronFlavour", int)
    pt = _flat(jets, "pt")
    abseta = np.abs(_flat(jets, "eta"))
    score = np.nan_to_num(_flat(jets, tagger), nan=-1.0)
    passL = score > wp_L          # same comparison as pocket_coffea.lib.jets.btagging
    passT = score > wp_T

    # MC efficiencies (map clamps |eta| and pT to its range)
    effmap = _cset(eff_file)["btag_eff_2024"]
    group = group_of(sample)
    eL = np.asarray(effmap.evaluate(group, "L", flav, abseta, pt))
    eT = np.asarray(effmap.evaluate(group, "T", flav, abseta, pt))
    eT = np.minimum(eT, eL * (1 - 1e-6))   # T is a subset of L

    # SF inputs clamped to the BTV validity range
    pt_sf = np.clip(pt, SF_PT_RANGE[0] + 1e-3, SF_PT_RANGE[1] - 1e-3)
    abseta_sf = np.clip(abseta, 0.0, SF_ABSETA_MAX - 1e-4)
    btv = _cset(btv_file)
    light = flav == 0
    heavy = ~light

    def sf(wp, syst_bc, syst_light):
        out = np.ones_like(pt)
        if heavy.any():
            out[heavy] = btv["UParTAK4_comb"].evaluate(syst_bc, wp, flav[heavy], abseta_sf[heavy], pt_sf[heavy])
        if light.any():
            out[light] = btv["UParTAK4_light"].evaluate(syst_light, wp, flav[light], abseta_sf[light], pt_sf[light])
        return out

    # Floor the MC probability of each class: a class with a tiny MC probability
    # (e.g. eL ~ eT for an "L not T" jet) would otherwise turn a small error on
    # the efficiencies into an arbitrarily large per-jet weight.
    p_mc = np.maximum(np.where(passT, eT, np.where(passL, eL - eT, 1.0 - eL)), P_MC_MIN)

    def event_weight(syst_bc="central", syst_light="central"):
        sL, sT = sf("L", syst_bc, syst_light), sf("T", syst_bc, syst_light)
        p_data = np.where(passT, sT * eT, np.where(passL, sL * eL - sT * eT, 1.0 - sL * eL))
        # P_data < 0 is possible only for inconsistent SF/eff combinations
        # (e.g. SF_L*eL > 1); clip to keep the weight finite and non-negative
        ratio = np.clip(p_data, 0.0, None) / p_mc
        return ak.to_numpy(ak.prod(ak.unflatten(ratio, counts), axis=1))

    out = {"nominal": event_weight()}
    for var in variations:
        kind, scheme = var.split("_", 1)            # bc|light, correlated|uncorrelated
        for d, label in (("up", "Up"), ("down", "Down")):
            syst = f"{d}_{scheme}"
            out[f"{var}{label}"] = (event_weight(syst, "central") if kind == "bc"
                                    else event_weight("central", syst))
    return out


class BTagFixedWPWeight(WeightWrapper):
    """sf_btag_fixedwp: 2024 fixed-WP (L, T) b-tagging weight, BTV method 1a.

    Configure the class attributes in the config before use:
      eff_file  -- efficiency map from derive_btag_eff_2024.py
      wp_L, wp_T, tagger -- from parameters.btagging.working_point["2024"]
    """
    name = "sf_btag_fixedwp"
    has_variations = True
    isMC_only = True
    _variations = list(VARIATIONS)

    eff_file = None
    wp_L = None
    wp_T = None
    tagger = "btagUParTAK4B"
    jet_collection = "JetGoodBtagAcc"

    def __init__(self, params=None, metadata=None):
        super().__init__(params, metadata)
        self._variations = list(VARIATIONS)

    def compute(self, events, size, shape_variation):
        jets = events[self.jet_collection]
        sample = events.metadata["sample"]
        if shape_variation == "nominal":
            w = btag_fixedwp_weight(jets, sample, self.eff_file, self.wp_L, self.wp_T,
                                    self.tagger, variations=VARIATIONS)
            return WeightDataMultiVariation(
                name=self.name,
                nominal=w["nominal"],
                variations=list(VARIATIONS),
                up=[w[f"{v}Up"] for v in VARIATIONS],
                down=[w[f"{v}Down"] for v in VARIATIONS],
            )
        # shape variations (JES/JER, lepton scale): nominal weight recomputed
        # with the varied jets, no SF variations
        w = btag_fixedwp_weight(jets, sample, self.eff_file, self.wp_L, self.wp_T, self.tagger)
        return WeightData(name=self.name, nominal=w["nominal"])
