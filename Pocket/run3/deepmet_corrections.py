# deepmet_corrections.py
#
# 2024 DeepMET (resolution tune) corrections, as derived in DY->mumu:
#   1. xy offset vs PV.npvsGood, data and MC        (derive_deepmet_xy_corrections.py)
#        px -= offset(datatype, "px", npvsGood), py -= offset(datatype, "py", npvsGood)
#   2. Gaussian smearing of MC px, py after step 1   (derive_deepmet_smearing.py)
#        px += sigma("px", npvsGood) * rng("px", run, luminosityBlock, event)   (same for py)
#      rng is a per-event deterministic N(0,1) (correctionlib HashPRNG); data are not smeared.
#
# corrected_deepmet(events, is_mc, xy_json, smear_json=None) returns a record (pt, phi, px, py);
# with smear_json=None only step 1 is applied.
import awkward as ak
import numpy as np
import correctionlib

_CSETS = {}


def _cset(path):
    if path not in _CSETS:
        _CSETS[path] = correctionlib.CorrectionSet.from_file(path)
    return _CSETS[path]


def corrected_deepmet(events, is_mc, xy_json, smear_json=None,
                      xy_correction="DeepMETResolutionTune_xy_offset", met="DeepMETResolutionTune"):
    met_obj = events[met]
    npv = ak.to_numpy(events.PV.npvsGood).astype(np.float64)
    px = ak.to_numpy(met_obj.pt * np.cos(met_obj.phi))
    py = ak.to_numpy(met_obj.pt * np.sin(met_obj.phi))

    offset = _cset(xy_json)[xy_correction]
    datatype = "mc" if is_mc else "data"
    px = px - offset.evaluate(datatype, "px", npv)
    py = py - offset.evaluate(datatype, "py", npv)

    if is_mc and smear_json:
        cs = _cset(smear_json)
        sigma, rng = cs["DeepMETResolutionTune_xy_smear_sigma"], cs["DeepMETResolutionTune_xy_smear_rng"]
        ids = [ak.to_numpy(events[f]).astype(np.int64) for f in ("run", "luminosityBlock", "event")]
        px = px + sigma.evaluate("px", npv) * rng.evaluate("px", *ids)
        py = py + sigma.evaluate("py", npv) * rng.evaluate("py", *ids)

    return ak.zip({"pt": np.hypot(px, py), "phi": np.arctan2(py, px), "px": px, "py": py})
