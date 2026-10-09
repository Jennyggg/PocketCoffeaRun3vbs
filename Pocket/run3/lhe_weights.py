# lhe_weights.py
#
# LHE scale and PDF weight wrappers, same definitions as in
# config_semilep_evaluateMC_qvgcal_addscaleweight_v10.py.
import warnings
import numpy as np
import awkward as ak
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation


class LHEScaleWeightWrapper(WeightWrapper):
    """LHE renormalization and factorization scale uncertainties.
    LHEScaleWeight is already normalized as w_var/w_nominal in NanoAOD.
      renorm_scale: down=LHEScaleWeight[:,1], up=LHEScaleWeight[:,7]
      fact_scale:   down=LHEScaleWeight[:,3], up=LHEScaleWeight[:,5]
    """
    name = "LHEScaleWeight"
    has_variations = True
    isMC_only = True
    _variations = ["renorm_scale", "fact_scale"]

    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)

    def compute(self, events, size, shape_variation):
        if shape_variation == "nominal":
            if not hasattr(events, "LHEScaleWeight"):
                ones = np.ones(size)
                return WeightDataMultiVariation(
                    name=self.name,
                    nominal=ones,
                    variations=self._variations,
                    up=[ones, ones],
                    down=[ones, ones],
                )
            w = events.LHEScaleWeight
            return WeightDataMultiVariation(
                name=self.name,
                nominal=np.ones(size),
                variations=self._variations,
                up=[ak.to_numpy(w[:, 7]), ak.to_numpy(w[:, 5])],
                down=[ak.to_numpy(w[:, 1]), ak.to_numpy(w[:, 3])],
            )
        else:
            return WeightData(name=self.name, nominal=np.ones(size))


_REPL_LO, _REPL_HI = 1, 101


def replica_envelope(vals, n_members):
    """Per-event mean and RMS over the PDF replica members.

    vals      : (n_events, 100) replica weights, already padded with 1.0
    n_members : (n_events,) how many LHEPdfWeight members the event really has

    Members the sample does not have are excluded rather than counted as 1.0.
    Returns (mean, rms) with mean=1, rms=0 for events with fewer than two
    usable replicas, where a spread is not defined.
    """
    vals = np.asarray(vals, dtype=float)
    n_members = np.asarray(n_members)

    idx = np.arange(_REPL_LO, _REPL_HI)[None, :]
    valid = (idx < n_members[:, None]) & np.isfinite(vals)
    count = valid.sum(axis=1)

    masked = np.where(valid, vals, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(masked, axis=1)
        # ddof=1: the replicas are a sample from the PDF distribution.
        rms = np.nanstd(masked, axis=1, ddof=1)

    enough = count >= 2
    mean = np.where(enough & np.isfinite(mean), mean, 1.0)
    rms = np.where(enough & np.isfinite(rms), rms, 0.0)
    return mean, rms


class LHEPdfWeightWrapper(WeightWrapper):
    """LHE PDF and alpha_S uncertainties from LHEPdfWeight.

    [1]-[100]: PDF replicas, reduced to a single 'pdf' variation per event
    (up/down = mean +- rms); [101]/[102]: alpha_S up/down. Samples with shorter
    sets use only the members they have; alpha_S gets unit weights when absent.
    """
    name = "LHEPdfWeight"
    has_variations = True
    isMC_only = True
    _variations = ["pdf", "alpha_S"]

    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)

    def compute(self, events, size, shape_variation):
        if shape_variation != "nominal":
            return WeightData(name=self.name, nominal=np.ones(size))

        ones = np.ones(size)
        if not hasattr(events, "LHEPdfWeight"):
            return WeightDataMultiVariation(
                name=self.name, nominal=ones, variations=self._variations,
                up=[ones] * len(self._variations),
                down=[ones] * len(self._variations),
            )

        w_raw = events.LHEPdfWeight
        n = ak.to_numpy(ak.num(w_raw))
        # pad so indices up to 102 always exist; padded members -> 1.0
        w = ak.fill_none(ak.pad_none(w_raw, 103, axis=1, clip=False), 1.0)

        def member(i):
            # only trust members the sample actually has
            return np.where(n > i, ak.to_numpy(w[:, i]), 1.0)

        replicas = ak.to_numpy(w[:, _REPL_LO:_REPL_HI])
        mean, rms = replica_envelope(replicas, n)

        return WeightDataMultiVariation(
            name=self.name,
            nominal=ones,
            variations=self._variations,
            up=[mean + rms, member(101)],
            down=[np.maximum(mean - rms, 0.0), member(102)],
        )
