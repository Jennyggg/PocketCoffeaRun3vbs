# lhe_weights.py
#
# LHE scale and PDF weight wrappers, same definitions as in
# config_semilep_evaluateMC_qvgcal_addscaleweight_v10.py.
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


# LHEPdfWeight = LHA IDs 325300-325402 (NNPDF31_nnlo_as_0118_mc_hessian_pdfas,
# 103 members), stored as w_member / w_nominal: [0] central (alpha_S = 0.118),
# [1]-[100] symmetric Hessian eigenvectors, [101] alpha_S = 0.116 (down),
# [102] alpha_S = 0.120 (up). The eigenvectors are NOT MC replicas, so the PDF
# uncertainty is the quadrature sum of (w_i - 1), not an RMS over members.
_EIG_LO, _EIG_HI = 1, 101
N_PDF_EIG = _EIG_HI - _EIG_LO
_ALPHAS_DOWN, _ALPHAS_UP = 101, 102
_N_MEMBERS = 103


def _padded_members(events):
    """(w, n): LHEPdfWeight padded to 103 members (missing -> 1.0) and the number
    of members each event really has."""
    w_raw = events.LHEPdfWeight
    n = ak.to_numpy(ak.num(w_raw))
    w = ak.fill_none(ak.pad_none(w_raw, _N_MEMBERS, axis=1, clip=False), 1.0)
    return w, n


def _member(w, n, i):
    """Member i where the sample has it, else 1.0 (no variation)."""
    vals = ak.to_numpy(w[:, i]).astype(float)
    return np.where((n > i) & np.isfinite(vals), vals, 1.0)


def hessian_eigen_weights(w, n):
    """(n_events, 100) eigenvector weights; members the sample lacks -> 1.0."""
    return np.stack([_member(w, n, i) for i in range(_EIG_LO, _EIG_HI)], axis=1)


def hessian_total_error(eig):
    """Per-event symmetric Hessian PDF error sqrt(sum_i (w_i - 1)^2)."""
    eig = np.asarray(eig, dtype=float)
    return np.sqrt(np.sum((eig - 1.0) ** 2, axis=1))


def _unit_weights(name, variations, size):
    ones = np.ones(size)
    return WeightDataMultiVariation(
        name=name, nominal=ones, variations=variations,
        up=[ones] * len(variations), down=[ones] * len(variations),
    )


class LHEPdfWeightWrapper(WeightWrapper):
    """Combined PDF uncertainty from the 100 Hessian eigenvectors + alpha_S.

      pdf:     up/down = 1 +- sqrt(sum_i (LHEPdfWeight[i] - 1)^2), i = 1..100
               (down clipped at 0)
      alpha_S: up = LHEPdfWeight[102] (0.120), down = LHEPdfWeight[101] (0.116)

    Modifiers: LHEPdfWeight_pdf{Up,Down}, LHEPdfWeight_alpha_S{Up,Down}.
    The per-event quadrature sum is exact for the yield only when all events move
    together; use LHEPdfWeightEigenWrapper for the fully correlated treatment.
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
        if not hasattr(events, "LHEPdfWeight"):
            return _unit_weights(self.name, self._variations, size)

        w, n = _padded_members(events)
        err = hessian_total_error(hessian_eigen_weights(w, n))
        return WeightDataMultiVariation(
            name=self.name,
            nominal=np.ones(size),
            variations=self._variations,
            up=[1.0 + err, _member(w, n, _ALPHAS_UP)],
            down=[np.maximum(1.0 - err, 0.0), _member(w, n, _ALPHAS_DOWN)],
        )


class LHEPdfWeightEigenWrapper(WeightWrapper):
    """PDF uncertainty decomposed into the 100 independent Hessian eigenvectors.

      pdf_i:   up = LHEPdfWeight[i], down = 2 - LHEPdfWeight[i] (mirrored about 1,
               clipped at 0), i = 1..100
      alpha_S: up = LHEPdfWeight[102] (0.120), down = LHEPdfWeight[101] (0.116)

    Modifiers: LHEPdfWeightEigen_pdf_<i>{Up,Down}, LHEPdfWeightEigen_alpha_S{Up,Down}.
    Each pdf_i is one nuisance, correlated across processes and bins.
    """
    name = "LHEPdfWeightEigen"
    has_variations = True
    isMC_only = True
    _variations = [f"pdf_{i}" for i in range(_EIG_LO, _EIG_HI)] + ["alpha_S"]

    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)

    def compute(self, events, size, shape_variation):
        if shape_variation != "nominal":
            return WeightData(name=self.name, nominal=np.ones(size))
        if not hasattr(events, "LHEPdfWeight"):
            return _unit_weights(self.name, self._variations, size)

        w, n = _padded_members(events)
        eig = hessian_eigen_weights(w, n)
        up = [eig[:, k] for k in range(N_PDF_EIG)]
        down = [np.maximum(2.0 - eig[:, k], 0.0) for k in range(N_PDF_EIG)]
        return WeightDataMultiVariation(
            name=self.name,
            nominal=np.ones(size),
            variations=self._variations,
            up=up + [_member(w, n, _ALPHAS_UP)],
            down=down + [_member(w, n, _ALPHAS_DOWN)],
        )
