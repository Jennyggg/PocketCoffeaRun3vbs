# cut_functions_qvgsf.py
# Cut functions for the ParticleNet QvG scale factor derivation.
# Z+jets (quark-enriched) and Dijet (gluon-enriched) selections.

import awkward as ak
import numpy as np
from pocket_coffea.lib.cut_definition import Cut

Z_MASS    = 91.1876
MZ_WINDOW = 20.0


# ── Skim-level cuts ──────────────────────────────────────────────────────────

def njet_skim(events, params, **kwargs):
    """At least one raw jet with pT > 15 GeV (skim runs before object preselection)."""
    return ak.num(events.Jet[events.Jet.pt > 15.0]) >= 1

njet_skim_cut = Cut(name="njet_skim", params={}, function=njet_skim)


# ── Z+jets selection ─────────────────────────────────────────────────────────

def select_zjets(events, params, **kwargs):
    """
    Z+jets event selection (quark-enriched probe jet, per AN-22-140):
      - >=2 tight OS muons with pT > 20 GeV and |eta| < 2.4
      - |mll - mZ| < 20 GeV
      - pT(ll) > 15 GeV
      - >=1 jet with pT > 20 GeV
      - Δφ(j0, Z) > 2.7
      - if j2 exists: pT(j2) < pT(Z)
    """
    muons = events.MuonGood
    nmu   = ak.num(muons)

    # At least 2 muons passing tighter pT cut
    mu0_pt = ak.fill_none(ak.firsts(muons).pt, 0.0)
    mu1_pt = ak.fill_none(ak.firsts(muons[:, 1:]).pt, 0.0)
    tight_pt = (mu0_pt > params["mu_pt"]) & (mu1_pt > params["mu_pt"])

    has_2mu = (nmu >= 2) & tight_pt

    # OS requirement: charges have opposite sign
    mu0_q = ak.fill_none(ak.firsts(muons).charge, 0)
    mu1_q = ak.fill_none(ak.firsts(muons[:, 1:]).charge, 0)
    os_pair = mu0_q * mu1_q < 0

    # Z mass window and pT
    mll_ok = np.abs(events.mll - Z_MASS) < MZ_WINDOW
    ptz_ok = events.ptz > params["ptz_min"]

    # At least one jet
    n_jets = ak.num(events.JetGood)
    has_1jet = n_jets >= 1

    # Leading jet pT > threshold
    j0_pt_ok = events.zjets_probe_pt > params["jet_pt"]

    # Δφ(j0, Z) > 2.7
    dphi_ok = events.dphi_j0_Z > params["dphi_min"]

    # Subleading jet pT < pT(Z)
    j1_below_ptz = events.j1_pt < events.ptz

    # MC-only: PU-vertex veto (True for data)
    genvtx_ok = ak.fill_none(events.genvtx_ok, True)

    mask = (
        has_2mu
        & os_pair
        & mll_ok
        & ptz_ok
        & has_1jet
        & j0_pt_ok
        & dphi_ok
        & j1_below_ptz
        & genvtx_ok
    )
    return ak.fill_none(mask, False)


zjets_cut = Cut(
    name="zjets",
    params={
        "mu_pt":    26.0,
        "ptz_min":  15.0,
        "jet_pt":   20.0,
        "dphi_min":  2.7,
    },
    function=select_zjets,
)


def zjets_quark(events, params, **kwargs):
    """Z+jets + tag jet is a quark jet (|partonFlavour| in 1-5)."""
    base = select_zjets(events, params, **kwargs)
    pf   = events.zjets_probe_pf
    is_q = (np.abs(pf) >= 1) & (np.abs(pf) <= 5)
    return base & ak.fill_none(is_q, False)


zjets_quark_cut = Cut(
    name="zjets_quark",
    params={
        "mu_pt":    26.0,
        "ptz_min":  15.0,
        "jet_pt":   20.0,
        "dphi_min":  2.7,
    },
    function=zjets_quark,
)


def zjets_gluon(events, params, **kwargs):
    """Z+jets + tag jet is a gluon jet (partonFlavour == 21)."""
    base = select_zjets(events, params, **kwargs)
    pf   = events.zjets_probe_pf
    is_g = pf == 21
    return base & ak.fill_none(is_g, False)


zjets_gluon_cut = Cut(
    name="zjets_gluon",
    params={
        "mu_pt":    26.0,
        "ptz_min":  15.0,
        "jet_pt":   20.0,
        "dphi_min":  2.7,
    },
    function=zjets_gluon,
)


def zjets_undef(events, params, **kwargs):
    """Z+jets + tag jet is undetermined (partonFlavour == 0)."""
    base = select_zjets(events, params, **kwargs)
    pf   = events.zjets_probe_pf
    is_u = pf == 0
    return base & ak.fill_none(is_u, False)


zjets_undef_cut = Cut(
    name="zjets_undef",
    params={
        "mu_pt":    26.0,
        "ptz_min":  15.0,
        "jet_pt":   20.0,
        "dphi_min":  2.7,
    },
    function=zjets_undef,
)


def select_zjets_notrigcut(events, params, **kwargs):
    """
    Loose Z+jets pre-selection without the offline muon pT plateau cut.
    Used to plot mu0_pt/mu1_pt and inspect the DoubleMuon trigger turn-on.
    Applies all other Z+jets cuts (OS, Z mass window, ptz, dphi, j1 balance).
    No MC-only PU/gen-matching cuts — this is a data/MC shape category.
    """
    muons = events.MuonGood
    nmu   = ak.num(muons)
    has_2mu = nmu >= 2

    mu0_q = ak.fill_none(ak.firsts(muons).charge, 0)
    mu1_q = ak.fill_none(ak.firsts(muons[:, 1:]).charge, 0)
    os_pair = mu0_q * mu1_q < 0

    mll_ok  = np.abs(events.mll - Z_MASS) < MZ_WINDOW
    ptz_ok  = events.ptz > params["ptz_min"]
    has_1jet = ak.num(events.JetGood) >= 1
    j0_pt_ok = events.zjets_probe_pt > params["jet_pt"]
    dphi_ok  = events.dphi_j0_Z > params["dphi_min"]
    j1_below_ptz = events.j1_pt < events.ptz

    mask = has_2mu & os_pair & mll_ok & ptz_ok & has_1jet & j0_pt_ok & dphi_ok & j1_below_ptz
    return ak.fill_none(mask, False)


zjets_notrigcut = Cut(
    name="zjets_notrigcut",
    params={
        "ptz_min":  15.0,
        "jet_pt":   20.0,
        "dphi_min":  2.7,
    },
    function=select_zjets_notrigcut,
)


def select_zjets_noMCcut(events, params, **kwargs):
    """
    Z+jets full selection (including muon pT plateau cut) but WITHOUT the
    MC-only genvtx and gen-jet matching requirements.
    Used to compare data/MC after offline cuts but before MC PU cleanup.
    """
    muons = events.MuonGood
    nmu   = ak.num(muons)

    mu0_pt = ak.fill_none(ak.firsts(muons).pt, 0.0)
    mu1_pt = ak.fill_none(ak.firsts(muons[:, 1:]).pt, 0.0)
    tight_pt = (mu0_pt > params["mu_pt"]) & (mu1_pt > params["mu_pt"])
    has_2mu  = (nmu >= 2) & tight_pt

    mu0_q = ak.fill_none(ak.firsts(muons).charge, 0)
    mu1_q = ak.fill_none(ak.firsts(muons[:, 1:]).charge, 0)
    os_pair = mu0_q * mu1_q < 0

    mll_ok       = np.abs(events.mll - Z_MASS) < MZ_WINDOW
    ptz_ok       = events.ptz > params["ptz_min"]
    has_1jet     = ak.num(events.JetGood) >= 1
    j0_pt_ok     = events.zjets_probe_pt > params["jet_pt"]
    dphi_ok      = events.dphi_j0_Z > params["dphi_min"]
    j1_below_ptz = events.j1_pt < events.ptz

    mask = has_2mu & os_pair & mll_ok & ptz_ok & has_1jet & j0_pt_ok & dphi_ok & j1_below_ptz
    return ak.fill_none(mask, False)


zjets_noMCcut = Cut(
    name="zjets_noMCcut",
    params={
        "mu_pt":    26.0,
        "ptz_min":  15.0,
        "jet_pt":   20.0,
        "dphi_min":  2.7,
    },
    function=select_zjets_noMCcut,
)


def select_dijet_noMCcut(events, params, **kwargs):
    """
    Dijet full selection (including j0_pt plateau cut) but WITHOUT the
    MC-only genvtx and gen-jet matching requirements.
    Used to compare data/MC after offline cuts but before MC PU cleanup.
    """
    n_jets    = ak.num(events.JetGood)
    has_2jets = n_jets >= 2

    j0_pt = ak.fill_none(ak.firsts(events.JetGood).pt, 0.0)
    j1_pt = ak.fill_none(ak.firsts(events.JetGood[:, 1:]).pt, 0.0)
    pt_ok = (j0_pt > params["lead_jet_pt"]) & (j1_pt > params["sublead_jet_pt"])

    dphi_ok    = events.dphi_jj > params["dphi_min"]
    balance_ok = events.dijet_balance < params["balance_max"]

    mask = has_2jets & pt_ok & dphi_ok & balance_ok
    return ak.fill_none(mask, False)


dijet_noMCcut = Cut(
    name="dijet_noMCcut",
    params={
        "lead_jet_pt":     55.0,
        "sublead_jet_pt":  30.0,
        "dphi_min":         2.7,
        "balance_max":      0.3,
    },
    function=select_dijet_noMCcut,
)


# ── Dijet selection ──────────────────────────────────────────────────────────

def select_dijet_notrigcut(events, params, **kwargs):
    """
    Loose dijet pre-selection without the leading jet pT threshold.
    Used to plot j0_pt and inspect the PFJet40 trigger turn-on plateau.
    Applies all other dijet cuts: back-to-back and balance (no eta restriction
    on probe jet — SF is derived across the full |eta| range up to 4.7).
    """
    n_jets    = ak.num(events.JetGood)
    has_2jets = n_jets >= 2

    j1_pt  = ak.fill_none(ak.firsts(events.JetGood[:, 1:]).pt, 0.0)
    j1_ok  = j1_pt > params["sublead_jet_pt"]

    dphi_ok    = events.dphi_jj > params["dphi_min"]
    balance_ok = events.dijet_balance < params["balance_max"]

    mask = has_2jets & j1_ok & dphi_ok & balance_ok
    return ak.fill_none(mask, False)


dijet_notrigcut = Cut(
    name="dijet_notrigcut",
    params={
        "sublead_jet_pt":  30.0,
        "dphi_min":         2.7,
        "balance_max":      0.3,
    },
    function=select_dijet_notrigcut,
)


def select_dijet(events, params, **kwargs):
    """
    Dijet event selection (gluon-enriched probe jet, per AN-22-140):
      - leading jet (j0) pT > 55 GeV (above PFJet40 plateau; fires trigger)
      - subleading jet (j1, probe) pT > 30 GeV (no trigger requirement)
      - Δφ(j0, j1) > 2.7 (back-to-back)
      - third-jet balance < 0.3
      - MC only: |GenVtx_z - PV_z| < 0.2 and j0, j1 gen-matched
    No eta restriction on probe jet — SF is derived in bins of probe |eta|
    covering the full range up to 4.7.
    """
    n_jets    = ak.num(events.JetGood)
    has_2jets = n_jets >= 2

    j0_pt = ak.fill_none(ak.firsts(events.JetGood).pt, 0.0)
    j1_pt = ak.fill_none(ak.firsts(events.JetGood[:, 1:]).pt, 0.0)
    pt_ok = (j0_pt > params["lead_jet_pt"]) & (j1_pt > params["sublead_jet_pt"])

    dphi_ok    = events.dphi_jj > params["dphi_min"]
    balance_ok = events.dijet_balance < params["balance_max"]

    genvtx_ok = ak.fill_none(events.genvtx_ok, True)

    mask = has_2jets & pt_ok & dphi_ok & balance_ok & genvtx_ok
    return ak.fill_none(mask, False)


_DIJET_PARAMS = {
    "lead_jet_pt":    55.0,
    "sublead_jet_pt": 30.0,
    "dphi_min":        2.7,
    "balance_max":     0.3,
}

dijet_cut = Cut(name="dijet", params=_DIJET_PARAMS, function=select_dijet)


def dijet_quark(events, params, **kwargs):
    """Dijet + probe jet is a quark jet (|partonFlavour| in 1-5)."""
    base = select_dijet(events, params, **kwargs)
    pf   = events.dijet_probe_pf
    is_q = (np.abs(pf) >= 1) & (np.abs(pf) <= 5)
    return base & ak.fill_none(is_q, False)


dijet_quark_cut = Cut(name="dijet_quark", params=_DIJET_PARAMS, function=dijet_quark)


def dijet_gluon(events, params, **kwargs):
    """Dijet + probe jet is a gluon jet (partonFlavour == 21)."""
    base = select_dijet(events, params, **kwargs)
    pf   = events.dijet_probe_pf
    is_g = pf == 21
    return base & ak.fill_none(is_g, False)


dijet_gluon_cut = Cut(name="dijet_gluon", params=_DIJET_PARAMS, function=dijet_gluon)


def dijet_undef(events, params, **kwargs):
    """Dijet + probe jet is undetermined (partonFlavour == 0)."""
    base = select_dijet(events, params, **kwargs)
    pf   = events.dijet_probe_pf
    is_u = pf == 0
    return base & ak.fill_none(is_u, False)


dijet_undef_cut = Cut(name="dijet_undef", params=_DIJET_PARAMS, function=dijet_undef)


# ── Dijet ZeroBias selection (no leading-jet pT threshold) ───────────────────
# Same as select_dijet but without the 580 GeV leading-jet cut so that
# ZeroBias data (and low-pT QCD MC) can populate the probe-jet pT range
# down to 30 GeV.  MC-only genvtx/gen-matching cuts are retained.

_DIJET_ZB_PARAMS = {
    "lead_jet_pt":    30.0,
    "sublead_jet_pt": 30.0,
    "dphi_min":        2.7,
    "balance_max":     0.3,
}


def select_dijet_zb(events, params, **kwargs):
    """
    Dijet selection for ZeroBias analysis: leading jet pT > 30 GeV (no trigger
    plateau requirement), probe pT > 30 GeV, Δφ > dphi_min, balance < balance_max,
    and MC-only genvtx/gen-matching cuts.
    """
    n_jets    = ak.num(events.JetGood)
    has_2jets = n_jets >= 2

    j0_pt  = ak.fill_none(ak.firsts(events.JetGood).pt,         0.0)
    j1_pt  = ak.fill_none(ak.firsts(events.JetGood[:, 1:]).pt,  0.0)
    pt_ok  = (j0_pt > params["lead_jet_pt"]) & (j1_pt > params["sublead_jet_pt"])

    dphi_ok    = events.dphi_jj > params["dphi_min"]
    balance_ok = events.dijet_balance < params["balance_max"]

    genvtx_ok = ak.fill_none(events.genvtx_ok, True)

    mask = has_2jets & pt_ok & dphi_ok & balance_ok & genvtx_ok
    return ak.fill_none(mask, False)


dijet_zb_cut = Cut(name="dijet_zb", params=_DIJET_ZB_PARAMS, function=select_dijet_zb)


def dijet_zb_quark(events, params, **kwargs):
    """Dijet ZB + probe jet is a quark jet (|partonFlavour| in 1–5)."""
    base = select_dijet_zb(events, params, **kwargs)
    pf   = events.dijet_probe_pf
    is_q = (np.abs(pf) >= 1) & (np.abs(pf) <= 5)
    return base & ak.fill_none(is_q, False)


dijet_zb_quark_cut = Cut(name="dijet_zb_quark", params=_DIJET_ZB_PARAMS, function=dijet_zb_quark)


def dijet_zb_gluon(events, params, **kwargs):
    """Dijet ZB + probe jet is a gluon jet (partonFlavour == 21)."""
    base = select_dijet_zb(events, params, **kwargs)
    pf   = events.dijet_probe_pf
    return base & ak.fill_none(pf == 21, False)


dijet_zb_gluon_cut = Cut(name="dijet_zb_gluon", params=_DIJET_ZB_PARAMS, function=dijet_zb_gluon)


def dijet_zb_undef(events, params, **kwargs):
    """Dijet ZB + probe jet is undetermined (partonFlavour == 0)."""
    base = select_dijet_zb(events, params, **kwargs)
    pf   = events.dijet_probe_pf
    is_u = pf == 0
    return base & ak.fill_none(is_u, False)


dijet_zb_undef_cut = Cut(name="dijet_zb_undef", params=_DIJET_ZB_PARAMS, function=dijet_zb_undef)


def select_dijet_zb_noMCcut(events, params, **kwargs):
    """
    ZeroBias dijet selection without MC-only genvtx/gen-matching cuts.
    Used for data-MC comparison plots where both data and MC see the same mask.
    """
    n_jets    = ak.num(events.JetGood)
    has_2jets = n_jets >= 2

    j0_pt  = ak.fill_none(ak.firsts(events.JetGood).pt,         0.0)
    j1_pt  = ak.fill_none(ak.firsts(events.JetGood[:, 1:]).pt,  0.0)
    pt_ok  = (j0_pt > params["lead_jet_pt"]) & (j1_pt > params["sublead_jet_pt"])

    dphi_ok    = events.dphi_jj > params["dphi_min"]
    balance_ok = events.dijet_balance < params["balance_max"]

    mask = has_2jets & pt_ok & dphi_ok & balance_ok
    return ak.fill_none(mask, False)


dijet_zb_noMCcut = Cut(name="dijet_zb_noMCcut", params=_DIJET_ZB_PARAMS, function=select_dijet_zb_noMCcut)


# ── DiPFJetAve dijet selections ───────────────────────────────────────────────
# One set of categories per trigger (dj40 … dj500).
# Probe jet = j1 (subleading), same as the existing Dijet selection.
# The offline cut is on (pT_j0 + pT_j1)/2 (dijet_ptavg) to ensure the event
# is on the trigger plateau.  HLT bits are stored by the workflow as
# ev["hlt_dj40"], ev["hlt_dj60"], …
#
# Per-trigger spec: (hlt_key, ptavg_min_GeV)
_DJ_SPEC = [
    ("dj40",   48.0),
    ("dj60",   72.0),
    ("dj80",   96.0),
    ("dj140", 168.0),
    ("dj200", 240.0),
    ("dj260", 310.0),
    ("dj320", 370.0),
    ("dj400", 450.0),
    ("dj500", 550.0),
]

_DIJET_DJ_PARAMS = {
    "lead_jet_pt":    30.0,   # loose j0 minimum; plateau set by ptavg cut
    "sublead_jet_pt": 30.0,   # j1 (probe) must be on PNetQvG plateau
    "dphi_min":        2.7,
    "balance_max":     0.3,
}


def _mk_dj_notrigcut_fn(hlt_key):
    """Diagnostic: basic dijet presel + HLT fired, NO ptavg cut.
    Used to plot dijet_ptavg and inspect the trigger turn-on plateau."""
    def _f(events, params, **kwargs):
        n_jets     = ak.num(events.JetGood)
        has_2j     = n_jets >= 2
        j0_pt      = ak.fill_none(ak.firsts(events.JetGood).pt,       0.0)
        j1_pt      = ak.fill_none(ak.firsts(events.JetGood[:, 1:]).pt, 0.0)
        pt_ok      = (j0_pt > params["lead_jet_pt"]) & (j1_pt > params["sublead_jet_pt"])
        dphi_ok    = events.dphi_jj > params["dphi_min"]
        balance_ok = events.dijet_balance < params["balance_max"]
        hlt_ok     = ak.fill_none(events[f"hlt_{hlt_key}"], True)
        return ak.fill_none(has_2j & pt_ok & dphi_ok & balance_ok & hlt_ok, False)
    return _f


def _mk_dj_all_fn(hlt_key, ptavg_min):
    """Full DiPFJetAve selection with ptavg plateau cut and MC-only gen cuts."""
    def _f(events, params, **kwargs):
        n_jets      = ak.num(events.JetGood)
        has_2j      = n_jets >= 2
        j0_pt       = ak.fill_none(ak.firsts(events.JetGood).pt,       0.0)
        j1_pt       = ak.fill_none(ak.firsts(events.JetGood[:, 1:]).pt, 0.0)
        pt_ok       = (j0_pt > params["lead_jet_pt"]) & (j1_pt > params["sublead_jet_pt"])
        dphi_ok     = events.dphi_jj > params["dphi_min"]
        balance_ok  = events.dijet_balance < params["balance_max"]
        ptavg_ok    = events.dijet_ptavg > ptavg_min
        hlt_ok    = ak.fill_none(events[f"hlt_{hlt_key}"], True)
        genvtx_ok = ak.fill_none(events.genvtx_ok, True)
        return ak.fill_none(
            has_2j & pt_ok & dphi_ok & balance_ok & ptavg_ok & hlt_ok
            & genvtx_ok, False)
    return _f


def _mk_dj_noMCcut_fn(hlt_key, ptavg_min):
    """DiPFJetAve selection with ptavg cut but WITHOUT MC-only gen cuts.
    For data-MC kinematic comparisons after the offline plateau cut."""
    def _f(events, params, **kwargs):
        n_jets     = ak.num(events.JetGood)
        has_2j     = n_jets >= 2
        j0_pt      = ak.fill_none(ak.firsts(events.JetGood).pt,       0.0)
        j1_pt      = ak.fill_none(ak.firsts(events.JetGood[:, 1:]).pt, 0.0)
        pt_ok      = (j0_pt > params["lead_jet_pt"]) & (j1_pt > params["sublead_jet_pt"])
        dphi_ok    = events.dphi_jj > params["dphi_min"]
        balance_ok = events.dijet_balance < params["balance_max"]
        ptavg_ok   = events.dijet_ptavg > ptavg_min
        hlt_ok     = ak.fill_none(events[f"hlt_{hlt_key}"], True)
        return ak.fill_none(has_2j & pt_ok & dphi_ok & balance_ok & ptavg_ok & hlt_ok, False)
    return _f


def _mk_dj_quark_fn(hlt_key, ptavg_min):
    _base = _mk_dj_all_fn(hlt_key, ptavg_min)
    def _f(events, params, **kwargs):
        base = _base(events, params, **kwargs)
        pf   = events.dijet_probe_pf
        return base & ak.fill_none((np.abs(pf) >= 1) & (np.abs(pf) <= 5), False)
    return _f


def _mk_dj_gluon_fn(hlt_key, ptavg_min):
    _base = _mk_dj_all_fn(hlt_key, ptavg_min)
    def _f(events, params, **kwargs):
        base = _base(events, params, **kwargs)
        pf   = events.dijet_probe_pf
        return base & ak.fill_none(pf == 21, False)
    return _f


def _mk_dj_undef_fn(hlt_key, ptavg_min):
    _base = _mk_dj_all_fn(hlt_key, ptavg_min)
    def _f(events, params, **kwargs):
        base = _base(events, params, **kwargs)
        pf   = events.dijet_probe_pf
        return base & ak.fill_none(pf == 0, False)
    return _f


# Build dict of Cut objects: dijet_dj_cuts[hlt_key][variant]
dijet_dj_cuts = {}
for _hlt_key, _ptavg_min in _DJ_SPEC:
    dijet_dj_cuts[_hlt_key] = {
        "notrigcut": Cut(
            name=f"dijet_{_hlt_key}_notrigcut", params=_DIJET_DJ_PARAMS,
            function=_mk_dj_notrigcut_fn(_hlt_key)),
        "all": Cut(
            name=f"dijet_{_hlt_key}_all", params=_DIJET_DJ_PARAMS,
            function=_mk_dj_all_fn(_hlt_key, _ptavg_min)),
        "noMCcut": Cut(
            name=f"dijet_{_hlt_key}_noMCcut", params=_DIJET_DJ_PARAMS,
            function=_mk_dj_noMCcut_fn(_hlt_key, _ptavg_min)),
        "quark": Cut(
            name=f"dijet_{_hlt_key}_quark", params=_DIJET_DJ_PARAMS,
            function=_mk_dj_quark_fn(_hlt_key, _ptavg_min)),
        "gluon": Cut(
            name=f"dijet_{_hlt_key}_gluon", params=_DIJET_DJ_PARAMS,
            function=_mk_dj_gluon_fn(_hlt_key, _ptavg_min)),
        "undef": Cut(
            name=f"dijet_{_hlt_key}_undef", params=_DIJET_DJ_PARAMS,
            function=_mk_dj_undef_fn(_hlt_key, _ptavg_min)),
    }
