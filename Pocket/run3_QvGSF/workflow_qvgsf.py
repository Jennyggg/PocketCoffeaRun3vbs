# workflow_qvgsf.py
# PocketCoffea processor for ParticleNet QvG scale factor derivation.
# Implements two selections:
#   1. Z+jets (quark-enriched per AN-22-140): double-muon triggered, Z->mumu + >=1 jet
#      Probe jet = leading jet recoiling against Z (j0)
#   2. Dijet (gluon-enriched per AN-22-140): PFJet40-triggered, back-to-back dijets
#      Tag jet  = leading jet j0 (fires trigger, NOT measured for QvG)
#      Probe jet = subleading jet j1 (unbiased, measured for QvG score)
#
# Probe jet properties are stored as event-level scalars so PocketCoffea's
# standard histogram infrastructure can fill 3D (score x pT x |eta|) histograms.

import awkward as ak
import numpy as np
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.leptons import lepton_selection
from pocket_coffea.lib.jets import jet_selection


def veto_jer_forward_unmatched(jets, variations):
    """
    Run 3 AK4 Puppi jets:
    Disable JER smearing for unmatched jets in 2.5 < |eta| < 3.0.
    These jets have a known issue where JER smearing is incorrectly applied
    to unmatched jets in this eta region.
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
    jets["pt"]   = ak.where(veto_pt,   jets.pt   / jets.pt_sf_jer,   jets.pt)
    jets["mass"] = ak.where(veto_mass, jets.mass / jets.mass_sf_jer, jets.mass)

    # Systematics
    if "pt_JER_up" in jets.fields:
        jets["pt_JER_up"]   = ak.where(veto_pt,   jets.pt, jets.pt_JER_up)
    if "pt_JER_down" in jets.fields:
        jets["pt_JER_down"] = ak.where(veto_pt,   jets.pt, jets.pt_JER_down)
    if "mass_JER_up" in jets.fields:
        jets["mass_JER_up"]   = ak.where(veto_mass, jets.mass, jets.mass_JER_up)
    if "mass_JER_down" in jets.fields:
        jets["mass_JER_down"] = ak.where(veto_mass, jets.mass, jets.mass_JER_down)
    for var in jets.fields:
        if "pt_JES" in var and ("up" in var or "down" in var):
            jets[var] = ak.where(veto_pt,   jets[var] / jets.pt_sf_jer,   jets[var])
        if "mass_JES" in var and ("up" in var or "down" in var):
            jets[var] = ak.where(veto_mass, jets[var] / jets.mass_sf_jer, jets[var])

    return jets


Z_MASS    = 91.1876   # GeV
MZ_WINDOW = 20.0      # |mll - mZ| < 20 GeV  →  80 < mll < 111 GeV


class QvGSFProcessor(BaseProcessorABC):
    """
    Processor for ParticleNet QvG scale factor derivation.

    Stores event-level tag-jet fields:
      - tag_jet_pt, tag_jet_abseta, tag_jet_qvgscore, tag_jet_partonFlavour
      - mll          (Z+jets region: di-muon invariant mass)
      - dijet_region (1 if event passes dijet pre-selection, else 0)
      - zjets_region (1 if event passes Z+jets pre-selection, else 0)

    The actual categories (Zjets_all, Zjets_quark, Zjets_gluon, Dijet_all, …)
    are defined as Cut objects in config_qvgsf.py, using these flags.
    """

    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    # ── object preselection ──────────────────────────────────────────────────
    def apply_object_preselection(self, variation):
        ev = self.events

        # Good muons (loose pT 10 GeV used here; tighter cut in cut functions)
        ev["MuonGood"] = lepton_selection(ev, "Muon", self.params)

        # Undo JER smearing for unmatched jets in 2.5 < |eta| < 3.0 (MC only)
        if self._isMC:
            ev["Jet"] = veto_jer_forward_unmatched(ev.Jet, variation)

        # Lepton-cleaned good jets (pT > 15, eta < 4.7, jetId=6 from params)
        ev["JetGood"], _ = jet_selection(ev, "Jet", self.params, self._year,
                                         leptons_collection="MuonGood")

    # ── object counting (required abstract method) ───────────────────────────
    def count_objects(self, variation):
        ev = self.events
        ev["nMuonGood"] = ak.num(ev.MuonGood)
        ev["nJetGood"]  = ak.num(ev.JetGood)

    # ── extra processing after preselection ──────────────────────────────────
    def process_extra_after_presel(self, variation):
        ev = self.events

        jets  = ev.JetGood
        muons = ev.MuonGood
        nj    = ak.num(jets)
        nmu   = ak.num(muons)

        # ── Z+jets tag jet = leading jet ────────────────────────────────────
        # Build quantities needed by the Zjets cut function too
        j0   = ak.firsts(jets)
        j0_pt  = ak.fill_none(j0.pt,  0.0)
        j0_eta = ak.fill_none(j0.eta, 0.0)

        # leading/subleading muons
        mu0 = ak.firsts(muons)
        mu1 = ak.firsts(jets[:, 0:0])   # placeholder; overwritten below
        has_2mu = nmu >= 2
        mu1 = ak.where(has_2mu, ak.firsts(muons[:, 1:]), ak.firsts(muons[:, 0:0]))

        # Muon pT scalars — stored for trigger turn-on inspection
        ev["mu0_pt"] = ak.fill_none(mu0.pt, 0.0)
        ev["mu1_pt"] = ak.where(has_2mu, ak.fill_none(ak.firsts(muons[:, 1:]).pt, 0.0), 0.0)

        # di-muon invariant mass (set to -1 when <2 muons)
        mll = ak.where(
            has_2mu,
            (mu0 + mu1).mass,
            ak.full_like(has_2mu, -1.0, dtype=float),
        )
        ev["mll"] = ak.fill_none(mll, -1.0)

        # pT(ll) = transverse momentum of the Z candidate
        ptz = ak.where(
            has_2mu,
            (mu0 + mu1).pt,
            ak.full_like(has_2mu, 0.0, dtype=float),
        )
        ev["ptz"] = ak.fill_none(ptz, 0.0)

        # Subleading jet pT (0 if absent)
        j1    = ak.pad_none(jets, 2)[:, 1]
        j1_pt = ak.fill_none(j1.pt, 0.0)
        ev["j1_pt"] = j1_pt

        # Average pT of two leading jets — used for DiPFJetAve plateau cuts
        ev["dijet_ptavg"] = 0.5 * (j0_pt + j1_pt)

        # ── Dijet: probe jet = j1 (subleading, unbiased by trigger) ────────────
        j0_abseta = np.abs(j0_eta)
        j1_raw    = ak.firsts(jets[:, 1:])

        # Third-jet balance for dijet
        j2_pt   = ak.fill_none(ak.firsts(ak.pad_none(jets, 3)[:, 2:]).pt, 0.0)
        pt_avg  = 0.5 * (j0_pt + ak.fill_none(ak.firsts(jets[:, 1:]).pt, 0.0))
        balance = ak.where(pt_avg > 0, j2_pt / pt_avg, 0.0)
        ev["dijet_balance"] = balance

        # Δφ(j0, j1) for dijet
        dphi_jj = ak.fill_none(
            np.abs(ak.firsts(jets).delta_phi(ak.firsts(jets[:, 1:]))), 0.0
        )
        ev["dphi_jj"] = dphi_jj

        # Δφ(j0, Z) for Z+jets
        dphi_j0_Z = ak.where(
            has_2mu & (nj >= 1),
            ak.fill_none(np.abs(j0.delta_phi(mu0 + mu1)), 0.0),
            ak.full_like(has_2mu, 0.0, dtype=float),
        )
        ev["dphi_j0_Z"] = dphi_j0_Z

        # ── Store probe jet fields ────────────────────────────────────────────
        # Z+jets probe = j0 (leading jet recoiling against Z)
        # Dijet probe  = j1 (subleading jet, unbiased by PFJet40 trigger)
        # Dijet lead   = j0 (leading jet; also unbiased for ZeroBias categories)
        # All sets stored for all events; cut functions select the relevant ones.

        _has_pf = hasattr(ev.JetGood, "partonFlavour")

        # Z+jets probe jet
        ev["zjets_probe_pt"]     = j0_pt
        ev["zjets_probe_abseta"] = j0_abseta
        ev["zjets_probe_score"]  = ak.fill_none(j0.btagPNetQvG, -1.0)
        ev["zjets_probe_pf"]     = (
            ak.fill_none(j0.partonFlavour, 0)
            if _has_pf else ak.zeros_like(j0_pt, dtype=int)
        )

        # Leading jet pT (j0) — stored for trigger turn-on inspection
        ev["j0_pt"] = j0_pt

        # Dijet probe jet (j1 = subleading, unbiased by PFJet40)
        ev["dijet_probe_pt"]     = ak.fill_none(j1_raw.pt,           0.0)
        ev["dijet_probe_abseta"] = ak.fill_none(np.abs(j1_raw.eta),  0.0)
        ev["dijet_probe_score"]  = ak.fill_none(j1_raw.btagPNetQvG, -1.0)
        ev["dijet_probe_pf"]     = (
            ak.fill_none(j1_raw.partonFlavour, 0)
            if _has_pf else ak.zeros_like(j0_pt, dtype=int)
        )

        # Dijet leading jet as probe (j0) — for ZeroBias, where the trigger fires
        # on bunch crossing (not pT), so j0 is also an unbiased QvG probe.
        ev["dijet_lead_pt"]     = j0_pt
        ev["dijet_lead_abseta"] = j0_abseta
        ev["dijet_lead_score"]  = ak.fill_none(j0.btagPNetQvG, -1.0)
        ev["dijet_lead_pf"]     = (
            ak.fill_none(j0.partonFlavour, 0)
            if _has_pf else ak.zeros_like(j0_pt, dtype=int)
        )

        # ── Per-trigger DiPFJetAve HLT fired flags ───────────────────────────
        # For data: read from HLT collection (False if field absent in stream).
        # For MC:   always True — no trigger requirement applied to simulation.
        _DJ_HLT = {
            "dj40":  "DiPFJetAve40",
            "dj60":  "DiPFJetAve60",
            "dj80":  "DiPFJetAve80",
            "dj140": "DiPFJetAve140",
            "dj200": "DiPFJetAve200",
            "dj260": "DiPFJetAve260",
            "dj320": "DiPFJetAve320",
            "dj400": "DiPFJetAve400",
            "dj500": "DiPFJetAve500",
        }
        for _key, _trig in _DJ_HLT.items():
            if self._isMC:
                ev[f"hlt_{_key}"] = ak.ones_like(ev.event, dtype=bool)
            elif _trig in ev.HLT.fields:
                ev[f"hlt_{_key}"] = ev.HLT[_trig]
            else:
                # HLT path not stored in this stream (e.g. ZeroBias NanoAOD)
                ev[f"hlt_{_key}"] = ak.zeros_like(ev.event, dtype=bool)

        # ── MC-only pileup-vertex and gen-jet matching cuts ──────────────────
        # MC-only PU-vertex veto: |GenVtx_z - PV_z| < 0.2 cm
        if self._isMC:
            genvtx_z = ev.GenVtx.z
            pv_z     = ak.fill_none(ev.PV.z, 0.0)
            ev["genvtx_ok"] = np.abs(genvtx_z - pv_z) < 0.2
        else:
            ev["genvtx_ok"] = ak.ones_like(ev.nJetGood, dtype=bool)
