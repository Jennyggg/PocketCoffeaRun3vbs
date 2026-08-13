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

    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        # Lazy cache: populated on first apply_object_preselection call for a given year.
        # Keyed by year so the same processor instance can handle multiple years.
        self._classifiers = {}


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


        veto_criteria = SimpleNamespace(
            object_preselection = {
                "Electron": {
                    "pt": 10.0,
                    "eta": 2.4,
                    "id": "cutBased", #"mvaNoIso", uncomment this for NanoAODv12
                },
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
                "Electron": {
                    "pt": 35.0,
                    "eta": 2.4,
                    "id": "cutBased", #"mvaNoIso", uncomment this for NanoAODv12
                },
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
        ev["ElectronVeto"] = lepton_selection(ev, "Electron", veto_criteria) 

        mask_ele_loose = (ev.Electron.pt > 35) & (np.abs(ev.Electron.eta) < 2.4) & (ev.Electron.cutBased >=1 )
        ev["ElectronLoose"] = ev.Electron[mask_ele_loose]

        ev["MuonLoose"]     = lepton_selection(ev, "Muon", loose_criteria)


        mask_muon_tight = (
            (ev.Muon.pt>26) & (np.abs(ev.Muon.eta)<2.4) & (ev.Muon.pfRelIso04_all<0.15) & (ev.Muon.tightId) & (np.abs(ev.Muon.dxy) < 0.2) & (np.abs(ev.Muon.dz) < 0.5)
        )

        mask_muon_loose = (
            (ev.Muon.pt>26) & (np.abs(ev.Muon.eta)<2.4) & (ev.Muon.looseId)
        )

        # The inverted "good" muons are those passing the loose selection but not the tight/
        ev["MuonGood"] = ev.Muon[mask_muon_loose & ~mask_muon_tight ]

        mask_ele_tight = (
            ( ev.Electron.pt > 35 ) & ( np.abs(ev.Electron.eta) < 2.4) & ( ev.Electron.cutBased >= 3 )
            & (
                ((np.abs(ev.Electron.dxy) < 0.05) & (np.abs(ev.Electron.eta) < 1.5) & (np.abs(ev.Electron.dz) < 0.1))
                | ((np.abs(ev.Electron.dxy) < 0.1) & (np.abs(ev.Electron.eta) >= 1.5) & (np.abs(ev.Electron.eta) < 2.4) & (np.abs(ev.Electron.dz) < 0.2))
                )
        )

        ev["ElectronGood"] = ev.Electron[mask_ele_loose & ~mask_ele_tight ]



        # Leptons (mu+e) and ordered in pt
        leptons = ak.with_name(
            ak.concatenate([ev.MuonGood, ev.ElectronGood], axis=1),
            "PtEtaPhiMCandidate",
        )

        loose_lep = ak.with_name(
            ak.concatenate([ev.MuonLoose, ev.ElectronLoose], axis=1),
            "PtEtaPhiMCandidate",
        )

        veto_lep = ak.with_name(
            ak.concatenate([ev.MuonVeto, ev.ElectronVeto], axis=1),
            "PtEtaPhiMCandidate",
        )

        ev["LeptonLoose"] = loose_lep[ak.argsort(loose_lep.pt, ascending=False)]
        ev["LeptonVeto"] = veto_lep[ak.argsort(veto_lep.pt, ascending=False)]
        ev["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        lead_lep = ak.firsts(ev.LeptonGood)
        lead_lep_loose = ak.firsts(ev.LeptonLoose)

        ev["lead_lep"] = lead_lep
        ev["MuonGoodLead"] = ak.firsts(ev.MuonGood[ak.argsort(ev.MuonGood.pt, ascending=False)])
        ev["ElectronGoodLead"] = ak.firsts(ev.ElectronGood[ak.argsort(ev.ElectronGood.pt, ascending=False)])
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
        for i in range(max(1, int(ak.max(ak.num(lepton_sorted, axis=1), initial=0)))):  # always define at least lepton1
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


        jets_sorted = ev.JetGood[ak.argsort(ev.JetGood.pt, ascending=False)]
        lepton_sorted = ev.LeptonGood[ak.argsort(ev.LeptonGood.pt, ascending=False)]
        n_jets = max(4, int(np.max(ak.num(jets_sorted, axis=1))))
        for i in range(n_jets):
            ev[f'jet{i+1}'] = ak.firsts(jets_sorted[:, i:i+1])
        for i in range(max(1, int(ak.max(ak.num(lepton_sorted, axis=1), initial=0)))):  # always define at least lepton1
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
