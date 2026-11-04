# workflow.py
import awkward as ak
import numpy as np
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator


import vector


vector.register_awkward()





class VBSSemileptonicProcessor(BaseProcessorABC):
    """
        - Build LeptonGood and JetGood (lepton-clean)
        - Identifies VBS tagging jets as the pair with the highest mjj
        - Reconstructs the hadronic W with two non-VBS jets that minimize |m-80.4|
        - Calculates auxiliary variables for histograms (mt, pt/eta, dR, etc.)
    """

    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    # 1) object-level preselection
    def apply_object_preselection(self, variation):
        ev = self.events
        ev["Electron", "etaSC"] = ev.Electron.eta + ev.Electron.deltaEtaSC

        # Good Leptons

        ev["MuonGood"]     = ev.Muon[(ev.Muon.pt > 26.0) & (np.abs(ev.Muon.eta) < 2.4)]

        ev["ElectronGood"] = ev.Electron[( ev.Electron.pt > 35 ) & ( np.abs(ev.Electron.eta) < 2.4) ]


        leptons = ak.with_name(
            ak.concatenate([ev.MuonGood, ev.ElectronGood], axis=1),
            "PtEtaPhiMCandidate",
        )


        ev["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        ev["JetGood"] = ev.Jet[(ev.Jet.pt > 30.0) & (np.abs(ev.Jet.eta) < 4.7)]
        ev["FatJetGood"] = ev.FatJet[(ev.FatJet.pt > 200.0) & (np.abs(ev.FatJet.eta) < 2.4)]

    def count_objects(self, variation):
        ev = self.events
        ev["nLeptonGood"]   = ak.num(ev.LeptonGood)
        ev["nJetGood"]      = ak.num(ev.JetGood)
        ev["nFatJetGood"] = ak.num(ev.FatJetGood)