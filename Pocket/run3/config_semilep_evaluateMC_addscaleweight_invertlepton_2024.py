# config_semilep_evaluateMC_addscaleweight_invertlepton_2024.py
#
# Inverted-lepton (fake-lepton) version of config_semilep_evaluateMC_addscaleweight_2024.py.
# Runs on the skimmed NanoAOD of config_2024_skim.py (USE_SKIMMED_DATASETS, below
# the sample lists); the skim only cuts on raw lepton pT/eta (no ID/isolation),
# so the inverted-isolation leptons are kept. W+jets pT(lnu) < 40 stitching:
# config_semilep_evaluateMC_addscaleweight_invertlepton_2024_stitch.py.
#
# Not applied yet (to be derived for 2024): QvG SF, WvsQCD/ZvsQCD SF, tau21 SF.
# Not applied: sf_btag -- BTV provides only fixed-WP UParTAK4 SFs for 2024 (no
# shape SF); the b-tag efficiency histograms below are for deriving the maps.
#
# Requires PocketCoffea branch update-upstream-2024
#   (PYTHONPATH=/afs/cern.ch/work/j/jinw/public/PocketCoffea_update).
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_nPVgood, goldenJson, eventFlags, get_JetVetoMap
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import HistConf, Axis
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.weights.common.weights_run3 import SF_ele_trigger
from pocket_coffea.parameters import defaults
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation, WeightLambda
import pocket_coffea.lib.jets as _pc_jets
import json
import numpy as np
if not hasattr(_pc_jets, "get_jersmear_SFunc"):
    raise RuntimeError(
        "This PocketCoffea cannot evaluate the 2024 JER uncertainties (new SF format): "
        "use PYTHONPATH=/afs/cern.ch/work/j/jinw/public/PocketCoffea_update"
    )

import workflow_invertlepton_2024, deepmet_corrections, custom_cut_functions, custom_cut_functions_2024, lhe_weights
from workflow_invertlepton_2024 import VBSSemileptonicProcessor
from lhe_weights import LHEScaleWeightWrapper, LHEPdfWeightWrapper, LHEPdfWeightEigenWrapper
from custom_cut_functions_2024 import (
    trigger_pd_overlap_cut,
    vbs_semileptonic_corrmet_presel,
    vbs_semileptonic_w_pt_corrmet_stitch_presel
)

from custom_cut_functions import (
    nLepton_skim_cut,
    nJet_skim_cut,
    whad_window_cut_bveto_e,
    msd_window_cut_bveto_e,
    whad_windowinvert_cut_bveto_e,
    msd_windowinvert_cut_bveto_e,
    whad_window_cut_baccept_e,
    msd_window_cut_baccept_e,
    whad_window_cut_bveto_mu,
    msd_window_cut_bveto_mu,
    whad_windowinvert_cut_bveto_mu,
    msd_windowinvert_cut_bveto_mu,
    whad_window_cut_baccept_mu,
    msd_window_cut_baccept_mu
)

for _mod in (workflow_invertlepton_2024, deepmet_corrections, custom_cut_functions, custom_cut_functions_2024, lhe_weights):
    cloudpickle.register_pickle_by_value(_mod)

localdir = os.path.dirname(os.path.abspath(__file__))
DEEPMET_CORRECTIONS = {
    "xy_json": f"{localdir}/output_METcorr_2024/deepmet_xy/deepmet_xy_corrections_2024.json",
    "smear_json": f"{localdir}/output_METcorr_applied_2024/deepmet_smearing/deepmet_xy_smearing_2024.json",
}
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_run3_invertlepton.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/plotting.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    f"{localdir}/params/jets_calibration_fullvar.yaml",
    f"{localdir}/params/fakelepton_weights_noiso_3j.yaml",
    f"{localdir}/params/classifiers_v8.yaml",
    f"{localdir}/params/classifiers_v8_2024.yaml",
    # Last: 2024 lumi, pileup, triggers, JERC, lepton SFs.
    # (params/lumi.yaml, pileup.yaml and lepton_scale_factors.yaml are NOT
    #  loaded: their 2024 entries are era-C-only lumi, the 2025 PU profile and
    #  a private muon SF file.)
    f"{localdir}/params/params_2024.yaml",
    update=True,
)

from coffea.lookup_tools import extractor


fake_muon_weights = {}
fake_electron_weights = {}
for y in parameters.fakeleptonweights.keys():
    ext = extractor()
    ext.add_weight_sets([
        f"muonFakeWeight {parameters.fakeleptonweights[y]['Muon']['nominal'][0]} {parameters.fakeleptonweights[y]['Muon']['file'][0]}",
        f"muonFakeWeight_kin_up {parameters.fakeleptonweights[y]['Muon']['kin_up'][0]} {parameters.fakeleptonweights[y]['Muon']['file_kin_up'][0]}",
        f"muonFakeWeight_kin_down {parameters.fakeleptonweights[y]['Muon']['kin_down'][0]} {parameters.fakeleptonweights[y]['Muon']['file_kin_down'][0]}",
        f"muonFakeWeight_contamination_up {parameters.fakeleptonweights[y]['Muon']['contamination_up'][0]} {parameters.fakeleptonweights[y]['Muon']['file_contamination_up'][0]}",
        f"muonFakeWeight_contamination_down {parameters.fakeleptonweights[y]['Muon']['contamination_down'][0]} {parameters.fakeleptonweights[y]['Muon']['file_contamination_down'][0]}",
        f"electronFakeWeight {parameters.fakeleptonweights[y]['Electron']['nominal'][0]} {parameters.fakeleptonweights[y]['Electron']['file'][0]}",
        f"electronFakeWeight_kin_up {parameters.fakeleptonweights[y]['Electron']['kin_up'][0]} {parameters.fakeleptonweights[y]['Electron']['file_kin_up'][0]}",
        f"electronFakeWeight_kin_down {parameters.fakeleptonweights[y]['Electron']['kin_down'][0]} {parameters.fakeleptonweights[y]['Electron']['file_kin_down'][0]}",
        f"electronFakeWeight_contamination_up {parameters.fakeleptonweights[y]['Electron']['contamination_up'][0]} {parameters.fakeleptonweights[y]['Electron']['file_contamination_up'][0]}",
        f"electronFakeWeight_contamination_down {parameters.fakeleptonweights[y]['Electron']['contamination_down'][0]} {parameters.fakeleptonweights[y]['Electron']['file_contamination_down'][0]}",
        ])
    ext.finalize()
    ev = ext.make_evaluator()
    fake_muon_weights[y] = {
        "nominal": ev[f"muonFakeWeight"],
        "kin_up": ev[f"muonFakeWeight_kin_up"],
        "kin_down": ev[f"muonFakeWeight_kin_down"],
        "contamination_up": ev[f"muonFakeWeight_contamination_up"],
        "contamination_down": ev[f"muonFakeWeight_contamination_down"],

    }
    fake_electron_weights[y] = {
        "nominal": ev[f"electronFakeWeight"],
        "kin_up": ev[f"electronFakeWeight_kin_up"],
        "kin_down": ev[f"electronFakeWeight_kin_down"],
        "contamination_up": ev[f"electronFakeWeight_contamination_up"],
        "contamination_down": ev[f"electronFakeWeight_contamination_down"],
    }

import awkward as ak
from pocket_coffea.lib.weights import WeightWrapper

class MuonGoodLeadWeight(WeightWrapper):
    name = "muon_inverttight_to_fake"
    has_variations = True
    isMC_only = False
    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)
        self._variations = ["kin", "contamination"]
    def compute(self, events, size, shape_variation):
        year = events.metadata["year"]
        mu = events.MuonGoodLead
        has_mu = ~ak.is_none(mu)
        # IMPORTANT: X=pt, Y=eta
        pt  = ak.where(has_mu, mu.pt, 0.0)
        eta = ak.where(has_mu, mu.eta, 0.0)
        pt  = ak.where(has_mu, np.clip(pt, 26.0, 100.0), pt)
        eta = ak.where(has_mu, np.clip(eta, -2.4, 2.4), eta)
        nominal = fake_muon_weights[year]["nominal"](pt, eta)
        nominal = ak.where(has_mu, nominal, 1.0)
        kin_up      = fake_muon_weights[year]["kin_up"](pt, eta)
        kin_up      = ak.where(has_mu, kin_up, 1.0)
        kin_up = ak.fill_none(kin_up,1.0)
        kin_down    = fake_muon_weights[year]["kin_down"](pt, eta)
        kin_down    = ak.where(has_mu, kin_down, 1.0)
        kin_down = ak.fill_none(kin_down,1.0)
        contamination_up      = fake_muon_weights[year]["contamination_up"](pt, eta)
        contamination_up      = ak.where(has_mu, contamination_up, 1.0)
        contamination_up = ak.fill_none(contamination_up,1.0)
        contamination_down    = fake_muon_weights[year]["contamination_down"](pt, eta)
        contamination_down    = ak.where(has_mu, contamination_down, 1.0)
        contamination_down = ak.fill_none(contamination_down,1.0)

        if shape_variation == "nominal":
            return WeightDataMultiVariation(
                name=self.name,
                nominal=nominal,
                variations=["kin", "contamination"],
                up=[kin_up,
                    contamination_up],
                down=[kin_down,
                    contamination_down],
            )
        else:
            return WeightData(name=self.name, nominal=nominal)

class ElectronGoodLeadWeight(WeightWrapper):
    name = "electron_inverttight_to_fake"
    has_variations = True
    isMC_only = False
    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)
        self._variations = ["kin", "contamination"]
    def compute(self, events, size, shape_variation):
        year = events.metadata["year"]
        ele = events.ElectronGoodLead
        has_ele = ~ak.is_none(ele)
        # IMPORTANT: X=pt, Y=eta
        pt  = ak.where(has_ele, ele.pt, 0.0)
        eta = ak.where(has_ele, ele.eta, 0.0)
        pt  = ak.where(has_ele, np.clip(pt, 35.0, 100.0), pt)
        eta = ak.where(has_ele, np.clip(eta, -2.4, 2.4), eta)
        nominal = fake_electron_weights[year]["nominal"](pt, eta)
        nominal = ak.where(has_ele, nominal, 1.0)
        kin_up      = fake_electron_weights[year]["kin_up"](pt, eta)
        kin_up      = ak.where(has_ele, kin_up, 1.0)
        kin_up = ak.fill_none(kin_up,1.0)
        kin_down    = fake_electron_weights[year]["kin_down"](pt, eta)
        kin_down    = ak.where(has_ele, kin_down, 1.0)
        kin_down = ak.fill_none(kin_down,1.0)
        contamination_up      = fake_electron_weights[year]["contamination_up"](pt, eta)
        contamination_up      = ak.where(has_ele, contamination_up, 1.0)
        contamination_up = ak.fill_none(contamination_up,1.0)
        contamination_down    = fake_electron_weights[year]["contamination_down"](pt, eta)
        contamination_down    = ak.where(has_ele, contamination_down, 1.0)
        contamination_down = ak.fill_none(contamination_down,1.0)

        if shape_variation == "nominal":
            return WeightDataMultiVariation(
                name=self.name,
                nominal=nominal,
                variations=["kin", "contamination"],
                up=[kin_up,
                    contamination_up],
                down=[kin_down,
                    contamination_down],
            )
        else:
            return WeightData(name=self.name, nominal=nominal)


########################################################################
# Samples (names as in the dataset JSONs' metadata "sample")
########################################################################

TAIL = "_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"

DY_SAMPLES = [f"DYto2{l}-2Jets_Bin-{b}{TAIL}" for l in ("E", "Mu", "Tau")
              for b in ("MLL-10to50", "0J-MLL-50", "1J-MLL-50", "2J-MLL-50")]
# W+jets: flavour-split 0J + flavour-inclusive pT(lnu)-binned 1J/2J (pT > 40).
# The pT(lnu) < 40 part of 1J/2J comes from W_LOWPT_SAMPLES in the _stitch config.
W_SAMPLES = (
    [f"Wto{l}Nu-2Jets_Bin-0J{TAIL}" for l in ("E", "Mu", "Tau")]
    + [f"WtoLNu-2Jets_Bin-{j}J-PTLNu-{pt}{TAIL}" for j in (1, 2)
       for pt in ("40to100", "100to200", "200to400", "400to600", "600")]
)
W_LOWPT_SAMPLES = [f"Wto{l}Nu-2Jets_Bin-{j}J{TAIL}" for l in ("E", "Mu", "Tau") for j in (1, 2)]
TOP_SAMPLES = [
    "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
    "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
    "TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
    "TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
    "TBbarto2Q-s-channel_TuneCP5_13p6TeV_powheg-pythia8",
    "TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8",
    "TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8",
    "TbarWplus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8",
    "TWminus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8",
]
OTHER_BKG_SAMPLES = [
    "ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8",
    f"WGtoLNuG-1Jets{TAIL}",
    "VBFtoLNu_TuneCP5_13p6TeV_madgraph-pythia8",
]
# Private VV samples (EWK signal + QCD-VV background). The 2024 production is
# stored as NANOEDMAODSIM, which coffea cannot read; the dataset JSONs point to
# the flat NanoAOD conversion (private_2024_conversion/) in
# eosuser /eos/cms/store/group/phys_smp/jinw/2024_VBS_sig.
PRIVATE_VV_FLAT_READY = True
VV_TYPES = ["ssWWunpolarized", "osWWunpolarized_Wptojj_Wmtolv", "osWWunpolarized_Wptolv_Wmtojj",
            "WZunpolarized_Wmtolv_Ztojj", "WZunpolarized_Wptolv_Ztojj"]
SIGNAL_SAMPLES = [f"{v}{TAIL}" for v in VV_TYPES]
QCDVV_SAMPLES = [f"{v}_QCD{TAIL}" for v in VV_TYPES]
DATA_SAMPLES = ["Muon0", "Muon1", "EGamma0", "EGamma1"]

# Main config: W+jets = flavour-split 0J + PTLNu-binned 1J/2J. The pT(lnu) < 40
# part of 1J/2J (W_LOWPT_SAMPLES) is run by the _stitch config.
MC_SAMPLES = DY_SAMPLES + W_SAMPLES + TOP_SAMPLES + OTHER_BKG_SAMPLES
if PRIVATE_VV_FLAT_READY:
    MC_SAMPLES += SIGNAL_SAMPLES + QCDVV_SAMPLES

INCLUDE_DATA = True     # process DATA_SAMPLES together with the MC in this config

########################################################################
# Input: skimmed NanoAOD
########################################################################
# FNAL skim semilep_vbs_skim_run3_v2 (config_2024_skim.py cuts), hadd-ed; the 28
# hadd outputs missing at FNAL were redone on eosuser (2024_skim_v2_hadd_fix).
# The MC normalisation uses the pre-skim sum_genweights stored in this JSON,
# so every file of a dataset must be processed (see MC_MAX_FILES below).
USE_SKIMMED_DATASETS = True
SKIMMED_DATASETS_JSON = f"{localdir}/datasets/skimmed_dataset_definition_hadd_2024_new_fixed.json"

DATA_ERAS = ["C", "D", "E", "F", "G", "H", "I"]   # all of 2024
MC_MAX_FILES = 0        # 0 = all files (required with USE_SKIMMED_DATASETS)

ALL_ERAS = ["C", "D", "E", "F", "G", "H", "I"]
if DATA_ERAS:
    unknown = set(DATA_ERAS) - set(ALL_ERAS)
    if unknown:
        raise ValueError(f"Unknown 2024 eras {unknown}, choose from {ALL_ERAS}")
    lumi_2024 = parameters.lumi.picobarns["2024"]
    parameters.lumi.picobarns["2024"]["tot"] = float(sum(lumi_2024[e] for e in DATA_ERAS))


def _dataset_json(sample):
    """Dataset JSON for a sample, restricted to the requested eras / number of files.

    The restricted copies are written to datasets/subsets/ (the original JSONs are
    never modified)."""
    path = f"{localdir}/datasets/{sample}.json"
    is_data = sample in DATA_SAMPLES
    if not (is_data and DATA_ERAS) and not (not is_data and MC_MAX_FILES):
        return path
    with open(path) as f:
        content = json.load(f)
    out = {}
    for name, ds in content.items():
        if ds["metadata"].get("year") != "2024":
            continue
        if is_data:
            if ds["metadata"].get("era") not in DATA_ERAS:
                continue
        else:
            ds = {**ds, "files": ds["files"][:MC_MAX_FILES]}
        out[name] = ds
    tag = f"eras{''.join(DATA_ERAS)}" if is_data else f"max{MC_MAX_FILES}files"
    subset_dir = f"{localdir}/datasets/subsets"
    os.makedirs(subset_dir, exist_ok=True)
    subset_path = f"{subset_dir}/{sample}_2024_{tag}.json"
    with open(subset_path, "w") as f:
        json.dump(out, f, indent=4)
    return subset_path


def _skimmed_dataset_json(samples, tag, data_samples=None, skimmed_json=None):
    """Subset of the skimmed dataset definition holding only `samples` (data
    restricted to DATA_ERAS), written to datasets/subsets/skimmed_2024_<tag>.json.

    Raises if a requested sample is not in the skimmed JSON, and if MC_MAX_FILES
    is set: the skimmed MC is normalised with the pre-skim sum_genweights of the
    full dataset (dataset metadata), so processing a subset of its files would
    silently under-normalise it."""
    data_samples = DATA_SAMPLES if data_samples is None else data_samples
    skimmed_json = skimmed_json or SKIMMED_DATASETS_JSON
    if MC_MAX_FILES and any(s not in data_samples for s in samples):
        raise ValueError("MC_MAX_FILES must be 0 with skimmed datasets: their MC "
                         "normalisation uses the pre-skim sum_genweights of all files.")
    with open(skimmed_json) as f:
        content = json.load(f)
    out = {}
    for name, ds in content.items():
        md = ds["metadata"]
        if md["sample"] not in samples or md.get("year") != "2024":
            continue
        if md["sample"] in data_samples and DATA_ERAS and md.get("era") not in DATA_ERAS:
            continue
        out[name] = ds
    found = {ds["metadata"]["sample"] for ds in out.values()}
    missing = [s for s in samples if s not in found]
    if missing:
        raise ValueError(f"Samples not found in {skimmed_json}: {missing}")
    subset_dir = f"{localdir}/datasets/subsets"
    os.makedirs(subset_dir, exist_ok=True)
    subset_path = f"{subset_dir}/skimmed_2024_{tag}.json"
    with open(subset_path, "w") as f:
        json.dump(out, f, indent=4)
    return subset_path









########################################################################
# Weights
########################################################################
# PDF uncertainty from the 100 Hessian eigenvectors of LHEPdfWeight (lhe_weights.py):
#   "combined": LHEPdfWeight      -> pdf = 1 +- sqrt(sum_i (w_i-1)^2), alpha_S
#   "eigen":    LHEPdfWeightEigen -> pdf_1..pdf_100 (up w_i, down 2-w_i), alpha_S
#   "both":     both sets of modifiers (pick one in the datacard; alpha_S appears twice)
# "eigen" adds 2x100 weight variations to every histogram.
#PDF_UNCERTAINTY = "combined"
#PDF_WEIGHTS = {
#    "combined": ["LHEPdfWeight"],
#    "eigen": ["LHEPdfWeightEigen"],
#    "both": ["LHEPdfWeight", "LHEPdfWeightEigen"],
#}[PDF_UNCERTAINTY]

#WEIGHTS = ["genWeight", "lumi", "XS", "pileup",
#           "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
#           "sf_ele_id", "sf_ele_reco", "sf_ele_trigger",
#           "LHEScaleWeight", *PDF_WEIGHTS, "sf_partonshower_isr", "sf_partonshower_fsr"]
#WEIGHT_VARIATIONS = ["pileup",
#                     "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
#                     "sf_ele_id", "sf_ele_reco", "sf_ele_trigger",
#                     "LHEScaleWeight", *PDF_WEIGHTS, "sf_partonshower_isr", "sf_partonshower_fsr"]
#SHAPE_VARIATIONS = ["jet_calibration", "electron_scale_and_smearing", "muons_scale_and_resolution"]

WEIGHTS = ["genWeight", "lumi", "XS", "pileup",
           "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
           "sf_ele_id", "sf_ele_reco", "sf_ele_trigger",
           "muon_inverttight_to_fake","electron_inverttight_to_fake"]
WEIGHT_VARIATIONS = ["muon_inverttight_to_fake","electron_inverttight_to_fake"]
SHAPE_VARIATIONS = []




SKIM = [
    get_nPVgood(1),
    eventFlags,
    goldenJson,
    nLepton_skim_cut,
    nJet_skim_cut,
    trigger_pd_overlap_cut,   # IsoMu24 / Ele30_WPTight_Gsf, Muon-EGamma overlap removed in data
    get_JetVetoMap(),
]

_RESOLVED_BINS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
_BOOSTED_BINS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
_BOOSTED_WWvsWZ_BINS = [0.0, 0.075, 0.15, 0.25, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
_RESOLVED_WWvsWZ_BINS = [0.0, 0.35, 0.4, 0.425, 0.45, 0.5, 0.55, 0.575, 0.6, 0.625, 0.65, 1.0]


VARIABLES = {
    "nJets":      HistConf([Axis(coll="events", field="nJetGood", bins=12, start=0, stop=12, label="N(jets)")]),
    "nBJetTight":     HistConf([Axis(coll="events", field="nBJetTight", bins=8, start=0, stop=8, label="N(b jets UParT tight)")]),
    "nBJetLoose":    HistConf([Axis(coll="events", field="nBJetLoose", bins=8, start=0, stop=8, label="N(b jets UParT loose)")]),
    "nCentralJets": HistConf([Axis(coll="events", field="nCentralJetsGood", bins=12, start=0, stop=12, label="N(Central Jets)")]),
    "nFatJets": HistConf([Axis(coll="events", field="nFatJetGood", bins=4, start=0, stop=4, label="N(Fat Jets)")]),
    "nLeptonLoose":      HistConf([Axis(coll="events", field="nLeptonLoose", bins=4, start=0, stop=4, label="N(Lepton Loose)")]),
    "nFatJetCandidate": HistConf([Axis(coll="events", field="nFatJetCandidate", bins=4, start=0, stop=4, label="N(Candidate Fat Jets)")]),
    "nMuonGood":     HistConf([Axis(coll="events", field="nMuonGood", bins=6, start=0, stop=6, label="N(muon good)")]),
    "nElectronGood":      HistConf([Axis(coll="events", field="nElectronGood", bins=6, start=0, stop=6, label="N(electron good)")]),
    "nLeptonGood":    HistConf([Axis(coll="events", field="nLeptonGood", bins=6, start=0, stop=6, label="N(lepton good)")]),

    'npvs':   HistConf([Axis(coll="PV", field="npvs", bins=20, start=0, stop=100, label="nPV")]),
    'npvsGood':   HistConf([Axis(coll="PV", field="npvsGood", bins=20, start=0, stop=100, label=r"$nPV_{good}$")]),
    # NanoAODv15: the PF MET collection is called PFMET
    "met":        HistConf([Axis(coll="PFMET", field="pt", bins=50, start=0, stop=250, label=r"PF $p_T^{miss}$ [GeV]")]),
    "met_phi":    HistConf([Axis(coll="PFMET", field="phi", bins=50, start=-4, stop=4, label=r"PF $\phi^{miss}$")]),
    "puppimet_phi":    HistConf([Axis(coll="PuppiMET", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]")]),
    "puppimet":        HistConf([Axis(coll="PuppiMET", field="pt", bins=50, start=0, stop=250, label=r"$p_T^{miss}$ [GeV]")]),
    "deepmet_resolution_tune_phi":    HistConf([Axis(coll="DeepMETResolutionTune", field="phi", bins=50, start=-4, stop=4, label=r"deep $ \phi^{miss}$ resolution tune [GeV]")]),
    "deepmet_resolution_tune":        HistConf([Axis(coll="DeepMETResolutionTune", field="pt", bins=50, start=0, stop=250, label=r"deep $p_T^{miss}$ resolution tune [GeV]")]),
    "deepmet_response_tune_phi":    HistConf([Axis(coll="DeepMETResponseTune", field="phi", bins=50, start=-4, stop=4, label=r"deep $ \phi^{miss}$ response tune [GeV]")]),
    "deepmet_response_tune":        HistConf([Axis(coll="DeepMETResponseTune", field="pt", bins=50, start=0, stop=250, label=r"deep $p_T^{miss}$ response tune [GeV]")]),
    "mt_w_lep":   HistConf([Axis(coll="events", field="mt_w_leptonic", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ [GeV]")]),
    "mt_w_lep_deepresolution": HistConf([Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ (deep MET resolution tune) [GeV]")]),
    "mt_w_lep_deepresponse": HistConf([Axis(coll="events", field="mt_w_leptonic_deepMET_responsetune", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ (deep MET response tune) [GeV]")]),
    # DeepMET resolution tune after xy offset + MC smearing
    "deepmet_resolution_tune_corr":     HistConf([Axis(coll="DeepMETResolutionTuneCorr", field="pt", bins=50, start=0, stop=250, label=r"DeepMET res. tune corr. $p_T^{miss}$ [GeV]")]),
    "deepmet_resolution_tune_corr_phi": HistConf([Axis(coll="DeepMETResolutionTuneCorr", field="phi", bins=32, start=-3.2, stop=3.2, label=r"DeepMET res. tune corr. $\phi^{miss}$")]),
    "mt_w_lep_deepresolution_corr": HistConf([Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune_corr", bins=40, start=0, stop=200, label=r"$m_T(W_{lep})$ (DeepMET res. tune corr.) [GeV]")]),
    "neutrino_pz":  HistConf([Axis(coll="events", field="neutrino_pz", bins=50, start=0, stop=250, label=r"$p_z^{\nu}$ [GeV]")]),
    "neutrino_eta":  HistConf([Axis(coll="events", field="neutrino_eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{\nu}$ [GeV]")]),

    "mjj_vbs":    HistConf([Axis(coll="vbsjets", field="mass", bins=50, start=0, stop=4000, label=r"$M_{jj}^{VBS}$ [GeV]")]),
    "deta_vbs":   HistConf([Axis(coll="vbsjets", field="delta_eta", bins=36, start=0, stop=9.0, label=r"$|\Delta\eta_{jj}^{VBS}|$")]),

    # NanoAODv15 has no Jet_jetId: plot the recomputed one
    "jet_id":   HistConf([Axis(coll="JetGood", field="jetId_corrected", bins=10, start=0, stop=10, label="Jet id")]),
    "jet_rel_iso":  HistConf([Axis(coll="LeptonGood", field="jetRelIso", bins=50, start=0, stop=2, label="Jet iso in lep")]),
    "dxy_mu":   HistConf([Axis(coll="LeptonGood", field="dxy", bins=50, start=0, stop=0.5, label="dxy mu")]),
    "dxy_ele":  HistConf([Axis(coll="LeptonGood", field="dxy", bins=50, start=0, stop=0.2, label="dxy ele")]),
    "dz_mu":    HistConf([Axis(coll="LeptonGood", field="dz", bins=50, start=0, stop=1, label="dz mu")]),
    "dz_ele":   HistConf([Axis(coll="LeptonGood", field="dz", bins=50, start=0, stop=0.5, label="dz ele")]),

    "m_jj_w":     HistConf([Axis(coll="w_had_jets", field="mass", bins=80, start=0, stop=150, label=r"$M_{jj}^{W\,had}$ [GeV]")]),
    "pt_jj_w":     HistConf([Axis(coll="w_had_jets", field="pt", bins=40, start=0, stop=210, label=r"$p_T(jj^{W\,had})$ [GeV]")]),
    "dR_w_had":   HistConf([Axis(coll="w_had_jets", field="dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(jj)^{W\,had}$")]),
    "eta_w_had1":   HistConf([Axis(coll="w_had_jet1", field="eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j1_{W\,had})$")]),
    "eta_w_had2":   HistConf([Axis(coll="w_had_jet2", field="eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j2_{W\,had})$")]),
    "pt_w_had1":   HistConf([Axis(coll="w_had_jet1", field="pt", bins=60, start=0.0, stop=300.0, label=r"$p_T(j1_{W\,had})$ [GeV]")]),
    "pt_w_had2":   HistConf([Axis(coll="w_had_jet2", field="pt", bins=60, start=0.0, stop=300.0, label=r"$p_T(j2_{W\,had})$ [GeV]")]),
    "phi_w_had1":   HistConf([Axis(coll="w_had_jet1", field="phi", bins=48, start=-4.0, stop=4.0, label=r"$\phi(j1_{W\,had})$")]),
    "phi_w_had2":   HistConf([Axis(coll="w_had_jet2", field="phi", bins=48, start=-4.0, stop=4.0, label=r"$\phi(j2_{W\,had})$")]),
    "pt_tag1":    HistConf([Axis(coll="jet1", field="pt", bins=60, start=0, stop=300, label=r"$p_T(j_1)$ [GeV]")]),
    "pt_tag2":    HistConf([Axis(coll="jet2", field="pt", bins=60, start=0, stop=300, label=r"$p_T(j_2)$ [GeV]")]),
    "eta_tag1":   HistConf([Axis(coll="jet1", field="eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)$")]),
    "eta_tag2":   HistConf([Axis(coll="jet2", field="eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)$")]),
    "phi_tag1":   HistConf([Axis(coll="jet1", field="phi", bins=48, start=-4., stop=4., label=r"$\phi(j_1)$")]),
    "phi_tag2":   HistConf([Axis(coll="jet2", field="phi", bins=48, start=-4., stop=4., label=r"$\phi(j_2)$")]),
    "eta_lead_lep":   HistConf([Axis(coll="lead_lep", field="eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{lead\ lep}$ ")]),
    "pt_lead_lep":   HistConf([Axis(coll="lead_lep", field="pt", bins=40, start=0.0, stop=300.0, label=r"$p_T^{lead\ lep}$ [GeV]")]),
    "phi_lead_lep":   HistConf([Axis(coll="lead_lep", field="phi", bins=32, start=-4.0, stop=4.0, label=r"$\phi^{lead\ lep}$ ")]),
    "wleptonic_eta":    HistConf([Axis(coll="events", field="wleptonic_eta", bins=48, start=-2.4, stop=2.4,   label=r"$\eta(W_{lep})$")]),
    "wleptonic_pt":    HistConf([Axis(coll="events", field="wleptonic_pt", bins=40, start=0.0, stop=300.0,   label=r"$p_T(W_{lep})$")]),
    "wleptonic_phi":    HistConf([Axis(coll="events", field="wleptonic_phi", bins=32, start=-4.0, stop=4.0,   label=r"$\phi(W_{lep})$")]),
    "fj_pt":    HistConf([Axis(coll="candidate_boost1", field="pt",  bins=60, start=150, stop=1000, label=r"$p_T(J^{W})$ [GeV]")]),
    "fj_eta":   HistConf([Axis(coll="candidate_boost1", field="eta", bins=48, start=-2.4, stop=2.4,   label=r"$\eta(J^{W})$")]),
    "fj_msd":   HistConf([Axis(coll="candidate_boost1", field="msoftdrop", bins=40, start=0,   stop=200,   label=r"$m_{SD}(J^{W})$ [GeV]")]),
    "fj_t21":   HistConf([Axis(coll="candidate_boost1", field="tau21", bins=32, start=0, stop=1.1,   label=r"$\tau_{21}$")]),
    "fj_XqqVsQCD":   HistConf([Axis(coll="candidate_boost1", field="particleNet_XqqVsQCD", bins=40, start=0, stop=1,   label=r"J^{W} XqqVsQCD")]),
    "fj_WvsQCD":   HistConf([Axis(coll="candidate_boost1", field="particleNetWithMass_WvsQCD", bins=40, start=0, stop=1,   label=r"J^{W} WvsQCD (no SF)")]),
    "fj_ZvsQCD":   HistConf([Axis(coll="candidate_boost1", field="particleNetWithMass_ZvsQCD", bins=40, start=0, stop=1,   label=r"J^{W} ZvsQCD (no SF)")]),
    "z_lep":   HistConf([Axis(coll="events", field="z_lep", bins=40, start=-1.0, stop=1.0, label=r"$Zepp. lepton$")]),
    "z_fat":      HistConf([Axis(coll="events", field="z_fat", bins=40, start=-1.0, stop=1.0, label=r"$Zepp. boosted jet$")]),
    "centrality_resolved":  HistConf([Axis(coll="events", field="centrality_resolved", bins=40, start=-5.0, stop=5.0, label=r"$Centrality_{resolved}$")]),
    "centrality_boosted":   HistConf([Axis(coll="events", field="centrality_boosted", bins=40, start=-5.0, stop=5.0, label=r"$Centrality_{boosted}$")]),
    "w_had_jet1_resolved_PNetQvG": HistConf([Axis(coll="w_had_jet1", field="btagPNetQvG", bins=40, start=0, stop=1.0, label=r"$ParticleNetQvG(W^{had.} j_1)$ (no SF)")]),
    "w_had_jet2_resolved_PNetQvG": HistConf([Axis(coll="w_had_jet2", field="btagPNetQvG", bins=40, start=0, stop=1.0, label=r"$ParticleNetQvG(W^{had.} j_2)$ (no SF)")]),
    "w_had_jet1_resolved_DeepFlavQG": HistConf([Axis(coll="w_had_jet1", field="btagDeepFlavQG", bins=40, start=0, stop=1.0, label=r"$DeepJetGvQ(W^{had.} j_1)$")]),
    "w_had_jet2_resolved_DeepFlavQG": HistConf([Axis(coll="w_had_jet2", field="btagDeepFlavQG", bins=40, start=0, stop=1.0, label=r"$DeepJetGvQ(W^{had.} j_2)$")]),
    "pt_vbsjet1":    HistConf([Axis(coll="vbsjet1", field="pt", bins=60, start=0, stop=300, label=r"$p_T(j_1)^{VBS}$ [GeV]")]),
    "pt_vbsjet2":    HistConf([Axis(coll="vbsjet2", field="pt", bins=60, start=0, stop=300, label=r"$p_T(j_2)^{VBS}$ [GeV]")]),
    "eta_vbsjet1":   HistConf([Axis(coll="vbsjet1", field="eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)^{VBS}$")]),
    "eta_vbsjet2":   HistConf([Axis(coll="vbsjet2", field="eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)^{VBS}$")]),
    "phi_vbsjet1":   HistConf([Axis(coll="vbsjet1", field="phi", bins=48, start=-4., stop=4., label=r"$\phi(j_1)^{VBS}$")]),
    "phi_vbsjet2":   HistConf([Axis(coll="vbsjet2", field="phi", bins=48, start=-4., stop=4., label=r"$\phi(j_2)^{VBS}$")]),
    "PNetQvG_vbsjet1":  HistConf([Axis(coll="vbsjet1", field="btagPNetQvG", bins=50, start=0, stop=1, label=r"$ParticleNetQvG(j_1)^{VBS}$ (no SF)")]),
    "PNetQvG_vbsjet2":  HistConf([Axis(coll="vbsjet2", field="btagPNetQvG", bins=50, start=0, stop=1, label=r"$ParticleNetQvG(j_2)^{VBS}$ (no SF)")]),
    "DeepFlavQG_vbsjet1":  HistConf([Axis(coll="vbsjet1", field="btagDeepFlavQG", bins=50, start=0, stop=1, label=r"$DeepJetGvQ(j_1)^{VBS}$")]),
    "DeepFlavQG_vbsjet2":  HistConf([Axis(coll="vbsjet2", field="btagDeepFlavQG", bins=50, start=0, stop=1, label=r"$DeepJetGvQ(j_2)^{VBS}$")]),
    "bjet_pt":    HistConf([Axis(coll="BJetLoose", field="pt", bins=60, start=0, stop=300, label=r"$p_T(b loose)$ [GeV]")]),
    "bjet_eta":   HistConf([Axis(coll="BJetLoose", field="eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(b loose)$")]),
    "bjet_phi":   HistConf([Axis(coll="BJetLoose", field="phi", bins=48, start=-4., stop=4., label=r"$\phi(b loose)$")]),

    # BDTs: 2022-trained models (params/classifiers_v8_2024.yaml), no tagger SFs
    "bdt_boosted_mu_2D":       HistConf([
        Axis(coll="events", field="bdt_boosted_mu", bins=_BOOSTED_BINS, label="BDT mu boosted"),
        Axis(coll="events", field="bdt_boosted_WW_WZ", bins=_BOOSTED_WWvsWZ_BINS, label="BDT boosted WW vs WZ")]),
    "bdt_resolved_mu_2D":       HistConf([
        Axis(coll="events", field="bdt_resolved_mu", bins=_RESOLVED_BINS, label="BDT mu resolved (WW)"),
        Axis(coll="events", field="bdt_resolved_WW_WZ", bins=_RESOLVED_WWvsWZ_BINS, label="BDT resolved WW vs WZ")]),
    "bdt_boosted_e_2D":       HistConf([
        Axis(coll="events", field="bdt_boosted_e", bins=_BOOSTED_BINS, label="BDT e boosted (WW)"),
        Axis(coll="events", field="bdt_boosted_WW_WZ", bins=_BOOSTED_WWvsWZ_BINS, label="BDT boosted WW vs WZ")]),
    "bdt_resolved_e_2D":       HistConf([
        Axis(coll="events", field="bdt_resolved_e", bins=_RESOLVED_BINS, label="BDT e resolved (WW)"),
        Axis(coll="events", field="bdt_resolved_WW_WZ", bins=_RESOLVED_WWvsWZ_BINS, label="BDT resolved WW vs WZ")]),
    "bdt_boosted_mu":       HistConf([Axis(coll="events", field="bdt_boosted_mu", bins=_BOOSTED_BINS, label="BDT mu boosted")]),
    "bdt_resolved_mu":       HistConf([Axis(coll="events", field="bdt_resolved_mu", bins=_RESOLVED_BINS, label="BDT mu resolved")]),
    "bdt_boosted_e":       HistConf([Axis(coll="events", field="bdt_boosted_e", bins=_BOOSTED_BINS, label="BDT e boosted")]),
    "bdt_resolved_e":       HistConf([Axis(coll="events", field="bdt_resolved_e", bins=_RESOLVED_BINS, label="BDT e resolved")]),

}


def make_cfg(mc_samples, preselection, include_data=True, tag="invertlepton"):
    process_samples = (list(DATA_SAMPLES) if include_data else []) + list(mc_samples)
    if USE_SKIMMED_DATASETS:
        process_jsons = [_skimmed_dataset_json(process_samples, tag)]
    else:
        process_jsons = [_dataset_json(s) for s in process_samples]
    return Configurator(
        parameters=parameters,
        datasets={
            "jsons": process_jsons,
            "filter": {
                "samples": process_samples,
                "year": ["2024"],
            },
        },
        workflow=VBSSemileptonicProcessor,
        workflow_options={"deepmet_corrections": DEEPMET_CORRECTIONS},
        skim=SKIM,
        preselections=[preselection],
        categories={
            "baseline": [passthrough],
            "boosted_e": [msd_window_cut_bveto_e],
            "boosted_e_WCR": [msd_windowinvert_cut_bveto_e],
            "boosted_e_TTCR": [msd_window_cut_baccept_e],
            "resolved_e":  [whad_window_cut_bveto_e],
            "resolved_e_WCR" :[whad_windowinvert_cut_bveto_e],
            "resolved_e_TTCR" :[whad_window_cut_baccept_e],
            "boosted_mu": [msd_window_cut_bveto_mu],
            "resolved_mu":  [whad_window_cut_bveto_mu],
            "resolved_mu_WCR" :[whad_windowinvert_cut_bveto_mu],
            "resolved_mu_TTCR" :[whad_window_cut_baccept_mu],
            "boosted_mu_WCR": [msd_windowinvert_cut_bveto_mu],
            "boosted_mu_TTCR": [msd_window_cut_baccept_mu],
        },
        weights_classes=common_weights + [SF_ele_trigger, LHEScaleWeightWrapper, LHEPdfWeightWrapper, LHEPdfWeightEigenWrapper, MuonGoodLeadWeight,ElectronGoodLeadWeight],
        weights={"common": {"inclusive": WEIGHTS}},
        variations={
            "weights": {"common": {"inclusive": WEIGHT_VARIATIONS}},
            "shape": {"common": {"inclusive": SHAPE_VARIATIONS}},
        },
        variables=VARIABLES,
    )


cfg = make_cfg(MC_SAMPLES, vbs_semileptonic_corrmet_presel, include_data=INCLUDE_DATA, tag="invertlepton")
