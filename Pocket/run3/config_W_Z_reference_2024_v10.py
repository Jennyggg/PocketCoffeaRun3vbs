# config_W_Z_reference_2024_v10.py
#
# 2024 data/MC check in standard W+jets and DY selections, with the same
# objects and corrections as config_semilep_evaluateMC_addscaleweight_2024.py
# (workflow_evaluate_2024). Replaces config_W_Z_reference_2024(_new).py, which
# used era C only, the 2025 pileup profile and no electron trigger SF.
#
# Requires PocketCoffea branch update-upstream-2024
#   (PYTHONPATH=/afs/cern.ch/work/j/jinw/public/PocketCoffea_update).
import os, json, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_nPVgood, goldenJson, eventFlags, get_JetVetoMap
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import HistConf, Axis
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.weights.common.weights_run3 import SF_ele_trigger
from pocket_coffea.parameters import defaults
import pocket_coffea.lib.jets as _pc_jets

if not hasattr(_pc_jets, "get_jersmear_SFunc"):
    raise RuntimeError(
        "This PocketCoffea cannot evaluate the 2024 JER uncertainties (new SF format): "
        "use PYTHONPATH=/afs/cern.ch/work/j/jinw/public/PocketCoffea_update"
    )

import workflow_evaluate_2024, custom_cut_functions, custom_cut_functions_2024
from workflow_evaluate_2024 import VBSSemileptonicProcessor
from custom_cut_functions_2024 import trigger_pd_overlap_cut
from custom_cut_functions import (
    nLepton_skim_cut,
    zjets_sel,
    zjets_e_sel,
    zjets_mu_sel,
    wjets_sel,
    wjets_e_sel,
    wjets_mu_sel,
)

for _mod in (workflow_evaluate_2024, custom_cut_functions, custom_cut_functions_2024):
    cloudpickle.register_pickle_by_value(_mod)

localdir = os.path.dirname(os.path.abspath(__file__))

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_run3.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/plotting.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    f"{localdir}/params/params_2024.yaml",
    update=True,
)

# Also run the JES/JER and lepton-scale shape variations (slower, ~x40 passes)
WITH_SHAPE_VARIATIONS = False

TAIL = "_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"
DY_SAMPLES = [f"DYto2{l}-2Jets_Bin-{b}{TAIL}" for l in ("E", "Mu", "Tau")
              for b in ("MLL-10to50", "0J-MLL-50", "1J-MLL-50", "2J-MLL-50")]
# Flavour-split jet-binned W+jets: complete and non-overlapping, no pT stitching needed
W_SAMPLES = [f"Wto{l}Nu-2Jets_Bin-{j}J{TAIL}" for l in ("E", "Mu", "Tau") for j in (0, 1, 2)]
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
DATA_SAMPLES = ["Muon0", "Muon1", "EGamma0", "EGamma1"]
SAMPLES = DY_SAMPLES + W_SAMPLES + TOP_SAMPLES + OTHER_BKG_SAMPLES + DATA_SAMPLES

########################################################################
# Quick-test switches
#   DATA_ERAS:    2024 data eras to process, e.g. ["C"] or ["C", "D"];
#                 MC is normalised to their summed luminosity.
#                 [] = all eras (C-I, full 2024 lumi).
#   MC_MAX_FILES: process at most N files per MC dataset; 0 = all files.
#                 (MC is normalised by the sum of genweights of the processed
#                 events, so this only costs statistics)
########################################################################
DATA_ERAS = ["C"]
MC_MAX_FILES = 50

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

print(f"[config_W_Z_reference_2024_v10] data eras: {DATA_ERAS or 'all'}, "
      f"MC lumi: {parameters.lumi.picobarns['2024']['tot']} pb-1, "
      f"MC files per dataset: {MC_MAX_FILES or 'all'}")

WEIGHTS = ["genWeight", "lumi", "XS", "pileup",
           "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
           "sf_ele_id", "sf_ele_reco", "sf_ele_trigger"]
WEIGHT_VARIATIONS = [w for w in WEIGHTS if w not in ("genWeight", "lumi", "XS")]
variations = {"weights": {"common": {"inclusive": WEIGHT_VARIATIONS}}}
if WITH_SHAPE_VARIATIONS:
    variations["shape"] = {"common": {"inclusive": ["jet_calibration", "electron_scale_and_smearing", "muons_scale_and_resolution"]}}

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [_dataset_json(s) for s in SAMPLES],
        "filter": {
            "samples": SAMPLES,
            "year": ["2024"],
        },
    },
    workflow=VBSSemileptonicProcessor,

    skim=[
        get_nPVgood(1),
        eventFlags,
        goldenJson,
        nLepton_skim_cut,
        trigger_pd_overlap_cut,   # IsoMu24 / Ele30_WPTight_Gsf, Muon-EGamma overlap removed in data
        get_JetVetoMap(),
    ],
    preselections=[passthrough],

    categories={
        "baseline": [passthrough],
        "wjets_reference": [wjets_sel],
        "wjets_e_reference": [wjets_e_sel],
        "wjets_mu_reference": [wjets_mu_sel],
        "zjets_reference": [zjets_sel],
        "zjets_e_reference": [zjets_e_sel],
        "zjets_mu_reference": [zjets_mu_sel],
    },

    weights_classes=common_weights + [SF_ele_trigger],
    weights={"common": {"inclusive": WEIGHTS}},
    variations=variations,

    variables={
        "nJets":      HistConf([Axis(coll="events", field="nJetGood", bins=12, start=0, stop=12, label="N(jets)")]),
        "nBJetTight":     HistConf([Axis(coll="events", field="nBJetTight", bins=8, start=0, stop=8, label="N(b jets UParT tight)")]),
        "nBJetLoose":    HistConf([Axis(coll="events", field="nBJetLoose", bins=8, start=0, stop=8, label="N(b jets UParT loose)")]),
        "nFatJets": HistConf([Axis(coll="events", field="nFatJetGood", bins=4, start=0, stop=4, label="N(Fat Jets)")]),
        "nFatJetCandidate": HistConf([Axis(coll="events", field="nFatJetCandidate", bins=4, start=0, stop=4, label="N(Candidate Fat Jets)")]),
        "nMuonGood":     HistConf([Axis(coll="events", field="nMuonGood", bins=6, start=0, stop=6, label="N(muon good)")]),
        "nElectronGood":      HistConf([Axis(coll="events", field="nElectronGood", bins=6, start=0, stop=6, label="N(electron good)")]),
        "nLeptonVeto":    HistConf([Axis(coll="events", field="nLeptonVeto", bins=6, start=0, stop=6, label="N(lepton veto)")]),
        'npvs':   HistConf([Axis(coll="PV", field="npvs", bins=50, start=0, stop=100, label="nPV")]),
        'npvsGood':   HistConf([Axis(coll="PV", field="npvsGood", bins=50, start=0, stop=100, label=r"$nPV_{good}$")]),
        'rho':    HistConf([Axis(coll="Rho", field="fixedGridRhoFastjetAll", bins=50, start=0, stop=60, label=r"$\rho$")]),

        # Leptons
        "pt_lep1":   HistConf([Axis(coll="lepton1", field="pt", bins=50, start=0.0, stop=250.0, label=r"$p_T(\ell_1)$ [GeV]")]),
        "eta_lep1":  HistConf([Axis(coll="lepton1", field="eta", bins=48, start=-2.4, stop=2.4, label=r"$\eta(\ell_1)$")]),
        "phi_lep1":  HistConf([Axis(coll="lepton1", field="phi", bins=32, start=-3.2, stop=3.2, label=r"$\phi(\ell_1)$")]),
        "pt_lep2":   HistConf([Axis(coll="lepton2", field="pt", bins=50, start=0.0, stop=250.0, label=r"$p_T(\ell_2)$ [GeV]")]),
        "eta_lep2":  HistConf([Axis(coll="lepton2", field="eta", bins=48, start=-2.4, stop=2.4, label=r"$\eta(\ell_2)$")]),
        "m_ll":      HistConf([Axis(coll="events", field="mll", bins=60, start=60.0, stop=120.0, label=r"$m_{\ell\ell}$ [GeV]")]),
        "jet_rel_iso":  HistConf([Axis(coll="LeptonGood", field="jetRelIso", bins=50, start=0, stop=2, label="Jet iso in lep")]),
        "dxy_lep":   HistConf([Axis(coll="LeptonGood", field="dxy", bins=50, start=-0.2, stop=0.2, label="dxy")]),
        "dz_lep":    HistConf([Axis(coll="LeptonGood", field="dz", bins=50, start=-0.5, stop=0.5, label="dz")]),

        # MET and mT
        "puppimet":        HistConf([Axis(coll="PuppiMET", field="pt", bins=50, start=0, stop=250, label=r"PUPPI $p_T^{miss}$ [GeV]")]),
        "puppimet_phi":    HistConf([Axis(coll="PuppiMET", field="phi", bins=32, start=-3.2, stop=3.2, label=r"PUPPI $\phi^{miss}$")]),
        "pfmet":           HistConf([Axis(coll="PFMET", field="pt", bins=50, start=0, stop=250, label=r"PF $p_T^{miss}$ [GeV]")]),
        "deepmet_resolution_tune":     HistConf([Axis(coll="DeepMETResolutionTune", field="pt", bins=50, start=0, stop=250, label=r"DeepMET resolution tune $p_T^{miss}$ [GeV]")]),
        "deepmet_resolution_tune_phi": HistConf([Axis(coll="DeepMETResolutionTune", field="phi", bins=32, start=-3.2, stop=3.2, label=r"DeepMET resolution tune $\phi^{miss}$")]),
        "mt_w_lep":   HistConf([Axis(coll="events", field="mt_w_leptonic", bins=40, start=0, stop=200, label=r"$m_T(W_{lep})$ (PUPPI) [GeV]")]),
        "mt_w_lep_deepresolution": HistConf([Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune", bins=40, start=0, stop=200, label=r"$m_T(W_{lep})$ (DeepMET resolution tune) [GeV]")]),
        "wleptonic_pt":    HistConf([Axis(coll="events", field="wleptonic_pt", bins=40, start=0.0, stop=300.0, label=r"$p_T(W_{lep})$ [GeV]")]),

        # Jets
        "HT":         HistConf([Axis(coll="events", field="ht", bins=35, start=0, stop=1400, label=r"$H_T$ [GeV]")]),
        "pt_jet1":    HistConf([Axis(coll="jet1", field="pt", bins=60, start=0, stop=300, label=r"$p_T(j_1)$ [GeV]")]),
        "pt_jet2":    HistConf([Axis(coll="jet2", field="pt", bins=60, start=0, stop=300, label=r"$p_T(j_2)$ [GeV]")]),
        "eta_jet1":   HistConf([Axis(coll="jet1", field="eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)$")]),
        "eta_jet2":   HistConf([Axis(coll="jet2", field="eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)$")]),
        "phi_jet1":   HistConf([Axis(coll="jet1", field="phi", bins=32, start=-3.2, stop=3.2, label=r"$\phi(j_1)$")]),
        "jet_id":     HistConf([Axis(coll="JetGood", field="jetId_corrected", bins=8, start=0, stop=8, label="Jet id")]),
        "PNetQvG_jet1":  HistConf([Axis(coll="jet1", field="btagPNetQvG", bins=40, start=0, stop=1, label=r"ParticleNet QvG$(j_1)$ (no SF)")]),
        "UParTB_jet1":   HistConf([Axis(coll="jet1", field="btagUParTAK4B", bins=40, start=0, stop=1, label=r"UParT B$(j_1)$")]),
        "mjj_vbs":    HistConf([Axis(coll="vbsjets", field="mass", bins=50, start=0, stop=2500, label=r"$M_{jj}^{VBS}$ [GeV]")]),
        "deta_vbs":   HistConf([Axis(coll="vbsjets", field="delta_eta", bins=36, start=0, stop=9.0, label=r"$|\Delta\eta_{jj}^{VBS}|$")]),

        # Fat jet
        "fj_pt":    HistConf([Axis(coll="candidate_boost1", field="pt",  bins=40, start=200, stop=1000, label=r"$p_T(J)$ [GeV]")]),
        "fj_eta":   HistConf([Axis(coll="candidate_boost1", field="eta", bins=48, start=-2.4, stop=2.4, label=r"$\eta(J)$")]),
        "fj_msd":   HistConf([Axis(coll="candidate_boost1", field="msoftdrop", bins=42, start=40, stop=250, label=r"$m_{SD}(J)$ [GeV]")]),
        "fj_t21":   HistConf([Axis(coll="candidate_boost1", field="tau21", bins=32, start=0, stop=1.1, label=r"$\tau_{21}(J)$")]),
        "fj_WvsQCD":   HistConf([Axis(coll="candidate_boost1", field="particleNetWithMass_WvsQCD", bins=40, start=0, stop=1, label=r"$J$ WvsQCD (no SF)")]),
    },
)
