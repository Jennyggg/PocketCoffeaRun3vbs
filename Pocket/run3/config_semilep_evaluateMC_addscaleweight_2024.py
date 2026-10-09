# config_semilep_evaluateMC_addscaleweight_2024.py
#
# 2024 (Summer24, NanoAODv15) version of
# config_semilep_evaluateMC_qvgcal_addscaleweight_v10.py, processing the
# unskimmed NanoAOD listed in the datasets of config_2024_skim.py.
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
import pocket_coffea.lib.jets as _pc_jets

if not hasattr(_pc_jets, "get_jersmear_SFunc"):
    raise RuntimeError(
        "This PocketCoffea cannot evaluate the 2024 JER uncertainties (new SF format): "
        "use PYTHONPATH=/afs/cern.ch/work/j/jinw/public/PocketCoffea_update"
    )

import workflow_evaluate_2024, custom_cut_functions, custom_cut_functions_2024, lhe_weights
from workflow_evaluate_2024 import VBSSemileptonicProcessor
from lhe_weights import LHEScaleWeightWrapper, LHEPdfWeightWrapper
from custom_cut_functions_2024 import trigger_pd_overlap_cut
from custom_cut_functions import (
    nLepton_skim_cut,
    nJet_skim_cut,
    vbs_semileptonic_presel,
    vbs_semileptonic_w_pt_stitch_presel,
    met_skim_cut,
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

for _mod in (workflow_evaluate_2024, custom_cut_functions, custom_cut_functions_2024, lhe_weights):
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
    f"{localdir}/params/jets_calibration_fullvar.yaml",
    f"{localdir}/params/classifiers_v8.yaml",
    f"{localdir}/params/classifiers_v8_2024.yaml",
    # Last: 2024 lumi, pileup, triggers, JERC, lepton SFs.
    # (params/lumi.yaml, pileup.yaml and lepton_scale_factors.yaml are NOT
    #  loaded: their 2024 entries are era-C-only lumi, the 2025 PU profile and
    #  a private muon SF file.)
    f"{localdir}/params/params_2024.yaml",
    update=True,
)

########################################################################
# Samples (unskimmed NanoAOD, same JSONs as config_2024_skim.py)
########################################################################
def _ds(name):
    return f"{localdir}/datasets/{name}.json"

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

MC_SAMPLES = DY_SAMPLES + W_SAMPLES + TOP_SAMPLES + OTHER_BKG_SAMPLES
if PRIVATE_VV_FLAT_READY:
    MC_SAMPLES += SIGNAL_SAMPLES + QCDVV_SAMPLES

########################################################################
# Weights
########################################################################
WEIGHTS = ["genWeight", "lumi", "XS", "pileup",
           "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
           "sf_ele_id", "sf_ele_reco", "sf_ele_trigger",
           "LHEScaleWeight", "LHEPdfWeight", "sf_partonshower_isr", "sf_partonshower_fsr"]
WEIGHT_VARIATIONS = ["pileup",
                     "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                     "sf_ele_id", "sf_ele_reco", "sf_ele_trigger",
                     "LHEScaleWeight", "LHEPdfWeight", "sf_partonshower_isr", "sf_partonshower_fsr"]
SHAPE_VARIATIONS = ["jet_calibration", "electron_scale_and_smearing", "muons_scale_and_resolution"]

SKIM = [
    get_nPVgood(1),
    eventFlags,
    goldenJson,
    nLepton_skim_cut,
    nJet_skim_cut,
    met_skim_cut,
    trigger_pd_overlap_cut,   # IsoMu24 / Ele30_WPTight_Gsf, Muon-EGamma overlap removed in data
    get_JetVetoMap(),
]

_RESOLVED_BINS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
_BOOSTED_BINS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
_BOOSTED_WWvsWZ_BINS = [0.0, 0.075, 0.15, 0.25, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
_RESOLVED_WWvsWZ_BINS = [0.0, 0.35, 0.4, 0.425, 0.45, 0.5, 0.55, 0.575, 0.6, 0.625, 0.65, 1.0]

# b-tag efficiency maps for the fixed-WP SF method (MC only)
def _btag_eff_hist(coll):
    return HistConf(
        [
            Axis(coll=coll, field="hadronFlavour", bins=[0, 4, 5, 6], label="hadron flavour"),
            Axis(coll=coll, field="pt", bins=[30, 50, 70, 100, 140, 200, 300, 600, 1000], label=r"$p_T$ [GeV]"),
            Axis(coll=coll, field="abseta", bins=[0.0, 0.8, 1.6, 2.5], label=r"$|\eta|$"),
        ],
        exclude_samples=DATA_SAMPLES, variations=False, only_categories=["baseline"],
    )

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

    # b-tag efficiency maps (denominator: JetGood |eta|<2.5; numerators: L, T)
    "btageff_all":   _btag_eff_hist("JetGoodBtagAcc"),
    "btageff_loose": _btag_eff_hist("BJetLoose"),
    "btageff_tight": _btag_eff_hist("BJetTight"),
}


def make_cfg(mc_samples, preselection, include_data=True):
    data_samples = DATA_SAMPLES if include_data else []
    jsons = [_ds(s) for s in mc_samples] + [_ds(d) for d in data_samples]
    return Configurator(
        parameters=parameters,
        datasets={
            "jsons": jsons,
            "filter": {
                "samples": mc_samples + data_samples,
                "year": ["2024"],
            },
        },
        workflow=VBSSemileptonicProcessor,
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
        weights_classes=common_weights + [SF_ele_trigger, LHEScaleWeightWrapper, LHEPdfWeightWrapper],
        weights={"common": {"inclusive": WEIGHTS}},
        variations={
            "weights": {"common": {"inclusive": WEIGHT_VARIATIONS}},
            "shape": {"common": {"inclusive": SHAPE_VARIATIONS}},
        },
        variables=VARIABLES,
    )


cfg = make_cfg(MC_SAMPLES, vbs_semileptonic_presel)
