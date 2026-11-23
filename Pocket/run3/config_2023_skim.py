
# config_2023_skim.py
#
# 2023 (NanoAODv12: 22Sep2023 data, Run3Summer23 / Run3Summer23BPix MC) version
# of config_2024_skim.py. Same skim cuts. Datasets: datasets/*_2023.json, built
# from datasets_definitions_2023_{bkg,data}.json (make_dataset_definitions_2023.py).
# Signal and QCD-VV samples not available yet for 2023 -> not listed.
#
# Triggers (params/triggers.yaml, 2023_preBPix / 2023_postBPix): IsoMu24 OR
# Ele30_WPTight_Gsf, as in 2024. Both unprescaled in Run2023C/D; EGM 2023 trigger
# SFs exist only for Ele30, MUO 2023 SFs for IsoMu24 (from pT 26 GeV), so the 2024
# offline thresholds (mu pT > 26, e pT > 35 GeV) apply unchanged.
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters import defaults

import workflow_skim, custom_cut_functions
from workflow_skim import VBSSemileptonicProcessor
from custom_cut_functions import (
    nLepton_eventskim_cut,   # >=1 e (pT>35) or mu (pT>26), |eta|<2.4
    nJet_eventskim_cut,      # N(AK4, pT>30) + N(AK8, pT>200) >= 3
)

cloudpickle.register_pickle_by_value(workflow_skim)
cloudpickle.register_pickle_by_value(custom_cut_functions)

localdir = os.path.dirname(os.path.abspath(__file__))

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_run3.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/plotting.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    update=True,
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/Muon0_2023.json",
            f"{localdir}/datasets/Muon1_2023.json",
            f"{localdir}/datasets/EGamma0_2023.json",
            f"{localdir}/datasets/EGamma1_2023.json",
            f"{localdir}/datasets/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8_2023.json",
            f"{localdir}/datasets/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8_2023.json",
            f"{localdir}/datasets/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8_2023.json",
            f"{localdir}/datasets/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8_2023.json",
            f"{localdir}/datasets/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_2023.json",
            f"{localdir}/datasets/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_2023.json",
            f"{localdir}/datasets/TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_2023.json",
            f"{localdir}/datasets/TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_2023.json",
            f"{localdir}/datasets/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_2023.json",
            f"{localdir}/datasets/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_2023.json",
            f"{localdir}/datasets/VBFtoLNu_TuneCP5_13p6TeV_madgraph-pythia8_2023.json",
            f"{localdir}/datasets/WGtoLNuG-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_2023.json",
            f"{localdir}/datasets/WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8_2023.json",
            f"{localdir}/datasets/WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8_2023.json",
            f"{localdir}/datasets/WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8_2023.json",
            f"{localdir}/datasets/ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8_2023.json",
        ],
        "filter": {
            "samples": [
                "DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "Muon0",
                "Muon1",
                "EGamma0",
                "EGamma1",
                "TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
                "TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
                "TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8",
                "TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8",
                "TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
                "VBFtoLNu_TuneCP5_13p6TeV_madgraph-pythia8",
                "WGtoLNuG-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_PTLNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8",
                "WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8",
                "WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8",
                "ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8",
            ],
            "year": ["2023_preBPix", "2023_postBPix"],
        },
    },

    workflow=VBSSemileptonicProcessor,

    # Skim: loose cuts applied before NanoAOD files are written out (same as 2024).
    # All cuts combined with AND logic. Designed to be inclusive enough
    # to support any downstream CR or SR selection without re-skimming.
    skim=[
        get_nPVgood(1),
        eventFlags,
        goldenJson,
        get_HLTsel(primaryDatasets=["Muon", "EGamma"]),   # IsoMu24 OR Ele30_WPTight_Gsf
        nLepton_eventskim_cut,   # >=1 lepton (e pT>35 or mu pT>26, |eta|<2.4)
        nJet_eventskim_cut,      # N(AK4, pT>30) + N(AK8, pT>200) >= 3
    ],

    preselections=[passthrough],

    categories={
        "baseline": [passthrough],
    },

    # No weights / variations / variables: processing halts after the skim
    # and one skimmed NanoAOD file per chunk is written to save_skimmed_files.
    weights_classes=[],
    weights={"common": {"inclusive": []},},
    variations={"weights": {"common": {"inclusive": []}}},
    variables={},

    save_skimmed_files="root://eosuser.cern.ch//eos/cms/store/group/phys_smp/jinw/2023_skim",
    # Output path on T3_US_FNALLPC shared EOS space (xrootd endpoint), as used for 2024:
    #save_skimmed_files="root://cmseos.fnal.gov//eos/uscms/store/group/lnujj/semilep_vbs_skim_run3_2023/",
)
