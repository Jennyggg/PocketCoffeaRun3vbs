
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters import defaults

import workflow_skim, custom_cut_functions
from workflow_skim import VBSSemileptonicProcessor
from custom_cut_functions import (
    nLepton_eventskim_cut,   # >=1 e (pT>35) or mu (pT>30)
    nJet_eventskim_cut,      # N(AK4, pT>30) >= 2
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
    f"{localdir}/params/params_2024.yaml",
    update=True,
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/DYto2E-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2E-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2E-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2E-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2Mu-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2Mu-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2Mu-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2Tau-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2Tau-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2Tau-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2Tau-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/Muon0.json",
            f"{localdir}/datasets/Muon1.json",
            f"{localdir}/datasets/EGamma0.json",
            f"{localdir}/datasets/EGamma1.json",
            f"{localdir}/datasets/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8.json",
            f"{localdir}/datasets/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8.json",
            f"{localdir}/datasets/TbarWplus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8.json",
            f"{localdir}/datasets/TBbarto2Q-s-channel_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8.json",
            f"{localdir}/datasets/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/TWminus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/VBFtoLNu_TuneCP5_13p6TeV_madgraph-pythia8.json",
            f"{localdir}/datasets/WGtoLNuG-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoENu-2Jets_Bin-0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoENu-2Jets_Bin-1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoENu-2Jets_Bin-2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoMuNu-2Jets_Bin-0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoMuNu-2Jets_Bin-1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoMuNu-2Jets_Bin-2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoTauNu-2Jets_Bin-0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoTauNu-2Jets_Bin-1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoTauNu-2Jets_Bin-2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8.json",
            f"{localdir}/datasets/ssWWunpolarized_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/ssWWunpolarized_QCD_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/osWWunpolarized_Wptojj_Wmtolv_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/osWWunpolarized_Wptojj_Wmtolv_QCD_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/osWWunpolarized_Wptolv_Wmtojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/osWWunpolarized_Wptolv_Wmtojj_QCD_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WZunpolarized_Wmtolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WZunpolarized_Wmtolv_Ztojj_QCD_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WZunpolarized_Wptolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WZunpolarized_Wptolv_Ztojj_QCD_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json"
        ],
        "filter": {
            "samples": [
                "DYto2E-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2E-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2E-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2E-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2Mu-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2Mu-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2Mu-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2Tau-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2Tau-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2Tau-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "DYto2Tau-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "Muon0",
                "Muon1",
                "EGamma0",
                "EGamma1",
                "TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
                "TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8",
                "TbarWplus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8",
                "TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
                "TBbarto2Q-s-channel_TuneCP5_13p6TeV_powheg-pythia8",
                "TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8",
                "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TWminus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8",
                "VBFtoLNu_TuneCP5_13p6TeV_madgraph-pythia8",
                "WGtoLNuG-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoENu-2Jets_Bin-0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoENu-2Jets_Bin-1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoENu-2Jets_Bin-2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoMuNu-2Jets_Bin-0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoMuNu-2Jets_Bin-1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoMuNu-2Jets_Bin-2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoTauNu-2Jets_Bin-0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoTauNu-2Jets_Bin-1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoTauNu-2Jets_Bin-2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8",
                "ssWWunpolarized_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "ssWWunpolarized_QCD_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "osWWunpolarized_Wptojj_Wmtolv_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "osWWunpolarized_QCD_Wptojj_Wmtolv_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "osWWunpolarized_Wptolv_Wmtojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "osWWunpolarized_QCD_Wptolv_Wmtojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WZunpolarized_Wmtolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WZunpolarized_Wmtolv_Ztojj_QCD_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WZunpolarized_Wptolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                "WZunpolarized_Wptolv_Ztojj_QCD_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"

            ],
            "year": ["2024"],
            # "nfiles": {
            #     "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8": 50,
            #     "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8": 50,
            # },
        },
    },

    workflow=VBSSemileptonicProcessor,

    # Skim: loose cuts applied before NanoAOD files are written out.
    # All cuts combined with AND logic. Designed to be inclusive enough
    # to support any downstream CR or SR selection without re-skimming.
    skim=[
        get_nPVgood(1),
        eventFlags,
        goldenJson,
        get_HLTsel(primaryDatasets=["Muon", "EGamma"]),
        nLepton_eventskim_cut,   # >=1 lepton (e pT>35 or mu pT>30)
        nJet_eventskim_cut,      # N(AK4, pT>30) + N(AK8, pT>30) >= 3
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

    # Output path on T3_US_FNALLPC shared EOS space (xrootd endpoint).
    #save_skimmed_files="root://eosuser.cern.ch//eos/cms/store/group/phys_smp/jinw/2024_skim",
    save_skimmed_files="root://cmseos.fnal.gov//eos/uscms/store/group/lnujj/semilep_vbs_skim_run3/",
)
