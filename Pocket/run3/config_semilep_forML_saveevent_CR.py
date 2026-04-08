# example_config_semileptonic.py
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags, get_JetVetoMap
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import HistConf, Axis
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.weights.common.weights_run3 import SF_ele_trigger
from pocket_coffea.parameters import defaults
import numpy as np
import awkward as ak
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation, WeightLambda
from pocket_coffea.lib.scale_factors import sf_pileup_reweight
from pocket_coffea.lib.columns_manager import ColOut

import workflow_forML_updatesel, custom_cut_functions
from workflow_forML_updatesel import VBSSemileptonicProcessor
from custom_cut_functions import (
    nLepton_skim_cut,
    nJet_skim_cut,
    vbs_semileptonic_presel,
    whad_window_cut_e,
    met_skim_cut,
    qcd_validate,
    whad_window_cut_bveto_e,
    msd_window_cut_bveto_e,
    whad_windowinvert_cut_bveto_e,
    msd_windowinvert_cut_bveto_e,
    whad_window_cut_baccept_e,
    msd_window_cut_baccept_e,
    whad_window_cut_mu,
    whad_window_cut_bveto_mu,
    msd_window_cut_bveto_mu,
    whad_windowinvert_cut_bveto_mu,
    msd_windowinvert_cut_bveto_mu,
    whad_window_cut_baccept_mu,
    msd_window_cut_baccept_mu
)

METDef = "DeepMETResolutionTune"
MTDef = "mt_w_leptonic_deepMET_resolutiontune"

# class PileupWeight(WeightWrapper):
#     name = "PileupWeight"
#     has_variations = True

#     def __init__(self, parameters, metadata):
#         super().__init__(parameters, metadata)
#         self.year = metadata["year"]
#         self._variations = parameters.pileupJSONfiles[self.year]["variations"]
#         self.params = parameters

#     def compute(self, events, size, shape_variation):
#         if shape_variation == "nominal":
#             sf, sfup, sfdown = sf_pileup_reweight(self.params, events, self.year)
#             sf_data = {
#                 "nominal": sf,
#                 "up": sfup,
#                 "down": sfdown
#             }
#             return WeightDataMultiVariation(
#                 name=self.name,
#                 nominal=sf_data["nominal"],
#                 variations=self._variations["up"] + self._variations["down"],
#                 up=[sf_data[var] for var in self._variations["up"]],
#                 down=[sf_data[var] for var in self._variations["down"]]
#             )
#         else:
#             return WeightData(
#                 name=self.name,
#                 nominal=np.ones(size),
#             )


cloudpickle.register_pickle_by_value(workflow_forML_updatesel)
cloudpickle.register_pickle_by_value(custom_cut_functions)

localdir = os.path.dirname(os.path.abspath(__file__))


default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_run3.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/plotting.yaml",
    f"{localdir}/params/lumi.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    f"{localdir}/params/pileup.yaml",
    update=True,
)

#PileupWeight = WeightLambda.wrap_func(
#    name="PileupWeight",
#    function=lambda params, metadata, events, size, shape_variations:
#        sf_pileup_reweight(params, events, metadata["year"]),
#    has_variations=True  # no list of variations it means only up and down
#    )

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            #######
            ## RUN 2 BKG
            # #########
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            #
            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # XSEC STUDIES
            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            
            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17_2.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17_2.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17_2.json",
            

            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix.json",
            
            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix2.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix2.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix2.json",
            
            #END XSEC STUDIES

            #f"{localdir}/datasets/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8.json",

            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_17.json",
            

            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_17.json",

            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",

            # f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",

            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",

            # f"{localdir}/datasets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.json",
            # f"{localdir}/datasets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.json",
            # f"{localdir}/datasets/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.json",
            # f"{localdir}/datasets/ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.json",
            # f"{localdir}/datasets/ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            # f"{localdir}/datasets/ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            # f"{localdir}/datasets/WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            # f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            # f"{localdir}/datasets/WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/WZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8.json",
            
            # #########
            # ## RUN 2 SIGNAL
            # ########
            # f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
 
            ########
            ## RUN 3 BKG
            ########
            #f"{localdir}/datasets/WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/TTto2L2Nu_TuneCP5_ERDOn_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",'''

            f"{localdir}/datasets/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8.json",
            f"{localdir}/datasets/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8.json",
            f"{localdir}/datasets/TbarWplus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/TWminus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/TT_TuneCP5_13p6TeV_powheg-pythia8.json",
            f"{localdir}/datasets/WW_TuneCP5_13p6TeV_pythia8.json",
            f"{localdir}/datasets/WZ_TuneCP5_13p6TeV_pythia8.json",
            f"{localdir}/datasets/ZZ_TuneCP5_13p6TeV_pythia8.json",
            f"{localdir}/datasets/WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8.json",

            # #########
            # ## RUN 3 SIGNAL
            # ########
            f"{localdir}/datasets/ssWWunpolarized_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/osWWunpolarized_Wptojj_Wmtolv_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/osWWunpolarized_Wptolv_Wmtojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WZunpolarized_Wmtolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WZunpolarized_Wptolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",


            #f"{localdir}/datasets/WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
            #f"{localdir}/datasets/WmWmJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
            #########
            ## SOME DATA
            #########
            f"{localdir}/datasets/SingleMuon.json", ## 2017B Single Muon dataset
            f"{localdir}/datasets/EGamma.json",
            f"{localdir}/datasets/Muon.json"
            
        ],
        "filter": {
            "samples": [
                
            #########
            ## RUN 2 BKG
            #########
            # "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",

            # "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            #"WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",

            # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            
            # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17_2",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17_2",
            # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17_2",
            

            # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix",
            # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix",
            
            # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix2",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix2",
            # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix2",
            
            #"WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8", 
            #"WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_17", 
            #"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            #"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8", 
            #"DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_17", 
            #"DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8", 

            # "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",

            # "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8", 
            # "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", 
            # "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8", 
            # "ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8",
            # "ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8", 
            # "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8", 
            # "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8", 
            # # "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV", 
            # # "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV", 
            # # "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8", 
            # # "WZZ_TuneCP5_13TeV-amcatnlo-pythia8", 
            # # "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8", 
            # # "WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8", 
            # # "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8", 
            # # "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8", 
        
                        
            # #########
            # ## RUN 2 SIGNAL
            # ########
            #"WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            
            ########
            ## RUN 3 BKG
            ########
            #"WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            #"WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8",
            #"TTto2L2Nu_TuneCP5_ERDOn_13p6TeV_powheg-pythia8",
            #"TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            #"DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",



            #"DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            "DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_PTLNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_PTLL-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"TT_TuneCP5_13p6TeV_powheg-pythia8",
            "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
            "TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
            "TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
            "TbarWplus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8",
            "TWminus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8",
            "WZ_TuneCP5_13p6TeV_pythia8",
            "WW_TuneCP5_13p6TeV_pythia8",
            "ZZ_TuneCP5_13p6TeV_pythia8",
            "WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8",
            # #######
            # # RUN 3 SIGNAL
            # #######
            #"ssWWunpolarized_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"osWWunpolarized_Wptojj_Wmtolv_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"osWWunpolarized_Wptolv_Wmtojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WZunpolarized_Wmtolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WZunpolarized_Wptolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8",
            #"WmWmJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8",

            #########
            ## SOME DATA
            #########
            #"SingleMuon", ## 2017B Single Muon dataset
            #"EGamma",
            #"Muon"
            ],
            "year": ["2022_postEE"]
        },
    },
    workflow=VBSSemileptonicProcessor,
    workflow_options = {"dump_columns_as_arrays_per_chunk": "root://eosuser.cern.ch//eos/user/j/jinw/VBS_forML/output_VBS_data2022postEE_deepmet_resolution_PTLLStitch_JetStitch_jet50_save_updatesel_jetflavor/"},
    
    skim=[
        get_nPVgood(1),    # nPV>0
        eventFlags,        # PileupID
        goldenJson,        
        nLepton_skim_cut,
        nJet_skim_cut,  
        met_skim_cut,
        get_HLTsel(primaryDatasets=["SingleMuon", "EGamma"]),
        get_JetVetoMap()
    ],

    # 2) preselections 
    preselections=[vbs_semileptonic_presel],

   
    categories={
        "resolved_mu_TTCR" :[whad_window_cut_baccept_mu],
        "resolved_e_TTCR" :[whad_window_cut_baccept_e],
        "resolved_mu_WCR" :[whad_windowinvert_cut_bveto_mu],
        "resolved_e_WCR" :[whad_windowinvert_cut_bveto_e],
    },

   
    weights_classes=common_weights+[SF_ele_trigger],#+[PileupWeight],
    #weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "PileupWeight", "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele
    #weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "pileup", "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_btag","sf_btag_calib"]}},
    weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "pileup", "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_btag"]}},
    #variations={"weights": {"common": {"inclusive": ["pileup", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_btag"]}}}, #"pileup"
    #variations={"weights": {"common": {"inclusive": ["pileup", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_btag"]}},
    #           "shape": {"common": {"inclusive": ['jet_calibration', 'electron_scale_and_smearing', 'muons_scale_and_resolution']}}
    #       }, #"pileup"
    variations={"weights": {"common": {"inclusive": []}}},
    variables={},
    columns = {
        "common": {
           "bycategory": {
               "resolved_mu_WCR": [
                   ColOut("events",["nJetGood","nCentralJetsGood",MTDef, "z_lep","centrality_resolved","vbsjet1_PNetQvG","vbsjet2_PNetQvG","vbsjet1_DeepFlavQG","vbsjet2_DeepFlavQG","w_had_jet1_resolved_PNetQvG","w_had_jet2_resolved_PNetQvG","w_had_jet1_resolved_DeepFlavQG","w_had_jet2_resolved_DeepFlavQG","vbs_dR","dR_w_had_vbs1","dR_w_had_vbs2","lead_wlep_vbsjet1_dR"]),
                   ColOut("events",["nJetGood","nCentralJetsGood"]),
                   ColOut(METDef, ["pt","phi"]),
                   ColOut("w_had_jets", ["mass","pt","phi","eta"]),
                   ColOut("vbsjets", ["delta_eta","mass", "delta_phi"]),
                   ColOut("jet1",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet2",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet3",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet4",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("lepton1",["eta","phi","pt"]),
                   ColOut("mass",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1", f"jet1_{METDef}",f"jet2_{METDef}",f"jet3_{METDef}",f"jet4_{METDef}",f"lepton1_{METDef}"]),
                   ColOut("dR",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1", f"jet1_{METDef}",f"jet2_{METDef}",f"jet3_{METDef}",f"jet4_{METDef}",f"lepton1_{METDef}"]),
                   ColOut("deta",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1"]),
                   ],
               "resolved_e_WCR": [
                   ColOut("events",["nJetGood","nCentralJetsGood",MTDef, "z_lep","centrality_resolved","vbsjet1_PNetQvG","vbsjet2_PNetQvG","vbsjet1_DeepFlavQG","vbsjet2_DeepFlavQG","w_had_jet1_resolved_PNetQvG","w_had_jet2_resolved_PNetQvG","w_had_jet1_resolved_DeepFlavQG","w_had_jet2_resolved_DeepFlavQG","vbs_dR","dR_w_had_vbs1","dR_w_had_vbs2","lead_wlep_vbsjet1_dR"]),
                   ColOut("events",["nJetGood","nCentralJetsGood"]),
                   ColOut(METDef, ["pt","phi"]),
                   ColOut("w_had_jets", ["mass","pt","phi","eta"]),
                   ColOut("vbsjets", ["delta_eta","mass", "delta_phi"]),
                   ColOut("jet1",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet2",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet3",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet4",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("lepton1",["eta","phi","pt"]),
                   ColOut("mass",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1", f"jet1_{METDef}",f"jet2_{METDef}",f"jet3_{METDef}",f"jet4_{METDef}",f"lepton1_{METDef}"]),
                   ColOut("dR",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1", f"jet1_{METDef}",f"jet2_{METDef}",f"jet3_{METDef}",f"jet4_{METDef}",f"lepton1_{METDef}"]),
                   ColOut("deta",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1"]),
                   ],
               "resolved_mu_TTCR": [
                   ColOut("events",["nJetGood","nCentralJetsGood",MTDef, "z_lep","centrality_resolved","vbsjet1_PNetQvG","vbsjet2_PNetQvG","vbsjet1_DeepFlavQG","vbsjet2_DeepFlavQG","w_had_jet1_resolved_PNetQvG","w_had_jet2_resolved_PNetQvG","w_had_jet1_resolved_DeepFlavQG","w_had_jet2_resolved_DeepFlavQG","vbs_dR","dR_w_had_vbs1","dR_w_had_vbs2","lead_wlep_vbsjet1_dR"]),
                   ColOut("events",["nJetGood","nCentralJetsGood"]),
                   ColOut(METDef, ["pt","phi"]),
                   ColOut("w_had_jets", ["mass","pt","phi","eta"]),
                   ColOut("vbsjets", ["delta_eta","mass", "delta_phi"]),
                   ColOut("jet1",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet2",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet3",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet4",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("lepton1",["eta","phi","pt"]),
                   ColOut("mass",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1", f"jet1_{METDef}",f"jet2_{METDef}",f"jet3_{METDef}",f"jet4_{METDef}",f"lepton1_{METDef}"]),
                   ColOut("dR",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1", f"jet1_{METDef}",f"jet2_{METDef}",f"jet3_{METDef}",f"jet4_{METDef}",f"lepton1_{METDef}"]),
                   ColOut("deta",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1"]),
                   ],
               "resolved_e_TTCR": [
                   ColOut("events",["nJetGood","nCentralJetsGood",MTDef, "z_lep","centrality_resolved","vbsjet1_PNetQvG","vbsjet2_PNetQvG","vbsjet1_DeepFlavQG","vbsjet2_DeepFlavQG","w_had_jet1_resolved_PNetQvG","w_had_jet2_resolved_PNetQvG","w_had_jet1_resolved_DeepFlavQG","w_had_jet2_resolved_DeepFlavQG","vbs_dR","dR_w_had_vbs1","dR_w_had_vbs2","lead_wlep_vbsjet1_dR"]),
                   ColOut("events",["nJetGood","nCentralJetsGood"]),
                   ColOut(METDef, ["pt","phi"]),
                   ColOut("w_had_jets", ["mass","pt","phi","eta"]),
                   ColOut("vbsjets", ["delta_eta","mass", "delta_phi"]),
                   ColOut("jet1",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet2",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet3",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("jet4",["eta","phi","pt", "btagPNetQvG", "btagDeepFlavB", "partonFlavour"]),
                   ColOut("lepton1",["eta","phi","pt"]),
                   ColOut("mass",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1", f"jet1_{METDef}",f"jet2_{METDef}",f"jet3_{METDef}",f"jet4_{METDef}",f"lepton1_{METDef}"]),
                   ColOut("dR",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1", f"jet1_{METDef}",f"jet2_{METDef}",f"jet3_{METDef}",f"jet4_{METDef}",f"lepton1_{METDef}"]),
                   ColOut("deta",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_lepton1", "jet3_jet4", "jet3_lepton1", "jet4_lepton1"]),
                   ],
             }
        }
    },
)
