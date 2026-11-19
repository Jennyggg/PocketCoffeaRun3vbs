# config_btag_eff_2024_stitch.py
#
# W+jets pT(lnu) < 40 slices (W_LOWPT_SAMPLES, with the stitching preselection)
# for the 2024 b-tag efficiency pass; see config_btag_eff_2024.py.
import cloudpickle
import config_btag_eff_2024 as eff_cfg
import config_semilep_evaluateMC_addscaleweight_2024 as main_cfg
from custom_cut_functions_2024 import vbs_semileptonic_w_pt_corrmet_stitch_presel

cloudpickle.register_pickle_by_value(eff_cfg)

cfg = eff_cfg.make_eff_cfg(main_cfg.W_LOWPT_SAMPLES, vbs_semileptonic_w_pt_corrmet_stitch_presel,
                           tag="stitch")
