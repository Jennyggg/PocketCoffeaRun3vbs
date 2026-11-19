# config_semilep_evaluateMC_addscaleweight_invertlepton_2024_stitch.py
#
# W+jets stitching companion of config_semilep_evaluateMC_addscaleweight_invertlepton_2024.py:
# the flavour-split WtoENu/MuNu/TauNu-2Jets_Bin-1J/2J samples restricted to
# gen pT(W) < 40 GeV, which the PTLNu-binned samples of the main config do not cover.
# MC only; the main invertlepton config runs the data.
import cloudpickle
import config_semilep_evaluateMC_addscaleweight_invertlepton_2024 as main_cfg
from custom_cut_functions_2024 import vbs_semileptonic_w_pt_corrmet_stitch_presel

cloudpickle.register_pickle_by_value(main_cfg)

cfg = main_cfg.make_cfg(
    main_cfg.W_LOWPT_SAMPLES,
    vbs_semileptonic_w_pt_corrmet_stitch_presel,
    include_data=False,
    tag="invertlepton_stitch",
)
