# config_semilep_evaluateMC_addscaleweight_2024_stitch.py
#
# W+jets stitching companion of config_semilep_evaluateMC_addscaleweight_2024.py
# (same role as config_semilep_evaluateMC_qvgcal_addscaleweight_v10_stitch.py):
# the flavour-split WtoENu/MuNu/TauNu-2Jets_Bin-1J/2J samples restricted to
# gen pT(W) < 40 GeV, which the PTLNu-binned samples of the main config do not cover.
# MC only; the main config runs the data. Input (skimmed or not) follows the
# USE_SKIMMED_DATASETS switch of the main config.
import cloudpickle
import config_semilep_evaluateMC_addscaleweight_2024 as main_cfg
from custom_cut_functions_2024 import vbs_semileptonic_w_pt_corrmet_stitch_presel

cloudpickle.register_pickle_by_value(main_cfg)

cfg = main_cfg.make_cfg(
    main_cfg.W_LOWPT_SAMPLES,
    vbs_semileptonic_w_pt_corrmet_stitch_presel,
    include_data=False,
    tag="stitch",
)
