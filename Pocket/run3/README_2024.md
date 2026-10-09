# 2024 (Summer24, NanoAODv15) processing

Requires PocketCoffea branch `update-upstream-2024` of
https://github.com/Jennyggg/PocketCoffea (upstream main + local analysis patches).

## Configs
| Config | Content |
|---|---|
| `config_W_Z_reference_2024_v10.py` | W+jets / DY reference regions for data/MC checks. Quick-test switches at the top: `DATA_ERAS` (e.g. `["C"]`, `[]` = all eras; MC normalised to the selected eras' lumi) and `MC_MAX_FILES` (`0` = all). |
| `config_semilep_evaluateMC_addscaleweight_2024.py` | Main VBS semileptonic selection (2024 version of the 2022 v10 config), without QvG / WvsQCD / ZvsQCD / tau21 SFs and without `sf_btag` (no 2024 shape SF). |
| `config_semilep_evaluateMC_addscaleweight_2024_stitch.py` | W+jets low-pT(W) stitching companion (MC only). |

Supporting files: `workflow_evaluate_2024.py`, `custom_cut_functions(_2024).py`,
`lhe_weights.py`, `params/params_2024.yaml` (lumi, PU, triggers, JERC, lepton SFs),
`params/classifiers_v8_2024.yaml` (2022 BDTs, paths relative to `params/`),
`datasets/*.json`, `private_2024_conversion/` (EDM -> flat NanoAOD for the private VV samples).

## Running at the Purdue AF (`--executor dask@purdue-af`)
1. Conda env under `/work/users/$USER/` (required by the executor) with
   `coffea==0.7.29`, `awkward<2`, `dask-gateway`, `xrootd`, `prometheus_client`, `xgboost`;
   then `pip install -e` the PocketCoffea `update-upstream-2024` branch.
2. `voms-proxy-init -voms cms` (the executor copies the proxy to `/work/users/$USER/`).
3. Check access to the private 2024 VV files (dataset JSONs point to
   `root://eosuser.cern.ch//eos/cms/store/group/phys_smp/jinw/2024_VBS_sig/...`), e.g. `xrdcp` one file.
4. `pocket-coffea run --cfg config_W_Z_reference_2024_v10.py -o <out> --executor dask@purdue-af`

`custom_run_options_2024.yaml` is for lxplus condor (it sets lxplus AFS paths).
