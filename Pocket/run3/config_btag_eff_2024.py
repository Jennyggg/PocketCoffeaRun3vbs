# config_btag_eff_2024.py
#
# MC b-tagging efficiency pass for the 2024 fixed-working-point b-tag weight
# (BTV method 1a). BTV provides only the SFs (UParTAK4_comb / UParTAK4_light);
# the MC efficiencies eps_L, eps_T(flavour, pT, |eta|) must come from our own MC.
#
# Selection: the preselection of config_semilep_evaluateMC_addscaleweight_2024.py
# (no b-tag requirement; the b-veto / b-accept cuts are only in the categories),
# all jets of JetGoodBtagAcc (= the jets counted by nBJetLoose / nBJetTight).
# Nominal only: no shape or weight variations.
#
# Only a fraction of the files of each MC dataset is processed. The pre-skim
# sum_genweights totals are therefore removed from the dataset metadata, so
# PocketCoffea normalises each dataset with sum(skimRescaleGenWeight*genWeight)
# of the processed events: the relative weights between datasets (needed when
# several samples are combined into one map) stay correct.
#
# W+jets pT(lnu) < 40 slices: config_btag_eff_2024_stitch.py.
# Maps: python derive_btag_eff_2024.py <output_all.coffea> [<stitch output_all.coffea>]
import os, json, math
import cloudpickle
import config_semilep_evaluateMC_addscaleweight_2024 as main_cfg
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import HistConf, Axis
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.weights.common.weights_run3 import SF_ele_trigger
from custom_cut_functions_2024 import vbs_semileptonic_corrmet_presel

cloudpickle.register_pickle_by_value(main_cfg)

localdir = os.path.dirname(os.path.abspath(__file__))

FILE_FRACTION = 0.05   # fraction of each dataset's skimmed files to process
MIN_FILES = 3          # ...but at least this many (or all, if fewer)

# Binning of the maps. eta is filled signed (abseta is not a stored field) and
# folded into |eta| by derive_btag_eff_2024.py; pT overflow goes to the last bin.
PT_BINS = [30, 50, 70, 100, 140, 200, 300, 600, 1000]
ABSETA_BINS = [0.0, 0.6, 1.2, 1.8, 2.5]
ETA_BINS = sorted({-x for x in ABSETA_BINS} | set(ABSETA_BINS))

_btag = main_cfg.parameters.btagging.working_point["2024"]
BTAG_ALGO = _btag["btagging_algorithm"]                   # btagUParTAK4B
WP_L, WP_T = float(_btag["btagging_WP"]["L"]), float(_btag["btagging_WP"]["T"])
SCORE_BINS = [0.0, WP_L, WP_T, 1.0]   # fail L | pass L, fail T | pass T

WEIGHTS = ["genWeight", "lumi", "XS", "pileup",
           "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
           "sf_ele_id", "sf_ele_reco", "sf_ele_trigger"]

VARIABLES = {
    "btag_eff": HistConf(
        [
            # numeric bins: [0,4) udsg, [4,5) c, [5,6) b (categorical axes only work for
            # metadata/custom fields in PocketCoffea)
            Axis(coll="JetGoodBtagAcc", field="hadronFlavour", type="variable", bins=[0, 4, 5, 6],
                 label="hadron flavour"),
            Axis(coll="JetGoodBtagAcc", field="pt", type="variable", bins=PT_BINS,
                 label=r"jet $p_T$ [GeV]"),
            Axis(coll="JetGoodBtagAcc", field="eta", type="variable", bins=ETA_BINS,
                 label=r"jet $\eta$"),
            Axis(coll="JetGoodBtagAcc", field=BTAG_ALGO, type="variable", bins=SCORE_BINS,
                 label=f"{BTAG_ALGO} (fail L / L not T / T)"),
        ],
        variations=False,
    ),
    # event-level cross-check of the region b-tag multiplicities
    "nBJetLoose": HistConf([Axis(coll="events", field="nBJetLoose", bins=6, start=0, stop=6,
                                 label="N(b jets UParT loose)")], variations=False),
    "nBJetTight": HistConf([Axis(coll="events", field="nBJetTight", bins=6, start=0, stop=6,
                                 label="N(b jets UParT tight)")], variations=False),
}


def eff_dataset_json(samples, tag):
    """Subset of the skimmed MC datasets of `samples`: FILE_FRACTION of the files
    (at least MIN_FILES), evenly spread over each dataset, without the pre-skim
    sum_genweights totals (see the header)."""
    with open(main_cfg.SKIMMED_DATASETS_JSON) as f:
        content = json.load(f)
    out = {}
    for name, ds in content.items():
        md = ds["metadata"]
        if md["sample"] not in samples or md.get("year") != "2024" or str(md.get("isMC")) != "True":
            continue
        files = ds["files"]
        n = min(len(files), max(MIN_FILES, math.ceil(FILE_FRACTION * len(files))))
        step = len(files) / n
        md = {k: v for k, v in md.items() if k not in ("sum_genweights", "sum_signOf_genweights")}
        out[name] = {**ds, "metadata": md, "files": [files[int(i * step)] for i in range(n)]}
    missing = sorted(set(samples) - {ds["metadata"]["sample"] for ds in out.values()})
    if missing:
        raise ValueError(f"MC samples not found in {main_cfg.SKIMMED_DATASETS_JSON}: {missing}")
    os.makedirs(f"{localdir}/datasets/subsets", exist_ok=True)
    path = f"{localdir}/datasets/subsets/skimmed_2024_btageff_{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=4)
    return path


def make_eff_cfg(samples, preselection, tag):
    return Configurator(
        parameters=main_cfg.parameters,
        datasets={
            "jsons": [eff_dataset_json(samples, tag)],
            "filter": {"samples": list(samples), "year": ["2024"]},
        },
        workflow=main_cfg.VBSSemileptonicProcessor,
        workflow_options={"deepmet_corrections": main_cfg.DEEPMET_CORRECTIONS},
        skim=main_cfg.SKIM,
        preselections=[preselection],
        categories={"baseline": [passthrough]},
        weights_classes=common_weights + [SF_ele_trigger],
        weights={"common": {"inclusive": WEIGHTS}},
        variations={"weights": {"common": {"inclusive": []}},
                    "shape": {"common": {"inclusive": []}}},
        variables=VARIABLES,
    )


cfg = make_eff_cfg(main_cfg.MC_SAMPLES, vbs_semileptonic_corrmet_presel, tag="main")
