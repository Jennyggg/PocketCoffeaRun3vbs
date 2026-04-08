# config_qvgsf.py
# PocketCoffea configuration for ParticleNet QvG scale factor derivation.
# 2022 postEE only.
#
# Two control regions:
#   Zjets  → Z+jets (gluon-enriched), Muon data + DY MC
#   Dijet  → Dijet  (quark-enriched), JetMET data + QCD MC
#
# Run with:
#   python runner.py --cfg config_qvgsf.py -o output_qvgsf_XXXXXX --executor dask@lxplus

import os, cloudpickle
import numpy as np
import awkward as ak
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_nPVgood, goldenJson, eventFlags, get_JetVetoMap
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.triggers import get_trigger_mask_byprimarydataset
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import HistConf, Axis
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.parameters import defaults

# ── Sample-aware HLT selection ────────────────────────────────────────────────
# QCD MC: no trigger requirement (DiPFJetAve fires on pT, not QvG score — unbiased for QvG)
# DY MC:  DoubleMuon trigger
# Data:   per-primaryDataset trigger (DoubleMuon / JetMET DiPFJetAve OR / ZeroBias)
def _hlt_sel(events, params, processor_params, year, isMC, sample, **kwargs):
    if isMC:
        if sample == "QCD":
            return ak.ones_like(events.event, dtype=bool)
        return get_trigger_mask_byprimarydataset(
            events, processor_params.HLT_triggers, year, isMC,
            primaryDatasets=["DoubleMuon"],
        )
    return get_trigger_mask_byprimarydataset(
        events, processor_params.HLT_triggers, year, isMC,
        primaryDatasets=params["primaryDatasets"],
    )

hlt_sel = Cut(
    name="HLT_sel",
    params={"primaryDatasets": ["DoubleMuon", "JetMET", "ZeroBias"]},
    function=_hlt_sel,
)

import workflow_qvgsf, cut_functions_qvgsf
from workflow_qvgsf import QvGSFProcessor
from cut_functions_qvgsf import (
    njet_skim_cut,
    zjets_cut,
    zjets_quark_cut,
    zjets_gluon_cut,
    zjets_undef_cut,
    dijet_zb_cut,
    dijet_zb_quark_cut,
    dijet_zb_gluon_cut,
    dijet_zb_undef_cut,
    dijet_zb_noMCcut,
    zjets_notrigcut,
    zjets_noMCcut,
    dijet_dj_cuts,
)

cloudpickle.register_pickle_by_value(workflow_qvgsf)
cloudpickle.register_pickle_by_value(cut_functions_qvgsf)

localdir = os.path.dirname(os.path.abspath(__file__))

# ── Parameters ───────────────────────────────────────────────────────────────
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_qvgsf.yaml",
    f"{localdir}/params/triggers_qvgsf.yaml",
    f"{localdir}/params/lumi.yaml",
    f"{localdir}/params/event_flags.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    f"{localdir}/params/plotting.yaml",
    update=True,
)

# ── Dataset JSON files ────────────────────────────────────────────────────────
# Use only the jet-multiplicity-binned samples (0J, 1J, 2J) + MLL-10to50.
# The PTLL-binned 1J/2J samples overlap with the jet-binned ones and would
# require stitching weights — avoid mixing them here.
DY_JSONS = [
    f"{localdir}/datasets/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
    f"{localdir}/datasets/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
    f"{localdir}/datasets/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
    f"{localdir}/datasets/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
]

QCD_JSONS = [
    f"{localdir}/datasets/QCD-4Jets_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
    f"{localdir}/datasets/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
]

QCDFLAT_JSONS = [
    f"{localdir}/datasets/QCD_PT-15to7000_Flat2022_TuneCP5_13p6TeV_pythia8.json",
]

MUON_JSONS = [
    f"{localdir}/datasets/Muon_2022E.json",
    f"{localdir}/datasets/Muon_2022F.json",
    f"{localdir}/datasets/Muon_2022G.json",
]

JETMET_JSONS = [
    f"{localdir}/datasets/JetMET_2022E.json",
    f"{localdir}/datasets/JetMET_2022F.json",
    f"{localdir}/datasets/JetMET_2022G.json",
]

ZEROBIAS_JSONS = [
    f"{localdir}/datasets/ZeroBias_2022E.json",
    f"{localdir}/datasets/ZeroBias_2022F.json",
    f"{localdir}/datasets/ZeroBias_2022G.json",
]

# ── Histogram bin definitions ────────────────────────────────────────────────
_PT_BINS  = [30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200,
             220, 240, 260, 290, 320, 360, 400, 450, 500, 8000]
_ETA_BINS = [0.0, 0.7, 1.3, 2.5, 3.0, 4.7]
# Z+jets uses the same bins as dijet
_PT_BINS_ZJETS  = _PT_BINS
_ETA_BINS_ZJETS = _ETA_BINS
_SCORE_BINS     = 20

# DiPFJetAve trigger keys and category name prefix
_DJ_KEYS = ["dj40", "dj60", "dj80", "dj140", "dj200", "dj260", "dj320", "dj400", "dj500"]

# ── Category lists for only_categories ───────────────────────────────────────
_ZB_CATS    = ["Dijet_ZB_all", "Dijet_ZB_quark", "Dijet_ZB_gluon",
               "Dijet_ZB_undef", "Dijet_ZB_noMCcut"]
_DJ_SF_CATS = [f"Dijet_{k.upper()}_{v}"
               for k in _DJ_KEYS
               for v in ("all", "quark", "gluon", "undef")]
_DJ_NOTRIG_CATS = [f"Dijet_{k.upper()}_notrigcut" for k in _DJ_KEYS]

# ── Configurator ─────────────────────────────────────────────────────────────
cfg = Configurator(
    parameters=parameters,

    datasets={
        "jsons": DY_JSONS + QCD_JSONS + MUON_JSONS + JETMET_JSONS + ZEROBIAS_JSONS,
        "filter": {
            "samples": [
                #"DYto2L-2Jets",  # unchanged — reuse output_ZB_fix
                #"QCD",            # rerun: HLT cut removed + J40lumi scale added
                #"QCD_Flat",
                "Muon",          # unchanged — reuse output_ZB_fix
                "JetMET",
                "ZeroBias",       # rerun: HLT trigger path fixed (ZeroBias_v* → ZeroBias)
            ]
        },
    },

    workflow=QvGSFProcessor,

    # ── Skim (applied before any category) ──────────────────────────────────
    skim=[
        get_nPVgood(1),      # at least 1 good PV
        eventFlags,          # MET filters
        goldenJson,          # golden JSON (data only)
        njet_skim_cut,       # at least 1 good jet
        # HLT: sample-aware selection (see hlt_sel above)
        # - QCD MC: no requirement (ZeroBias is trigger-unbiased)
        # - DY MC:  DoubleMuon trigger
        # - Data:   per-primaryDataset trigger (DoubleMuon/JetMET/ZeroBias)
        hlt_sel,
        get_JetVetoMap(),
    ],

    preselections=[passthrough],

    # ── Categories ─────────────────────────────────────────────────────────
    # All categories are applied to all samples.  For data, partonFlavour=0
    # so Zjets_quark/gluon and Dijet_*_quark/gluon will be empty for data.
    categories={
        # ── Z+jets region (unchanged) ────────────────────────────────────
        "Zjets_all":       [zjets_cut],
        "Zjets_quark":     [zjets_quark_cut],
        "Zjets_gluon":     [zjets_gluon_cut],
        "Zjets_undef":     [zjets_undef_cut],
        # No muon pT plateau cut: inspect DoubleMuon trigger turn-on
        "Zjets_notrigcut": [zjets_notrigcut],
        # After offline cuts, before MC-only PU/gen-matching cuts
        "Zjets_noMCcut":   [zjets_noMCcut],

        # ── ZeroBias dijet region (unchanged, provides low-pT probe) ─────
        "Dijet_ZB_all":     [dijet_zb_cut],
        "Dijet_ZB_quark":   [dijet_zb_quark_cut],
        "Dijet_ZB_gluon":   [dijet_zb_gluon_cut],
        "Dijet_ZB_undef":   [dijet_zb_undef_cut],
        "Dijet_ZB_noMCcut": [dijet_zb_noMCcut],

        # ── DiPFJetAve dijet regions (supplement ZeroBias at higher pT) ──
        # For each trigger 5 categories are auto-generated:
        #   _notrigcut  — basic presel + HLT fired, no ptavg cut → ptavg histogram
        #   _all        — + ptavg plateau cut + MC-only gen cuts → SF derivation
        #   _quark/_gluon/_undef  — flavor subcategories of _all
        **{
            f"Dijet_{k.upper()}_{v}": [dijet_dj_cuts[k][v]]
            for k in _DJ_KEYS
            for v in ("notrigcut", "all", "quark", "gluon", "undef")
        },
    },

    # ── Weights ──────────────────────────────────────────────────────────────
    # Muon SFs (id, iso, trigger) are applied inclusively.
    # In the dijet region QCD events have no selected muons so the product
    # is trivially 1.0.  Jet trigger SFs are not standard POG corrections and
    # are instead absorbed into the data/MC efficiency ratio directly.
    # QCD MC is kept at full-lumi normalisation; per-trigger lumi rescaling
    # for comparison plots is applied in post-processing.
    weights_classes=common_weights,
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
                "pileup",
                "sf_partonshower_isr",
                "sf_partonshower_fsr",
            ]
        },
        "bysample": {
            # Muon SFs only for Z+jets samples (QCD has no selected muons)
            "DYto2L-2Jets": {
                "inclusive": ["sf_mu_id", "sf_mu_iso", "sf_mu_trigger"],
            },
        },
    },
    variations={
        "weights": {
            "common": {
                "inclusive": ["pileup","sf_partonshower_isr","sf_partonshower_fsr"]
            },
            "bysample": {
                "DYto2L-2Jets": {
                    "inclusive": ["sf_mu_id", "sf_mu_iso", "sf_mu_trigger"],
                },
            },
        },
        "shape": {
            "common": {
                "inclusive": ['jet_calibration']
            },
            "bysample": {
                "DYto2L-2Jets": {
                    "inclusive": ['muons_scale_and_resolution'],
                },
            },
        }

    },

    # ── Histograms ───────────────────────────────────────────────────────────
    variables={
        # ── Z+jets probe jet: 3D SF histogram ───────────────────────────────
        "zjets_probe_score_pt_eta": HistConf([
            Axis(coll="events", field="zjets_probe_score",
                 bins=_SCORE_BINS, start=0.0, stop=1.0,
                 label="btagPNetQvG (Z+jets probe jet)"),
            Axis(coll="events", field="zjets_probe_pt",
                 bins=_PT_BINS_ZJETS,
                 label=r"$p_T$ (Z+jets probe jet) [GeV]"),
            Axis(coll="events", field="zjets_probe_abseta",
                 bins=_ETA_BINS_ZJETS,
                 label=r"$|\eta|$ (Z+jets probe jet)"),
        ]),
        # 1D cross-checks
        "zjets_probe_score": HistConf([
            Axis(coll="events", field="zjets_probe_score",
                 bins=50, start=0.0, stop=1.0,
                 label="btagPNetQvG (Z+jets probe jet)"),
        ]),
        "zjets_probe_pt": HistConf([
            Axis(coll="events", field="zjets_probe_pt",
                 bins=50, start=0, stop=500,
                 label=r"$p_T$ (Z+jets probe jet) [GeV]"),
        ]),
        "zjets_probe_abseta": HistConf([
            Axis(coll="events", field="zjets_probe_abseta",
                 bins=47, start=0, stop=4.7,
                 label=r"$|\eta|$ (Z+jets probe jet)"),
        ]),
        # di-muon invariant mass
        "mll": HistConf([
            Axis(coll="events", field="mll",
                 bins=60, start=60, stop=120,
                 label=r"$m_{\mu\mu}$ [GeV]"),
        ]),
        # pT(Z)
        "ptz": HistConf([
            Axis(coll="events", field="ptz",
                 bins=50, start=0, stop=300,
                 label=r"$p_T^Z$ [GeV]"),
        ]),

        # ── Dijet ZeroBias probe jets: 3D SF histograms ──────────────────────
        # j1 (subleading) probe
        "dijet_zb_probe_score_pt_eta": HistConf([
            Axis(coll="events", field="dijet_probe_score",
                 bins=_SCORE_BINS, start=0.0, stop=1.0,
                 label="btagPNetQvG (Dijet ZB probe jet)"),
            Axis(coll="events", field="dijet_probe_pt",
                 bins=_PT_BINS,
                 label=r"$p_T$ (Dijet ZB probe jet) [GeV]"),
            Axis(coll="events", field="dijet_probe_abseta",
                 bins=_ETA_BINS,
                 label=r"$|\eta|$ (Dijet ZB probe jet)"),
        ], only_categories=_ZB_CATS),
        # j0 (leading) probe — ZeroBias fires on bunch crossing, not pT, so j0
        # is also an unbiased QvG probe.
        "dijet_zb_lead_score_pt_eta": HistConf([
            Axis(coll="events", field="dijet_lead_score",
                 bins=_SCORE_BINS, start=0.0, stop=1.0,
                 label="btagPNetQvG (Dijet ZB lead jet)"),
            Axis(coll="events", field="dijet_lead_pt",
                 bins=_PT_BINS,
                 label=r"$p_T$ (Dijet ZB lead jet) [GeV]"),
            Axis(coll="events", field="dijet_lead_abseta",
                 bins=_ETA_BINS,
                 label=r"$|\eta|$ (Dijet ZB lead jet)"),
        ], only_categories=["Dijet_ZB_all", "Dijet_ZB_noMCcut"]),

        # ── DiPFJetAve probe jet (j1, subleading): 3D SF histogram ──────────
        # One histogram filled for all DJ trigger categories; the category axis
        # in the coffea output separates per-trigger data/MC automatically.
        "dijet_dj_probe_score_pt_eta": HistConf([
            Axis(coll="events", field="dijet_probe_score",
                 bins=_SCORE_BINS, start=0.0, stop=1.0,
                 label="btagPNetQvG (Dijet DJ probe jet)"),
            Axis(coll="events", field="dijet_probe_pt",
                 bins=_PT_BINS,
                 label=r"$p_T$ (Dijet DJ probe jet) [GeV]"),
            Axis(coll="events", field="dijet_probe_abseta",
                 bins=_ETA_BINS,
                 label=r"$|\eta|$ (Dijet DJ probe jet)"),
        ], only_categories=_DJ_SF_CATS),
        # 1D kinematic cross-checks for DiPFJetAve categories
        "dijet_probe_score": HistConf([
            Axis(coll="events", field="dijet_probe_score",
                 bins=50, start=0.0, stop=1.0,
                 label="btagPNetQvG (Dijet probe jet)"),
        ], only_categories=_DJ_SF_CATS),
        "dijet_probe_pt": HistConf([
            Axis(coll="events", field="dijet_probe_pt",
                 bins=50, start=0, stop=600,
                 label=r"$p_T$ (Dijet probe jet) [GeV]"),
        ], only_categories=_DJ_SF_CATS),
        "dijet_probe_abseta": HistConf([
            Axis(coll="events", field="dijet_probe_abseta",
                 bins=47, start=0, stop=4.7,
                 label=r"$|\eta|$ (Dijet probe jet)"),
        ], only_categories=_DJ_SF_CATS),

        # ── Average pT of leading jet pair: trigger turn-on inspection ───────
        # Filled WITHOUT the ptavg cut so the full turn-on shape is visible.
        # Data: events that fired the specific DiPFJetAve trigger (or ZeroBias).
        # MC:   same kinematic presel, no trigger cut.
        # Scale MC to trigger lumi in post-processing for comparison.
        "dijet_ptavg": HistConf([
            Axis(coll="events", field="dijet_ptavg",
                 bins=120, start=0, stop=600,
                 label=r"$(p_T^{j_0}+p_T^{j_1})/2$ [GeV]"),
        ], only_categories=["Dijet_ZB_noMCcut"] + _DJ_NOTRIG_CATS),

        # ── Dijet kinematics ─────────────────────────────────────────────────
        "dphi_jj": HistConf([
            Axis(coll="events", field="dphi_jj",
                 bins=32, start=0, stop=3.2,
                 label=r"$|\Delta\phi(j_0,j_1)|$"),
        ], only_categories=_ZB_CATS + _DJ_SF_CATS),
        "dijet_balance": HistConf([
            Axis(coll="events", field="dijet_balance",
                 bins=40, start=0, stop=1,
                 label=r"$p_T^{j_2} / \langle p_T^{j_{0,1}}\rangle$"),
        ], only_categories=_ZB_CATS + _DJ_SF_CATS),

        # ── Z+jets trigger turn-on ───────────────────────────────────────────
        "mu0_pt": HistConf([
            Axis(coll="events", field="mu0_pt",
                 bins=60, start=0, stop=120,
                 label=r"Leading muon $p_T$ [GeV]"),
        ]),
        "mu1_pt": HistConf([
            Axis(coll="events", field="mu1_pt",
                 bins=60, start=0, stop=120,
                 label=r"Subleading muon $p_T$ [GeV]"),
        ]),

        # ── General ─────────────────────────────────────────────────────────
        "nJets": HistConf([
            Axis(coll="events", field="nJetGood",
                 bins=12, start=0, stop=12, label="N(jets)"),
        ]),
        "nMuons": HistConf([
            Axis(coll="events", field="nMuonGood",
                 bins=6, start=0, stop=6, label="N(muons good)"),
        ]),
    },
)
