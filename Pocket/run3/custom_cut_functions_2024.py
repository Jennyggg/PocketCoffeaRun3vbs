# custom_cut_functions_2024.py
import awkward as ak
import numpy as np
from pocket_coffea.lib.cut_definition import Cut


def _hlt_or(events, paths):
    mask = np.zeros(len(events), dtype=bool)
    for path in paths:
        if path in events.HLT.fields:
            mask = mask | ak.to_numpy(events.HLT[path])
    return mask


def trigger_with_pd_overlap_removal(events, params, processor_params, year, isMC, sample, **kwargs):
    """Single-lepton trigger with Muon/EGamma primary-dataset overlap removal.

    MC:            OR of the muon and electron triggers.
    Muon* data:    muon triggers.
    EGamma* data:  electron triggers AND NOT muon triggers, so an event that is
                   in both primary datasets is only taken from Muon*.

    get_HLTsel(primaryDatasets=[...]) applies the OR to every data stream, which
    counts events present in both primary datasets twice.
    """
    triggers = processor_params.HLT_triggers[year]
    mu = _hlt_or(events, triggers[params["muon_pd"]])
    ele = _hlt_or(events, triggers[params["egamma_pd"]])
    if isMC:
        return mu | ele
    if sample.startswith(params["muon_pd"]):
        return mu
    if sample.startswith(params["egamma_pd"]):
        return ele & ~mu
    raise ValueError(f"Data sample {sample} is neither a {params['muon_pd']} nor an {params['egamma_pd']} stream")


trigger_pd_overlap_cut = Cut(
    name="HLT_singlelepton_PDoverlap",
    params={"muon_pd": "Muon", "egamma_pd": "EGamma"},
    function=trigger_with_pd_overlap_removal,
)


# W+jets reference selections with the MET cut on the corrected DeepMET resolution tune
# (DeepMETResolutionTuneCorr: xy offset + MC smearing; needs workflow_options["deepmet_corrections"]).
from custom_cut_functions import select_wjets, select_wjets_e, select_wjets_mu

wjets_corrmet_sel = Cut(
    name="wjet_corrmet",
    params={"met_def": "deepmet_resolution_corr", "met_pt": 30.0},
    function=select_wjets,
)
wjets_e_corrmet_sel = Cut(
    name="wjet_e_corrmet",
    params={"met_def": "deepmet_resolution_corr", "met_pt": 30.0},
    function=select_wjets_e,
)
wjets_mu_corrmet_sel = Cut(
    name="wjet_mu_corrmet",
    params={"met_def": "deepmet_resolution_corr", "met_pt": 30.0},
    function=select_wjets_mu,
)
