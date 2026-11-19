"""Compound-identity exclusions and internal-only score-calibration folds.

External reference MoA labels are retained for reporting, never used to choose
the training memberships, calibrate a threshold or select a model.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

SEEDS = (42, 43, 44, 45, 46)
ARMS = ("proteomics", "ecfp_proteomics")
NATIVE = {"pxd027164": "st131", "pxd063721": "ptf2"}


def compound_name(dataset, case):
    if dataset == "smx":
        return "Sulfamethoxazole"
    if dataset == "meropenem":
        return "Meropenem"
    if dataset == "inhouse_corramycin":
        return "Corramycin"
    name = case[2:] if dataset == "yu" and case.startswith("Yu") else case
    return {"Polymyxin": "Polymyxin B", "Penicillin": "Penicillin G", "CSA13": "CSA-13"}.get(
        name, name
    )


def canonical_case(dataset, case):
    if dataset == "eclipse":
        return {"JH29": "Ciprofloxacin", "JH55": "Kanamycin", "JH6": "Imipenem"}[case]
    if dataset == "inhouse_corramycin":
        return "inhouse_corramycin"
    return str(case)


def inventory(cases, metadata, labels):
    """Match exact antibiotic identities; never use structural-similarity guesses."""
    training = metadata.loc[metadata.compound_id.isin(labels)].copy()
    if len(training) != len(labels) or not training.compound_si.str.casefold().is_unique:
        raise ValueError("Ambiguous training compound names")
    lookup = dict(zip(training.compound_si.str.casefold(), training.compound_id))
    result = cases[["dataset", "case", "expected"]].copy()
    result["case_key"] = result.dataset + "::" + result.case.str.lower()
    result["compound_name"] = [compound_name(r.dataset, r.case) for r in result.itertuples()]
    result["compound_key"] = result.compound_name.str.casefold()
    result["training_compound"] = result.compound_key.map(lookup).fillna("")
    result["context"] = ["omit_" + cid if cid else "full51" for cid in result.training_compound]
    if result.case_key.duplicated().any():
        raise ValueError("Duplicated external application case")
    return result


def fit_key(train, candidates):
    value = json.dumps(dict(train=sorted(train), candidates=sorted(candidates)), sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def make_plan(labels, cases, calibration_folds):
    """Keep the historical three folds, removing the test identity from all stages."""
    universe = set(labels)
    classes = sorted(set(labels.values()))
    if len(calibration_folds) != 3:
        raise ValueError("Exactly three fixed calibration folds are required")
    tests = [set(f["test"]) for f in calibration_folds]
    if set.union(*tests) != universe or sum(map(len, tests)) != len(universe):
        raise ValueError("Calibration folds must partition the in-house compounds")
    for f in calibration_folds:
        if set(f["train"]) != universe - set(f["test"]):
            raise ValueError("Incorrect calibration complement")
    jobs = {}

    def add(train, predict=(), keep=False):
        train, predict = sorted(train), sorted(predict)
        if not train or set(train) & set(predict):
            raise ValueError("Training/prediction overlap")
        if {labels[c] for c in train} != set(classes):
            raise ValueError("A training fold lost a complete class")
        key = fit_key(train, classes)
        job = jobs.setdefault(key, dict(train=train, predict=[], keep_checkpoint=False))
        if job["train"] != train:
            raise ValueError("Fit-key collision")
        job["predict"] = sorted(set(job["predict"]) | set(predict))
        job["keep_checkpoint"] |= keep
        return key

    contexts = []
    for name, group in cases.groupby("context", sort=True):
        omitted = set(group.training_compound) - {""}
        if len(omitted) > 1:
            raise ValueError("Each context may omit only one compound")
        develop = universe - omitted
        calibration = []
        for i, f in enumerate(calibration_folds):
            train, test = set(f["train"]) - omitted, set(f["test"]) - omitted
            calibration.append(
                dict(fold=i, train=sorted(train), test=sorted(test), job=add(train, test))
            )
        contexts.append(
            dict(
                context=name,
                omitted=sorted(omitted),
                develop=sorted(develop),
                main_job=add(develop, keep=True),
                calibration=calibration,
                external_cases=sorted(group.case_key),
            )
        )
    result = dict(
        compound_labels=labels,
        classes=classes,
        contexts=contexts,
        jobs=jobs,
        seeds=list(SEEDS),
        arms=list(ARMS),
        total_fits=len(jobs) * len(SEEDS) * len(ARMS),
        external_labels_used=False,
        fixed_gene_panel=True,
        uncertainty_method="low_max_score",
        false_alarm_target=0.10,
        calibration_folds=3,
    )
    validate_plan(result, cases)
    return result


def validate_plan(plan, cases):
    labels, jobs = plan["compound_labels"], plan["jobs"]
    for job in jobs.values():
        if set(job["train"]) & set(job["predict"]):
            raise ValueError("Training/prediction overlap")
        if {labels[c] for c in job["train"]} != set(plan["classes"]):
            raise ValueError("Missing training class")
    assigned = []
    for context in plan["contexts"]:
        omitted, develop = set(context["omitted"]), set(context["develop"])
        if omitted & develop or set(jobs[context["main_job"]]["train"]) != develop:
            raise ValueError("Incorrect main training membership")
        tests = []
        for fold in context["calibration"]:
            if set(fold["train"]) != develop - set(fold["test"]):
                raise ValueError("Incorrect context calibration split")
            if set(jobs[fold["job"]]["train"]) != set(fold["train"]):
                raise ValueError("Incorrect calibration fit")
            if omitted & (set(fold["train"]) | set(fold["test"])):
                raise ValueError("External test identity entered calibration")
            tests.extend(fold["test"])
        if len(tests) != len(set(tests)) or set(tests) != develop:
            raise ValueError("Calibration does not cover the development library exactly once")
        for key in context["external_cases"]:
            row = cases.set_index("case_key").loc[key]
            if row.training_compound and row.training_compound not in omitted:
                raise ValueError("An external compound remains in its training library")
        assigned.extend(context["external_cases"])
    if len(assigned) != len(set(assigned)) or set(assigned) != set(cases.case_key):
        raise ValueError("External cases must be assigned exactly once")
