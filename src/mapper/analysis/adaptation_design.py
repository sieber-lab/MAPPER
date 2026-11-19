"""Identity-grouped Yu leave-one-antibiotic-out adaptation design.

All measurements of an antibiotic share one group across domains. Inner folds
hold out complete groups, preserving the earlier in-house three-fold partition.
New Yu identities are allocated by class and fold size with a fixed seed.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

SEEDS = (42, 43, 44, 45, 46)
CALIBRATION_MODES = ("inhouse", "pooled")


def assign_folds(labels, original_folds):
    folds = {}
    for index, fold in enumerate(original_folds):
        for group in fold["test"]:
            if group in folds:
                raise ValueError("Repeated original calibration identity")
            folds[group] = index
    if len(original_folds) != 3 or not set(folds) <= set(labels):
        raise ValueError("Invalid original calibration partition")
    rng = np.random.default_rng(20261028)
    for label in sorted(set(labels.values())):
        pending = sorted(group for group in labels if labels[group] == label and group not in folds)
        for group in rng.permutation(pending):
            counts = [
                (
                    sum(labels[g] == label and f == i for g, f in folds.items()),
                    sum(f == i for f in folds.values()),
                    i,
                )
                for i in range(3)
            ]
            folds[str(group)] = min(counts)[2]
    if set(folds) != set(labels):
        raise ValueError("Incomplete calibration allocation")
    return folds


def make_plan(labels, cases, original_folds):
    classes = sorted(set(labels.values()))
    assignments = assign_folds(labels, original_folds)
    universe = set(labels)
    jobs = {}

    def add(train, predict=(), keep=False):
        train = sorted(train)
        if not train or set(train) & set(predict):
            raise ValueError("Empty or overlapping fit membership")
        if {labels[group] for group in train} != set(classes):
            raise ValueError("A calibration fit lost a complete candidate class")
        key = hashlib.sha256(json.dumps(train).encode()).hexdigest()[:16]
        job = jobs.setdefault(key, dict(train=train, predict=[], keep_checkpoint=False))
        if job["train"] != train:
            raise ValueError("Fit-key collision")
        job["predict"] = sorted(set(job["predict"]) | set(predict))
        job["keep_checkpoint"] |= bool(keep)
        return key

    contexts = []
    if cases.case_key.duplicated().any() or cases.group_id.duplicated().any():
        raise ValueError("Each Yu antibiotic must have one primary case")
    for row in cases.sort_values("compound_name", kind="stable").itertuples():
        excluded = row.group_id
        if excluded not in universe:
            raise ValueError("Unknown outer test identity")
        develop = universe - {excluded}
        calibration = []
        for fold in range(3):
            test = {g for g in develop if assignments[g] == fold}
            train = develop - test
            calibration.append(
                dict(fold=fold, test=sorted(test), train=sorted(train), job=add(train, test))
            )
        contexts.append(
            dict(
                context=row.case_key.replace("::", "_"),
                case_key=row.case_key,
                compound_name=row.compound_name,
                omitted=excluded,
                develop=sorted(develop),
                main_job=add(develop, keep=True),
                calibration=calibration,
            )
        )
    plan = dict(
        compound_labels=labels,
        classes=classes,
        assignments=assignments,
        contexts=contexts,
        jobs=jobs,
        seeds=list(SEEDS),
        total_fits=len(jobs) * len(SEEDS),
        feature_setting="proteomics_only",
        genes=873,
        representation="log2FC_no_significance_selection",
        calibration_modes=list(CALIBRATION_MODES),
        primary_calibration_mode="inhouse",
        false_alarm_target=0.10,
        fixed_gene_panel=True,
        compound_grouped_across_sources=True,
        test_outcomes_used_for_model_or_cutoff_selection=False,
    )
    validate_plan(plan, cases)
    return plan


def validate_plan(plan, cases):
    labels, jobs = plan["compound_labels"], plan["jobs"]
    universe = set(labels)
    if {c["case_key"] for c in plan["contexts"]} != set(cases.case_key):
        raise ValueError("Outer case inventory changed")
    for job in jobs.values():
        if set(job["train"]) & set(job["predict"]):
            raise ValueError("Identity leakage between training and calibration")
        if {labels[group] for group in job["train"]} != set(plan["classes"]):
            raise ValueError("Missing candidate class in training")
    for context in plan["contexts"]:
        omitted = context["omitted"]
        develop = set(context["develop"])
        if develop != universe - {omitted}:
            raise ValueError("Outer identity not excluded exactly")
        if set(jobs[context["main_job"]]["train"]) != develop:
            raise ValueError("Incorrect deployment training membership")
        tests = []
        for fold in context["calibration"]:
            train, test = set(fold["train"]), set(fold["test"])
            if train != develop - test or omitted in train | test:
                raise ValueError("Outer test identity entered calibration")
            if set(jobs[fold["job"]]["train"]) != train:
                raise ValueError("Calibration fit has the wrong training set")
            tests.extend(fold["test"])
        if len(tests) != len(set(tests)) or set(tests) != develop:
            raise ValueError("Calibration must cover every development identity once")
