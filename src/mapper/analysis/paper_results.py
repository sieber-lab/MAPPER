"""Independently recompute manuscript-facing quantities from frozen tables."""

from pathlib import Path
import json
import sys
import traceback
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from mapper.paths import repository_root

ROOT = repository_root()
HERE = ROOT / "results/reports"
HERE.mkdir(parents=True, exist_ok=True)
P = ROOT / "data/figure_inputs"
R = []


def read(g, n):
    return pd.read_csv(P / g / n, float_precision="round_trip")


def near(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=2e-12)


def run(name, fun):
    try:
        R.append(dict(check=name, status="pass", result=fun()))
    except Exception:
        R.append(dict(check=name, status="investigate", traceback=traceback.format_exc()))
    (HERE / "recomputed_results.json").write_text(
        json.dumps(R, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
        + "\n"
    )
    print(name, R[-1]["status"], flush=True)


def primary():
    data = read("baselines", "primary_predictions.csv")
    summary = read("baselines", "performance_metrics.csv")
    results = []
    for arm, d in data.groupby("arm"):
        assert len(d) == 255 and d.CompoundID.nunique() == 51
        assert not d.duplicated(["iteration", "CompoundID"]).any()
        values = {k: [] for k in ["top1_accuracy", "top2_accuracy", "macro_f1"]}
        for _, x in d.groupby("iteration"):
            assert len(x) == 51
            values["top1_accuracy"].append(x.True_MoA.eq(x.predicted).mean())
            values["top2_accuracy"].append(
                (x.True_MoA.eq(x.predicted) | x.True_MoA.eq(x.second)).mean()
            )
            values["macro_f1"].append(f1_score(x.True_MoA, x.predicted, average="macro"))
        for metric, v in values.items():
            estimate = float(np.mean(v))
            saved = summary.query("arm==@arm and metric==@metric").estimate.item()
            near(estimate, saved)
        results.append(dict(arm=arm, **{k: float(np.mean(v)) for k, v in values.items()}))
    return results


def confusion_and_calibration():
    calls = read("baselines", "primary_predictions.csv")
    conf = read("baselines", "main_model_confusion_source.csv")
    for row in conf.itertuples():
        d = calls.query("arm==@row.arm")
        count = (d.True_MoA.eq(row.Reference_MoA) & d.predicted.eq(row.Predicted_MoA)).sum() / 5
        near(count, row.mean_compounds_per_repeat)
        near(100 * count / (d.True_MoA.eq(row.Reference_MoA).sum() / 5), row.percentage)
    bins = read("baselines", "calibration_bins.csv")
    for row in bins.itertuples():
        d = calls.query("arm==@row.arm and bin==@row.bin")
        if len(d) == 0:
            continue
        near(d.Score.mean(), row.mean_score)
        near(d.event.mean(), row.observed_event_frequency)
        assert d.CompoundID.nunique() == row.n_distinct_compounds
    return dict(confusion_cells=len(conf), calibration_bins=len(bins))


def screen():
    d = read("model_screen", "all_fold_metrics.csv")
    assert len(d) == 4800
    assert (
        d.groupby("configuration").size().eq(25).all()
        and not d.duplicated(["configuration", "cv_cycle"]).any()
    )
    winners = read("model_screen", "summary_winners_all.csv")
    for row in winners.itertuples():
        means = d[d.model_type.eq(row.model)].groupby("var_tag")[row.metric].mean()
        assert means.idxmax() == row.best
        near(means.max(), row.mean)
        wide = d[d.model_type.eq(row.model)].pivot(
            index="cv_cycle", columns="var_tag", values=row.metric
        )
        test = stats.friedmanchisquare(*wide.to_numpy().T)
        near(test.pvalue, row.friedman_p_raw)
    table = read("model_screen", "table_s7_pr_auc_with_ci.csv")
    for label, model, tag, mean, sd, n, lo, hi in table.itertuples(index=False, name=None):
        a = d[d.model_type.eq(model) & d.var_tag.eq(tag)].pr_auc
        near(
            [a.mean(), a.std(), *stats.t.interval(0.95, 24, loc=a.mean(), scale=stats.sem(a))],
            [mean, sd, lo, hi],
        )
    checks = read("model_screen", "raw_checks.csv")
    for c in [
        "valid_profile_candidate_counts",
        "shared_fold_manifest",
        "original_compound_score_cache_matches",
        "independently_verified_compound_metrics",
    ]:
        assert checks[c].all()
    return dict(
        fold_evaluations=len(d),
        configurations=d.configuration.nunique(),
        winner_conditions=len(winners),
        confidence_intervals=len(table),
    )


def figure3():
    d = read("figure3", "metric_source.csv")
    om = read("figure3", "omnibus_tests.csv")
    pairs = read("figure3", "pairwise_tests.csv")
    for row in om.itertuples():
        x = d[(d.model == row.model) & (d.metric_key == row.metric)].pivot(
            index=["iteration", "fold"], columns="arm", values="value"
        )
        assert x.shape == (25, 3)
        s = stats.friedmanchisquare(*x.to_numpy().T)
        near(
            [s.statistic, s.pvalue, min(1, 3 * s.pvalue)],
            [row.chi_square, row.p_raw, row.p_bonferroni],
        )
    for row in pairs.itertuples():
        x = d[(d.model == row.model) & (d.metric_key == row.metric)]
        near(
            [x[x.arm == row.arm_a].value.mean(), x[x.arm == row.arm_b].value.mean()],
            [row.mean_a, row.mean_b],
        )
    import itertools

    for (model, metric), part in d.groupby(["model", "metric_key"]):
        wide = part.pivot(index=["iteration", "fold"], columns="arm", values="value")
        a = wide.to_numpy()
        n, k = a.shape
        ranks = stats.rankdata(a, axis=1)
        sums = ranks.sum(axis=0)
        s2 = ((ranks * ranks).sum() - k * n * (k + 1.0) ** 2 / 4.0) / (k - 1.0)
        t2 = ((sums - n * (k + 1.0) / 2.0) ** 2).sum() / s2
        df = n * k - k - n + 1
        variance = s2 * (2.0 * n * (k - 1.0)) / df * (1.0 - t2 / (n * (k - 1.0)))
        combinations = list(itertools.combinations(range(k), 2))
        raw = np.array(
            [
                2 * stats.t.sf(abs(sums[i] - sums[j]) / np.sqrt(variance), df)
                for i, j in combinations
            ]
        )
        order = np.argsort(raw)
        adjusted = np.empty(len(raw))
        adjusted[order] = np.minimum(
            1,
            np.minimum.accumulate((raw[order] * len(raw) / np.arange(1, len(raw) + 1))[::-1])[::-1],
        )
        saved = pairs[(pairs.model == model) & (pairs.metric == metric)]
        for (i, j), p, q in zip(combinations, raw, adjusted):
            labels = {wide.columns[i], wide.columns[j]}
            row = saved[saved.apply(lambda r: {r.arm_a, r.arm_b} == labels, axis=1)].iloc[0]
            near([p, q], [row.p_raw, row.q_bh])
    # Independent repeated-measures ANOVA from a balanced subjects x conditions matrix.
    enc = read("encoders", "rm_anova_input.csv")
    saved = read("encoders", "anova_summary.csv")
    estimates = {}
    for metric in ["pr_auc", "roc_auc", "mcc"]:
        x = enc.pivot(index="cv_cycle", columns="method", values=metric).to_numpy()
        n, k = x.shape
        grand = x.mean()
        ss_cond = n * ((x.mean(axis=0) - grand) ** 2).sum()
        ss_error = ((x - x.mean(axis=0) - x.mean(axis=1)[:, None] + grand) ** 2).sum()
        f = (ss_cond / (k - 1)) / (ss_error / ((n - 1) * (k - 1)))
        estimates[metric] = dict(F=f, p=stats.f.sf(f, k - 1, (n - 1) * (k - 1)))
    (HERE / "encoder_anova_independent.json").write_text(json.dumps(estimates, indent=2) + "\n")
    return dict(
        training_omnibus_tests=len(om),
        training_pairwise_means=len(pairs),
        encoder_anova=estimates,
        saved_anova=saved.to_dict("records"),
    )


def external():
    cases = read("external_final", "current_case_predictions.csv")
    scores = read("external_final", "current_class_scores.csv")
    other = read("external_comparison", "case_comparison_with_similarity.csv")
    pd.testing.assert_frame_equal(cases, other)
    assert len(scores) == 43 * 3 * 9 and len(cases) == 43 * 3
    rows = []
    for arm, x in cases.groupby("arm"):
        assert x.case_key.nunique() == 43 and x.compound_key.nunique() == 29
        assert x.compound_excluded.all()
        assert x.train_compounds.eq(np.where(x.training_compound.notna(), 50, 51)).all()
        assert x.uncertain.eq(x.score <= x.score_cutoff).all()
        for row in x.itertuples():
            z = scores[(scores.arm == arm) & (scores.case_key == row.case_key)].sort_values("rank")
            assert z["rank"].tolist() == list(range(1, 10)) and z.candidate.nunique() == 9
            assert z.iloc[0].candidate == row.predicted
            near(z.iloc[0].score, row.score)
            assert z.loc[z.candidate.eq(row.expected), "rank"].item() == row.reference_rank
        known = x[~x.uncertain]
        distant = x[x.distant_unseen_identity]
        rows.append(
            dict(
                arm=arm,
                top1=int(x.reference_rank.le(1).sum()),
                top2=int(x.reference_rank.le(2).sum()),
                top3=int(x.reference_rank.le(3).sum()),
                certain=len(known),
                certain_correct=int(known.reference_rank.eq(1).sum()),
                distant=len(distant),
                distant_correct=int(distant.reference_rank.eq(1).sum()),
                balanced=float(x.reference_rank.eq(1).groupby(x.expected).mean().mean()),
            )
        )
    return rows


def yu():
    calls = read("yu", "yu_paired_predictions.csv")
    summary = read("yu", "yu_outcome_counts.csv")
    rows = []
    for setting, x in calls.groupby("training_setting"):
        assert len(x) == 19 and x.case_key.nunique() == 19
        correct = x.predicted.eq(x.expected)
        s = summary[summary.training_setting == setting].iloc[0]
        actual = dict(
            correct=int(correct.sum()),
            correct_certain=int((correct & ~x.uncertain).sum()),
            incorrect_certain=int((~correct & ~x.uncertain).sum()),
            correct_uncertain=int((correct & x.uncertain).sum()),
            incorrect_uncertain=int((~correct & x.uncertain).sum()),
        )
        for k, v in actual.items():
            near(v, s[k])
        near(100 * correct.groupby(x.expected).mean().mean(), s.balanced_accuracy)
        rows.append(dict(setting=setting, **actual, balanced=s.balanced_accuracy))
    return rows


def structural():
    calls = read("structural_groups", "all_candidate_scores_by_cv.csv")
    groups = read("structural_groups", "compound_similarity_groups.csv")
    metrics = read("structural_groups", "performance_summary.csv")
    examples = set(groups[groups.illustrated_example].CompoundID)
    assert examples == set(groups.sort_values("mean_tanimoto").head(6).CompoundID)
    detail = read("structural_detail", "compound_predictions_by_cv.csv")
    piv = detail.groupby(["CompoundID", "arm"]).true_class_score.mean().unstack()
    assert (piv.proteomics_ecfp < piv.proteomics_only).all()
    results = metrics[metrics.metric.eq("top1_accuracy")].to_dict("records")
    return dict(
        six_examples=list(sorted(examples)),
        all_six_correct_class_scores_reduced_by_ecfp=True,
        performance=results,
        candidate_schema=list(calls.columns),
    )


def qc():
    profiles = read("qc", "development_profiles.csv")
    summ = read("qc", "replicate_summaries.csv")
    counts = profiles.groupby("compound_id").size()
    assert (
        len(profiles) == 200
        and len(counts) == 51
        and counts.value_counts().to_dict() == {4: 47, 3: 4}
    )
    a = summ[summ.group_type.eq("Antibiotic")]
    c = summ[summ.group_type.eq("Solvent control")]
    return dict(
        profiles=len(profiles),
        counts=counts.value_counts().to_dict(),
        three_profile_compounds=counts[counts.eq(3)].index.tolist(),
        operator_counts=profiles.operator.value_counts().to_dict(),
        protein_group_range=[
            int(profiles.observed_protein_groups.min()),
            int(profiles.observed_protein_groups.max()),
        ],
        protein_groups_median=float(profiles.observed_protein_groups.median()),
        missing_median=float(profiles.missing_fraction.median()),
        antibiotic_correlation_median=float(a.median_pairwise_pearson.median()),
        antibiotic_protein_cv_median=float(a.median_cv_complete.median()),
        control_correlation_median=float(c.median_pairwise_pearson.median()),
        control_protein_cv_median=float(c.median_cv_complete.median()),
        control_groups=len(c),
    )


def bootstrap_selection():
    d = read("baselines", "selection_summary.csv")
    assert d.n_resamples.eq(2000).all()
    p = ROOT / "data/analysis/configuration_selection"
    iterations = pd.read_csv(p / "policy_iterations.csv")
    summ = pd.read_csv(p / "policy_summary.csv")
    return dict(
        selection_summary=d.to_dict("records"),
        iteration_columns=list(iterations),
        policy_summary=summ.to_dict("records"),
    )


def omission_association():
    import itertools

    part = read("difficulty", "class_summary.csv").query('method=="low_max_score" and scope=="ood"')
    x = stats.rankdata(part.class_size)
    y = stats.rankdata(part.flag_fraction)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    observed = float(x @ y / denom)
    values = np.asarray(list(itertools.permutations(y)), float) @ x / denom
    p = float(np.mean(abs(values) >= abs(observed) - 1e-12))
    saved = (
        read("difficulty", "association_tests.csv")
        .query('method=="low_max_score" and scope=="ood" and predictor=="class_size"')
        .iloc[0]
    )
    near([observed, p], [saved.spearman, saved.p_raw])
    return dict(spearman=observed, exact_p=p, permutations=len(values))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    for name, fun in [
        ("primary_multiclass_metrics", primary),
        ("confusion_and_calibration", confusion_and_calibration),
        ("all_model_screen", screen),
        ("figure3_statistics", figure3),
        ("external_scores_ranks_flags_and_exclusions", external),
        ("yu_adaptation", yu),
        ("structural_coverage", structural),
        ("proteomics_QC", qc),
        ("selection_summary", bootstrap_selection),
        ("class_size_exact_test", omission_association),
    ]:
        run(name, fun)

    if any(row["status"] != "pass" for row in R):
        raise SystemExit(1)
