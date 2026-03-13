# uncertainty/uncertainty_estimator.py
from __future__ import annotations

from pathlib import Path
import glob
import json
import math
from typing import Dict, Tuple, List, Optional, Iterable, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from joblib import load

try:
    from analysis.plot_style import save_figure
except Exception:  # pragma: no cover - fallback for non-notebook use
    save_figure = None



CANON_MAP = {
    'Fatty acid':      ['Fatty acid','Fatty acid synthesis'],
    'Nitrofurans':     ['Nitrofurans'],
    'Folic acid':      ['Folic acid','Folic acid metabolism','Folic acid biosynthesis'],
    'Membrane':        ['Membrane','Membrane disruption','Membrane integrity'],
    'RNA synth.':      ['RNA synth.','RNA Synthesis','RNA synthesis'],
    'LPS synth.':      ['LPS synth.','lipopolysaccharide synthesis','LPS synthesis'],
    'DNA repl.':       ['DNA repl.','DNA replication','DNA Replication'],
    'Cell wall':       ['Cell wall','Cell wall biosynthesis','Cell-wall biosynthesis'],
    'Protein syn.':    ['Protein syn.','Protein synthesis','Protein Synthesis'],
}

COLORS = [
    "#264653","#287271","#2a9d8f","#8ab17d",
    "#e9c46a","#efb366","#f4a261","#ee8959","#e76f51"
]

DISPLAY_ORDER = [
    'Fatty acid','Nitrofurans','Folic acid','Membrane',
    'RNA synth.','LPS synth.','DNA repl.','Cell wall','Protein syn.'
]





class UncertaintyEstimator:
    """
    End-to-end uncertainty estimator over TabM prediction artifacts.

    Pipeline:
      - load prediction artifacts from PRED_ROOT
      - aggregate to per-compound predictions (PredScore, Margin, ...)
      - compute uncertainty features:
          SeedStd, AugVar, experts-derived metrics (if experts npz present),
          replicate-aware features (BioVar, RepAgree, RepMarginStd/IQR)
      - apply sequential decision:
          1) hard floor on PredScore / Margin (class-aware + absolute) -> override to uncertain
          2) else meta-model -> p_error and flag
      - expose final per-compound table as `.out`
      - provide plotting helpers

    Parameters
    ----------
    pred_root : str | Path
        Output directory produced by predict_primary_tabm.py (contains CSVs + optional experts).
    meta_dir : str | Path
        Directory containing `meta_model.joblib` and `meta_artifact.json`.
    out_dir : str | Path
        Directory where results (CSV, plots) will be written.
    abs_min_score_floor : float
        Absolute score floor for PredScore. If PredScore < this, floor triggers.
    score_floor_factor : float
        Relative floor factor multiplied by (1/C) where C = number of MoAs in predictions.
        score_floor = max(abs_min_score_floor, score_floor_factor * 1/C)
    margin_floor : float
        Optional margin floor. If <= 0, disabled.
    keep_raw_meta_scores : bool
        If True, keep raw meta p_error before floor override.
    fillna_value : float
        Value to fill NaNs in meta feature vector.
    output_prefix : str | None
        Optional prefix added to plot output filenames to avoid overwriting.
    """



    @staticmethod
    def _canonical_moa_name(moa: str) -> str:
        """Return the canonical class name for a MoA string."""
        if moa is None:
            return "Unknown"
        s = str(moa).strip()
        for canon, variants in CANON_MAP.items():
            if s in variants:
                return canon
        return s  # fallback: return itself if not matched

    @staticmethod
    def _moa_to_color(moa: str) -> str:
        """Map a canonical MoA to its corresponding color."""
        canon = UncertaintyEstimator._canonical_moa_name(moa)
        if canon in DISPLAY_ORDER:
            idx = DISPLAY_ORDER.index(canon)
            return COLORS[idx]
        return "#aaaaaa"  # fallback gray

    # -----------------------------
    # Construction
    # -----------------------------
    def __init__(
        self,
        pred_root: str | Path,
        meta_dir: str | Path,
        out_dir: str | Path,
        abs_min_score_floor: float = 0.01,
        score_floor_factor: float = 0.25,
        margin_floor: float = 0.01,
        keep_raw_meta_scores: bool = True,
        fillna_value: float = 0.0,
        output_prefix: str | None = None,
    ):
        self.PRED_ROOT = Path(pred_root)
        self.META_DIR = Path(meta_dir)
        self.OUT_DIR = Path(out_dir)
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)

        self.abs_min_score_floor = float(abs_min_score_floor)
        self.score_floor_factor = float(score_floor_factor)
        self.margin_floor = float(margin_floor)
        self.keep_raw_meta_scores = bool(keep_raw_meta_scores)
        self.fillna_value = float(fillna_value)
        prefix = str(output_prefix).strip() if output_prefix is not None else ""
        if prefix and not prefix.endswith("_"):
            prefix = f"{prefix}_"
        self.output_prefix = prefix

        self.pairwise_mean = None
        self.long_by_seed = None
        self.experts_row_metrics = None
        self.sample_summary = None
        self.comp_summary = None
        self.full_comp_tbl = None

        self.meta_clf = None
        self.meta_art = {}
        self.FEAT_ORDER = []
        self.meta_threshold = 0.5
        self.negate_feats = {"PredScore", "Margin"}

        self.out = None

    def _prefixed(self, name: str) -> str:
        return f"{self.output_prefix}{name}" if self.output_prefix else name

    # -----------------------------
    # Public orchestrator
    # -----------------------------
    def run(self) -> pd.DataFrame:
        """Execute full pipeline and return the final per-compound table."""
        self._load_prediction_artifacts()
        self._aggregate_predictions()
        self._maybe_load_experts()
        self._attach_uncertainty_metrics()
        self._load_meta_model()
        self._apply_meta_with_floor()
        self._write_csv()
        return self.out.copy()

    # -----------------------------
    # I/O helpers
    # -----------------------------
    def _load_prediction_artifacts(self) -> None:
        """Load pairwise mean and long-by-seed CSVs."""
        mean_path = self.PRED_ROOT / "pairwise_predictions.csv"
        long_path = self.PRED_ROOT / "pairwise_predictions_by_seed.csv"
        if not mean_path.exists():
            raise FileNotFoundError(f"Missing {mean_path}")
        if not long_path.exists():
            raise FileNotFoundError(f"Missing {long_path}")

        pm = pd.read_csv(mean_path)
        need = {"SampleID", "CompoundID", "CandidateMoA", "Aug_Index", "Score"}
        if not need.issubset(pm.columns):
            raise ValueError(f"{mean_path} missing required columns; found {pm.columns.tolist()}")
        pm["SampleID"] = pm["SampleID"].astype(str)
        pm["CompoundID"] = pm["CompoundID"].astype(str)
        pm["CandidateMoA"] = pm["CandidateMoA"].astype(str)
        pm["Aug_Index"] = pd.to_numeric(pm["Aug_Index"], errors="coerce").astype(int)
        pm["Score"] = pd.to_numeric(pm["Score"], errors="coerce")

        lb = pd.read_csv(long_path)
        need2 = {"RowID", "SampleID", "CompoundID", "CandidateMoA", "Aug_Index", "Seed", "Score"}
        if not need2.issubset(lb.columns):
            raise ValueError(f"{long_path} missing required columns; found {lb.columns.tolist()}")
        lb["RowID"] = lb["RowID"].astype(str)
        lb["SampleID"] = lb["SampleID"].astype(str)
        lb["CompoundID"] = lb["CompoundID"].astype(str)
        lb["CandidateMoA"] = lb["CandidateMoA"].astype(str)
        lb["Aug_Index"] = pd.to_numeric(lb["Aug_Index"], errors="coerce").astype(int)
        lb["Seed"] = pd.to_numeric(lb["Seed"], errors="coerce").astype(int)
        lb["Score"] = pd.to_numeric(lb["Score"], errors="coerce")

        self.pairwise_mean = pm
        self.long_by_seed = lb

    def _maybe_load_experts(self) -> None:
        """Load experts npz metrics if available (concatenated across seeds)."""
        files = sorted(glob.glob(str(self.PRED_ROOT / "pairwise_experts_seed*.npz")))
        if not files:
            self.experts_row_metrics = None
            return

        rows = []
        for p in files:
            npz = np.load(p, allow_pickle=True)
            X = npz["experts"]  # (R,K,C) or (R,T,K,C)
            row_ids = npz["row_ids"].astype(str)
            if X.ndim == 3:
                X = X[:, None, :, :]  # (R,1,K,C)
            R, T, K, C = X.shape

            eps = 1e-12
            P = np.clip(X, eps, 1 - eps)
            P = P / P.sum(axis=-1, keepdims=True)

            P_mean = P.mean(axis=(1, 2))
            H_pred = self._entropy(P_mean)
            H_exp = self._entropy(P).mean(axis=(1, 2))
            MI = H_pred - H_exp

            max_over_ck = P.max(axis=-1)  # (R,T,K)
            VarMax = max_over_ck.var(axis=(1, 2))

            modal = P.argmax(axis=-1)  # (R,T,K)
            modal_flat = modal.reshape(R, -1)
            VR = np.empty(R, dtype=float)
            KCons = np.empty(R, dtype=float)
            for i in range(R):
                _, cnts = np.unique(modal_flat[i], return_counts=True)
                m = cnts.max()
                agree = m / (T * K)
                KCons[i] = agree
                VR[i] = 1.0 - agree

            M = T * K
            P_flat = P.reshape(R, M, C)
            MeanPairKL = np.zeros(R, dtype=float)
            if M > 1:
                rng = np.random.default_rng(12345)
                for r in range(R):
                    acc, cnt = 0.0, 0
                    n_pairs = min(256, M * (M - 1) // 2)
                    for _ in range(n_pairs):
                        i, j = rng.integers(0, M, size=2)
                        if i == j:
                            continue
                        pi = P_flat[r, i]
                        pj = P_flat[r, j]
                        acc += (pi * (np.log(pi) - np.log(pj))).sum()
                        cnt += 1
                    MeanPairKL[r] = acc / max(cnt, 1)

            AvgMaxProbExp = P.max(axis=-1).mean(axis=(1, 2))
            top_ens = P_mean.argmax(axis=-1)
            rows_idx = np.arange(R)
            VarP_top = P[rows_idx, :, :, top_ens].var(axis=(1, 2))

            mdf = pd.DataFrame(
                {
                    "RowID": row_ids,
                    "PredEntropy": H_pred,
                    "ExpEntropy": H_exp,
                    "EpistemicMI": MI,
                    "MeanPairKL": MeanPairKL,
                    "KConsistency": KCons,
                    "VariationRatio": VR,
                    "AvgMaxProbExp": AvgMaxProbExp,
                    "VarP_top": VarP_top,
                    "VarMax": VarMax,
                }
            )
            rows.append(mdf)

        self.experts_row_metrics = pd.concat(rows, axis=0, ignore_index=True).assign(
            RowID=lambda d: d["RowID"].astype(str)
        )

    # -----------------------------
    # Aggregations & features
    # -----------------------------
    def _aggregate_predictions(self) -> None:
        """Match predict script aggregation to sample- and compound-level."""
        assert self.pairwise_mean is not None

        pair = self.pairwise_mean
        pivot = (
            pair.pivot_table(
                index="SampleID", columns="CandidateMoA", values="Score", aggfunc="mean"
            ).sort_index(axis=1)
        )
        cmp_by_sample = pd.Series(pivot.index, index=pivot.index).map(
            lambda s: str(s).split("_")[0]
        )

        cols = pivot.columns.to_numpy()
        M = pivot.to_numpy()
        Mf = np.where(np.isnan(M), -np.inf, M)
        top1 = np.argmax(Mf, axis=1)
        order = np.argsort(-Mf, axis=1)
        top2 = np.where(Mf.shape[1] >= 2, order[:, 1], order[:, 0])

        sample_summary = pd.DataFrame(
            {
                "SampleID": pivot.index.astype(str),
                "CompoundID": cmp_by_sample.values.astype(str),
                "PredictedMoA": cols[top1].astype(str),
                "PredScore": M[np.arange(M.shape[0]), top1].astype(float),
                "SecondMoA": cols[top2].astype(str),
                "SecondScore": M[np.arange(M.shape[0]), top2].astype(float),
                "Margin": (
                    M[np.arange(M.shape[0]), top1] - M[np.arange(M.shape[0]), top2]
                ).astype(float),
            }
        ).set_index("SampleID")

        comp_scores = pivot.copy()
        comp_scores["CompoundID"] = cmp_by_sample.values.astype(str)
        comp_mean = comp_scores.groupby("CompoundID").mean(numeric_only=True)

        cols2 = comp_mean.columns.to_numpy()
        M2 = comp_mean.to_numpy()
        M2f = np.where(np.isnan(M2), -np.inf, M2)
        t1c = np.argmax(M2f, axis=1)
        ord2 = np.argsort(-M2f, axis=1)
        t2c = np.where(M2f.shape[1] >= 2, ord2[:, 1], ord2[:, 0])

        comp_summary = pd.DataFrame(
            {
                "CompoundID": comp_mean.index.astype(str),
                "PredictedMoA": cols2[t1c].astype(str),
                "PredScore": M2[np.arange(M2.shape[0]), t1c].astype(float),
                "SecondMoA": cols2[t2c].astype(str),
                "SecondScore": M2[np.arange(M2.shape[0]), t2c].astype(float),
                "Margin": (
                    M2[np.arange(M2.shape[0]), t1c] - M2[np.arange(M2.shape[0]), t2c]
                ).astype(float),
            }
        ).set_index("CompoundID")

        self.sample_summary = sample_summary
        self.comp_summary = comp_summary

    def _attach_uncertainty_metrics(self) -> None:
        """Compute and attach SeedStd, AugVar, experts metrics and replicate-aware features."""
        assert self.comp_summary is not None
        assert self.long_by_seed is not None
        assert self.pairwise_mean is not None

        comp_summary = self.comp_summary
        df_join = self.long_by_seed.copy()
        metr_cols = [
            "PredEntropy",
            "ExpEntropy",
            "EpistemicMI",
            "MeanPairKL",
            "KConsistency",
            "VariationRatio",
            "AvgMaxProbExp",
            "VarP_top",
            "VarMax",
        ]

        # Join experts if available
        if self.experts_row_metrics is not None:
            df_join = df_join.merge(self.experts_row_metrics, on="RowID", how="left")

        # SeedStd
        per_seed = df_join.groupby(["CompoundID", "CandidateMoA", "Seed"], as_index=False)[
            "Score"
        ].mean()
        seed_std = (
            per_seed.groupby(["CompoundID", "CandidateMoA"])["Score"]
            .std(ddof=0)
            .unstack("CandidateMoA")
        )

        # AugVar: var across augmentations per (seed,sample) then average across seeds
        aug_unit = (
            df_join.groupby(["CompoundID", "CandidateMoA", "Seed", "SampleID"])["Score"]
            .var(ddof=0)
            .fillna(0.0)
            .reset_index(name="AugVar_unit")
        )
        aug_seed = (
            aug_unit.groupby(["CompoundID", "CandidateMoA", "Seed"])["AugVar_unit"]
            .mean()
            .reset_index(name="AugVar_seed")
        )
        aug_cls = (
            aug_seed.groupby(["CompoundID", "CandidateMoA"])["AugVar_seed"]
            .mean()
            .unstack("CandidateMoA")
        )

        # Experts-derived pair means (if present)
        per_pair_m = None
        if self.experts_row_metrics is not None:
            dfj2 = df_join.dropna(subset=["CandidateMoA", "CompoundID"])
            per_pair_m = dfj2.groupby(["CompoundID", "CandidateMoA"])[metr_cols].mean()

        # Replicate-aware features from pairwise mean
        rep_mean = (
            self.pairwise_mean.groupby(["CompoundID", "SampleID", "CandidateMoA"], as_index=False)[
                "Score"
            ]
            .mean()
            .rename(columns={"Score": "Score"})
        )
        rep_wide = rep_mean.pivot(
            index=["CompoundID", "SampleID"], columns="CandidateMoA", values="Score"
        )

        cols = rep_wide.columns.to_numpy()
        X = rep_wide.to_numpy()
        Xf = np.where(np.isnan(X), -np.inf, X)
        order = np.argsort(-Xf, axis=1)
        top1_idx = order[:, 0]
        top2_idx = np.where(Xf.shape[1] >= 2, order[:, 1], order[:, 0])

        rep_df = rep_wide.reset_index()
        rep_df["rep_top1"] = cols[top1_idx]
        rep_df["rep_margin"] = X[np.arange(X.shape[0]), top1_idx] - X[
            np.arange(X.shape[0]), top2_idx
        ]

        comp_pred = comp_summary["PredictedMoA"]
        rep_df["CompPred"] = rep_df["CompoundID"].map(comp_pred)
        rep_df["agree"] = (rep_df["rep_top1"].astype(str) == rep_df["CompPred"].astype(str)).astype(
            float
        )

        rep_long = rep_wide.reset_index().melt(
            id_vars=["CompoundID", "SampleID"],
            var_name="CandidateMoA",
            value_name="Score",
        )
        rep_long = rep_long.merge(
            comp_summary.reset_index()[["CompoundID", "PredictedMoA"]],
            on="CompoundID",
            how="left",
        )
        rep_long_pred = rep_long[rep_long["CandidateMoA"] == rep_long["PredictedMoA"]]

        bio_var = rep_long_pred.groupby("CompoundID")["Score"].var(ddof=0).rename("BioVar")
        rep_marg_std = rep_df.groupby("CompoundID")["rep_margin"].std(ddof=0).rename(
            "RepMarginStd"
        )
        rep_marg_iqr = (
            rep_df.groupby("CompoundID")["rep_margin"].quantile(0.75)
            - rep_df.groupby("CompoundID")["rep_margin"].quantile(0.25)
        ).rename("RepMarginIQR")
        rep_agree = rep_df.groupby("CompoundID")["agree"].mean().rename("RepAgree")

        def take_for_pred(wide_like: pd.DataFrame, name: str, df: pd.DataFrame):
            vals = []
            for cid, row in df.iterrows():
                pred = str(row["PredictedMoA"])
                try:
                    if isinstance(wide_like.index, pd.MultiIndex):
                        vals.append(wide_like.loc[(cid, pred)])
                    else:
                        vals.append(wide_like.loc[cid, pred])
                except Exception:
                    vals.append(np.nan)
            df[name] = vals
            return df

        agg = comp_summary.copy()
        agg = take_for_pred(seed_std, "SeedStd", agg)
        agg = take_for_pred(aug_cls, "AugVar", agg)

        if per_pair_m is not None:
            for m in metr_cols:
                agg = take_for_pred(per_pair_m[m], m, agg)
        else:
            for m in metr_cols:
                if m not in agg.columns:
                    agg[m] = np.nan

        agg["BioVar"] = agg.index.map(bio_var)
        agg["RepMarginStd"] = agg.index.map(rep_marg_std)
        agg["RepMarginIQR"] = agg.index.map(rep_marg_iqr)
        agg["RepAgree"] = agg.index.map(rep_agree)

        self.full_comp_tbl = agg.reset_index()

    # -----------------------------
    # Meta model & sequential guard
    # -----------------------------
    def _load_meta_model(self) -> None:
        """Load meta model and artifact; prepare feature order and preprocessing."""
        model_path = self.META_DIR / "meta_model.joblib"
        artifact_path = self.META_DIR / "meta_artifact.json"
        if not model_path.exists() or not artifact_path.exists():
            raise FileNotFoundError(f"Missing meta export under {self.META_DIR}")

        self.meta_clf = load(model_path)
        with open(artifact_path, "r") as f:
            self.meta_art = json.load(f)

        self.FEAT_ORDER = list(self.meta_art.get("feature_order", []))
        if not self.FEAT_ORDER:
            raise ValueError("meta_artifact.json missing 'feature_order'")

        self.meta_threshold = float(self.meta_art.get("meta_threshold", 0.5))
        self.negate_feats = set(
            self.meta_art.get("preprocess", {}).get("negate", ["PredScore", "Margin"])
        )

    def _apply_meta_with_floor(self) -> None:
        """Apply sequential decision: floor override → meta-model; build final table."""
        assert self.full_comp_tbl is not None

        # Build meta features
        Z = self.full_comp_tbl.copy()
        for c in self.FEAT_ORDER:
            if c not in Z.columns:
                Z[c] = np.nan
        for c in self.negate_feats:
            if c in Z.columns:
                Z[c] = -pd.to_numeric(Z[c], errors="coerce")

        X_meta = (
            Z[self.FEAT_ORDER].apply(pd.to_numeric, errors="coerce").fillna(self.fillna_value).values
        )
        p_error_meta = self.meta_clf.predict_proba(X_meta)[:, 1]
        flag_meta = (p_error_meta >= self.meta_threshold).astype(int)

        # Floor computation (class-aware)
        assert self.pairwise_mean is not None
        num_classes = int(self.pairwise_mean["CandidateMoA"].astype(str).nunique())
        uniform = 1.0 / max(num_classes, 1)
        score_floor_rel = self.score_floor_factor * uniform
        score_floor = max(self.abs_min_score_floor, score_floor_rel)

        df = self.full_comp_tbl.copy()
        df["p_error_meta"] = p_error_meta
        df["flag_meta_raw"] = flag_meta
        df["score_floor"] = score_floor
        df["margin_floor"] = self.margin_floor

        # Evaluate floors
        reasons = []
        for _, r in df.iterrows():
            rr = []
            if float(r.get("PredScore", np.nan)) < score_floor:
                rr.append("PredScore<floor")
            if self.margin_floor > 0 and float(r.get("Margin", np.nan)) < self.margin_floor:
                rr.append("Margin<floor")
            reasons.append(";".join(rr))
        df["floor_reason"] = reasons
        df["floor_applied"] = df["floor_reason"].str.len() > 0

        if self.keep_raw_meta_scores:
            df["p_error_raw_meta"] = df["p_error_meta"].copy()
        df.loc[df["floor_applied"], "p_error_meta"] = 1.0
        df.loc[df["floor_applied"], "flag_meta_raw"] = 1

        df = df.rename(columns={"p_error_meta": "p_error", "flag_meta_raw": "flag_meta"})

        # Reorder for convenience
        cols_front = [
            "CompoundID",
            "PredictedMoA",
            "PredScore",
            "SecondMoA",
            "SecondScore",
            "Margin",
            "p_error",
            "flag_meta",
            "floor_applied",
            "floor_reason",
            "score_floor",
            "margin_floor",
        ]
        cols_keep = cols_front + [c for c in df.columns if c not in cols_front]
        self.out = df[cols_keep]

    def _write_csv(self) -> None:
        assert self.out is not None
        out_csv = self.OUT_DIR / "compound_uncertainty_meta.csv"
        self.out.to_csv(out_csv, index=False)

    # -----------------------------
    # Plotting helpers
    # -----------------------------
    def plot_p_error_bar(
        self,
        title: str | None = None,
        save: bool = True,
        compound_ids: Iterable[str] | None = None,
        label_map: Mapping[str, str] | None = None,
    ) -> Path | None:
        """
        Bar plot of p_error per compound.

        Parameters
        ----------
        title : optional str
            Plot title.
        save : bool
            Whether to save the plot to OUT_DIR.
        compound_ids : optional iterable of str
            If provided, only these CompoundID values are plotted. If None, all compounds are used.
        label_map : optional mapping of str to str
            Optional display labels for CompoundID values (does not affect filtering).
        """
        assert self.out is not None
        df = self.out.copy()

        if compound_ids is not None:
            sel = {str(c) for c in compound_ids}
            df = df[df["CompoundID"].astype(str).isin(sel)].copy()
        if df.empty:
            raise ValueError("No compounds left after applying compound_ids filter in plot_p_error_bar().")

        # Canonicalize MoAs
        df["CanonMoA"] = df["PredictedMoA"].map(self._canonical_moa_name)
        # Base MoA color
        df["base_color"] = df["CanonMoA"].map(self._moa_to_color)
        # Override color with grey if p_error >= threshold
        df["color"] = np.where(
            df["p_error"] >= self.meta_threshold,
            "#aaaaaa",           # grey for uncertain compounds
            df["base_color"],    # MoA color otherwise
        )

        df = df.sort_values("p_error").reset_index(drop=True)
        labels = df["CompoundID"].astype(str)
        if label_map:
            labels = labels.map(lambda c: label_map.get(c, c))

        plt.figure(figsize=(max(10, len(df)*0.25), 6))
        plt.bar(
            x=np.arange(len(df)),
            height=df["p_error"],
            color=df["color"],
            edgecolor="black",
            linewidth=0.4
        )

        plt.axhline(self.meta_threshold, color="red", linestyle="--", linewidth=1.3)

        # Mark floor overrides
        for i, applied in enumerate(df["floor_applied"]):
            if applied:
                plt.plot([i, i], [0, df.loc[i, "p_error"]],
                         linestyle=":", color="black", linewidth=0.8)

        plt.xlabel("")
        plt.ylabel("Predicted error probability (p_error)")
        plt.title(title or "Meta-model uncertainty across compounds")

        if len(df) <= 30:
            plt.xticks(np.arange(len(df)), labels, rotation=45, ha="right")
        else:
            step = max(1, len(df)//30)
            xt = np.arange(0, len(df), step)
            plt.xticks(xt, labels.iloc[xt], rotation=45, ha="right")

        # **Fixed y-axis to [0, 1] for consistent threshold height**
        plt.ylim(0, 1)

        # Legend using canonical names in DISPLAY_ORDER + uncertain grey
        handles = [
            Patch(facecolor=COLORS[i], edgecolor="black", label=DISPLAY_ORDER[i])
            for i in range(len(DISPLAY_ORDER))
        ]
        # Add explicit entry for uncertain compounds (grey)
        handles.append(
            Patch(facecolor="#aaaaaa", edgecolor="black", label="Uncertain (p_error ≥ threshold)")
        )
        plt.legend(handles=handles, title="Predicted MoA / Uncertainty",
                   bbox_to_anchor=(1.05,1), loc="upper left")

        plt.tight_layout()

        path = None
        if save:
            base = self.OUT_DIR / self._prefixed("compound_p_error_barplot")
            if save_figure is None:
                path = base.with_suffix(".png")
                plt.savefig(path, dpi=200)
            else:
                save_figure(plt.gcf(), base, dpi=200)
                path = base.with_suffix(".png")
        plt.show()
        return path

    def plot_top3_moas(
        self,
        save: bool = True,
        compound_ids: Iterable[str] | None = None,
        label_map: Mapping[str, str] | None = None,
    ) -> Path | None:
        """
        Grouped bar plot of top-3 predicted MoAs per compound.

        Parameters
        ----------
        save : bool
            Whether to save the plot to OUT_DIR.
        compound_ids : optional iterable of str
            If provided, only these CompoundID values are plotted (and their scores used).
            If None, all compounds are used.
        label_map : optional mapping of str to str
            Optional display labels for CompoundID values (does not affect filtering).
        """
        assert self.out is not None
        pairwise_csv = self.PRED_ROOT / "pairwise_predictions.csv"
        if not pairwise_csv.exists():
            raise FileNotFoundError(pairwise_csv)

        df = self.out.copy()
        if compound_ids is not None:
            sel = {str(c) for c in compound_ids}
            df = df[df["CompoundID"].astype(str).isin(sel)].copy()
        if df.empty:
            raise ValueError("No compounds left after applying compound_ids filter in plot_top3_moas().")

        pairwise_df = pd.read_csv(pairwise_csv)
        per_pair = pairwise_df.groupby(["CompoundID", "CandidateMoA"], as_index=False)["Score"].mean()
        comp_wide = per_pair.pivot(index="CompoundID", columns="CandidateMoA", values="Score")

        # Order compounds by increasing p_error (on the filtered df)
        order = df.sort_values("p_error")["CompoundID"].astype(str).tolist()
        comp_wide = comp_wide.reindex(order)

        # canonicalize column MoAs
        comp_wide.columns = [self._canonical_moa_name(c) for c in comp_wide.columns]

        # Extract top3
        rows = []
        for cid, row in comp_wide.iterrows():
            vals = row.dropna()
            if vals.empty:
                continue
            top3 = vals.sort_values(ascending=False).head(3)
            for rank, (moa, prob) in enumerate(top3.items()):
                rows.append({"CompoundID": cid, "MoA": moa, "Prob": float(prob), "Rank": rank})

        top3 = pd.DataFrame(rows)
        if top3.empty:
            raise RuntimeError("No rows for top-3 plot (after filtering / NaNs).")

        top3["color"] = top3["MoA"].map(self._moa_to_color)

        x_centers = np.arange(len(order))
        bar_width = 0.24
        offsets = {r: (r - 1)*bar_width for r in [0,1,2]}

        top3["x"] = top3.apply(
            lambda r: x_centers[order.index(r["CompoundID"])] + offsets[int(r["Rank"])],
            axis=1
        )

        plt.figure(figsize=(max(10, len(order)*0.28), 6))
        plt.bar(
            top3["x"],
            top3["Prob"],
            width=bar_width*0.95,
            color=top3["color"],
            edgecolor="black",
            linewidth=0.4
        )

        plt.ylabel("Probability")
        plt.xlabel("")
        plt.title("Top-3 predicted MoAs per compound")
        labels = order
        if label_map:
            labels = [label_map.get(cid, cid) for cid in order]
        plt.xticks(x_centers, labels, rotation=45, ha="right")
        plt.ylim(0, 1)

        # Legend strictly in DISPLAY_ORDER
        handles = [
            Patch(facecolor=COLORS[i], edgecolor="black", label=DISPLAY_ORDER[i])
            for i in range(len(DISPLAY_ORDER))
        ]
        plt.legend(handles=handles, title="MoA",
                   bbox_to_anchor=(1.02,1), loc="upper left", frameon=False)

        plt.tight_layout()

        path = None
        if save:
            base = self.OUT_DIR / self._prefixed("compound_top3_moa_grouped_bars")
            if save_figure is None:
                path = base.with_suffix(".png")
                plt.savefig(path, dpi=200)
            else:
                save_figure(plt.gcf(), base, dpi=200)
                path = base.with_suffix(".png")
        plt.show()
        return path


    def plot_margin_vs_p_error(
        self,
        annotate_n: int = 10,
        save: bool = True,
        compound_ids: Iterable[str] | None = None,
        label_map: Mapping[str, str] | None = None,
    ) -> Path | None:
        """
        Scatter of Margin vs p_error, colored by canonical MoA.

        Parameters
        ----------
        annotate_n : int
            Number of uncertain high-margin points to annotate.
        save : bool
            Whether to save the plot to OUT_DIR.
        compound_ids : optional iterable of str
            If provided, only these CompoundID values are plotted. If None, all compounds are used.
        label_map : optional mapping of str to str
            Optional display labels for CompoundID values (does not affect filtering).
        """
        assert self.out is not None
        df = self.out.copy()

        if compound_ids is not None:
            sel = {str(c) for c in compound_ids}
            df = df[df["CompoundID"].astype(str).isin(sel)].copy()
        if df.empty:
            raise ValueError("No compounds left after applying compound_ids filter in plot_margin_vs_p_error().")

        df["CanonMoA"] = df["PredictedMoA"].map(self._canonical_moa_name)
        df["color"] = df["CanonMoA"].map(self._moa_to_color)

        df["uncertain"] = (df["p_error"] >= self.meta_threshold).astype(int)

        plt.figure(figsize=(7.5,5.5))
        plt.scatter(
            df["Margin"],
            df["p_error"],
            s=32,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
            c=df["color"]
        )

        plt.axhline(self.meta_threshold, color="red", linestyle="--", linewidth=1.1)

        # Annotate uncertain high-margin
        cand = df[df["uncertain"]==1].sort_values(
            ["Margin","p_error"], ascending=[False,False]
        ).head(annotate_n)

        for _, r in cand.iterrows():
            label = r["CompoundID"]
            if label_map:
                label = label_map.get(str(r["CompoundID"]), label)
            plt.annotate(label,
                         (r["Margin"], r["p_error"]),
                         xytext=(4,4),
                         textcoords="offset points",
                         fontsize=8)

        plt.xlabel("Margin (top1 - top2)")
        plt.ylabel("p_error (meta-model)")
        plt.title("Margin vs. p_error (canonical MoA coloring)")
        plt.grid(alpha=0.25)

        # Legend in DISPLAY_ORDER
        handles = [
            Patch(facecolor=COLORS[i], edgecolor="black", label=DISPLAY_ORDER[i])
            for i in range(len(DISPLAY_ORDER))
        ]
        plt.legend(handles=handles, title="MoA", loc="lower right", frameon=False)

        plt.tight_layout()

        path = None
        if save:
            base = self.OUT_DIR / self._prefixed("compound_margin_vs_p_error_scatter")
            if save_figure is None:
                path = base.with_suffix(".png")
                plt.savefig(path, dpi=200)
            else:
                save_figure(plt.gcf(), base, dpi=200)
                path = base.with_suffix(".png")
        plt.show()
        return path

    # Optional: contribution “waterfall” plots (requires Calibrated LR over PolyFeatures).
    def plot_waterfall(
        self,
        compound_ids: Iterable[str] | None = None,
        max_pos: int = 6,
        max_neg: int = 6,
        save: bool = False,
        label_map: Mapping[str, str] | None = None,
    ) -> List[Path]:
        """
        Plot meta-model logit contributions per base feature for given compounds.
        Requires the meta-model to be a CalibratedClassifierCV wrapping a Pipeline
        (StandardScaler, PolynomialFeatures, LogisticRegression).

        Parameters
        ----------
        compound_ids : iterable of str or None
            Which compounds to plot. If None, all compounds in self.out are used.
        max_pos : int
            Max number of strongest positive contributors to show.
        max_neg : int
            Max number of strongest negative contributors to show.
        save : bool
            Whether to save PNG/PDF/SVG to OUT_DIR.
        label_map : optional mapping of str to str
            Optional display labels for CompoundID values (does not affect filtering).
        """
        assert self.out is not None
        if compound_ids is None:
            compound_ids = self.out["CompoundID"].tolist()

        scaler, poly, lr = self._extract_inner_lr_pipeline()

        # Build risk-space features (negate where needed, align order)
        Z = self.out.set_index("CompoundID")[self.FEAT_ORDER].copy()
        for c in self.negate_feats:
            if c in Z:
                Z[c] = -pd.to_numeric(Z[c], errors="coerce")
        for c in self.FEAT_ORDER:
            if c not in Z:
                Z[c] = 0.0
        X_risk = Z[self.FEAT_ORDER].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        poly_names = poly.get_feature_names_out(self.FEAT_ORDER)
        coef = lr.coef_.ravel()
        intercept = float(lr.intercept_.ravel()[0])
        X_std = scaler.transform(X_risk.values)
        X_poly = poly.transform(X_std)
        contrib_terms = X_poly * coef  # (N, n_poly_features)

        paths = []
        for cid in compound_ids:
            if cid not in X_risk.index:
                continue
            idx = X_risk.index.get_loc(cid)
            contrib_row = contrib_terms[idx, :]
            contrib_map = self._aggregate_by_base_feature(contrib_row, poly_names, self.FEAT_ORDER)
            c = pd.Series(contrib_map)

            pos = c[c > 0.0].sort_values(ascending=False).head(max_pos)
            neg = c[c < 0.0].sort_values(ascending=True).head(max_neg)
            feat_names = list(pos.index) + list(neg.index)
            values = np.r_[pos.values, neg.values]
            colors = ["#b22222"] * len(pos) + ["#228b22"] * len(neg)

            row = self.out.set_index("CompoundID").loc[cid]
            label = cid
            if label_map:
                label = label_map.get(str(cid), label)
            title = f"{label} — MoA={row.get('PredictedMoA','?')} | p_error={row.get('p_error', np.nan):.3f} | Margin={row.get('Margin', np.nan):.3f}"

            fig, ax = plt.subplots(figsize=(8, 4.6))
            y_pos = np.arange(len(feat_names))
            ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.6)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("Logit contribution (meta-model)")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feat_names)
            plt.tight_layout()
            p = None
            if save:
                base = self.OUT_DIR / self._prefixed(f"waterfall_{cid}")
                if save_figure is None:
                    p = base.with_suffix(".png")
                    plt.savefig(p, dpi=200)
                else:
                    save_figure(fig, base, dpi=200)
                    p = base.with_suffix(".png")
                paths.append(p)
            plt.show()

        return paths

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return -(p * np.log(p)).sum(axis=-1)

    def _extract_inner_lr_pipeline(self):
        """
        Extract (StandardScaler, PolynomialFeatures, LogisticRegression) from a
        CalibratedClassifierCV wrapping a sklearn Pipeline. Tries to be version-robust.
        """
        clf = self.meta_clf
        try:
            cc = clf.calibrated_classifiers_[0]
        except Exception as e:
            raise RuntimeError(
                "Unexpected meta-model: expected CalibratedClassifierCV with calibrated_classifiers_."
            ) from e

        base = getattr(cc, "base_estimator", None) or getattr(cc, "estimator", None)
        if base is None:
            for attr in ("classifier", "model", "pipeline"):
                base = getattr(cc, attr, None) or base
        if base is None:
            raise RuntimeError("Could not locate wrapped estimator in calibrated classifier.")

        try:
            steps = dict(base.named_steps)
        except Exception as e:
            raise RuntimeError(f"Wrapped estimator is not a sklearn Pipeline. Got: {type(base)}") from e

        scaler = steps.get("standardscaler", None)
        poly = steps.get("polynomialfeatures", None)
        lr = steps.get("logisticregression", None)
        missing = [n for n, obj in [("StandardScaler", scaler), ("PolynomialFeatures", poly), ("LogisticRegression", lr)] if obj is None]
        if missing:
            raise RuntimeError(f"Pipeline missing steps: {missing}. Found steps: {list(steps.keys())}")
        return scaler, poly, lr

    @staticmethod
    def _aggregate_by_base_feature(
        contrib_row: np.ndarray, poly_names: np.ndarray, base_feats: List[str]
    ) -> Dict[str, float]:
        from collections import defaultdict

        acc = defaultdict(float)
        for val, pname in zip(contrib_row, poly_names):
            terms = [t.strip() for t in pname.split(" ")]
            bases = set([t.split("^")[0] for t in terms])
            for b in bases:
                if b in base_feats:
                    acc[b] += float(val)
        return {f: acc.get(f, 0.0) for f in base_feats}
