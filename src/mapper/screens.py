"""Run the fixed feature/encoder screens using their recorded configurations."""

import json
from pathlib import Path

from .paths import repository_root


def configurations(study):
    path = repository_root() / "data/configurations/screens.json"
    return [item for item in json.loads(path.read_text()) if item["study"] == study]


def prepare_configuration(config):
    from .training import load_base_data, make_feature_dicts, assemble_pairwise_training_table

    root = repository_root()
    data = load_base_data(
        base_dir=root / "data/inhouse",
        **{
            "emb_" + key: root / "data/embeddings" / name
            for key, name in [
                ("sem", "250808_semanticAugmentation_embeddings.pkl"),
                ("sentence", "250808_sentenceAugmentation_embeddings.pkl"),
                ("word", "250808_wordAugmentation_embeddings.pkl"),
            ]
        },
    )
    features = make_feature_dicts(
        data["X_proteomics"], data["X_ecfp"], data["X_infoalign"], data["X_mic"], data["X_gc"]
    )
    table, values, target = assemble_pairwise_training_table(
        data["text_embeddings"][config["augmentation"]],
        config["encoder"],
        data["y_labels"],
        data["X_proteomics"].index,
        features,
        config["stored_feature_order"],
    )
    return table, values, target


def fit_configuration(config, output, leave_one_out=False, pilot=None):
    import pandas as pd
    from .evaluation import evaluate_model, evaluate_model_LOO

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    table, values, target = prepare_configuration(config)
    values = pd.DataFrame(values, index=table.index)
    common = dict(
        model_type=config["model"],
        X=values,
        y=target,
        groups=table["CompoundID"],
        sample_ids=table["SampleID"],
        candidate_moas=table["Text_MoA"],
        predictions_dir=str(output),
        run_name=config["original_configuration"],
    )
    if leave_one_out:
        return evaluate_model_LOO(
            **common,
            seed_repeats=1 if pilot else 5,
            base_seed=42,
            return_experts=True,
            test_compounds=[pilot[2]] if pilot else None,
        )
    per_sample = table.drop_duplicates("SampleID").set_index("SampleID")
    return evaluate_model(
        **common,
        true_moa_by_sample=per_sample["True_MoA"],
        compound_by_sample=per_sample["CompoundID"],
        cv_splits=config["cv_splits"],
        seed_splits=config["seed_splits"],
        base_seed=config["base_seed"],
        cv_iterations=config["cv_iterations"],
        folds_manifest=str(repository_root() / config["fold_manifest"]),
        fold_selection=[pilot[:2]] if pilot else None,
    )
