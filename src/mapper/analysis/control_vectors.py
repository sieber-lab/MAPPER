"""Deterministic class anchors and compound-label permutations."""

import numpy as np
import pandas as pd

BANK_SEEDS = [202609250, 202609251, 202609252]
NEUTRAL_WORDS = """window curtain carpet shelf lantern pillow basket ladder bridge
garden meadow pebble ribbon button kettle teapot saucer umbrella jacket mitten
notebook pencil eraser folder envelope stamp parcel ticket suitcase cabin balcony
chimney doorway ceiling staircase cupboard drawer mirror clock blanket cushion
sand snow rain cloud wind sun moon star evening morning autumn summer winter
spring valley hill mountain river stream lake shore beach island harbor pathway
village town street square plaza fountain bench fence gate wall tower tunnel
canvas paint brush chalk paper cardboard string rope knot sail mast wheel spoke
handle hinge latch lock key bell flute drum trumpet violin piano guitar melody
rhythm dance song poem story book chapter page letter number pattern shape angle
circle oval triangle rectangle stripe dot shade color silver amber violet orange""".split()


def make_random_banks(semantic):
    centroids = semantic.mean(axis=1)
    mean = centroids.mean(axis=0)
    scale = centroids.std(axis=0, ddof=1)
    residual = semantic - centroids[:, None, :]
    jitter_sd = np.sqrt(np.mean(residual**2, axis=(0, 1)))
    banks = {}
    for i, seed in enumerate(BANK_SEEDS):
        anchors = mean + np.random.default_rng(seed).normal(size=centroids.shape) * scale
        banks[f"random_fixed_{i}"] = np.repeat(anchors[:, None, :], 10, axis=1)
    noise = np.random.default_rng(202609253).normal(size=semantic.shape) * jitter_sd
    banks["random_augmented"] = banks["random_fixed_0"] + noise
    return banks, {"centroid_mean": mean, "centroid_sd": scale, "jitter_sd": jitter_sd}


def nonsense_sequences(classes):
    assert len(NEUTRAL_WORDS) == len(set(NEUTRAL_WORDS))
    rows = []
    for i, name in enumerate(classes):
        rng = np.random.default_rng(202619250 + i)
        pool = rng.choice(NEUTRAL_WORDS, 32, replace=False)
        words = rng.choice(pool, 240, replace=True)
        for aug in range(1, 11):
            rows.append(
                {
                    "MoA": name,
                    "augmentation": aug,
                    "text": " ".join(rng.permutation(words)) + ".",
                    "word_inventory": "|".join(sorted(pool)),
                    "seed": 202619250 + i,
                }
            )
    return pd.DataFrame(rows)


def label_permutation(compound_labels, split, iteration, fold, realization):
    train = sorted(split["train"])
    labels = np.array([compound_labels[c] for c in train])
    seed = 202709250 + 10000 * realization + 100 * iteration + fold
    shuffled = np.random.default_rng(seed).permutation(labels)
    return dict(zip(train, shuffled.tolist())), seed
