from pathlib import Path
import csv
import shutil

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

RAW_DIR = Path(
    "center_match_diagnostics/raw_candidates"
)

NORMALIZED_DIR = Path(
    "center_match_diagnostics/normalized_candidates"
)

OUTPUT_ROOT = Path(
    "center_unique_candidates"
)

UNIQUE_RAW_DIR = (
    OUTPUT_ROOT / "raw"
)

UNIQUE_NORMALIZED_DIR = (
    OUTPUT_ROOT / "normalized"
)

CLUSTER_CSV = (
    OUTPUT_ROOT / "clusters.csv"
)

UNIQUE_RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

UNIQUE_NORMALIZED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Similarity threshold
#
# Higher = stricter.
#
# 0.96 means two glyphs need to be extremely similar
# before we consider them duplicates.
#
# We intentionally start conservative so we do NOT accidentally
# merge, for example, a 3 and an 8.
# ------------------------------------------------------------

DUPLICATE_THRESHOLD = 0.96


# ============================================================
# SIMILARITY
# ============================================================

def similarity_score(a, b):
    """
    Normalized correlation between two already-normalized
    candidate glyphs.
    """

    if a is None or b is None:
        return 0.0

    if a.shape != b.shape:
        b = cv2.resize(
            b,
            (
                a.shape[1],
                a.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    a = a.astype(
        np.float32
    )

    b = b.astype(
        np.float32
    )

    a_std = a.std()
    b_std = b.std()

    if (
        a_std < 1e-6
        or b_std < 1e-6
    ):
        return 0.0

    a = (
        a - a.mean()
    ) / a_std

    b = (
        b - b.mean()
    ) / b_std

    return float(
        np.mean(
            a * b
        )
    )


# ============================================================
# LOAD CANDIDATES
# ============================================================

def load_candidates():
    """
    Load candidate_XXXX_norm.png and locate its corresponding
    candidate_XXXX.png raw image.
    """

    candidates = []

    normalized_paths = sorted(
        NORMALIZED_DIR.glob(
            "candidate_*_norm.png"
        )
    )

    for norm_path in normalized_paths:

        stem = norm_path.stem

        # candidate_0001_norm
        candidate_id = stem.replace(
            "_norm",
            ""
        )

        raw_path = (
            RAW_DIR
            / f"{candidate_id}.png"
        )

        if not raw_path.exists():
            print(
                f"Warning: missing raw image "
                f"for {candidate_id}"
            )
            continue

        normalized = cv2.imread(
            str(norm_path),
            cv2.IMREAD_GRAYSCALE
        )

        if normalized is None:
            continue

        candidates.append(
            {
                "id": candidate_id,
                "raw_path": raw_path,
                "norm_path": norm_path,
                "normalized": normalized,
            }
        )

    return candidates


# ============================================================
# GREEDY CLUSTERING
# ============================================================

def cluster_candidates(candidates):
    """
    Greedy clustering.

    Each new candidate is compared against the representative
    image of every existing cluster.

    If similarity >= threshold, it joins that cluster.

    Otherwise a new cluster is created.
    """

    clusters = []

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        best_cluster_index = None
        best_score = -1.0

        for cluster_index, cluster in enumerate(
            clusters
        ):
            representative = (
                cluster["representative"]
                ["normalized"]
            )

            score = similarity_score(
                candidate["normalized"],
                representative
            )

            if score > best_score:
                best_score = score
                best_cluster_index = (
                    cluster_index
                )

        if (
            best_cluster_index is not None
            and best_score
            >= DUPLICATE_THRESHOLD
        ):
            cluster = clusters[
                best_cluster_index
            ]

            cluster["members"].append(
                {
                    "candidate": candidate,
                    "similarity": best_score,
                }
            )

        else:
            clusters.append(
                {
                    "representative": candidate,
                    "members": [
                        {
                            "candidate": candidate,
                            "similarity": 1.0,
                        }
                    ],
                }
            )

        if (
            index % 10 == 0
            or index == len(candidates)
        ):
            print(
                f"Processed "
                f"{index}/{len(candidates)} "
                f"-> {len(clusters)} clusters"
            )

    return clusters


# ============================================================
# SAVE CLUSTERS
# ============================================================

def save_clusters(clusters):
    rows = []

    for cluster_number, cluster in enumerate(
        clusters,
        start=1
    ):

        representative = (
            cluster["representative"]
        )

        cluster_name = (
            f"unique_{cluster_number:03d}"
        )

        # ----------------------------------------------------
        # SAVE REPRESENTATIVE RAW IMAGE
        # ----------------------------------------------------

        raw_destination = (
            UNIQUE_RAW_DIR
            / f"{cluster_name}.png"
        )

        shutil.copy2(
            representative["raw_path"],
            raw_destination
        )

        # ----------------------------------------------------
        # SAVE REPRESENTATIVE NORMALIZED IMAGE
        # ----------------------------------------------------

        norm_destination = (
            UNIQUE_NORMALIZED_DIR
            / f"{cluster_name}_norm.png"
        )

        shutil.copy2(
            representative["norm_path"],
            norm_destination
        )

        # ----------------------------------------------------
        # MAPPING CSV
        # ----------------------------------------------------

        members = cluster[
            "members"
        ]

        member_ids = [
            member[
                "candidate"
            ]["id"]
            for member in members
        ]

        similarities = [
            member[
                "similarity"
            ]
            for member in members
        ]

        rows.append(
            {
                "cluster": cluster_name,
                "representative":
                    representative["id"],
                "member_count":
                    len(members),
                "members":
                    ";".join(member_ids),
                "similarities":
                    ";".join(
                        f"{score:.4f}"
                        for score in similarities
                    ),
            }
        )

    with open(
        CLUSTER_CSV,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "cluster",
                "representative",
                "member_count",
                "members",
                "similarities",
            ]
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    candidate_count,
    clusters
):
    sizes = [
        len(
            cluster["members"]
        )
        for cluster in clusters
    ]

    print()
    print(
        "--- CENTER CANDIDATE DEDUPE COMPLETE ---"
    )

    print(
        f"Original candidates: "
        f"{candidate_count}"
    )

    print(
        f"Unique clusters: "
        f"{len(clusters)}"
    )

    print(
        f"Duplicates removed: "
        f"{candidate_count - len(clusters)}"
    )

    if sizes:

        print(
            f"Largest cluster: "
            f"{max(sizes)} images"
        )

        print(
            f"Average cluster size: "
            f"{np.mean(sizes):.2f}"
        )

    print()

    print(
        f"Unique raw images: "
        f"{UNIQUE_RAW_DIR.resolve()}"
    )

    print(
        f"Unique normalized images: "
        f"{UNIQUE_NORMALIZED_DIR.resolve()}"
    )

    print(
        f"Cluster mapping: "
        f"{CLUSTER_CSV.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    candidates = (
        load_candidates()
    )

    if not candidates:
        print(
            "No normalized center candidates found."
        )

        print(
            f"Expected files in: "
            f"{NORMALIZED_DIR.resolve()}"
        )

        return

    print(
        f"Loaded {len(candidates)} "
        f"center candidates."
    )

    print(
        f"Duplicate threshold: "
        f"{DUPLICATE_THRESHOLD:.2f}"
    )

    print()

    clusters = cluster_candidates(
        candidates
    )

    save_clusters(
        clusters
    )

    print_summary(
        len(candidates),
        clusters
    )


if __name__ == "__main__":
    main()