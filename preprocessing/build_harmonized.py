# ============================================================
# STAGE 5E.2 — BUILD HART-COMPATIBLE HARMONIZED BENCHMARK
#
# Representation:
#   30 Hz
#   4.0 seconds
#   120 samples
#   hop = 30 samples = 1 sec = 75% overlap
#   3-axis accelerometer only
#   unit = g
#
# Classes:
#   0 walking
#   1 sitting
#   2 standing
#   3 stairs
#
# IMPORTANT:
#   Built from original/raw signals, NOT from harmonized_v1
#   50-sample windows.
#
# UCI-HAR:
#   original dataset contains 128-sample overlapping windows.
#   We reconstruct continuous activity segments using their
#   known 50% overlap, then perform proper 50 -> 30 Hz
#   polyphase resampling.
# ============================================================

import os
import sys
import re
import gc
import json
import shutil
import hashlib
import warnings
from pathlib import Path
from fractions import Fraction
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.signal import resample_poly

# The original cell called IPython's display(), unavailable in a plain Python script.
def display(obj):
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(os.environ.get(
    "HAR_PROJECT_ROOT",
    str(Path.home() / "projects" / "HAR_Q1_2026"),
)).expanduser().resolve()

# Keep all historical project outputs untouched: write ONLY into isolated replay root.
REPLAY_ROOT = Path(os.environ.get(
    "HAR_REPLAY_ROOT",
    str(PROJECT_ROOT / "repro_stage5e2_replay"),
)).expanduser().resolve()
if REPLAY_ROOT == PROJECT_ROOT or REPLAY_ROOT == PROJECT_ROOT.parent:
    raise RuntimeError("Refusing to use project or its parent as the replay output directory")

RAW_ROOT = (
    PROJECT_ROOT /
    "datasets" /
    "raw"
)

OLD_PROCESSED_ROOT = (
    PROJECT_ROOT /
    "datasets" /
    "processed" /
    "harmonized_v1"
)

# Release-specific source splits are packaged alongside the preprocessing script.
# Override HAR_SPLIT_SOURCE_ROOT only if using a verified alternative split release.
OLD_SPLIT_ROOT = Path(os.environ.get(
    "HAR_SPLIT_SOURCE_ROOT",
    str(Path(__file__).resolve().parents[1] / "splits"),
)).expanduser().resolve()

OUT_ROOT = (
    REPLAY_ROOT /
    "datasets" /
    "processed" /
    "harmonized_hart_v1"
)

SPLIT_ROOT = (
    REPLAY_ROOT /
    "splits" /
    "harmonized_hart_v1"
)

MANIFEST_ROOT = (
    REPLAY_ROOT /
    "manifests"
)

TABLE_ROOT = (
    REPLAY_ROOT /
    "paper_tables"
)


if (REPLAY_ROOT / "datasets" / "processed" / "harmonized_hart_v1").exists():
    raise RuntimeError(f"Replay outputs already exist: {REPLAY_ROOT}. Choose a new HAR_REPLAY_ROOT.")

# Fail before creating any output if the Stage-3 split inputs are missing.
# These paths MUST point to the source project, never to the replay directory.
for _dataset in ("wisdm", "uci_har", "mhealth", "pamap2", "motionsense"):
    _source_split = OLD_SPLIT_ROOT / f"{_dataset}_5fold.json"
    if not _source_split.is_file():
        raise FileNotFoundError(
            f"Missing original Stage-3 split: {_source_split}. "
            f"Do not create synthetic splits or use a replay output directory."
        )

print(f"[REPLAY 04.1] Read-only project: {PROJECT_ROOT}")
print(f"[REPLAY 04.1] Read original splits from: {OLD_SPLIT_ROOT}")
print(f"[REPLAY 04.1] Write new splits into: {SPLIT_ROOT}")
print(f"[REPLAY 04.1] Write processed arrays into: {OUT_ROOT}", flush=True)

for p in [
    OUT_ROOT,
    SPLIT_ROOT,
    MANIFEST_ROOT,
    TABLE_ROOT,
]:
    p.mkdir(
        parents=True,
        exist_ok=True
    )


TARGET_FS = 30.0

WINDOW_LENGTH = 120
HOP_LENGTH = 30

WINDOW_SECONDS = (
    WINDOW_LENGTH /
    TARGET_FS
)

HOP_SECONDS = (
    HOP_LENGTH /
    TARGET_FS
)


CLASS_NAMES = [
    "walking",
    "sitting",
    "standing",
    "stairs",
]


CLASS_TO_ID = {
    name: i
    for i, name
    in enumerate(
        CLASS_NAMES
    )
}


DATASETS = [
    "wisdm",
    "uci_har",
    "mhealth",
    "pamap2",
    "motionsense",
]


WISDM_MAP = {
    "A": "walking",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
}


UCI_MAP = {
    1: "walking",
    2: "stairs",
    3: "stairs",
    4: "sitting",
    5: "standing",
}


MHEALTH_MAP = {
    1: "standing",
    2: "sitting",
    4: "walking",
    5: "stairs",
}


PAMAP2_MAP = {
    2: "sitting",
    3: "standing",
    4: "walking",
    12: "stairs",
    13: "stairs",
}


MOTIONSENSE_MAP = {
    "wlk": "walking",
    "sit": "sitting",
    "std": "standing",
    "ups": "stairs",
    "dws": "stairs",
}


print("=" * 80)
print("STAGE 5E.2 — HART-COMPATIBLE DATASET BUILD")
print("=" * 80)

print("Python:", sys.executable)
print("Target Hz:", TARGET_FS)
print("Window:", WINDOW_LENGTH, "samples")
print("Window seconds:", WINDOW_SECONDS)
print("Hop:", HOP_LENGTH, "samples")
print("Overlap:", f"{1-HOP_LENGTH/WINDOW_LENGTH:.0%}")


# The validated runtime is documented; do not hardcode the virtualenv folder name.
print("Runtime environment:", sys.executable)


# ============================================================
# HELPERS
# ============================================================

G = 9.80665


def sha256_file(
    path,
    chunk_size=1024 * 1024
):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def find_dataset_root(name):

    p = (
        RAW_ROOT /
        name
    )

    if p.exists():
        return p

    return RAW_ROOT


def fill_missing(
    x
):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    if np.isfinite(
        x
    ).all():

        return x


    df = pd.DataFrame(
        x
    )

    # Equivalent spirit to HART forward-fill,
    # with bfill only for leading NaNs.
    df = df.ffill().bfill()

    x = df.to_numpy(
        dtype=np.float32
    )


    if not np.isfinite(
        x
    ).all():

        raise RuntimeError(
            "NaN/Inf remain after filling."
        )


    return x


def convert_acc_to_g(
    x,
    mode
):

    x = np.asarray(
        x,
        dtype=np.float32
    )


    if mode == "g":

        return x, "already_g"


    if mode == "ms2":

        return (
            x / G,
            "m_s2_to_g"
        )


    if mode == "auto":

        norm = np.linalg.norm(
            x,
            axis=1
        )

        median_norm = float(
            np.nanmedian(
                norm
            )
        )


        if median_norm > 3.0:

            return (
                x / G,
                "auto_m_s2_to_g"
            )

        return (
            x,
            "auto_already_g"
        )


    raise ValueError(
        mode
    )


def resample_to_30hz(
    x,
    source_fs
):

    x = fill_missing(
        x
    )

    if float(
        source_fs
    ) == TARGET_FS:

        return x.astype(
            np.float32,
            copy=False
        )


    ratio = Fraction(
        TARGET_FS /
        float(source_fs)
    ).limit_denominator(
        1000
    )


    y = resample_poly(
        x,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=0,
    )


    return y.astype(
        np.float32,
        copy=False
    )


def iter_activity_runs(
    labels,
    mapping
):

    """
    Split by ORIGINAL activity label.

    This is deliberate:
      PAMAP2 12 and 13 both -> stairs,
      UCI 2 and 3 both -> stairs,
    but should remain separate source runs.
    """

    labels = np.asarray(
        labels
    )


    start = None
    current_raw = None
    current_name = None


    for i, raw in enumerate(
        labels
    ):

        try:

            if isinstance(
                raw,
                str
            ):

                key = raw.strip()

            else:

                key = int(raw)

        except Exception:

            key = raw


        mapped = mapping.get(
            key
        )


        if mapped is None:

            if start is not None:

                yield (
                    start,
                    i,
                    current_raw,
                    current_name,
                )

                start = None
                current_raw = None
                current_name = None

            continue


        if start is None:

            start = i
            current_raw = key
            current_name = mapped

            continue


        if (
            key != current_raw
        ):

            yield (
                start,
                i,
                current_raw,
                current_name,
            )

            start = i
            current_raw = key
            current_name = mapped


    if start is not None:

        yield (
            start,
            len(labels),
            current_raw,
            current_name,
        )


# ============================================================
# DATASET BUILDER
# ============================================================

class DatasetBuilder:

    def __init__(
        self,
        dataset
    ):

        self.dataset = dataset

        self.windows = []
        self.rows = []

        self.window_counter = 0

        self.session_counter = 0

        self.skipped_short = 0

        self.input_sessions = 0

        self.unit_transforms = {}


    def add_session(
        self,
        x,
        subject,
        activity_name,
        source_label,
        source_fs,
        session_id,
        unit_mode,
        notes=None,
    ):

        self.input_sessions += 1


        x = np.asarray(
            x,
            dtype=np.float32
        )


        if (
            x.ndim != 2
            or
            x.shape[1] != 3
        ):

            raise RuntimeError(
                f"{self.dataset}: "
                f"bad session shape "
                f"{x.shape}"
            )


        if len(x) < 2:

            self.skipped_short += 1
            return


        x = fill_missing(
            x
        )


        x, transform = (
            convert_acc_to_g(
                x,
                unit_mode
            )
        )


        self.unit_transforms[
            transform
        ] = (
            self.unit_transforms.get(
                transform,
                0
            )
            + 1
        )


        x30 = resample_to_30hz(
            x,
            source_fs
        )


        if (
            len(x30)
            <
            WINDOW_LENGTH
        ):

            self.skipped_short += 1
            return


        label_id = CLASS_TO_ID[
            activity_name
        ]


        starts = range(
            0,
            len(x30)
            -
            WINDOW_LENGTH
            + 1,
            HOP_LENGTH
        )


        local_index = 0


        for start in starts:

            end = (
                start +
                WINDOW_LENGTH
            )


            window = np.array(
                x30[
                    start:end
                ],
                dtype=np.float32,
                copy=True
            )


            if (
                window.shape
                !=
                (
                    WINDOW_LENGTH,
                    3
                )
            ):

                continue


            if not np.isfinite(
                window
            ).all():

                raise RuntimeError(
                    f"{self.dataset}: "
                    "non-finite window"
                )


            self.window_counter += 1


            self.windows.append(
                window
            )


            self.rows.append({

                "window_id":
                    self.window_counter,

                "subject":
                    int(subject),

                "label":
                    activity_name,

                "label_id":
                    int(label_id),

                "source_label":
                    str(source_label),

                "source":
                    self.dataset,

                "session_id":
                    str(session_id),

                "session_window_index":
                    local_index,

                "source_fs":
                    float(source_fs),

                "target_fs":
                    TARGET_FS,

                "start_sample_30hz":
                    int(start),

                "end_sample_30hz":
                    int(end),

                "window_seconds":
                    WINDOW_SECONDS,

                "hop_seconds":
                    HOP_SECONDS,

                "unit":
                    "g",

                "notes":
                    (
                        ""
                        if notes is None
                        else str(notes)
                    ),
            })


            local_index += 1


    def finish(self):

        if not self.windows:

            raise RuntimeError(
                f"{self.dataset}: "
                "no windows generated."
            )


        X = np.stack(
            self.windows,
            axis=0
        ).astype(
            np.float32,
            copy=False
        )


        meta = pd.DataFrame(
            self.rows
        )


        del self.windows
        self.windows = []

        gc.collect()


        return X, meta


# ============================================================
# WISDM
# ============================================================

def process_wisdm():

    root = find_dataset_root(
        "wisdm"
    )


    files = sorted(
        root.rglob(
            "data_*_accel_phone.txt"
        )
    )


    if not files:

        raise FileNotFoundError(
            "WISDM phone accelerometer "
            "files not found."
        )


    print(
        "\nWISDM files:",
        len(files)
    )


    builder = DatasetBuilder(
        "wisdm"
    )


    for file in files:

        df = pd.read_csv(
            file,
            header=None,
            sep=",",
            engine="python",
            on_bad_lines="skip",
        )


        if df.shape[1] < 6:
            continue


        # Remove trailing ";" in z column.
        df.iloc[:, 5] = (
            df.iloc[:, 5]
            .astype(str)
            .str.replace(
                ";",
                "",
                regex=False
            )
        )


        for c in [
            0, 2, 3, 4, 5
        ]:

            df.iloc[:, c] = (
                pd.to_numeric(
                    df.iloc[:, c],
                    errors="coerce"
                )
            )


        df = df.dropna(
            subset=[
                df.columns[0],
                df.columns[1],
                df.columns[3],
                df.columns[4],
                df.columns[5],
            ]
        ).reset_index(
            drop=True
        )


        subject_values = (
            df.iloc[:, 0]
            .astype(int)
        )


        # File should correspond to one subject.
        subject = int(
            subject_values.mode().iloc[0]
        )


        labels = (
            df.iloc[:, 1]
            .astype(str)
            .str.strip()
            .to_numpy()
        )


        acc = (
            df.iloc[
                :,
                3:6
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        session_number = 0


        for (
            start,
            end,
            source_label,
            activity_name,
        ) in iter_activity_runs(
            labels,
            WISDM_MAP
        ):

            session_number += 1


            builder.add_session(

                x=acc[
                    start:end
                ],

                subject=subject,

                activity_name=
                    activity_name,

                source_label=
                    source_label,

                source_fs=20,

                session_id=(
                    f"{file.stem}_"
                    f"run{session_number}"
                ),

                unit_mode="auto",
            )


    return builder.finish(), {
        "files": len(files)
    }


# ============================================================
# MHEALTH
# ============================================================

def process_mhealth():

    root = find_dataset_root(
        "mhealth"
    )


    files = sorted(
        root.rglob(
            "mHealth_subject*.log"
        )
    )


    if not files:

        files = [
            p
            for p in root.rglob(
                "*.log"
            )
            if "mhealth" in
            str(p).lower()
        ]


    if not files:

        raise FileNotFoundError(
            "MHEALTH files not found."
        )


    print(
        "\nMHEALTH files:",
        len(files)
    )


    builder = DatasetBuilder(
        "mhealth"
    )


    for file in files:

        match = re.search(
            r"subject[_]?(\d+)",
            file.name,
            re.I
        )


        if not match:
            continue


        subject = int(
            match.group(1)
        )


        df = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            engine="python",
        )


        if df.shape[1] < 4:
            continue


        acc = (
            df.iloc[
                :,
                0:3
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .to_numpy(
                dtype=np.float32
            )
        )


        labels = (
            pd.to_numeric(
                df.iloc[:, -1],
                errors="coerce"
            )
            .fillna(-999)
            .astype(int)
            .to_numpy()
        )


        session_number = 0


        for (
            start,
            end,
            source_label,
            activity_name,
        ) in iter_activity_runs(
            labels,
            MHEALTH_MAP
        ):

            session_number += 1


            builder.add_session(

                acc[
                    start:end
                ],

                subject,

                activity_name,

                source_label,

                50,

                (
                    f"{file.stem}_"
                    f"run{session_number}"
                ),

                "ms2",
            )


    return builder.finish(), {
        "files": len(files)
    }


# ============================================================
# PAMAP2
# ============================================================

def process_pamap2():

    root = find_dataset_root(
        "pamap2"
    )


    all_files = sorted(
        root.rglob(
            "subject*.dat"
        )
    )


    protocol_files = [
        p
        for p in all_files
        if any(
            part.lower()
            ==
            "protocol"
            for part
            in p.parts
        )
    ]


    files = (
        protocol_files
        if protocol_files
        else all_files
    )


    if not files:

        raise FileNotFoundError(
            "PAMAP2 subject files "
            "not found."
        )


    print(
        "\nPAMAP2 files:",
        len(files)
    )


    builder = DatasetBuilder(
        "pamap2"
    )


    for file in files:

        match = re.search(
            r"subject(\d+)",
            file.stem,
            re.I
        )


        if not match:
            continue


        subject = int(
            match.group(1)
        )


        df = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            engine="python",
        )


        if df.shape[1] < 24:
            continue


        labels = (
            pd.to_numeric(
                df.iloc[:, 1],
                errors="coerce"
            )
            .fillna(-999)
            .astype(int)
            .to_numpy()
        )


        # chest accelerometer ±16g:
        # cols 21,22,23
        acc = (
            df.iloc[
                :,
                21:24
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .to_numpy(
                dtype=np.float32
            )
        )


        session_number = 0


        for (
            start,
            end,
            source_label,
            activity_name,
        ) in iter_activity_runs(
            labels,
            PAMAP2_MAP
        ):

            session_number += 1


            builder.add_session(

                acc[
                    start:end
                ],

                subject,

                activity_name,

                source_label,

                100,

                (
                    f"{file.stem}_"
                    f"run{session_number}"
                ),

                "ms2",
            )


    return builder.finish(), {
        "files": len(files)
    }


# ============================================================
# MOTIONSENSE
# ============================================================

def process_motionsense():

    root = find_dataset_root(
        "motionsense"
    )

    all_files = sorted(
        root.rglob(
            "sub_*.csv"
        )
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Use ONLY A_DeviceMotion_data.
    #
    # B_Accelerometer_data:
    #   x, y, z only
    #
    # A_DeviceMotion_data:
    #   gravity.*
    #   userAcceleration.*
    #   rotationRate.*
    #   attitude.*
    # --------------------------------------------------------

    files = []


    for file in all_files:

        parts_lower = [
            p.lower()
            for p in file.parts
        ]

        if (
            "a_devicemotion_data"
            not in parts_lower
        ):
            continue


        parent = file.parent.name


        # Activities used in harmonized benchmark:
        # dws / ups / wlk / sit / std
        if re.match(
            r"^(wlk|sit|std|ups|dws)_\d+$",
            parent
        ):

            files.append(
                file
            )


    if not files:

        raise FileNotFoundError(
            "MotionSense A_DeviceMotion_data "
            "files not found."
        )


    print(
        "All MotionSense CSV:",
        len(all_files)
    )

    print(
        "Selected A_DeviceMotion_data CSV:",
        len(files)
    )


    # --------------------------------------------------------
    # SANITY CHECK PATHS
    # --------------------------------------------------------

    print("\nExample files:")

    for p in files[:5]:

        print(
            " ",
            p
        )


    builder = DatasetBuilder(
        "motionsense"
    )


    required = [
        "gravity.x",
        "gravity.y",
        "gravity.z",
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z",
    ]


    bad_files = []


    for file in files:

        match = re.search(
            r"sub_(\d+)",
            file.stem
        )


        if not match:
            continue


        subject = int(
            match.group(1)
        )


        activity_code = (
            file.parent.name
            .split("_")[0]
        )


        activity_name = (
            MOTIONSENSE_MAP[
                activity_code
            ]
        )


        df = pd.read_csv(
            file
        )


        # Drop unnamed index column if present.
        unnamed = [
            c
            for c in df.columns
            if str(c).lower()
            .startswith(
                "unnamed"
            )
        ]

        if unnamed:

            df = df.drop(
                columns=unnamed
            )


        missing_cols = [
            c
            for c in required
            if c not in df.columns
        ]


        if missing_cols:

            bad_files.append({
                "file":
                    str(file),

                "missing":
                    missing_cols,
            })

            continue


        gravity = (
            df[
                [
                    "gravity.x",
                    "gravity.y",
                    "gravity.z",
                ]
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .to_numpy(
                dtype=np.float32
            )
        )


        user_acc = (
            df[
                [
                    "userAcceleration.x",
                    "userAcceleration.y",
                    "userAcceleration.z",
                ]
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .to_numpy(
                dtype=np.float32
            )
        )


        # ----------------------------------------------------
        # Raw acceleration:
        #
        # accelerometer = gravity + user acceleration
        #
        # DeviceMotion values are expressed in g.
        # ----------------------------------------------------

        acc = (
            gravity
            +
            user_acc
        )


        builder.add_session(

            x=acc,

            subject=subject,

            activity_name=
                activity_name,

            source_label=
                activity_code,

            source_fs=50,

            session_id=(
                file.parent.name
                + "_"
                + file.stem
            ),

            unit_mode="g",

            notes=(
                "MotionSense A_DeviceMotion_data; "
                "raw acceleration reconstructed as "
                "gravity + userAcceleration"
            ),
        )


    if bad_files:

        print(
            "\nWARNING — skipped files "
            "with missing columns:",
            len(bad_files)
        )

        print(
            bad_files[:3]
        )


    (
        X,
        meta
    ) = builder.finish()


    audit = {

        "all_csv_files":
            len(all_files),

        "selected_device_motion_files":
            len(files),

        "bad_files":
            len(bad_files),

        "input_source":
            "A_DeviceMotion_data",

        "acceleration_definition":
            "gravity + userAcceleration",

        "source_fs_hz":
            50,
    }


    return (
        (X, meta),
        audit
    )


# ============================================================
# UCI-HAR
#
# Original UCI-HAR inertial signals are 128-sample
# windows with 50% overlap.
#
# We reconstruct continuous sequences by verifying:
#
#   previous[64:128]
#        ~= current[0:64]
#
# Only windows with the SAME ORIGINAL activity label
# are joined.
# ============================================================

def locate_uci_root():

    root = find_dataset_root(
        "uci_har"
    )


    candidates = list(
        root.rglob(
            "total_acc_x_train.txt"
        )
    )


    if not candidates:

        raise FileNotFoundError(
            "UCI-HAR inertial signals "
            "not found."
        )


    file = candidates[0]

    # .../train/Inertial Signals/
    split_dir = (
        file.parent.parent
    )

    dataset_root = (
        split_dir.parent
    )


    return dataset_root


def load_uci_split(
    dataset_root,
    split
):

    split_dir = (
        dataset_root /
        split
    )

    signal_dir = (
        split_dir /
        "Inertial Signals"
    )


    suffix = split


    x = np.loadtxt(
        signal_dir /
        f"total_acc_x_{suffix}.txt",
        dtype=np.float32
    )

    y = np.loadtxt(
        signal_dir /
        f"total_acc_y_{suffix}.txt",
        dtype=np.float32
    )

    z = np.loadtxt(
        signal_dir /
        f"total_acc_z_{suffix}.txt",
        dtype=np.float32
    )


    windows = np.stack(
        [
            x,
            y,
            z,
        ],
        axis=-1
    )


    labels = np.loadtxt(
        split_dir /
        f"y_{suffix}.txt",
        dtype=int
    )


    subjects = np.loadtxt(
        split_dir /
        f"subject_{suffix}.txt",
        dtype=int
    )


    return (
        windows,
        labels,
        subjects
    )


def reconstruct_uci_sessions():

    root = locate_uci_root()

    stats = {
        "root": str(root),
        "pairs_checked": 0,
        "pairs_joined": 0,
        "overlap_mae": [],
        "reconstructed_sessions": 0,
    }

    sessions = []

    # UCI-HAR original windows:
    # 128 samples, 50% overlap -> 64 samples
    OVERLAP = 64

    # Numerical tolerance for checking whether
    # adjacent windows truly overlap.
    MAX_OVERLAP_MAE = 0.02


    for split in [
        "train",
        "test",
    ]:

        (
            windows,
            labels,
            subjects,
        ) = load_uci_split(
            root,
            split
        )


        for subject in np.unique(
            subjects
        ):

            indices = np.where(
                subjects == subject
            )[0]


            current_chunks = None
            current_prev_window = None
            current_raw_label = None
            session_number = 0


            def finalize():

                nonlocal current_chunks
                nonlocal current_prev_window
                nonlocal current_raw_label
                nonlocal session_number

                if current_chunks is None:
                    return


                sequence = np.concatenate(
                    current_chunks,
                    axis=0
                )


                mapped = UCI_MAP.get(
                    int(
                        current_raw_label
                    )
                )


                if mapped is not None:

                    session_number += 1

                    sessions.append({
                        "x":
                            sequence,

                        "subject":
                            int(subject),

                        "activity":
                            mapped,

                        "source_label":
                            int(
                                current_raw_label
                            ),

                        "session_id":
                            (
                                f"{split}_"
                                f"s{int(subject)}_"
                                f"run{session_number}"
                            ),
                    })


                    stats[
                        "reconstructed_sessions"
                    ] += 1


                current_chunks = None
                current_prev_window = None
                current_raw_label = None


            for idx in indices:

                raw_label = int(
                    labels[idx]
                )


                mapped = UCI_MAP.get(
                    raw_label
                )


                # Ignore LAYING and any labels
                # outside our harmonized 4 classes.
                if mapped is None:

                    finalize()
                    continue


                w = np.asarray(
                    windows[idx],
                    dtype=np.float32
                )


                # Start new sequence
                if current_chunks is None:

                    current_chunks = [
                        w.copy()
                    ]

                    current_prev_window = (
                        w
                    )

                    current_raw_label = (
                        raw_label
                    )

                    continue


                # Only identical ORIGINAL activity
                # labels are candidates for joining.
                if (
                    raw_label
                    ==
                    current_raw_label
                ):

                    stats[
                        "pairs_checked"
                    ] += 1


                    mae = float(
                        np.mean(
                            np.abs(
                                current_prev_window[
                                    OVERLAP:
                                ]
                                -
                                w[
                                    :OVERLAP
                                ]
                            )
                        )
                    )


                    stats[
                        "overlap_mae"
                    ].append(
                        mae
                    )


                    continuous = (
                        mae
                        <=
                        MAX_OVERLAP_MAE
                    )

                else:

                    continuous = False


                if continuous:

                    # First 64 samples already exist
                    # in previous window.
                    # Append only new half.
                    current_chunks.append(
                        w[
                            OVERLAP:
                        ].copy()
                    )

                    current_prev_window = (
                        w
                    )

                    stats[
                        "pairs_joined"
                    ] += 1


                else:

                    # Finish old activity sequence.
                    finalize()

                    # Begin a new one.
                    current_chunks = [
                        w.copy()
                    ]

                    current_prev_window = (
                        w
                    )

                    current_raw_label = (
                        raw_label
                    )


            # Last sequence of this subject
            finalize()


    overlap = np.asarray(
        stats[
            "overlap_mae"
        ],
        dtype=float
    )


    if len(overlap) > 0:

        stats[
            "overlap_mae_median"
        ] = float(
            np.median(
                overlap
            )
        )

        stats[
            "overlap_mae_p95"
        ] = float(
            np.percentile(
                overlap,
                95
            )
        )

        stats[
            "overlap_mae_max"
        ] = float(
            np.max(
                overlap
            )
        )


    # Do not store huge list in manifest
    del stats[
        "overlap_mae"
    ]


    return sessions, stats

def process_uci_har():

    sessions, audit = (
        reconstruct_uci_sessions()
    )


    print(
        "\nUCI-HAR reconstructed sessions:",
        len(sessions)
    )

    print(
        "Overlap pairs checked:",
        audit[
            "pairs_checked"
        ]
    )

    print(
        "Overlap pairs joined:",
        audit[
            "pairs_joined"
        ]
    )

    print(
        "Overlap MAE median:",
        audit.get(
            "overlap_mae_median"
        )
    )


    if (
        audit[
            "pairs_joined"
        ]
        ==
        0
    ):

        raise RuntimeError(
            "UCI-HAR reconstruction failed: "
            "no overlapping windows joined."
        )


    builder = DatasetBuilder(
        "uci_har"
    )


    for s in sessions:

        builder.add_session(

            s["x"],

            s["subject"],

            s["activity"],

            s[
                "source_label"
            ],

            50,

            s[
                "session_id"
            ],

            "g",

            notes=(
                "reconstructed from original "
                "128-sample 50%-overlap "
                "UCI-HAR windows"
            ),
        )


    return builder.finish(), audit


# ============================================================
# SUBJECT-ID ALIGNMENT WITH STAGE 3
# ============================================================

def align_subject_ids(dataset, meta):
    """Validate final IDs against immutable released subject-split definitions.

    The verified private replay compared subject IDs with Stage 3 metadata.
    The retained Stage 5E.2 manifest documented identity mappings for all
    five datasets. For a standalone public release, strictly verify identity
    against the original published split IDs. Do NOT silently reorder or
    reindex participants. If a mismatch occurs, abort the reproduction.
    """
    source = OLD_SPLIT_ROOT / f"{dataset}_5fold.json"
    split = json.loads(source.read_text(encoding="utf-8"))
    first = split["folds"][0]
    expected = set()
    for key in ("train_subjects", "val_subjects", "test_subjects"):
        expected.update(int(s) for s in first[key])
    found = set(int(s) for s in pd.to_numeric(meta["subject"]).astype(int).unique())
    if found != expected:
        raise RuntimeError(
            f"{dataset}: regenerated subject IDs differ from the fixed public splits. "
            f"found={sorted(found)} expected={sorted(expected)}. "
            "Do not relabel subjects without independently auditing the mapping."
        )
    return meta, {"type": "identity", "mapping": {}}


# ============================================================
# SPLIT VALIDATION
# ============================================================

def validate_against_splits(
    dataset,
    meta
):

    split_path = (
        OLD_SPLIT_ROOT /
        f"{dataset}_5fold.json"
    )


    with open(
        split_path,
        "r",
        encoding="utf-8"
    ) as f:

        split = json.load(f)


    available = set(
        meta[
            "subject"
        ]
        .astype(int)
        .unique()
        .tolist()
    )


    for fold in split[
        "folds"
    ]:

        expected = (
            set(
                map(
                    int,
                    fold[
                        "train_subjects"
                    ]
                )
            )
            |
            set(
                map(
                    int,
                    fold[
                        "val_subjects"
                    ]
                )
            )
            |
            set(
                map(
                    int,
                    fold[
                        "test_subjects"
                    ]
                )
            )
        )


        missing = (
            expected -
            available
        )


        if missing:

            raise RuntimeError(
                f"{dataset}: subjects "
                f"missing in HART dataset: "
                f"{sorted(missing)}"
            )


    shutil.copy2(
        split_path,
        SPLIT_ROOT /
        split_path.name
    )


# ============================================================
# SAVE ONE DATASET
# ============================================================

def save_dataset(
    dataset,
    X,
    meta,
    parser_audit,
    subject_mapping,
):

    out = (
        OUT_ROOT /
        dataset
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )


    X_path = (
        out /
        "X.npy"
    )

    meta_path = (
        out /
        "metadata.csv"
    )

    manifest_path = (
        out /
        "manifest.json"
    )


    np.save(
        X_path,
        X
    )


    meta.to_csv(
        meta_path,
        index=False
    )


    class_counts = (
        meta[
            "label"
        ]
        .value_counts()
        .reindex(
            CLASS_NAMES,
            fill_value=0
        )
        .to_dict()
    )


    manifest = {

        "dataset":
            dataset,

        "representation":
            "harmonized_hart_v1",

        "created_at":
            datetime.now()
            .isoformat(),

        "shape":
            list(
                X.shape
            ),

        "subjects":
            int(
                meta[
                    "subject"
                ].nunique()
            ),

        "classes":
            sorted(
                meta[
                    "label"
                ].unique()
                .tolist()
            ),

        "class_counts":
            {
                k: int(v)
                for k, v
                in class_counts.items()
            },

        "sampling_rate_hz":
            TARGET_FS,

        "window_length":
            WINDOW_LENGTH,

        "window_seconds":
            WINDOW_SECONDS,

        "hop_length":
            HOP_LENGTH,

        "hop_seconds":
            HOP_SECONDS,

        "overlap":
            (
                1 -
                HOP_LENGTH /
                WINDOW_LENGTH
            ),

        "channels":
            [
                "acc_x",
                "acc_y",
                "acc_z",
            ],

        "unit":
            "g",

        "window_label_policy":
            (
                "constant original-label "
                "runs only"
            ),

        "parser_audit":
            parser_audit,

        "subject_alignment":
            subject_mapping,

        "sha256": {
            "X.npy":
                sha256_file(
                    X_path
                ),

            "metadata.csv":
                sha256_file(
                    meta_path
                ),
        },
    }


    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )


    return manifest


# ============================================================
# PROCESS ALL DATASETS
# ============================================================

PROCESSORS = {
    "wisdm":
        process_wisdm,

    "uci_har":
        process_uci_har,

    "mhealth":
        process_mhealth,

    "pamap2":
        process_pamap2,

    "motionsense":
        process_motionsense,
}


all_manifests = {}
summary_rows = []
class_rows = []


for dataset in DATASETS:

    print("\n" + "=" * 80)
    print(
        "PROCESSING:",
        dataset.upper()
    )
    print("=" * 80)


    (
        (
            X,
            meta
        ),
        parser_audit,
    ) = (
        PROCESSORS[
            dataset
        ]()
    )


    # --------------------------------------------------------
    # SUBJECT IDS = SAME AS STAGE 3
    # --------------------------------------------------------

    (
        meta,
        subject_mapping,
    ) = align_subject_ids(
        dataset,
        meta
    )


    # --------------------------------------------------------
    # STRICT VALIDATION
    # --------------------------------------------------------

    if (
        X.ndim != 3
        or
        X.shape[1:]
        !=
        (
            WINDOW_LENGTH,
            3
        )
    ):

        raise RuntimeError(
            f"{dataset}: "
            f"bad X shape {X.shape}"
        )


    if len(X) != len(meta):

        raise RuntimeError(
            f"{dataset}: X/meta "
            "length mismatch."
        )


    if not np.isfinite(
        X[:min(
            len(X),
            5000
        )]
    ).all():

        raise RuntimeError(
            f"{dataset}: NaN/Inf."
        )


    labels = set(
        meta[
            "label_id"
        ]
        .astype(int)
        .unique()
        .tolist()
    )


    if labels != {
        0, 1, 2, 3
    }:

        raise RuntimeError(
            f"{dataset}: missing classes "
            f"{labels}"
        )


    if (
        meta[
            "subject"
        ].nunique()
        <
        5
    ):

        raise RuntimeError(
            f"{dataset}: too few subjects."
        )


    validate_against_splits(
        dataset,
        meta
    )


    manifest = save_dataset(

        dataset,

        X,

        meta,

        parser_audit,

        subject_mapping,
    )


    all_manifests[
        dataset
    ] = manifest


    counts = (
        meta[
            "label"
        ]
        .value_counts()
        .reindex(
            CLASS_NAMES,
            fill_value=0
        )
    )


    summary_rows.append({

        "dataset":
            dataset,

        "windows":
            int(
                len(meta)
            ),

        "subjects":
            int(
                meta[
                    "subject"
                ].nunique()
            ),

        "classes":
            int(
                meta[
                    "label"
                ].nunique()
            ),

        "fs_hz":
            TARGET_FS,

        "window_samples":
            WINDOW_LENGTH,

        "window_seconds":
            WINDOW_SECONDS,

        "hop_samples":
            HOP_LENGTH,

        "overlap":
            0.75,
    })


    for label in CLASS_NAMES:

        class_rows.append({

            "dataset":
                dataset,

            "class":
                label,

            "windows":
                int(
                    counts[
                        label
                    ]
                ),
        })


    print(
        f"{dataset}: "
        f"X={X.shape}, "
        f"subjects="
        f"{meta['subject'].nunique()}"
    )


    print(
        "Class counts:",
        {
            k: int(v)
            for k, v
            in counts.items()
        }
    )


    del X
    del meta

    gc.collect()


# ============================================================
# COPY CROSS-DATASET PROTOCOL
# ============================================================

cross_protocol = (
    OLD_SPLIT_ROOT /
    "cross_dataset_protocols.json"
)


if cross_protocol.exists():

    shutil.copy2(
        cross_protocol,
        SPLIT_ROOT /
        cross_protocol.name
    )


# ============================================================
# TABLES
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


class_df = pd.DataFrame(
    class_rows
)


summary_path = (
    TABLE_ROOT /
    "stage5e2_hart_dataset_summary.csv"
)


class_path = (
    TABLE_ROOT /
    "stage5e2_hart_class_counts.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


class_df.to_csv(
    class_path,
    index=False
)


# ============================================================
# MASTER MANIFEST
# ============================================================

master = {

    "stage":
        "5E.2",

    "created_at":
        datetime.now()
        .isoformat(),

    "representation":
        "harmonized_hart_v1",

    "methodology": {

        "source":
            "original/raw signals",

        "sampling_rate_hz":
            TARGET_FS,

        "resampling":
            "scipy.signal.resample_poly",

        "window_samples":
            WINDOW_LENGTH,

        "window_seconds":
            WINDOW_SECONDS,

        "hop_samples":
            HOP_LENGTH,

        "hop_seconds":
            HOP_SECONDS,

        "overlap":
            0.75,

        "channels":
            "3-axis accelerometer",

        "unit":
            "g",

        "classes":
            CLASS_NAMES,

        "label_policy":
            (
                "windows generated within "
                "constant original activity runs"
            ),

        "splits":
            (
                "same subject assignments "
                "as harmonized_v1"
            ),
    },

    "datasets":
        all_manifests,

    "validation":
        "PASS",
}


master_path = (
    MANIFEST_ROOT /
    "stage5e2_hart_dataset_manifest.json"
)


with open(
    master_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        master,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("HART DATASET SUMMARY")
print("=" * 80)

display(
    summary_df
)


print("\n" + "=" * 80)
print("HART CLASS COUNTS")
print("=" * 80)

display(
    class_df.pivot(
        index="dataset",
        columns="class",
        values="windows"
    ).fillna(0)
)


print("\nSaved benchmark:")
print(
    OUT_ROOT
)

print("\nSaved splits:")
print(
    SPLIT_ROOT
)

print("\nSaved summary:")
print(
    summary_path
)

print("\nSaved manifest:")
print(
    master_path
)


print("\n" + "=" * 80)
print("STAGE 5E.2 VALIDATION: PASS")
print("=" * 80)