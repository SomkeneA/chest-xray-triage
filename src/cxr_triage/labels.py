"""Canonical ChestX-ray14 finding labels and shared text."""

from __future__ import annotations

import logging

import numpy as np

# Order used by Wang et al. and by most public ChestX-ray14 reproductions.
CHEST14_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

CLINICAL_DISCLAIMER = (
    "Research and portfolio use only. This software is not a medical device "
    "and must not be used for diagnosis, screening, or clinical decision-making."
)

NIH_BOX_PAGE = "https://nihcc.app.box.com/v/ChestXray-NIHCC"
EXPECTED_IMAGE_COUNT = 112_120
EXPECTED_PATIENT_COUNT = 30_805

_LOG = logging.getLogger(__name__)
_WARNED_LABELS: set[str] = set()


def encode_labels(finding_labels: str, labels: tuple[str, ...] = CHEST14_LABELS):
    """Multi-hot encode a pipe-separated ChestX-ray14 finding string.

    ``No Finding`` and an empty string are the all-zero vector. Unknown tokens
    are ignored after a one-time warning so a stray label cannot crash training.
    """
    target = np.zeros(len(labels), dtype=np.float32)
    lookup = {name.lower(): index for index, name in enumerate(labels)}
    parts = [part.strip() for part in str(finding_labels).split("|") if part.strip()]
    for part in parts:
        key = part.lower().replace(" ", "_") if part.lower() == "pleural thickening" else part.lower()
        index = lookup.get(key)
        if part.lower() == "no finding":
            continue
        if index is None:
            if part not in _WARNED_LABELS:
                _LOG.warning("Ignoring unknown finding label %r", part)
                _WARNED_LABELS.add(part)
            continue
        target[index] = 1.0
    return target
