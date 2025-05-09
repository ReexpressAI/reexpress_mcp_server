# Copyright Reexpress AI, Inc. All rights reserved.

import constants

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import faiss
# import time
import constants
from collections import namedtuple

#### DataValidator
unlabeledLabel: int = -1
oodLabel: int = -99

maxLabelDisplayNameCharacters: int = 100
maxInputAttributeSize = 5000  # differs from Swift v1
minInputAttributeSize = 32  # differs from Swift v1

def getDefaultLabelName(label: int, abbreviated: bool = False) -> str:
    if label == unlabeledLabel:
        return "unlabeled"
    elif label == oodLabel:
        if abbreviated:
            return "OOD"
        else:
            return "out-of-distribution (OOD)"
    return ""


def isKnownValidLabel(label: int, numberOfClasses: int) -> bool:
    return 0 <= label < numberOfClasses


def isValidLabel(label: int, numberOfClasses: int) -> bool:
    if isKnownValidLabel(label=label, numberOfClasses=numberOfClasses):
        return True
    elif label == unlabeledLabel:
        return True
    elif label == oodLabel:
        return True
    return False


def allValidLabelsAsArray(numberOfClasses: int) -> list[int]:
    allValidLabels: list[int] = [oodLabel, unlabeledLabel]
    for label in range(numberOfClasses):
        allValidLabels.append(label)
    return allValidLabels