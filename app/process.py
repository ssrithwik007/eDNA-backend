import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            encoded[mapping[base], i] = 1.0
    return encoded

def standardize_sequence(seq):
    scaler = StandardScaler()
    encoded = one_hot_encode(seq)
    standardized = scaler.fit_transform(encoded)
    return standardized



    