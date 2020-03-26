"""Create a pandas dataframe."""
import math
import os
import sys

import pandas as pd
import librosa
import numpy as np
from kymatio.numpy import Scattering1D
from scipy.io import wavfile
import scipy.signal as sps


def load(path):
    """Load audio file and convert to mono."""
    x, sr = librosa.load(path, sr=16000, mono=True)
    return (sr, x)


# WIP
def pad_length(audio, max_len: int = None):
    """Pad the length of audio files."""
    matrix_len = len(audio)
    x_all = np.zeros((round(matrix_len/max_len), max_len))
    if max_len:
        # If it's too long, truncate it.
        if len(audio) > max_len:
            prev_length = 0
            for k in range(1, len(audio)//max_len):
                next_length = round(max_len * k)
                # start = (sr - len(x)) // 2
                if next_length != len(audio) and next_length <= len(audio):
                    x = audio[prev_length:next_length]
                    prev_length = next_length

                x_all[k, 0:max_len] = x
    return x_all


def make_wavlet(x):
    """Make wavlet coefficient tables."""
    J = 9
    Q = 16
    scattering = Scattering1D(J, x.shape[-1], Q, oversampling=1, max_order=3)
    Sx = scattering(x)
    return Sx


def make_mfcc(x):
    """Make an mfcc representation of the signal."""
    return librosa.feature.mfcc(x, n_mfcc=32)


def find_onset_env(sr, x):
    """Find signal onset strengths."""
    hop_length = 512
    return librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)


def find_tempo(sr, oenv):
    """Find the estimated tempo."""
    hop_length = 512
    return librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                              hop_length=hop_length)[0]


def make_dataframe(data_directory: str, cache=None, audio_pad: int = None,
                   save: bool = False):
    """Create a pandas dataframe with features."""
    if not cache:
        cache = "./data"
    cache = os.path.join(cache, "data_frame.pkl")
    data_directory = data_directory
    categories = [os.path.join(data_directory, cat) for cat
                  in os.listdir(data_directory)
                  if cat != '.DS_Store']
    rows = []
    for _, category in enumerate(categories):
        class_cat = [os.path.join(category, cat) for cat
                     in os.listdir(category) if cat != '.DS_Store']
        for item in class_cat:
            sr, x = load(item)
            # x = pad_length(x, max_len=30)
            wavelet = make_wavlet(x)
            mfcc = make_mfcc(x)
            onset_env = find_onset_env(sr, x)
            tempo = find_tempo(sr, onset_env)
            cat = item.split("/")[-2]
            columns = ['wavelet', 'mfcc', 'onset_env', 'tempo', 'category']
            data = [wavelet, mfcc, onset_env, tempo, cat]
            rows.append(dict(zip(columns, data)))
    df = pd.DataFrame.from_dict(rows, orient='columns')
    if save:
        df.to_pickle(cache, protocol=-1)
    return df
