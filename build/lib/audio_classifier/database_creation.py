"""Doc String."""
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


def pad_length(path, max_len: int = None):
    """Pad the length of audio files."""
    sr, audio = load(path)
    matrix_len = int(len(audio))
    x_all = np.zeros((matrix_len//max_len, max_len))
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
    return (sr, x_all)


def make_wavlet(path):
    """Make wavlet coefficient tables."""
    T = 2**14
    J = 9
    Q = 16
    scattering = Scattering1D(J, T, Q, oversampling=1, max_order=3)
    sampling_rate, data = wavfile.read(path)
    number_of_samples = round(len(data) * float(16000) / sampling_rate)
    data = data.sum(axis=1)/2
    x = sps.resample(data, number_of_samples)
    x = x / np.max(np.abs(x))
    print(x.shape)
    Sx = scattering(x)
    return Sx


def make_mfcc(x):
    """Make an mfcc representation of the signal."""
    return librosa.feature.mfcc(x)


def find_onset_env(sr, x):
    """Find signal onset strengths."""
    hop_length = 512
    return librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)


def find_tempo(sr, oenv):
    """Find the estimated tempo."""
    hop_length = 512
    return librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                              hop_length=hop_length)[0]


class DataBase:
    """Class that makes a pandas dataframe."""

    def __init__(self, data_directory: str, cache=None):
        """Initialize dataframe directory."""
        if not cache:
            cache = "./data"
        self.cache = os.path.join(cache, "data_frame.pkl")
        self.data_directory = data_directory

    def make_dataset(self, save: bool = False):
        """Create a pandas dataframe with features."""
        categories = [os.path.join(self.data_directory, cat) for cat
                      in os.listdir(self.data_directory)
                      if cat != '.DS_Store']
        rows = []
        for _, category in enumerate(categories):
            class_cat = [os.path.join(category, cat) for cat
                         in os.listdir(category) if cat != '.DS_Store']
            for item in class_cat:
                sr, x = load(item)
                # wavelet = make_wavlet(item)
                mfcc = make_mfcc(x)
                onset_env = find_onset_env(sr, x)
                tempo = find_tempo(sr, onset_env)
                cat = item.split("/")[-2]
                columns = ['mfcc', 'onset_env', 'tempo', 'category'] # 'wavelet'
                data = [mfcc, onset_env, tempo, cat] # wavelet
                rows.append(dict(zip(columns, data)))
        df = pd.DataFrame.from_dict(rows, orient='columns')
        if save:
            filename = self.cache
            df.to_pickle(filename)
        return df
