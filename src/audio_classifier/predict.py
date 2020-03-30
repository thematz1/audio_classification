"""."""
import os
import contextlib

import pytest
import librosa
import sklearn
import pandas as pd
import numpy as np
from tqdm import tqdm
from kymatio.numpy import Scattering1D

from src.audio_classifier.loader import _loader
from src.audio_classifier.create_dataframe_v1 import (_return_contrast,
                                                      _return_mfcc,
                                                      _return_spec_centroid,
                                                      _return_tempo,
                                                      _return_tonnetz,
                                                      _normalize)


def file_features(path, max_len: int = 30):
    """
    Extract features from audio file
    to make classification.
    """
    main_frame = pd.DataFrame()
    full_x, sr = _loader(path, max_len)
    target = 1
    for _, x in enumerate(full_x.values()):
        name = path.split("/")[-1]
        spectral_centroids = _return_spec_centroid(x, sr)
        chromagram = pd.Series(librosa.feature.chroma_stft(x,
                                                           sr=sr)[0])
        rms = pd.Series(librosa.feature.rms(x)[0])
        tonnetz = _return_tonnetz(x, sr)
        contrast = _return_contrast(x, sr)
        tempo = _return_tempo(x, sr)
        mfcc = _return_mfcc(x, target, name)
        row = {'file': name,
               'chromagram': chromagram,
               'spectral_centroids': spectral_centroids,
               'tempo': tempo,
               'contrast': contrast,
               'rms': rms,
               'tonnetz': tonnetz,
               'target': target
               }
        data_frame = pd.DataFrame(row).astype({'target': 'int16',
                                               'file': 'category'})

        data_frame = data_frame.apply(pd.to_numeric, errors='ignore',
                                      downcast='float')
        data_frame.loc[:, 'tempo'].fillna(method='ffill',
                                          downcast='float32',
                                          inplace=True)
        data_frame.set_index(['file', 'target'], inplace=True)
        data_frame = pd.merge(data_frame, mfcc, left_index=True,
                              right_index=True, how='outer')
        data_frame.set_index([data_frame.index, 'time'], inplace=True)
        data_frame = data_frame.swaplevel(1, 2)
        data_frame = data_frame.groupby(['file',
                                         'time',
                                         'target'])
        data_frame = data_frame.resample('1S', level=1).mean()
        data_frame = data_frame.droplevel(3)
        data_frame.reset_index(inplace=True)
        data_frame['time'] = data_frame.time.dt.nanosecond
        data_frame['time'] = data_frame['time'].astype('float32')
        data_frame['target'] = data_frame['target'].astype('int16')
        data_frame.set_index('file', inplace=True)
        main_frame = main_frame.append(data_frame)
    main_frame = main_frame.sort_index()
    main_frame = main_frame[main_frame.columns].apply(_normalize, axis=0)
    main_frame.index = main_frame.index.astype('category')
    X = main_frame[['time', 'chromagram', 'spectral_centroids', 'tempo',
                    'contrast', 'rms', 'tonnetz', 'mfcc_val']]
    y = main_frame['target']
    return X, y
