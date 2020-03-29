"""."""
import os
import contextlib

import librosa
import sklearn
import pandas as pd
import numpy as np
from tqdm import tqdm
from kymatio.numpy import Scattering1D

# path = '/Users/mathewzaharopoulos/Downloads/audio-classifier-keras-cnn-master/Samples'
path = 'samples_data'


def _normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# Work In Progress
def _make_wavlet(x):
    J = 9
    Q = 16
    scattering = Scattering1D(J, x.shape[-1], Q, oversampling=1, max_order=3)
    Sx = scattering(x)
    return Sx


def _return_mfcc(x, idx, name):
    mfcc = librosa.feature.mfcc(x)
    time_stamps = librosa.core.times_like(mfcc)
    mfcc = list(zip(mfcc, time_stamps))
    mfcc = [[idx, name, y, pos] for x, pos in mfcc for y in x]
    mfcc = (pd.DataFrame(mfcc,
            columns=['target', 'file', 'mfcc_val', 'time'])
            .astype({'target': 'category', 'file': 'category',
                     'mfcc_val': 'float32'}))
    mfcc['time'] = pd.to_datetime(mfcc.time).dt.nanosecond
    mfcc = mfcc.set_index(['target', 'file'])
    return mfcc


def _return_tonnetz(x, sr):
    har = librosa.effects.harmonic(x)
    tonnetz = pd.Series(librosa.feature.tonnetz(y=har, sr=sr)[0])
    return tonnetz


def _return_spec_centroid(x, sr):
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    spectral_centroids = pd.Series(spectral_centroids)
    return spectral_centroids


def _return_contrast(x, sr):
    S = np.abs(librosa.stft(x))
    contrast = pd.Series(librosa.feature.spectral_contrast(S=S, sr=sr)[0])
    return contrast


def _return_tempo(x, sr):
    oenv = librosa.onset.onset_strength(y=x, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=oenv,
                               sr=sr)[0]
    return int(tempo)


def create_dataframe(path_to_data_directory=path, save: str = '', verbose: bool = False):
    """Create a dataframe of audio features in pandas."""
    main_frame = pd.DataFrame()
    directory = [cat for cat in os.listdir(path_to_data_directory)
                 if cat != '.DS_Store']
    print('\n'.join([40*'=',
          f'Number of categories {len(directory)}', 40*'=']))
    for idx, category in enumerate(directory):
        files = [file for file in
                 os.listdir(os.path.join(path_to_data_directory, category))
                 if file != '.DS_Store']
        print(f'\nNumber of files in category {category},'
              f' (code: {idx}) -> {len(files)}\n')
        if verbose:
            files = tqdm(files)
        for audio in tqdm(files):
            name = audio
            target = int(idx)
            x, sr = librosa.load(os.path.join(path_to_data_directory,
                                 category, audio))
            rms = pd.Series(librosa.feature.rms(x)[0])
            tonnetz = _return_tonnetz(x, sr)

            mfcc = _return_mfcc(x, idx, name)
            row = {
                   'file': name,
                   'rms': rms,
                   'tonnetz': tonnetz,
                   'target': target
            }
            data_frame = pd.DataFrame(row).astype({'target': 'int8',
                                                  'file': 'category'})
            data_frame = data_frame.apply(pd.to_numeric, errors='ignore',
                                          downcast='float')
            data_frame.set_index(['file', 'target'], inplace=True)
            data_frame = pd.merge(data_frame, mfcc, left_index=True,
                                  right_index=True, how='outer')
            data_frame.set_index([data_frame.index, 'time'], inplace=True)
            main_frame = main_frame.append(data_frame)
    main_frame = main_frame.sort_index()
    main_frame = main_frame.apply(_normalize, axis=0)
    if save:
        main_frame.to_pickle(save, protocol=-1)
    return main_frame


@contextlib.contextmanager
def create_dataframe_context(path, save=''):
    """
    Context manager for create_dataframe function.

    Returns a dataframe.
    """
    try:
        yield create_dataframe(path, save)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    pass
