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
    mfcc['time'] = pd.to_datetime(mfcc.time, unit='s')
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


@pytest.mark.slow
def create_dataframe(path_to_data_directory=path, save: str = '',
                     verbose_prog: bool = False):
    """
    Create a dataframe of audio features in pandas.

    Sort categories by last digists following an underscore in category name.

    >>> path = 'samples_data'
    >>> df = create_dataframe(path)
    Category Name  |   code
    =======================
    example_1      |     0
    category_2     |     1
    >>> 
    """
    main_frame = pd.DataFrame()
    directory = sorted([cat for cat in os.listdir(path_to_data_directory)
                        if cat != '.DS_Store'], key=lambda x: x.split('_')[-1])
    if verbose_prog:
        print('\n'.join([40*'=',
              f'Number of categories {len(directory)}', 40*'=']))
    if not verbose_prog:
        str_len = len(str(max([x for x in directory])))
        title = 'Category Name'
        legend = f'{title}{(str_len - len(title)) * " "}  |   code'
        print(legend)
        print(len(legend) * '=')
        for idx, category in enumerate(directory):
            cat_len = len(category)
            print(f'{category}{(len(title) - (cat_len)) * " "}  |     {idx}')
    for idx, category in enumerate(directory):
        files = [file for file in
                 os.listdir(os.path.join(path_to_data_directory, category))
                 if file != '.DS_Store']
        if verbose_prog:
            print(f'\nNumber of files in category {category},'
                  f' (code: {idx}) -> {len(files)}\n')
            files = tqdm(files)
        for audio in files:
            name = audio
            target = int(idx)
            x, sr = librosa.load(os.path.join(path_to_data_directory,
                                 category, audio))
            spectral_centroids = _return_spec_centroid(x, sr)
            chromagram = pd.Series(librosa.feature.chroma_stft(x, sr=sr)[0])
            rms = pd.Series(librosa.feature.rms(x)[0])
            tonnetz = _return_tonnetz(x, sr)
            contrast = _return_contrast(x, sr)
            tempo = _return_tempo(x, sr)
            # wavlet = make_wavlet(x)
            mfcc = _return_mfcc(x, idx, name)
            row = {
                   'file': name,
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
            # data_frame = data_frame.rolling(50, center=True,
            #                                 win_type='bartlett',
            #                                 on='time').mean()
            main_frame = main_frame.append(data_frame)
    main_frame = main_frame.sort_index()
    main_frame = main_frame[main_frame.columns].apply(_normalize, axis=0)
    main_frame.index = main_frame.index.astype('category')
    if save:
        main_frame.to_pickle(save, protocol=-1)
    return main_frame


@pytest.mark.slow
@contextlib.contextmanager
def create_dataframe_context(path, save='', verbose_prog=False):
    """
    Context manager for create_dataframe function.

    Returns a dataframe.

    >>> path = 'audio_classification/samples_data'
    >>> with create_dataframe_context(path) as data:
    ...     df = data
    ... 
    Category Name  |   code
    =======================
    example_1      |     0
    category_2     |     1
    >>>
    """
    try:
        yield create_dataframe(path, save, verbose_prog)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    pass
