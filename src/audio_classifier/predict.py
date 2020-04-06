"""
TO-DO
-start incorperating test modules
-map results to waveform and graphed area
    * return times from pandas
    * plot x axis with times
"""
import os
import contextlib
import sys

import pytest
import librosa
import sklearn
import pandas as pd
import numpy as np
from tqdm import tqdm
from kymatio.numpy import Scattering1D
import matplotlib.pyplot as plt

import model as mdl
from loader import _loader
from create_dataframe_v1 import (_return_contrast,
                                 _return_mfcc,
                                 _return_spec_centroid,
                                 _return_tempo,
                                 _return_tonnetz,
                                 _normalize)


def file_features(path, max_len: int = 30):
    """
    Extract features from audio file \
    to make classification.

    -build a command from create_database for single file
        * break into further units (singletons)
    """
    main_frame = pd.DataFrame()
    audio, sr = _loader(path, max_len)
    target = 1
    for _, x in enumerate(audio.values()):
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
    X = X.reset_index()
    X = X.drop(columns='file')
    # y = y.reset_index()
    # y = y.drop(columns='file')
    return X


def split_results(x):
    """Split estamtes into structual judgements.

     <<<write IDE example doc

     """
    try:
        winners = [i for i in x if i == 0]
        winners = winners[:(len(winners)//20)]
        assert len(winners) == 1
    except AssertionError:
        print(len(winners))
    non_winners = [i for i in x if i == 1]
    non_winners = non_winners[:(len(non_winners)//20)]
    # assert len(non_winners) == 0
    return winners, non_winners


def build_pie(x):
    """Build pie chart showing ratio \
    of categories within one file.

     <<<write IDE example doc
     
     """
    a, b = split_results(x)
    labels = ['winners', 'non_winners']
    sizes = [len(a), len(b)]
    _, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.show()


def build_disp(estimate):
    """
    TO-DO:
    - create save option
  
    reference:
    sublot bg-color : https://stackoverflow.com/questions/9957637/how-can-i-set-the-background-color-on-specific-areas-of-a-pyplot-figure
    subplot bg-color2 : https://stackoverflow.com/questions/15861875/custom-background-sections-for-matplotlib-figure
    matplot-doc: https://matplotlib.org/2.1.0/gallery/subplots_axes_and_figures/axhspan_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axhspan-demo-py
    """
    x = sys.argv[1]
    signal, sample_rate = librosa.load(x)
    length_sig = round(len(signal) / sample_rate)
    length = int((length_sig // 40) + 1)
    time = np.linspace(0, len(signal) / sample_rate,
                       num=len(signal))
    fig, ax = plt.subplots(2, 1)
    plt.sca(ax[0])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.plot(time, signal)
    plt.xlabel("Time in seconds")
    # bgcolor = np.linspace(0, len(signal), num=length)
    count = 0
    nxt = 20
    for i in range(0, int(time[-1]), 30):
        if (estimate[count:nxt] == 1).any():
            plt.axvspan(i, (i + 29), facecolor='r', alpha=0.5) # use i+30 for no whitespace in graph
        else:
            plt.axvspan(i, (i + 29), facecolor='g', alpha=0.5)
        count += 20
        nxt += 20
    # plt.xticks(np.arange(0, length, 1))

    plt.sca(ax[1])
    plt.specgram(signal, NFFT=1024, scale='dB',
                 Fs=sample_rate, noverlap=900)
    plt.xlabel('Time in seconds')
    plt.ylabel('Frequency')
    # plt.ylim(20, 8000)
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """Command line functions:
 
    sys.argv1 = file for judgement estiamtes
    sys.argv2 = undefined
    """
    print("""\n   Estimate is in progress..
          Be patient.""")
    path = sys.argv[1]
    features = file_features(path)
    model = mdl.main()
    estimate = model.predict(features)
    count = 20
    while count < len(estimate):
        x = count
        prev = count - 20
        print(estimate[prev:x])
        count += 20
    build_pie(estimate)
    build_disp(estimate)

