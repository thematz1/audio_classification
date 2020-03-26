"""Test database creation  module."""
from random import (randint, uniform)
from pprint import pprint as pp
import os
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
import scipy.signal as sps
import pytest

from src.audio_classifier.create_dataframe import (load, pad_length,
                                                   make_wavlet, make_mfcc,
                                                   make_dataframe)


# sys.path.append('..')
# Unit tests
path = str(Path(__file__).parent.parent / "samples_data/")


def test_load():
    """Test the load audio function."""
    test_file = os.path.join(path, "category 1/01.wav")
    _, x = load(test_file)
    s, _ = librosa.load(test_file, sr=16000)
    assert len(s) == len(x)


@pytest.mark.skip(reason="WIP")
def test_pad_lengths():
    """
    Create an numpy array of audio segments of the length max length.

    Pads those that are too short.
    """
    test_file = os.path.join(path, "category 1/01.wav")
    max_len = randint(1, 60)
    _, t = load(test_file)
    num_splits = len(t)//max_len
    _, x = pad_length(t, max_len=max_len)
    assert len(x) == num_splits
    assert len(x[0]) == max_len
    if isinstance((len(t)/max_len), float):
        assert x[0][-1] == 0


def test_make_wavelet():
    """
    Random length test of wavlet opperation.

    Max-order 3. 16 voice per octave. SR = 16000. T is just over 1 second.
    """
    test_file = os.path.join(path, "category 1/01.wav")
    _, audio = load(test_file)
    x = make_wavlet(audio)
    J = 9
    expect = (J * 16) - (16*2)
    assert len(x) == expect


@pytest.fixture
def data(tmpdir):
    """Create pandas dataframe with tempdir for storage."""
    data = make_dataframe
    yield data, tmpdir


@pytest.mark.slow
def test_dataset_is_same_length_as_input(data):
    """Check that length of files in is the same as returned."""
    test_cat = [os.path.join(path, cat) for cat in os.listdir(path)
                if cat != '.DS_Store']
    test_len = len([os.path.join(category, cat) for category in test_cat
                    for cat in os.listdir(category) if cat != '.DS_Store'])
    test_cat = None
    data, _ = data
    data_test = data(path)
    assert len(data_test) == test_len


@pytest.mark.slow
def test_dataset_save_works(data, tmpdir):
    """Check that dataframe is pickling."""
    data, tmpdir = data
    data(path, cache=tmpdir, save=True)
    assert 'data_frame.pkl' in list(os.listdir(tmpdir))


@pytest.mark.slow
def test_dataset_has_proper_categories(data):
    """Test that the proper amound of categories is returned."""
    data, _ = data
    database = data(path)
    categories = database['category'].unique()
    num_categories = len([cat for cat in os.listdir(path)
                          if cat != '.DS_Store'])
    assert num_categories == len(categories)
