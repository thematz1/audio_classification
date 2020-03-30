import librosa

audio_path = 'samples_data/category_2/01.wav'


def _loader(audio_path, max_len: int = 30):
    x, sr = librosa.load(audio_path)
    split = sr * max_len
    total_splits = len(x) // split
    file_data = {}
    prev = 0
    for i in range(total_splits):
        nxt = prev + split
        file_data[i] = x[prev:nxt]
        prev = nxt
    return file_data, sr
