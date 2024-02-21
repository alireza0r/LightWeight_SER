# !pip install deeplake
# !pip install deeplake[audio]
# !pip install librosa

import deeplake
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def load_local(wav_path, sr, label_loc, label_dict):
    emotions = []
    raw = []
    length = []
    files = os.listdir(wav_path)
    for i, f in enumerate(files):
        try:
            path = os.path.join(wav_path, f)
            r, sr = librosa.load(path, sr=sr)
            raw.append(r)
            emotions.append(label_dict[f[label_loc]])
            length.append(len(r))
        except Exception as e:
            print(i, 'error:')
            print(e)
    return raw, emotions, length

def load_data(name="hub://activeloop/ravdess-emotional-speech-audio"):
    repetition = 1
    ds = deeplake.load(name)

    emotions = []
    raw = []
    length = []
    for i in range(len(ds)):
        if ds.repetitions[i].numpy() == repetition:
            try:
                raw.append(ds.audios[i].numpy())
                emotions.append(ds.emotions[i].numpy())
                length.append(len(ds.audios[i].numpy()))
            except:
                print(i, 'error')

    return raw, emotions, length

def length_equality(raws, lentgh, max_lenght=None):
  new_raw = []
  
  if not isinstance(max_lenght, int):
    max_lenght = max(lentgh)
      
  for r, l in zip(raws, lentgh):
    new_raw.append(librosa.util.fix_length(r, size=max_lenght, axis=0))

  return new_raw


def mel_extract(raws, max_size, sr=44100, n_mels=64):
  mels = []

  for r in raws:
    y = librosa.util.fix_length(r, size=max_size, axis=0)
    S = librosa.feature.melspectrogram(y=np.squeeze(y), sr=sr, n_mels=n_mels, n_fft=512)
    S_DB = librosa.power_to_db(S, ref=np.max)
    mels.append(S_DB)

  mels = np.stack(mels)
  return mels
