from utils import *

def do_preprocess(max_lenght=None):
  raw, label, length = load_data()

  if not isinstance(max_lenght, int):
    max_lenght = max(length)
  
  mels = mel_extract(raws=raw, sr=sr, max_size=max_lenght)

  max_label = np.max(label)
  min_label = np.min(label)
  print('Max of label:', max_label, 'Min of Label:', min_label)

  onehot_label = [[1 if (i+1)==l else 0 for i in range(max_label)] for l in label]
  onehot_label = np.array(onehot_label)
  print('OneHot shape:', onehot_label.shape)

  return raw, mels, onehot_label, label, length

def do_local_preprocess(wav_path, sr, label_loc, label_dict, max_lenght=None, n_mels=64):
  raw, label, length = load_local(wav_path, sr, label_loc, label_dict)

  if not isinstance(max_lenght, int):
    max_lenght = max(length)
  
  mels = mel_extract(raws=raw, sr=sr, max_size=max_lenght, n_mels=n_mels)

  max_label = np.max(label)
  min_label = np.min(label)
  print('Max of label:', max_label, 'Min of Label:', min_label)

  onehot_label = [[1 if i==l else 0 for i in range(min_label, max_label+1)] for l in label]
  onehot_label = np.array(onehot_label)
  print('OneHot shape:', onehot_label.shape)

  return raw, mels, onehot_label, label, length
