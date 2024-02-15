from util import *

def do_preprocess():
  raw, label, length = load_data()
  mels = mel_extract(raws=raw, max_size=max(length))

  max_label = np.max(label)
  min_label = np.min(label)
  print('Max of label:', max_label, 'Min of Label:', min_label)

  onehot_label = [[1 if (i+1)==l else 0 for i in range(max_label)] for l in label]
  onehot_label = np.array(onehot_label)
  print('OneHot shape:', onehot_label.shape)

  return raw, mels, onehot_label, label, length
