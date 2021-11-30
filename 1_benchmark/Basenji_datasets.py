import os, h5py, json
from sys import argv
import numpy as np
import tensorflow as tf

def feature_bytes(values):
    """Convert numpy arrays to bytes features."""
    values = values.flatten().tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

dataset_fname, out_dir = argv[1:]

# load nvwa-datasets
h5file = h5py.File(dataset_fname, 'r')

# define options
tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
# tfrecord
fold_labels = ['valid', 'test', 'train']
num_folds = 3
fold_mseqs = []

for fold_type in fold_labels:
    if fold_type == 'valid' or fold_type == 'train' :
        data = h5file["val_data"][:].astype(bool)
        label = h5file["val_label"][:].astype(float)
    else:
        data = h5file[fold_type+"_data"][:].astype(bool)
        label = h5file[fold_type+"_label"][:].astype(float)

    num_seqs = data.shape[0]
    fold_mseqs.append(num_seqs)

    tfr_file = os.path.join(out_dir, "tfrecords", fold_type + '-0.tfr')
    with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:

        for si in range(num_seqs):
            seq_1hot = data[si,:,3500:16500].swapaxes(0,-1)
            target = label[si,None,:]
            assert seq_1hot.shape == (13000, 4)
            # assert target.shape == (1, 134557)

            # hash to bytes
            features_dict = {
                'sequence': feature_bytes(seq_1hot),
                'target': feature_bytes(target)
            }

            # write example
            example = tf.train.Example(features=tf.train.Features(feature=features_dict))
            writer.write(example.SerializeToString())

h5file.close()

################################################################
# stats
################################################################
seq_length = 13000
pool_width = 13000 // 4
crop_bp = 0

stats_dict = {}
stats_dict['num_targets'] = label.shape[-1]
stats_dict['seq_length'] = seq_length
stats_dict['pool_width'] = pool_width
stats_dict['crop_bp'] = crop_bp

target_length = seq_length - 2*crop_bp
target_length = target_length // pool_width
stats_dict['target_length'] = target_length

for fi in range(num_folds):
	stats_dict['%s_seqs' % fold_labels[fi]] = fold_mseqs[fi]

with open('%s/statistics.json' % out_dir, 'w') as stats_json_out:
	json.dump(stats_dict, stats_json_out, indent=4)
