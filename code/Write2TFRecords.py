import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import os
import scipy.io
import shutil
import json
from StringIO import StringIO



# command line options
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dvs_data_dir', os.path.join('UCF101_DVS_Flow_2'),
                           """Brox data directory""")
tf.app.flags.DEFINE_string('brox_data_dir', os.path.join('UCF101_Brox_Flow'),
                           """DVS data directory""")
tf.app.flags.DEFINE_string('out_dir', os.path.join('UCF101_Flow_tfrecords'),
                           """Output directory""")
tf.app.flags.DEFINE_string('path_to_class_list', os.path.join(os.path.curdir, 'ucfTrainTestlist', 'classInd.txt'),
                           """Path to class index list""")


mask = r'*.[jp][pn]*'


if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_frames_to_record(key, example, examples_to_write):
    frames_list = example.feature_lists.feature_list[key]
    
    def _process_frames(image_list):
        concat_frames = np.concatenate(image_list)
        output = StringIO()
        format = 'PNG'
        PIL_frames = Image.fromarray(concat_frames)
        PIL_frames.save(output, format)
        contents = output.getvalue()
        output.close()
        return contents

    image_list = []
    for i, full_path in enumerate(examples_to_write):
        image = np.array(Image.open(full_path))
        image_list.append(image)
        
        # process 10 frames at a time
        if i % 10 == 0 and i != 0:
            contents = _process_frames(image_list)

            # add frame to tfrecord
            frames_list.feature.add().bytes_list.value.append(contents)
            image_list = []
       
    if image_list:
        contents = _process_frames(image_list)

        # add frame to tfrecord
        frames_list.feature.add().bytes_list.value.append(contents)    


# load data from folder
dvs_u_folders = os.listdir(os.path.join(FLAGS.dvs_data_dir, 'u'))
dvs_v_folders = os.listdir(os.path.join(FLAGS.dvs_data_dir, 'v'))
brox_u_folders = os.listdir(os.path.join(FLAGS.brox_data_dir, 'u'))
brox_v_folders = os.listdir(os.path.join(FLAGS.brox_data_dir, 'v'))


folders = list(set(brox_u_folders) & set(brox_v_folders) & set(dvs_u_folders) & set(dvs_v_folders))
print len(folders)


# read labels into dictionary
label_dict = {}
with open(FLAGS.path_to_class_list) as f:
    for line in f:
        (val, key) = line.split()
        label_dict[key.lower()] = int(val) - 1
    

for folder in folders:
    #folder_header = folder.split('/')[-2]

    # get label index
    fileparts = folder.split('_')
    label = fileparts[-3]
    label_idx = int(label_dict[label.lower()])
    
    
    tfrecords_filename = folder + '.tfrecords'
    write_to =  os.path.join(FLAGS.out_dir, tfrecords_filename)

    
    #check if tfrecords already exists
    if os.path.exists(write_to):
        continue
    
    writer = tf.python_io.TFRecordWriter(write_to)
    
    print('Writing contents of folder %s to tfrecords, label %i ...' % (folder, label_idx))
    brox_u_examples_to_write = glob.glob(os.path.join(FLAGS.brox_data_dir, 'u', folder, mask))
    brox_v_examples_to_write = glob.glob(os.path.join(FLAGS.brox_data_dir, 'v', folder, mask))
    dvs_u_examples_to_write = glob.glob(os.path.join(FLAGS.dvs_data_dir, 'u', folder, mask))
    dvs_v_examples_to_write = glob.glob(os.path.join(FLAGS.dvs_data_dir, 'v', folder, mask))


    # get height and width of brox and dvs image
    image = np.array(Image.open(dvs_u_examples_to_write[0]))
    dvs_dims = list(image.shape)

    
    image = np.array(Image.open(brox_u_examples_to_write[0]))
    brox_dims = list(image.shape)

    
    example = tf.train.SequenceExample()
    example.context.feature['video_name'].bytes_list.value.append(folder.encode('utf-8'))
    example.context.feature["label"].int64_list.value.append(label_idx)
    example.context.feature["brox_dims"].int64_list.value.extend(brox_dims)
    example.context.feature["dvs_dims"].int64_list.value.extend(dvs_dims)

    write_frames_to_record('brox_u', example, brox_u_examples_to_write)
    write_frames_to_record('brox_v', example, brox_v_examples_to_write)

    
    write_frames_to_record('dvs_u', example, dvs_u_examples_to_write)
    write_frames_to_record('dvs_v', example, dvs_v_examples_to_write)

    writer.write(example.SerializeToString())
    writer.close()
    
    #sanity check
    for example in tf.python_io.tf_record_iterator(write_to):
        result = tf.train.Example.FromString(example)
    
    
    #delete folders
    print('Removing folders %s...' % folder)
    shutil.rmtree(os.path.join(FLAGS.brox_data_dir, 'u', folder))
    shutil.rmtree(os.path.join(FLAGS.brox_data_dir, 'v', folder))
    shutil.rmtree(os.path.join(FLAGS.dvs_data_dir, 'u', folder))
    shutil.rmtree(os.path.join(FLAGS.dvs_data_dir, 'v', folder))
  
    
    
'''
paths = glob.glob(os.path.join(FLAGS.out_dir, '*.tfrecords'))

sess = tf.InteractiveSession()
for path in paths:
    for example in tf.python_io.tf_record_iterator(path):
        context, features = tf.parse_single_sequence_example(
            example,
            context_features={'video_name': tf.FixedLenFeature([], tf.string),
                              'label': tf.FixedLenFeature([], tf.int64),
                              'dvs_dims': tf.FixedLenFeature([2], tf.int64),
                              'brox_dims': tf.FixedLenFeature([2], tf.int64)
                             },
            sequence_features={'brox_u' : tf.FixedLenSequenceFeature([], dtype=tf.string),
                               'brox_v' : tf.FixedLenSequenceFeature([], dtype=tf.string),
                               'dvs_u' : tf.FixedLenSequenceFeature([], dtype=tf.string),
                               'dvs_v' : tf.FixedLenSequenceFeature([], dtype=tf.string)
                              })


        def _get_video_matrix(features, dims):
            num_cols = tf.squeeze(dims[0] * dims[1])
            print num_cols
            #frames in row order
            rand_idx = tf.random_uniform([], minval=0, maxval=tf.shape(features)[0], dtype=tf.int32)
            decoded_features = tf.reshape(tf.cast(tf.image.decode_png(features[rand_idx]), tf.float32),tf.stack([-1, num_cols],0))
            return decoded_features
        

        brox = _get_video_matrix(features['brox_u'], context['brox_dims'])        

        b =  brox.eval()
        print b.shape

        result = tf.train.Example.FromString(example)
        print path
    break
'''



