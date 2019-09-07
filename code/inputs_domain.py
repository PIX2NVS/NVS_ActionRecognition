import tensorflow as tf
import random
import inception_preprocessing

NUM_THREADS = 8

def _get_video_matrix(features, dims, rand_idx):
    num_cols = tf.squeeze(dims[0] * dims[1])                        
    #frames in row order
    
    #TODO: concatenate multiple blocks of frames - for loop?
    decoded_features = tf.reshape(tf.cast(tf.image.decode_png(features[rand_idx]), tf.float32),tf.stack([-1, num_cols],0))   
    return decoded_features

def create_dataset(filenames, input_sizes, batch_size, train):
    dataset = tf.data.TFRecordDataset(filenames)                                 
    dataset = dataset.shuffle(buffer_size=1000)
    resize_to = input_sizes
    
    def parser(record):

        context, features = tf.parse_single_sequence_example(                      
            record,
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


        video_name = context['video_name']                   
        label = tf.cast(context['label'], tf.int32)
        dvs_dims = tf.cast(context['dvs_dims'], tf.int32)
        brox_dims = tf.cast(context['brox_dims'], tf.int32)

        # stack dx and dy components
        rand_idx = tf.random_uniform([], minval=0, maxval=tf.shape(features['dvs_u'])[0]-1, dtype=tf.int32) 
        dvs_record = tf.concat([_get_video_matrix(features['dvs_u'], dvs_dims, rand_idx),          
                                _get_video_matrix(features['dvs_v'], dvs_dims, rand_idx)],1)         
        
        rand_idx = tf.random_uniform([], minval=0, maxval=tf.shape(features['brox_u'])[0]-1, dtype=tf.int32) 
        brox_record = tf.concat([_get_video_matrix(features['brox_u'], brox_dims, rand_idx), 
                                 _get_video_matrix(features['brox_v'], brox_dims, rand_idx)],1)
        print dvs_record, brox_record


        # preprocessing (only center crop atm)
        bbox = None
        '''
        brox_num_frames = tf.shape(brox_record)[0]; dvs_num_frames = tf.shape(dvs_record)[0]
        if train:
            # get a random starting index between 0 and record_shape[0]
            brox_idx = tf.random_uniform([], minval=0, maxval=brox_num_frames, dtype=tf.int32)
        else:
            # select the starting index based on center index
            brox_idx = brox_num_frames - brox_num_frames / 2
            
            
        # no spatial or temporal jitter
        dvs_idx = tf.cast(tf.ceil((tf.multiply(tf.truediv(brox_idx,brox_num_frames), 
                                               tf.cast(dvs_num_frames, tf.float64)))), tf.int32)
        '''
        
        brox_image, _ = _flow_preprocess(brox_record, brox_dims[0], brox_dims[1], resize_to['brox'], bbox, 
                                                    train=train)
        dvs_image, _ = _flow_preprocess(dvs_record, dvs_dims[0], dvs_dims[1], resize_to['dvs'], bbox, train=train)
        return {'brox': brox_image, 'dvs': dvs_image}, label


    dataset = dataset.map(parser, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)

    return dataset

def generate_batch_to_train(train_filenames, val_filenames, input_sizes, batch_size, keys):
    
    # zip datasets for each channel
    train_dataset = create_dataset(train_filenames, input_sizes, batch_size, train=True)
    val_dataset = create_dataset(val_filenames, input_sizes, batch_size, train=False)


    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    input_batch, labels_batch = iterator.get_next()
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(val_dataset)

    print input_batch, keys
    # filter batch based on keys
    input_batch = {k: v for k, v in input_batch.iteritems() if k in keys}          ################################
    print input_batch
    
    return input_batch, labels_batch, training_init_op, validation_init_op


def generate_batch_to_test(filenames, input_sizes, batch_size, keys):

    # zip datasets for each channel
    dataset = create_dataset(filenames, input_sizes, batch_size, train=False)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    input_batch, labels_batch = iterator.get_next()

    # filter batch based on keys
    input_batch = {k: v for k, v in input_batch.iteritems() if k == keys}

    
    return input_batch, labels_batch




def convert_2D_to_3D_tensor(example, input_size):
    example_shape = tf.shape(example)
    example = tf.reshape(example, [input_size[0], input_size[1], input_size[3], input_size[2]])         #### [224,224,20]---->[224,224,10,2]
    example = tf.transpose(example, [2, 0, 1, 3])       ##############[224,224,10,2]---->[10,224,224,2]
    return example


def _flow_preprocess(mv_record, height, width, input_size, bbox, train=True):
    mv_record_shape = tf.shape(mv_record)
    mv_bytes = 2*tf.multiply(height, width)
    temporal_depth = input_size[-1]
    
    mv_input = tf.random_crop(mv_record, [temporal_depth, mv_bytes])

    mv_input = tf.reshape(mv_input, [-1])


    # split each list element into x and y components and stack depth wise
    image_shape = tf.stack([2*temporal_depth, height, width])
    image_chunk = tf.reshape(mv_input, image_shape)
    image_chunk = tf.transpose(image_chunk, perm=[1, 2, 0])             #### [H,W,20]

    # cast image block as float
    image_chunk = tf.cast(image_chunk, tf.float32)
    
    def normalize_images(image_chunk):
        #mean subtraction and normalization
        mean, variance = tf.nn.moments(image_chunk, axes=[0,1])       ################    [H,W,20]  
        float_image = tf.subtract(image_chunk,tf.expand_dims(tf.expand_dims(mean,0),0))        ################ -mean?

        return float_image
    
    image_chunk = normalize_images(image_chunk)    

    
    # rescale between [0,1]
    image_chunk = tf.truediv(tf.subtract(image_chunk, tf.reduce_min(image_chunk)), 
                         tf.subtract(tf.reduce_max(image_chunk), tf.reduce_min(image_chunk)))     
    
    
    # if bbox provided don't distort and just crop and resize
    #bbox_distortion = False if bbox is not None else True
    bbox_distortion = True

    resized_image_chunk, distorted_bbox = inception_preprocessing.preprocess_image(image_chunk, input_size[0], input_size[1], is_training=train, depth=2*temporal_depth, color_distortion=False, bbox=bbox, bbox_distortion=bbox_distortion)
    float_image = tf.image.resize_images(resized_image_chunk, size=[input_size[0], input_size[1]])
    print float_image
    
    # convert to 3D tensor
    float_image = convert_2D_to_3D_tensor(float_image, input_size)
    
    return float_image, distorted_bbox



