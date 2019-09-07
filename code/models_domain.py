import tensorflow as tf
import i3d_flow

slim = tf.contrib.slim          ###########

def model(input_batch, num_classes, mode, dropout_keep_prob = 1.0):
    if mode == 'brox' or mode == 'joint':
        dropout_keep_prob = 1.0 if mode == 'joint' else dropout_keep_prob
        with tf.variable_scope('Flow', reuse=tf.AUTO_REUSE):
            brox_model = i3d_flow.InceptionI3d(num_classes, spatial_squeeze=True, final_endpoint='Mixed_5c')
            brox_end_point, _ = brox_model(input_batch['brox'], is_training=True, dropout_keep_prob = dropout_keep_prob)     

        with tf.variable_scope('Flow_Logits', reuse=tf.AUTO_REUSE):
            brox_end_point = tf.nn.avg_pool3d(brox_end_point, ksize=[1, 2, 7, 7, 1],
                         strides=[1, 1, 1, 1, 1], padding='VALID')
            brox_end_point = tf.squeeze(slim.fully_connected(brox_end_point, num_classes, activation_fn=None))


    if mode == 'dvs' or mode == 'joint':
        with tf.variable_scope('DVS', reuse=tf.AUTO_REUSE):
            dvs_model = i3d_flow.InceptionI3d(num_classes, spatial_squeeze=True, final_endpoint='Mixed_5c')
            dvs_end_point, _ = dvs_model(input_batch['dvs'], is_training=True, dropout_keep_prob = dropout_keep_prob)

        with tf.variable_scope('DVS_Logits', reuse=tf.AUTO_REUSE):
            dvs_end_point = tf.nn.avg_pool3d(dvs_end_point, ksize=[1, 2, 7, 7, 1],
                         strides=[1, 1, 1, 1, 1], padding='VALID')
            dvs_end_point = tf.squeeze(slim.fully_connected(dvs_end_point, num_classes, activation_fn=None))

    if mode == 'dvs': 
        brox_end_point = None
    if mode == 'brox': 
        dvs_end_point = None
        
    return dvs_end_point, brox_end_point

def compute_loss(brox_end_point, dvs_end_point, labels_batch, alpha, temperature, mode):
            
            
    # compute loss
    with tf.variable_scope('Loss'):

        #supervised loss
        labels = labels_batch; logits = dvs_end_point if mode == 'dvs' else brox_end_point
        print labels, logits
        supervised_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        if mode == 'joint':
            #transfer loss
            labels = tf.divide(brox_end_point, FLAGS.temperature); logits = tf.divide(dvs_end_point, FLAGS.temperature)
            #convert labels from sparse to dense
            softmax = tf.nn.softmax(labels)
            transfer_loss = tf.nn.softmax_cross_entropy_with_logits(labels = softmax, logits = logits)
            train_loss = FLAGS.alpha * FLAGS.temperature**2 * transfer_loss + supervised_loss
        else:
            train_loss = supervised_loss

        total_loss = tf.reduce_mean(train_loss)

    return total_loss, logits