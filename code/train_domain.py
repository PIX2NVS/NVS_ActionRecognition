from datetime import datetime
import os.path
import time
import math
import random
import glob

import numpy as np
import tensorflow as tf

from inputs_domain import generate_batch_to_train
from models_domain import model, compute_loss

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

slim = tf.contrib.slim

# command line options
FLAGS = tf.app.flags.FLAGS

# Data flags
tf.app.flags.DEFINE_string('flow_data_dir', os.path.join('..', 'UCF101_Flow_tfrecords'),
                           """Flow data directory""")
tf.app.flags.DEFINE_string('flow_ckpt', 'kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt',
                           """Brox flow checkpoint""")
# 'kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt'
tf.app.flags.DEFINE_string('save_dir', os.path.abspath('runs_dvs_v2'),
                           """Directory to save checkpoints and summaries""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_every', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('summary_every', 100,
                            """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_every', 500,
                            """How often to checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 70000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_classes', 101,
                            """Number of classes (UCF-101: 101, HMDB-51: 51).""")

# Input flags
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('dvs_image_size', 224,
                            """Spatial size of dvs input.""")
tf.app.flags.DEFINE_integer('brox_image_size', 224,
                            """Spatial size of brox input.""")
tf.app.flags.DEFINE_integer('dvs_depth', 10,
                            """Depth of dvs input.""")
tf.app.flags.DEFINE_integer('brox_depth', 10,
                            """Depth of brox input.""")

# Training flags
tf.app.flags.DEFINE_float('temperature', 1.0,
                          """Temperature.""")
tf.app.flags.DEFINE_float('alpha', 1.0,
                          """Alpha weighting between knowledge transfer and supervised learning.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-2,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('momentum_rate', 0.9,
                          """Momentum rate.""")

tf.app.flags.DEFINE_integer('decay_step', 10e3,
                            """Number of steps between decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 1.0,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """Decay rate for exponential moving average""")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.2,
                          """Dropout keep probability.""")

mode = 'dvs'

# summaries directory
train_summary_dir = os.path.join(FLAGS.save_dir, 'train')
val_summary_dir = os.path.join(FLAGS.save_dir, 'val')

# checkpoint directory
checkpoint_dir = os.path.join(FLAGS.save_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def compute_accuracy(scores, labels):
    labels = tf.cast(labels, tf.int64)
    predictions = tf.argmax(scores, 1)
    correct_predictions = tf.equal(predictions, labels)
    accuracy = 100 * tf.reduce_mean(tf.cast(correct_predictions, "float"))
    return accuracy, predictions


def train(loss, scope):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step,
                                               FLAGS.decay_step, FLAGS.learning_rate_decay_factor, staircase=True)

    lr_summary = tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                  scope=scope))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return train_op, lr_summary, global_step


def build_graph(graph):
    with graph.as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        # read files
        filenames = glob.glob(os.path.join(FLAGS.flow_data_dir, '*.tfrecords'))
        random.shuffle(filenames)

        def parse_files(train_filename, filenames):
            print('Parsing inputs: ')
            with open(train_filename) as f:
                files_to_parse = [os.path.splitext(line.split()[0].split('/')[1])[0] for line in f]
                filenames = [d for d in filenames for item in files_to_parse if item in d]
            return filenames

        input_sizes = {}
        input_sizes['dvs'] = [FLAGS.dvs_image_size, FLAGS.dvs_image_size, 2, FLAGS.dvs_depth]
        input_sizes['brox'] = [FLAGS.dvs_image_size, FLAGS.brox_image_size, 2, FLAGS.brox_depth]

        filename_to_parse = 'trainlist01.txt'
        train_filenames = parse_files(os.path.join(os.curdir, 'ucfTrainTestlist', filename_to_parse), filenames)

        # num epochs
        num_epochs = int(math.floor(FLAGS.max_steps * FLAGS.batch_size / len(train_filenames)))

        filename_to_parse = 'testlist01.txt'
        val_filenames = parse_files(os.path.join(os.curdir, 'ucfTrainTestlist', filename_to_parse), filenames)[:1000]

        keys = ['dvs', 'brox'] if mode == 'joint' else [mode]
        input_batch, labels_batch, train_init_op, val_init_op = generate_batch_to_train(train_filenames, val_filenames,
                                                                                        input_sizes, FLAGS.batch_size,
                                                                                        keys)

        dvs_end_point, brox_end_point = model(input_batch, FLAGS.num_classes, mode,
                                              dropout_keep_prob=FLAGS.dropout_keep_prob)
        # restore kinetics weights
        if mode == 'brox':
            flow_variable_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'Flow':
                    flow_variable_map[variable.name.replace(':0', '')] = variable
            flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
            dvs_end_point = None
        elif mode == 'dvs':
            flow_saver = None
        elif mode == 'joint':
            flow_saver = tf.train.Saver(tf.global_variables())

        # TODO: add checkpoint copy of kinetics weights for dvs model pretraining (name.replace('first_scope', 'second_scope'))

        total_loss, logits = compute_loss(brox_end_point, dvs_end_point, labels_batch, FLAGS.alpha, FLAGS.temperature,
                                          mode)

        # compute accuracy
        scores = tf.nn.softmax(logits)
        accuracy, _ = compute_accuracy(scores, labels_batch)

        # generate optimizer
        scope = 'DVS' if mode == 'dvs' or mode == 'joint' else 'Flow'
        with tf.variable_scope('Optimizer'):
            train_op, lr_summary, global_step = train(total_loss, scope)

        # generate summary op
        image_summaries = []
        for key in keys:
            flow_shape = tf.shape(input_batch[key])
            flow_pad = tf.zeros([flow_shape[0], flow_shape[2], flow_shape[3], 1])
            image_summaries.append(
                tf.summary.image(key, tf.concat([tf.squeeze(input_batch[key][:, 0, :, :, :2]), flow_pad], axis=-1)))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        loss_summary = tf.summary.scalar('loss', total_loss)
        train_summary_op = tf.summary.merge_all()
        val_summary_op = tf.summary.merge([image_summaries])

        return flow_saver, train_op, train_summary_op, val_summary_op, train_init_op, val_init_op, accuracy, total_loss, global_step, num_epochs


def run():
    # build graph
    graph = tf.Graph()
    flow_saver, train_op, train_summary_op, val_summary_op, train_init_op, val_init_op, accuracy, total_loss, global_step, num_epochs = build_graph(
        graph)

    # Writer for summaries
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph)
    val_summary_writer = tf.summary.FileWriter(val_summary_dir, graph)

    def restore_fn(sess):
        # Initialize variables
        # tf.global_variables_initializer()
        if FLAGS.flow_ckpt is not None:
            # restore flow variables
            flow_saver.restore(sess, FLAGS.flow_ckpt)
            tf.logging.info('Flow checkpoint restored')

    sv = tf.train.Supervisor(graph=graph, logdir=checkpoint_dir, init_fn=restore_fn, global_step=global_step,
                             save_summaries_secs=0, save_model_secs=0, summary_op=None)

    with sv.managed_session() as sess:
        # Writer for summaries
        # summary_writer = tf.summary.FileWriter(train_summary_dir, graph)
        for epoch in range(num_epochs):

            print('Training:\n')
            sess.run(train_init_op)
            while True:
                try:
                    if sv.should_stop():
                        break

                    start_time = time.time()
                    _, batch_accuracy, batch_loss, step, summary = sess.run(
                        [train_op, accuracy, total_loss, sv.global_step,
                         train_summary_op])
                    duration = time.time() - start_time

                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    if step % FLAGS.log_every == 0:
                        tf.logging.info(
                            'training step %d, accuracy = %.2f, loss = %.2f, (%.1f examples/sec; %.3f sec/batch)',
                            step, batch_accuracy, batch_loss, examples_per_sec, sec_per_batch)

                    if step % FLAGS.summary_every == 0:
                        train_summary_writer.add_summary(summary, step)
                        tf.logging.info('\nSaving summaries...\n')

                    if step % FLAGS.checkpoint_every == 0:
                        sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
                        print("\nSaved model checkpoint to {}\n".format(sv.save_path))
                except tf.errors.OutOfRangeError:
                    print('\nEnd of epoch %i\n', epoch)
                    break


            print('Validation:\n')
            sess.run(val_init_op)

            sum_accuracy = 0;
            num_batches = 0
            sum_loss = 0
            while True:
                try:
                    start_time = time.time()
                    batch_accuracy, batch_loss, summary = sess.run([accuracy, total_loss, val_summary_op])

                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    tf.logging.info('accuracy = %.2f, loss = %.2f, (%.1f examples/sec; %.3f sec/batch)',
                                    batch_accuracy, batch_loss, examples_per_sec, sec_per_batch)

                    sum_accuracy += batch_accuracy
                    sum_loss += batch_loss
                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            total_accuracy = float(sum_accuracy) / num_batches
            mean_loss = float(sum_loss) / num_batches

            tf.logging.info('\nstep %d, validation accuracy = %.2f, validation loss = %.2f\n', step, total_accuracy,
                            mean_loss)

            # write summary
            val_summary = tf.Summary()
            val_summary.ParseFromString(summary)
            val_summary.value.add(tag='accuracy', simple_value=total_accuracy)
            val_summary.value.add(tag='loss', simple_value=mean_loss)
            val_summary_writer.add_summary(val_summary, step)


if __name__ == '__main__':
    run()











