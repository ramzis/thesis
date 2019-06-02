from __future__ import print_function

import os
import argparse
import tensorflow as tf
from math import ceil
from pre import Pre

# For confusion matrix
import re
import tfplot
import itertools
import matplotlib
import numpy as np
from textwrap import wrap
from sklearn.metrics import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
# Model params
parser.add_argument('--data_desc', default='D1.json', type=str)
parser.add_argument('--data_path', default='raw', type=str)
parser.add_argument('--model_name', default='baseline', type=str)
# Augmentations
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--data_oversample', default=False, type=bool)
parser.add_argument('--data_augment', default=False, type=bool)
parser.add_argument('--n_augmentations', default=0, type=int)
parser.add_argument('--augment_dir', default='../scratch/augmented/', type=str)
# Training params
parser.add_argument('--data_folds', default=4, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)


# Confustion matrix data
def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name='MyFigure/image', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a list of labels which will be used to display the axis labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other items to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''

    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


# Reading the dataset
def read_images(imagepaths, labels, args, is_training=False):

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize images to a common size
    image = tf.image.resize_images(image, [args.img_dim, args.img_dim])

    # Subtract off the mean and divide by the variance of the pixels.
    norm_image = tf.image.per_image_standardization(image)

    min_queue_examples = int(640 * 0.4)
    num_preprocess_threads = 16
    # Create batches
    X, Y = tf.train.shuffle_batch(
        [norm_image, label],
        batch_size=args.batch_size,
        min_after_dequeue=min_queue_examples,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * args.batch_size)

    # Display the images in the visualizer.
    # if is_training:
    #    tf.summary.image('training', X, max_outputs=10)
    # else:
    #    tf.summary.image('validation', X, max_outputs=10)

    return X, Y


# Create model
def conv_net(x, n_classes, args, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)

        # Visualise conv2 layer activations
        for i in range(32):
            if not is_training:
                tf.summary.image('conv2_' + str(i), tf.reshape(conv2[0, :, :, i], [1, 108, 108, 1] ), max_outputs=64)

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=args.dropout_keep_prob, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Load data and perform training
def main(args):

    # Initialise data paths
    if os.path.isfile(args.data_desc):
        p = Pre().loadData(args.data_desc)
    else:
        exit(1)
        # p = Pre().createData(args.data_desc, args.data_path, data_thresh=4, verbose=True)
        # p.saveData(args.data_desc)

    # Cross-validation iterator with optional augmentation,
    # number of folds and training data oversampling
    if args.data_augment:
        data_iterator = p.get_augmented_train_and_test_data(
            n_augmentations=args.n_augmentations,
            n_splits=args.data_folds, balance=args.data_oversample,
            augmentation_dir=args.augment_dir)
    else:
        data_iterator = p.get_cv_train_and_test_data(
            n_splits=args.data_folds, balance=args.data_oversample)

    fold_no = 0

    for train_path, train_label, test_path, test_label in data_iterator:

        # Reset graph for each iteration
        tf.reset_default_graph()

        # Count the fold iterations
        fold_no += 1

        # Ensure same labels for train and validation data
        assert set(train_label) == set(test_label),\
            "Train and val labels don't correspond:\n{}\n{}".format(set(train_label), set(test_label))

        # Get number of classes from data labels
        num_classes = len(set(train_label))

        # Build the training data input
        X, Y = read_images(train_path, train_label, args, is_training=True)

        # Build the validation data input
        X_V, Y_V = read_images(test_path, test_label, args, is_training=False)

        # Create a graph for training
        logits_train = conv_net(X, num_classes, args, reuse=False, is_training=True)

        # Create a graph for testing that reuses the same weights and has no dropout
        logits_test = conv_net(X, num_classes, args, reuse=True, is_training=False)

        # Create a graph for validation that reuses the same weights and has no dropout
        logits_validation = conv_net(X_V, num_classes, args, reuse=True, is_training=False)

        with tf.name_scope("loss"):
            # Define loss and optimizer (with train logits, for dropout to take effect)
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_train, labels=Y))
            tf.summary.scalar("training", loss_op)

            # Define loss of validation
            validation_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_validation, labels=Y_V))
            tf.summary.scalar("validation", validation_op)

            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            train_op = optimizer.minimize(loss_op)


        with tf.name_scope("accuracy"):
            # Evaluate model (with test logits, for dropout to be disabled)
            correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar("training", accuracy)

            # Evaluate model (with validation logits, for dropout to be disabled)
            correct_pred_v = tf.equal(tf.argmax(logits_validation, 1), tf.cast(Y_V, tf.int64))
            accuracy_valid = tf.reduce_mean(tf.cast(correct_pred_v, tf.float32))
            tf.summary.scalar("validation", accuracy_valid)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Create a log writer, run 'tensorboard --logdir=./logs/nn_logs'
            dirs = ["./logs/nn_logs/", str(num_classes), args.model_name, str(args.data_folds), str(fold_no)]
            run_log_dir = os.path.join(*dirs)
            writer = tf.summary.FileWriter(run_log_dir, sess.graph)
            merged = tf.summary.merge_all()

            desc = ("net type: "            + "Basic CNN" +
                    "\nname: "              + str(args.model_name) +
                    "\nn_classes: "         + str(num_classes) +
                    "\nimg_size: "          + str(args.img_dim) +
                    "\nbatch_mode: "        + "shuffle" +
                    "\nbatch_size: "        + str(args.batch_size) +
                    "\ndata_folds: "        + str(args.data_folds) +
                    "\nnum_epochs: "        + str(args.num_epochs) +
                    "\nlearning_rate: "     + str(args.learning_rate) +
                    "\ndropout_keep_prob: " + str(args.dropout_keep_prob)
                    )

            # Run the initializer
            sess.run(init)

            # Add run description
            with open(run_log_dir + '/run_info.txt', 'w') as f:
                f.write(desc)

            # Start the data queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Training cycle
            samples = len(train_path)
            batches_per_epoch = int(ceil(float(samples) / args.batch_size))
            for step in range(1, batches_per_epoch * (args.num_epochs + 1)):

                # Track values every epoch
                if step % batches_per_epoch == 0:

                    # Run optimization and calculate batch loss and accuracy
                    _, loss, acc = sess.run([train_op, loss_op, accuracy])

                    # Run through an epoch of validation data
                    y_true = []
                    y_pred = []
                    valid_acc  = 0
                    for v in range(batches_per_epoch):
                        valid, logits_valid, y_v = sess.run([accuracy_valid, tf.argmax(logits_validation, 1), Y_V])
                        y_true += map(lambda x: str(x), y_v.tolist())
                        y_pred += map(lambda x: str(x), logits_valid.tolist())
                        valid_acc += valid

                    # Average validation batch accuracy
                    valid_acc = valid_acc / batches_per_epoch

                    # Validation data confusion matrix summary
                    img_d_summary = plot_confusion_matrix(
                        y_true,
                        y_pred,
                        map(lambda x: str(x), range(num_classes)),
                        tensor_name='ConfusionMatrix')

                    writer.add_summary(img_d_summary, step / batches_per_epoch)
                    summary = sess.run(merged)
                    writer.add_summary(summary, step / batches_per_epoch)

                    print("Epoch="                  + str(int(step / batches_per_epoch)) +
                          ", Minibatch Loss= "      + "{:.4f}".format(loss) +
                          ", Validation Accuracy= " + "{:.4f}".format(valid_acc) +
                          ", Training Accuracy= "   + "{:.3f}".format(acc))

                    print("True batch labels= "     + str(y_v) + "\n" +
                          "Assigned labels=   "     + str(logits_valid))

                else:
                    # Only run the optimization op (backprop)
                    sess.run(train_op)

            print("Optimization Finished!")

            # Free resources from session and stop queue threads
            coord.request_stop()
            coord.join(threads)
            sess.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
