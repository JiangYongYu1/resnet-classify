import tensorflow as tf
import argparse
import os
import resnet_model
import read_image
slim = tf.contrib.slim
FLAGS = None
train_image_size = 224
CHANAL = 3
classnum = 8


def main(_):
    lr = tf.Variable(0.0001, tf.float32)

    ckpt_path = os.path.join(FLAGS.model_dir, "coat_model.ckpt")

    with tf.name_scope('input'):
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, train_image_size, train_image_size, CHANAL])
        y_input = tf.placeholder(dtype=tf.float32, shape=[None, classnum])

    with tf.name_scope('resnet_model'):
        net, end_points = resnet_model.resnet_v2_152(x_input, classnum)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=end_points['predictions']))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_input, 1), tf.arg_max(end_points['predictions'], 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('input_train_image'):
        train_file_path = os.path.join(FLAGS.data_dir, 'train_00000.tfrecords')
        train_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file_path))
        train_images, train_labels = read_image.read_tfrecord(train_image_filename_queue, classnum, 100)

    with tf.name_scope('input_val_image'):
        test_file_path = os.path.join(FLAGS.data_dir, 'validation_00000.tfrecords')
        test_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(test_file_path))
        test_images, test_labels = read_image.read_tfrecord(test_image_filename_queue, classnum, 100)


    merged = tf.summary.merge_all()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.join(FLAGS.tb_dir), sess.graph)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for epoch in range(1000):
            #sess.run(tf.assign(lr, 0.0001 * (0.99 ** epoch)))
            for step in range(epoch * 100, (epoch + 1) * 100):
                batch_train_images, batch_train_labels = sess.run([train_images, train_labels])
                _, summary = sess.run([train_step, merged], feed_dict={x_input: batch_train_images, y_input: batch_train_labels})
                writer.add_summary(summary, step)
                if ((step+1)%25 == 0):
                    batch_test_images, batch_test_labels = sess.run([test_images, test_labels])
                    test_accu, test_loss = sess.run([accuracy, loss], feed_dict={x_input: batch_test_images, y_input: batch_test_labels})
                    print("After step: %d, the accuracy mean is %.5f, the test loss is %2.4f" % (step + 1, test_accu, test_loss))
        saver.save(sess, ckpt_path)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='', help='input data path')

    parser.add_argument('--model_dir', type=str, default='', help='output model path')

    parser.add_argument('--tb_dir', type=str, default='', help='tensorboard path')

    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)