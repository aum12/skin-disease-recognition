"""Image classification given a trained model as input.

This program imports a trained GraphDef protocol buffer and runs
inference on an input JPEG image (feed forward through neural network).

It outputs human readable strings of the top x predictions along 
with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Inspired by TensorFlow tutorial:
https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
# pylint: enable=unused-import,g-bad-import-order

from tensorflow.python.platform import gfile

#Input file flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '/tf_files/',
                          """Path to retrained model graph, """)
tf.app.flags.DEFINE_string('labels_dir', '/tf_files/',
                            """Path to class labels txt file.""")
tf.app.flags.DEFINE_string('model_name', 'retrained_graph.pb',
                          """File name of retrained model graph, """)
tf.app.flags.DEFINE_string('labels_name', 'retrained_labels.txt',
                            """Path to class labels txt file.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver.
  
  Args:
    Nothing.

  Returns:
    Nothing.
  """
  with gfile.FastGFile(os.path.join(
      FLAGS.model_dir, FLAGS.model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def get_label_list():
  """Creates a list containing class labels.

  Args:
    Nothing.

  Returns:
    label_list: A list containing class names specified in class txt file.
  """
  f_name = os.path.join(FLAGS.labels_dir, FLAGS.labels_name)
  if os.path.exists(f_name):
    with open(f_name, 'rb') as f:
      try:
        label_list = [line.rstrip('\n') for line in f]
      except:
        print("Could not read file:" + f_name)
        sys.exit()
  return label_list

def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing.
  """
  if not gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    print("Running inference on image: %s" % os.path.basename(FLAGS.image_file))
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

  # Get indicies of top x predictions
  top_x = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
  # Get list of human readable class labels
  class_labels = get_label_list()
  
  #print results
  for top_i in top_x: 
    print(class_labels[top_i] + ': %.2f%%' % (predictions[top_i] * 100))
  print('\n')


def main(_):
  image = (FLAGS.image_file if FLAGS.image_file else
           sys.exit("Please provide input image file using image_file flag."))
  run_inference_on_image(image)


if __name__ == '__main__':
  tf.app.run()
