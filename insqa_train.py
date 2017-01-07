#coding=utf-8
#! /usr/bin/env python3.4

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import insurance_qa_data_helpers
from insqa_cnn import InsQACNN
import operator
import Discriminator
#print tf.__version__
import time

now = int(time.time()) 
    
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")

vocab = insurance_qa_data_helpers.build_vocab()
alist = insurance_qa_data_helpers.read_alist()
raw = insurance_qa_data_helpers.read_raw()
x_train_1, x_train_2, x_train_3 = insurance_qa_data_helpers.load_data_6(vocab, alist, raw, FLAGS.batch_size)
testList, vectors = insurance_qa_data_helpers.load_test_and_vectors()
vectors = ''
print('x_train_1', np.shape(x_train_1))
print("Load done...")

val_file = '../../insuranceQA/test1'

precision = '../../insuranceQA/test1.acc'+timeStamp
#x_val, y_val = data_deepqa.load_data_val()

# Training
# ==================================================
def train_step(sess,cnn,x_batch_1, x_batch_2, x_batch_3):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x_1: x_batch_1,
              cnn.input_x_2: x_batch_2,
              cnn.input_x_3: x_batch_3,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
          
            
           
            _, step,  loss, accuracy = sess.run(
                [cnn.train_op, cnn.global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # train_summary_writer.add_summary(summaries, step)

def dev_step(sess,cnn,dev_size):
  scoreList = []
  
  for i in range(dev_size):
    batch_scores=[]
    for j in range(500/FLAGS.batch_size):
      x_test_1, x_test_2, x_test_3 = insurance_qa_data_helpers.load_data_val_6(testList, vocab, i*500+j*FLAGS.batch_size, FLAGS.batch_size)
      feed_dict = {
        cnn.input_x_1: x_test_1,
        cnn.input_x_2: x_test_2,
        cnn.input_x_3: x_test_3,
        cnn.dropout_keep_prob: 1.0
      }
      predicted =sess.run([cnn.cos_12], feed_dict)
     

      batch_scores.extend(predicted[0])
    index=  batch_scores.index(max(batch_scores))
    if int(testList[i*500+index].split()[0])==1:
      scoreList.append(1)
    else:
      scoreList.append(0)
  return sum(scoreList) *1.0 /len(scoreList)
def main():
  with tf.Graph().as_default():
    with tf.device("/gpu:1"):
      session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
      sess = tf.Session(config=session_conf)
      with sess.as_default(),open(precision,"w") as log:



          discriminator = Discriminator.Discriminator(
              sequence_length=x_train_1.shape[1],
              batch_size=FLAGS.batch_size,
              vocab_size=len(vocab),
              embedding_size=FLAGS.embedding_dim,
              filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
              num_filters=FLAGS.num_filters,
              l2_reg_lambda=FLAGS.l2_reg_lambda)

         
          sess.run(tf.global_variables_initializer())
          # Generate batches
          # Training loop. For each batch...
          for i in range(FLAGS.num_epochs):
              # try:
              x_batch_1, x_batch_2, x_batch_3 = insurance_qa_data_helpers.load_data_6(vocab, alist, raw, FLAGS.batch_size)
              train_step(sess,discriminator,x_batch_1, x_batch_2, x_batch_3)
              current_step = tf.train.global_step(sess, discriminator.global_step)
              if current_step % FLAGS.evaluate_every == 0:                
                if current_step % (FLAGS.evaluate_every *20) !=0:
                  precision_current=dev_step(sess,discriminator,100)
                  line=" %d epoch: precision %f"%(current_step,precision_current)
                else:
                  precision_current=dev_step(sess,discriminator,1800)
                  line="__________________\n%d epoch: precision %f"%(current_step,precision_current)
                log.write(line+"\n")
                print(line)
                  # if current_step % FLAGS.checkpoint_every == 0:
                  #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                  #     print("Saved model checkpoint to {}\n".format(path))
              # except Exception as e:
              #     print(e)
if __name__ == '__main__':
  main()