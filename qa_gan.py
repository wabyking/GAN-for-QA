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
import Generator
import random
#print tf.__version__

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
tf.flags.DEFINE_integer("pools_size", 5000, "The sampled set of a positive ample, which is bigger than 500")
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
precision = '../../insuranceQA/test1.gan'+timeStamp
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
     
    if batch_scores[0] == max(batch_scores):
      scoreList.append(1)
    else:
      scoreList.append(0)
  return sum(scoreList) *1.0 /len(scoreList)


def generateNegSamples(sess,model,size,neg=True):
  x1_set,x2_set,s3_set=[],[],[]
  for epoch in range(size):
    if neg:
      # i=random.randint(0, len(raw) - 1) 
      # start=random.randint(0, len(alist) - FLAGS.pools_size)
      # end =start+FLAGS.pools_size
      # # pools=
      # pools=alist[start:end]
      pools=np.random.choice(alist,size=[FLAGS.pools_size])
      
      # x1,x2,x3=insurance_qa_data_helpers.load_data_pair1(vocab,pools,raw[i])   # single question
      x1,x2,x3=insurance_qa_data_helpers.load_data_pair(vocab,pools,raw,FLAGS.pools_size)  # diversity question

      predicteds=[]
      for j in range(FLAGS.pools_size/FLAGS.batch_size):
        index_start=FLAGS.batch_size*j
        index_end=FLAGS.batch_size*(j+1)
      
        feed_dict = {
          model.input_x_1: x1[index_start:index_end],
          model.input_x_2: x2[index_start:index_end],
          model.input_x_3: x3[index_start:index_end],
          model.dropout_keep_prob: 1.0
        }
        predicted=sess.run(model.pred_score,feed_dict);
      
        predicteds.extend(predicted)
      exp_rating = np.exp(predicteds)
      prob = exp_rating / np.sum(exp_rating)
      
      print ".",
      neg_index = np.random.choice(np.arange(len(prob)), size=[500], p=prob)

      samples_x1,samples_x2,samples_x3=x1[neg_index],x2[neg_index],x3[neg_index]
    else:
       # samples_x1,samples_x2,samples_x3= insurance_qa_data_helpers.load_data_6(vocab, alist, raw, FLAGS.batch_size)
      # i=random.randint(0, len(raw) - 1) 
      # start=random.randint(0, len(alist) - FLAGS.pools_size)
      # end =start+FLAGS.pools_size
      # pools=alist[start:end]
      # pools=np.random.choice(alist,size=[FLAGS.pools_size])
      
      # x1,x2,x3=insurance_qa_data_helpers.load_data_pair(vocab,alist,raw,FLAGS.pools_size)
      # prob=np.ones(FLAGS.pools_size) * 1.0 / FLAGS.pools_size
      # neg_index = np.random.choice(np.arange(len(prob)), size=[500], p=prob)

      # samples_x1,samples_x2,samples_x3=x1[neg_index],x2[neg_index],x3[neg_index]
      samples_x1,samples_x2,samples_x3=insurance_qa_data_helpers.load_data_6(vocab, alist, raw, FLAGS.batch_size)
    x1_set.append(samples_x1)
    x2_set.append(samples_x2)
    s3_set.append(samples_x3)
  print "have sampled %d samples"% epoch
  return x1_set,x2_set,s3_set
def evaluation(sess,model,log):
  if isinstance(model,  Discriminator.Discriminator):
    model_type="Dis"
  else:
    model_type="Gen"
  current_step = tf.train.global_step(sess, model.global_step)
  if current_step % FLAGS.evaluate_every == 0:
      
      if current_step % (FLAGS.evaluate_every *20) !=0:
        precision_current=dev_step(sess,model,100)
        line=" %d epoch: %s precision %f"%(current_step,model_type,precision_current)
      else:
        precision_current=dev_step(sess,model,1800)
        line="__________________\n%d epoch: %s precision %f"%(current_step,model_type,precision_current)
      log.write(line+"\n")
      print(line)

def main():
  with tf.Graph().as_default():
    with tf.device("/gpu:0"):
      session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
      sess = tf.Session(config=session_conf)
      with sess.as_default(),open(precision,"w") as log :

          discriminator = Discriminator.Discriminator(
              sequence_length=x_train_1.shape[1],
              batch_size=FLAGS.batch_size,
              vocab_size=len(vocab),
              embedding_size=FLAGS.embedding_dim,
              filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
              num_filters=FLAGS.num_filters,
              l2_reg_lambda=FLAGS.l2_reg_lambda)
          generator = Generator.Generator(
              sequence_length=x_train_1.shape[1],
              batch_size=FLAGS.batch_size,
              vocab_size=len(vocab),
              embedding_size=FLAGS.embedding_dim,
              filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
              num_filters=FLAGS.num_filters,
              l2_reg_lambda=FLAGS.l2_reg_lambda)
         
          sess.run(tf.global_variables_initializer())

          for epoch in range(FLAGS.num_epochs):
              if epoch > -1:
                  # G generate negative for D, then train D
                  # generate_for_d(sess, generator, DIS_TRAIN_FILE)
                  # train_size = ut.file_len(DIS_TRAIN_FILE)
                for d_epoch in range(1):
                  print "start sample"
                  x1,x2,x3=generateNegSamples(sess,generator,100)
                  
                  for x_batchs1, x_batchs2, x_batchs3 in zip(x1,x2,x3):  # try:

                    for j in range(500/FLAGS.batch_size):
                      index_start=FLAGS.batch_size*j
                      index_end=FLAGS.batch_size*(j+1)
                      x_batch_1, x_batch_2, x_batch_3=x_batchs1[index_start:index_end], x_batchs2[index_start:index_end], x_batchs3[index_start:index_end]
                      train_step(sess,discriminator,x_batch_1, x_batch_2, x_batch_3)
                  evaluation(sess,discriminator,log)
              # Train G
              
              for g_epoch in range(1):  # 50
                  print "g_epoach: %d" % g_epoch
                  x1,x2,x3=generateNegSamples(sess,generator,100)
                  for i in range(1):
                    for x_batchs1, x_batchs2, x_batchs3 in zip(x1,x2,x3):  # try:

                      for j in range(500/FLAGS.batch_size):
                        index_start=FLAGS.batch_size*j
                        index_end=FLAGS.batch_size*(j+1)
                        x_batch_1, x_batch_2, x_batch_3=x_batchs1[index_start:index_end], x_batchs2[index_start:index_end], x_batchs3[index_start:index_end]


                        feed_dict = {
                          discriminator.input_x_1: x_batch_1,
                          discriminator.input_x_2: x_batch_2,
                          discriminator.input_x_3: x_batch_3,
                          discriminator.dropout_keep_prob: 1.0
                        }
                        neg_reward = sess.run(discriminator.neg_reward,feed_dict)
                       
                        feed_dict = {
                          generator.input_x_1: x_batch_1,
                          generator.input_x_2: x_batch_2,
                          generator.input_x_3: x_batch_3,
                          generator.reward: neg_reward,
                          generator.dropout_keep_prob: 1.0
                        }
                        _ ,step= sess.run([generator.gan_updates, generator.global_step],feed_dict)
                        print "have trained %d g_epoch" % step
                        
                        evaluation(sess,generator,log)






                 


if __name__ == '__main__':
  main()