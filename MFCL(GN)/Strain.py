from absl import flags
from absl import logging
import tensorflow as tf
import model as model_lib
import objective as obj_lib
import metrics
import numpy
import pickle
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D 
from tensorflow.keras.layers import GlobalAveragePooling2D, Add, Input, Flatten, Dropout, Conv2DTranspose, LeakyReLU
import os

import data as data_lib
import math
from tensorflow.keras import Model
# from random import randrange
# from tensorflow.keras.losses import binary_crossentropy
# from tensorflow.keras.optimizers import Adam
# import random

import resnet
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.datasets import cifar10
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

FLAGS = flags.FLAGS

(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

x_t=np.concatenate((X_test[0:5000], X_test[5000:10000]), axis=0)
y_t=np.concatenate((y_test[0:5000], y_test[5000:10000]), axis=0)
lolt=len(y_t); vayt=np.reshape(y_t, (lolt,))
T_X_test, T_y_test = shuffle(np.array(x_t), np.array(vayt))
T_Mydataset1 = tf.data.Dataset.from_tensor_slices(T_X_test)
T_Mydataset2 = tf.data.Dataset.from_tensor_slices(T_y_test)
datasetT = tf.data.Dataset.zip((T_Mydataset1, T_Mydataset2))


def try_restore_from_checkpoint(model, global_step, optimizer, Mdirectoryy):
  """Restores the latest ckpt if it exists, otherwise check FLAGS.checkpoint."""
  checkpoint = tf.train.Checkpoint(model=model, global_step=global_step, optimizer=optimizer)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=Mdirectoryy, max_to_keep=FLAGS.keep_checkpoint_max)
  latest_ckpt = checkpoint_manager.latest_checkpoint

  if latest_ckpt:
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
  elif FLAGS.checkpoint:
    checkpoint_manager2 = tf.train.CheckpointManager(tf.train.Checkpoint(model=model),directory=Mdirectoryy,max_to_keep=FLAGS.keep_checkpoint_max)
    checkpoint_manager2.checkpoint.restore(FLAGS.checkpoint).expect_partial()
    if FLAGS.zero_init_logits_layer:
      model = checkpoint_manager2.checkpoint.model
      output_layer_parameters = model.supervised_head.trainable_weights
      for x in output_layer_parameters:
        x.assign(tf.zeros_like(x))
  return checkpoint_manager


class ClassikhodamClass():
  def __init__(self, m):
    self.dam=[[], [], [], [], [], [], []]
    self.dam2=[[], [], [], [], []]
    self.name = m  # Worker ID
    self.strategy = tf.distribute.MirroredStrategy()
    self.topology = None

  # def classiKhodam(self, M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, iti):
  def classiKhodam(self, M, iti):
    if iti==FLAGS.tekrar-1:
      num_train_examples = 50000
      num_classes = 10
      Mdirectoryy='/tmp/simclr_test' 
      self.summary_writer = tf.summary.create_file_writer(Mdirectoryy) 
      with self.strategy.scope():
        self.dsFalse = data_lib.build_distributed_dataset(datasetT, FLAGS.testBatch, False, self.strategy, self.topology)  
        # self.ds = data_lib.build_distributed_dataset(datasetT, FLAGS.testBatch, True, self.strategy, self.topology)

    if iti==FLAGS.tekrar-1:
      with self.strategy.scope():
        self.model = model_lib.Model3(num_classes)          
        # Build LR schedule and optimizer.
        self.learning_rate = model_lib.WarmUpAndCosineDecay(FLAGS.learning_rate, num_train_examples)
        self.optimizer = model_lib.build_optimizer(self.learning_rate) 
        self.epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

        # Build metrics.
        self.all_metrics = []  # For summaries.
        self.weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
        self.total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
        self.all_metrics.extend([self.weight_decay_metric, self.total_loss_metric])
        if FLAGS.train_mode == 'pretrain':
          self.contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
          self.contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
          self.contrast_entropy_metric = tf.keras.metrics.Mean('train/contrast_entropy')
          self.all_metrics.extend([self.contrast_loss_metric, self.contrast_acc_metric, self.contrast_entropy_metric])   
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
          self.supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
          self.supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')
          self.all_metrics.extend([self.supervised_loss_metric, self.supervised_acc_metric])    
          self.train_steps = model_lib.get_train_steps(num_train_examples)
      
      # self.eval_steps = FLAGS.eval_steps or int(math.ceil(num_eval_examples / FLAGS.eval_batch_size))
 
    Mdirectoryy='/tmp/simclr_test'
    # Restore checkpoint if available.
    self.checkpoint_manager = try_restore_from_checkpoint(self.model, self.optimizer.iterations, self.optimizer, Mdirectoryy)
    self.checkpoint_steps = (FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * self.epoch_steps))
    self.steps_per_loop = self.checkpoint_steps
    print('self.steps_per_loop', self.steps_per_loop)  

    for i in range(FLAGS.NumofWorkers): 
      M[i].terroo=False
    def single_step(featuresb, blabels):
      with tf.GradientTape() as tape:
        self.projection_head_outputs, self.supervised_head_outputs, hiddens = self.model(featuresb, training=True)     
        self.loss = None
        if self.projection_head_outputs is not None:
          self.outputs = self.projection_head_outputs
          self.con_loss, self.logits_con, self.labels_con = obj_lib.add_contrastive_loss(
              self.outputs,
              hidden_norm=FLAGS.hidden_norm,
              temperature=FLAGS.temperature,
              strategy=self.strategy)
          if self.loss is None:
            self.loss = self.con_loss
          else:
            self.loss += self.con_loss
            
          metrics.update_pretrain_metrics_train(self.contrast_loss_metric,
                                                self.contrast_acc_metric,
                                                self.contrast_entropy_metric,
                                                self.con_loss, self.logits_con, self.labels_con)

        if self.supervised_head_outputs is not None:
          self.outputs = self.supervised_head_outputs
          # l = labels['labels']
          l = blabels
          if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            l = tf.concat([l, l], 0)
          self.sup_loss = obj_lib.add_supervised_loss(labels=l, logits=self.outputs)

          if self.loss is None:
            self.loss = self.sup_loss
          else:
            self.loss += self.sup_loss
          metrics.update_finetune_metrics_train(self.supervised_loss_metric,
                                                self.supervised_acc_metric, 
                                                self.sup_loss,
                                                l, self.outputs)
        self.weight_decay = model_lib.add_weight_decay(self.model, adjust_per_optimizer=True)
        self.weight_decay_metric.update_state(self.weight_decay)
        self.loss += self.weight_decay
        self.total_loss_metric.update_state(self.loss)
        self.loss = self.loss / self.strategy.num_replicas_in_sync
        # logging.info('Trainable variables:')
        # for var in self.model.trainable_variables:
        #   logging.info(var.name)
        # for yad in range(5):
        #   self.model.trainable_variables[yad].assign(self.autoencoder.trainable_variables[yad])         
        self.grads = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(self.grads, self.model.trainable_variables))
    
    with self.strategy.scope():
      self.Aziiterator1s = iter(M[0].server_ds) 
      self.Aziiterator2s = iter(M[1].server_ds) 
      self.Aziiterator3s = iter(M[2].server_ds) 
      self.Aziiterator4s = iter(M[3].server_ds) 
      self.Aziiterator5s = iter(M[4].server_ds)  
      self.Aziiterator6s = iter(M[5].server_ds) 
      self.Aziiterator7s = iter(M[6].server_ds) 
      self.Aziiterator8s = iter(M[7].server_ds) 
      self.Aziiterator9s = iter(M[8].server_ds) 
      self.Aziiterator10s = iter(M[9].server_ds) 
    
    TS=50;
    kam=int(self.train_steps/TS)    
    for fer in range(kam):
      lep=[];rep=[]
      for der in range(TS):
        images1, labels1 = next(self.Aziiterator1s)
        images2, labels2 = next(self.Aziiterator2s)
        images3, labels3 = next(self.Aziiterator3s)
        images4, labels4 = next(self.Aziiterator4s)
        images5, labels5 = next(self.Aziiterator5s)
        images6, labels6 = next(self.Aziiterator6s)
        images7, labels7 = next(self.Aziiterator7s)
        images8, labels8 = next(self.Aziiterator8s)
        images9, labels9 = next(self.Aziiterator9s)
        images10, labels10 = next(self.Aziiterator10s)

        features1, labels1 = images1, {'labels1': labels1}
        num_transforms = features1.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list1 = tf.split(features1, num_or_size_splits=num_transforms, axis=-1)
        features1 = tf.concat(features_list1, 0)  # (num_transforms * bsz, h, w, c)
        javab1 = M[0].encoder.predict(features1)

        features2, labels2 = images2, {'labels2': labels2}
        num_transforms = features2.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list2 = tf.split(features2, num_or_size_splits=num_transforms, axis=-1)
        features2 = tf.concat(features_list2, 0)  # (num_transforms * bsz, h, w, c)
        javab2 = M[1].encoder.predict(features2)

        features3, labels3 = images3, {'labels3': labels3}
        num_transforms = features3.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list3 = tf.split(features3, num_or_size_splits=num_transforms, axis=-1)
        features3 = tf.concat(features_list3, 0)  # (num_transforms * bsz, h, w, c)
        javab3 = M[2].encoder.predict(features3)

        features4, labels4 = images4, {'labels4': labels4}
        num_transforms = features4.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list4 = tf.split(features4, num_or_size_splits=num_transforms, axis=-1)
        features4 = tf.concat(features_list4, 0)  # (num_transforms * bsz, h, w, c)
        javab4 = M[3].encoder.predict(features4)

        features5, labels5 = images5, {'labels5': labels5}
        num_transforms = features5.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list5 = tf.split(features5, num_or_size_splits=num_transforms, axis=-1)
        features5 = tf.concat(features_list5, 0)  # (num_transforms * bsz, h, w, c)
        javab5 = M[4].encoder.predict(features5)

        features6, labels6 = images6, {'labels6': labels6}
        num_transforms = features6.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list6 = tf.split(features6, num_or_size_splits=num_transforms, axis=-1)
        features6 = tf.concat(features_list6, 0)  # (num_transforms * bsz, h, w, c)
        javab6 = M[5].encoder.predict(features6)

        features7, labels7 = images7, {'labels7': labels7}
        num_transforms = features7.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list7 = tf.split(features7, num_or_size_splits=num_transforms, axis=-1)
        features7 = tf.concat(features_list7, 0)  # (num_transforms * bsz, h, w, c)
        javab7 = M[6].encoder.predict(features7)

        features8, labels8 = images8, {'labels8': labels8}
        num_transforms = features8.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list8 = tf.split(features8, num_or_size_splits=num_transforms, axis=-1)
        features8 = tf.concat(features_list8, 0)  # (num_transforms * bsz, h, w, c)
        javab8 = M[7].encoder.predict(features8)

        features9, labels9 = images9, {'labels9': labels9}
        num_transforms = features9.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list9 = tf.split(features9, num_or_size_splits=num_transforms, axis=-1)
        features9 = tf.concat(features_list9, 0)  # (num_transforms * bsz, h, w, c)
        javab9 = M[8].encoder.predict(features9)

        features10, labels10 = images10, {'labels10': labels10}
        num_transforms = features10.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        features_list10 = tf.split(features10, num_or_size_splits=num_transforms, axis=-1)
        features10 = tf.concat(features_list10, 0)  # (num_transforms * bsz, h, w, c)
        javab10 = M[9].encoder.predict(features10)
        
        javabaB1=np.concatenate((javab1[0:FLAGS.serBatch], javab2[0:FLAGS.serBatch], javab3[0:FLAGS.serBatch], 
                              javab4[0:FLAGS.serBatch], javab5[0:FLAGS.serBatch], javab6[0:FLAGS.serBatch], 
                              javab7[0:FLAGS.serBatch], javab8[0:FLAGS.serBatch], javab9[0:FLAGS.serBatch], 
                              javab10[0:FLAGS.serBatch]), axis=0) 

        javabaB2=np.concatenate((javab1[FLAGS.serBatch:(2*FLAGS.serBatch)], javab2[FLAGS.serBatch:(2*FLAGS.serBatch)], 
                                javab3[FLAGS.serBatch:(2*FLAGS.serBatch)], 
                              javab4[FLAGS.serBatch:(2*FLAGS.serBatch)], javab5[FLAGS.serBatch:(2*FLAGS.serBatch)], 
                              javab6[FLAGS.serBatch:(2*FLAGS.serBatch)], javab7[FLAGS.serBatch:(2*FLAGS.serBatch)], 
                              javab8[FLAGS.serBatch:(2*FLAGS.serBatch)], javab9[FLAGS.serBatch:(2*FLAGS.serBatch)], 
                              javab10[FLAGS.serBatch:(2*FLAGS.serBatch)]), axis=0)


        el1=labels1['labels1']; el2=labels2['labels2']
        el3=labels3['labels3']; el4=labels4['labels4']; el5=labels5['labels5']
        el6=labels6['labels6']; el7=labels7['labels7']; el8=labels8['labels8'] 
        el9=labels9['labels9']; el10=labels10['labels10']
        labelsha=tf.concat([el1, el2, el3, el4, el5, el6, el7, el8, el9, el10], 0) 

        bagh1, bagh2, bagh3= shuffle(javabaB1, javabaB2, labelsha)

        javaba=np.concatenate((bagh1, bagh2), axis=0)
        javab=tf.convert_to_tensor(javaba, dtype=tf.float32)
        rep.append(javab)
        lep.append(bagh3)
        print(der)

      with self.strategy.scope():
        @tf.function           
        def train_multiple_steps(javabb, labelsb):
          for _ in tf.range(1):   
            with tf.name_scope(''):    
              self.strategy.run(single_step, (javabb, labelsb))

        global_step = self.optimizer.iterations
        cur_step = global_step.numpy()        
        print('*********cur_step', cur_step)
        print('......train_steps', self.train_steps)    

        # TS=self.train_steps 
        print('len', len(rep))
        i=0      
        while cur_step < (fer+1)*TS:  
          with self.summary_writer.as_default():          
            train_multiple_steps(rep[i], lep[i])
            i=i+1
            cur_step = global_step.numpy()
            self.checkpoint_manager.save(cur_step)
            logging.info('Completed: %d / %d steps', cur_step, self.train_steps)
            metrics.log_and_write_metrics_to_summary(self.all_metrics, cur_step, self.dam)
            tf.summary.scalar('learning_rate', self.learning_rate(tf.cast(global_step, dtype=tf.float32)), global_step)
            self.summary_writer.flush()
          for metric in self.all_metrics:
            metric.reset_states()

    logging.info('Training complete...')

    # if iti!=0 and iti%10==0:

    """Perform evaluation."""
    if FLAGS.train_mode == 'pretrain' and not FLAGS.lineareval_while_pretraining:
      logging.info('Skipping eval during pretraining without linear eval.')
      return
    # Build input pipeline.
    Mdirectoryy='/tmp/simclr_test'
    self.summary_writer2 = tf.summary.create_file_writer(Mdirectoryy)
    ckpt=self.checkpoint_manager.latest_checkpoint

    # Build metrics.
    with self.strategy.scope():
      self.contrastive_top_1_accuracy_metric = tf.keras.metrics.Mean('train/contrast_acc')
      self.label_top_1_accuracy = tf.keras.metrics.Accuracy('eval/label_top_1_accuracy')
      self.label_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(5, 'eval/label_top_5_accuracy')
      self.regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
      self.all_metrics2 = [self.regularization_loss, self.label_top_1_accuracy, self.label_top_5_accuracy, self.contrastive_top_1_accuracy_metric]

      # Restore checkpoint.
      logging.info('Restoring from %s', ckpt)
      checkpoint = tf.train.Checkpoint(model=self.model, global_step=tf.Variable(0, dtype=tf.int64))
      checkpoint.restore(ckpt).expect_partial()
      global_step = checkpoint.global_step
      logging.info('Performing eval at step %d', global_step.numpy())
    
    def single_step_eval(ejavab, elab):
      self.alkole2=ejavab
      self.malkole2=elab['labels']
      self.projection_head_outputs, self.supervised_head_outputs, hiddens = self.model(self.alkole2, training=False)              
      outputs = self.supervised_head_outputs         
      metrics.update_finetune_metrics_eval(self.label_top_1_accuracy, outputs, self.malkole2)
      cur_step = global_step.numpy()
      with self.summary_writer2.as_default():
        metrics.log_and_write_metrics_to_summary2(self.all_metrics2, cur_step, self.dam2)
        self.summary_writer.flush()
    

    with self.strategy.scope():
      # @tf.function
      def run_single_step(ejavab, elabels):
        self.strategy.run(single_step_eval, (ejavab, elabels))

    with self.strategy.scope():
      self.AziiteratorT = iter(self.dsFalse)
    

    eimages=0; elabels=0; efeatures=0; num_transforms=0; efeatures_list=0
    eimages, elabels = next(self.AziiteratorT)
    efeatures, elabels = eimages, {'labels': elabels}
    num_transforms = efeatures.shape[3] // 3
    num_transforms = tf.repeat(3, num_transforms)
    efeatures_list = tf.split(efeatures, num_or_size_splits=num_transforms, axis=-1)
    efeatures = tf.concat(efeatures_list, 0)  # (num_transforms * bsz, h, w, c)
      
    for jk in range(FLAGS.NumofWorkers):
      ejavab=0
      ejavab = M[jk].encoder.predict(efeatures)

      with self.strategy.scope():
        for i in range(1):
          run_single_step(ejavab, elabels)
    return




   
