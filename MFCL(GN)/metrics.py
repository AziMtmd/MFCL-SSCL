# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Training utilities."""

from absl import logging
import pandas as pd

import tensorflow.compat.v2 as tf

def update_pretrain_metrics_train(contrast_loss, contrast_acc, contrast_entropy,
                                  loss, logits_con, labels_con):
  """Updated pretraining metrics."""
  contrast_loss.update_state(loss)
  contrast_acc_val = tf.equal(tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
  contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
  contrast_acc.update_state(contrast_acc_val)

  prob_con = tf.nn.softmax(logits_con)
  entropy_con = -tf.reduce_mean(tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))
  contrast_entropy.update_state(entropy_con)


def update_finetune_metrics_train(supervised_loss_metric, supervised_acc_metric,
                                  loss, labels, logits):
  supervised_loss_metric.update_state(loss)
  label_acc = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, axis=1))
  label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
  supervised_acc_metric.update_state(label_acc)


def update_pretrain_metrics_eval(label_top_1_accuracy_metrics, outputs, labels):  
  # contrast_acc_val = tf.equal(tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
  # contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
  # contrastive_top_1_accuracy_metric.update_state(contrast_acc_val)

  label_top_1_accuracy_metrics.update_state(tf.argmax(labels, 1), tf.argmax(outputs, axis=1))
  # label_top_5_accuracy_metrics.update_state(labels, outputs)


def update_finetune_metrics_eval2(contrast_acc_metric_test, logits_con2, labels_con2):  
  print('logits_con2', len(logits_con2))
  print('labels_con2', len(labels_con2))
  contrast_acc_val = tf.equal(tf.argmax(labels_con2, 1), tf.argmax(logits_con2, axis=1))
  contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
  contrast_acc_metric_test.update_state(contrast_acc_val)


def update_finetune_metrics_eval(label_top_1_accuracy_metrics, outputs, labels):
  label_top_1_accuracy_metrics.update_state(tf.argmax(labels, 1), tf.argmax(outputs, axis=1))
  # label_top_5_accuracy_metrics.update_state(labels, outputs)


def _float_metric_value(metric):
  """Gets the value of a float-value keras metric."""
  return metric.result().numpy().astype(float)


def log_and_write_metrics_to_summary(all_metrics, global_step, dam):
  dl=0
  for metric in all_metrics:
    metric_value = _float_metric_value(metric)
    logging.info('Step: [%d] %s = %f', global_step, metric.name, metric_value)
    dam[dl].append(metric_value)
    dl=dl+1
    tf.summary.scalar(metric.name, metric_value, step=global_step)
    
  report=zip(dam[0], dam[1], dam[2], dam[3], dam[4], dam[5], dam[6]) 
  filename = 'azi.xlsx'
  pd.DataFrame(report).to_excel(filename, header=False, index=False)

def log_and_write_metrics_to_summary2(all_metrics2, global_step, dam2):
  dl=0
  for metric in all_metrics2:
    metric_value = _float_metric_value(metric)
    logging.info('Step: [%d] %s = %f', global_step, metric.name, metric_value)
    dam2[dl].append(metric_value)
    dl=dl+1
    tf.summary.scalar(metric.name, metric_value, step=global_step)
    
  report2=zip(dam2[0], dam2[1], dam2[3]) 
  filename2 = 'azi2.xlsx'
  pd.DataFrame(report2).to_excel(filename2, header=False, index=False)


