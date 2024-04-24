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
"""The main training pipeline."""

import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from absl import app
from absl import flags
from absl import logging
import data as data_lib
import metrics
import model as model_lib
import objective as obj_lib
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import keras
import sys
import tf_slim as slim
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

FLAGS = flags.FLAGS

flags.DEFINE_integer('train_batch_size', 128, 'Batch size for training.')
flags.DEFINE_integer('train_epochs', 100, 'Number of epochs to train for.')
flags.DEFINE_float('warmup_epochs', 10, 'Number of epochs of warmup.')
flags.DEFINE_string('dataset', 'cifar10', 'Name of a dataset.')
flags.DEFINE_integer('proj_out_dim', 128,'Number of head projection dimension.')
flags.DEFINE_integer('num_proj_layers', 3,'Number of non-linear head layers.')
flags.DEFINE_integer('resnet_depth', 18,'Depth of ResNet.')
flags.DEFINE_integer('image_size', 32, 'Input image size.')

flags.DEFINE_float('learning_rate', 1.5, 'Initial learning rate per batch size of 256.')
flags.DEFINE_enum('learning_rate_scaling', 'linear', ['linear', 'sqrt'],'How to scale the learning rate as a function of batch size.')
flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')
flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')
flags.DEFINE_string('train_split', 'train', 'Split for training.')
flags.DEFINE_integer('train_steps', 0, 'Number of steps to train for. If provided, overrides train_epochs.')
flags.DEFINE_integer('eval_steps', 1, 'Number of steps to eval for. If not provided, evals over entire dataset.')
flags.DEFINE_integer('eval_batch_size', 256, 'Batch size for eval.')
flags.DEFINE_integer('checkpoint_epochs', 1, 'Number of epochs between checkpoints/summaries.')
flags.DEFINE_integer('checkpoint_steps', 0,'Number of steps between checkpoints/summaries. If provided, overrides checkpoint_epochs.')
flags.DEFINE_string('eval_split', 'test', 'Split for evaluation.')
flags.DEFINE_bool('cache_dataset', False, 'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can improve performance.')
flags.DEFINE_enum('mode', 'train_then_eval', ['train', 'eval', 'train_then_eval'],'Whether to perform training or evaluation.')
flags.DEFINE_enum('train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')
flags.DEFINE_bool('lineareval_while_pretraining', True, 'Whether to finetune supervised head while pretraining.')
flags.DEFINE_string('checkpoint', None,'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')
flags.DEFINE_bool('zero_init_logits_layer', False,'If True, zero initialize layers after avg_pool for supervised learning.')
flags.DEFINE_integer('fine_tune_after_block', -1,'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning just the linear head.')
flags.DEFINE_string('master', None,'Address/name of the TensorFlow master to use. By default, use an in-process master.')
flags.DEFINE_string('model_dir', '/azi', 'Model directory for training.')
flags.DEFINE_string('data_dir', '/azi', 'Directory where dataset is stored.')
flags.DEFINE_enum('optimizer', 'lars', ['momentum', 'adam', 'lars'],'Optimizer to use.')
flags.DEFINE_float('momentum', 0.9,'Momentum parameter.')
flags.DEFINE_string('eval_name', None,'Name for eval.')
flags.DEFINE_integer('keep_checkpoint_max', 5,'Maximum number of checkpoints to keep.')
flags.DEFINE_integer('keep_hub_module_max', 1,'Maximum number of Hub modules to keep.')
flags.DEFINE_float('temperature', 0.3,'Temperature parameter for contrastive loss.')
flags.DEFINE_boolean('hidden_norm', True,'Temperature parameter for contrastive loss.')
flags.DEFINE_enum('proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],'How the head projection is done.')
flags.DEFINE_integer('ft_proj_selector', 0,'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')
flags.DEFINE_boolean('global_bn', True,'Whether to aggregate BN statistics across distributed cores.')
flags.DEFINE_integer('width_multiplier', 1, 'Multiplier to change width of network.')
flags.DEFINE_float('sk_ratio', 0.,'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')
flags.DEFINE_float('se_ratio', 0.,'If it is bigger than 0, it will enable SE.')
flags.DEFINE_float('color_jitter_strength', 1.0,'The strength of color jittering.')
flags.DEFINE_boolean('use_blur', True,'Whether or not to use Gaussian blur for augmentation during pretraining.')

def normalize_image(img):
    grads_norm = img[:,:,0]+ img[:,:,1]+ img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm

def plot_maps(i, img1, img2,vmin=0.3,vmax=0.7, mix_val=2):
    f = plt.figure(figsize=(15,45))
    plt.subplot(1,3,1)
    plt.imshow(img1,vmin=vmin, vmax=vmax, cmap="ocean")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap = "ocean")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(img1*mix_val+img2/mix_val, cmap = "ocean" )
    plt.axis("off")    
    plt.savefig('/azi/nik'+str(i))


def get_salient_tensors_dict(include_projection_head):
  """Returns a dictionary of tensors."""
  graph = tf.compat.v1.get_default_graph()
  result = {}
  for i in range(1, 5):
    result['block_group%d' % i] = graph.get_tensor_by_name('resnet/block_group%d/block_group%d:0' % (i, i))
  result['initial_conv'] = graph.get_tensor_by_name('resnet/initial_conv/Identity:0')
  result['initial_max_pool'] = graph.get_tensor_by_name('resnet/initial_max_pool/Identity:0')
  result['final_avg_pool'] = graph.get_tensor_by_name('resnet/final_avg_pool:0')
  result['logits_sup'] = graph.get_tensor_by_name('head_supervised/logits_sup:0')
  if include_projection_head:
    result['proj_head_input'] = graph.get_tensor_by_name('projection_head/proj_head_input:0')
    result['proj_head_output'] = graph.get_tensor_by_name('projection_head/proj_head_output:0')
  return result


def flops(mrd):
  session = tf.compat.v1.Session()
  graph = tf.compat.v1.get_default_graph()
  with graph.as_default():
    with session.as_default():
      run_meta = tf.compat.v1.RunMetadata()
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
  return flops.total_float_ops


def build_saved_model(model, include_projection_head=True):
  """Returns a tf.Module for saving to SavedModel."""
  class SimCLRModel(tf.Module):
    """Saved model for exporting to hub."""
    def __init__(self, model):
      self.model = model
      # This can't be called `trainable_variables` because `tf.Module` has a getter with the same name.
      self.trainable_variables_list = model.trainable_variables
    @tf.function
    def __call__(self, inputs, trainable):
      self.model(inputs, training=trainable)
      return get_salient_tensors_dict(include_projection_head)
  module = SimCLRModel(model)
  input_spec = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)
  module.__call__.get_concrete_function(input_spec, trainable=True)
  module.__call__.get_concrete_function(input_spec, trainable=False)
  return module


def save(model, global_step):
  """Export as SavedModel for finetuning and inference."""
  saved_model = build_saved_model(model)
  export_dir = os.path.join(FLAGS.model_dir, 'saved_model')
  checkpoint_export_dir = os.path.join(export_dir, str(global_step))
  if tf.io.gfile.exists(checkpoint_export_dir):
    tf.io.gfile.rmtree(checkpoint_export_dir)
  tf.saved_model.save(saved_model, checkpoint_export_dir)
  if FLAGS.keep_hub_module_max > 0:
    # Delete old exported SavedModels.
    exported_steps = []
    for subdir in tf.io.gfile.listdir(export_dir):
      if not subdir.isdigit():
        continue
      exported_steps.append(int(subdir))
    exported_steps.sort()
    for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
      tf.io.gfile.rmtree(os.path.join(export_dir, str(step_to_delete)))


def try_restore_from_checkpoint(model, global_step, optimizer):
  """Restores the latest ckpt if it exists, otherwise check FLAGS.checkpoint."""
  checkpoint = tf.train.Checkpoint(model=model, global_step=global_step, optimizer=optimizer)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint,directory=FLAGS.model_dir,max_to_keep=FLAGS.keep_checkpoint_max)
  latest_ckpt = checkpoint_manager.latest_checkpoint
  if latest_ckpt:
    # Restore model weights, global step, optimizer states
    logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
  elif FLAGS.checkpoint:
    # Restore model weights only, but not global step and optimizer states
    logging.info('Restoring from given checkpoint: %s', FLAGS.checkpoint)
    checkpoint_manager2 = tf.train.CheckpointManager(tf.train.Checkpoint(model=model),
        directory=FLAGS.model_dir,max_to_keep=FLAGS.keep_checkpoint_max)
    checkpoint_manager2.checkpoint.restore(FLAGS.checkpoint).expect_partial()
    if FLAGS.zero_init_logits_layer:
      model = checkpoint_manager2.checkpoint.model
      output_layer_parameters = model.supervised_head.trainable_weights
      logging.info('Initializing output layer parameters %s to zero', [x.op.name for x in output_layer_parameters])
      for x in output_layer_parameters:
        x.assign(tf.zeros_like(x))
  return checkpoint_manager


def json_serializable(val):
  try:
    json.dumps(val)
    return True
  except TypeError:
    return False


def perform_evaluation(model, builder, eval_steps, ckpt, strategy, topology):
  """Perform evaluation."""
  if FLAGS.train_mode == 'pretrain' and not FLAGS.lineareval_while_pretraining:
    logging.info('Skipping eval during pretraining without linear eval.')
    return
  # Build input pipeline.
  ds = data_lib.build_distributed_dataset(builder, FLAGS.eval_batch_size, False, strategy, topology)
  summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)
  # Build metrics.
  with strategy.scope():
    regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
    label_top_1_accuracy = tf.keras.metrics.Accuracy('eval/label_top_1_accuracy')
    label_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(5, 'eval/label_top_5_accuracy')
    all_metrics = [regularization_loss, label_top_1_accuracy, label_top_5_accuracy]
    # Restore checkpoint.
    logging.info('Restoring from %s', ckpt)
    checkpoint = tf.train.Checkpoint(model=model, global_step=tf.Variable(0, dtype=tf.int64))
    checkpoint.restore(ckpt).expect_partial()
    global_step = checkpoint.global_step
    logging.info('Performing eval at step %d', global_step.numpy())

  def single_step(features, labels):
    plt.savefig('/azi/foo.png')
    with tf.GradientTape() as saliency_tape:
      saliency_tape.watch(features)
      sp,supervised_head_outputs, msh = model(features, training=False)
      saliency_tape.watch(msh)
      log_prediction_proba = tf.math.log(tf.reduce_max(sp))
      # print('log_prediction_proba.shape', log_prediction_proba.shape)
      # print('log_prediction_proba', log_prediction_proba)
    saliency = saliency_tape.gradient(sp, features)
    # print('saliency', saliency.shape)
    for i in range(10):
      plot_maps(i, normalize_image(saliency[i]), normalize_image(features[i]))

    assert supervised_head_outputs is not None
    outputs = supervised_head_outputs
    l = labels['labels']
    metrics.update_finetune_metrics_eval(label_top_1_accuracy,label_top_5_accuracy, outputs, l)
    reg_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True)
    regularization_loss.update_state(reg_loss)

  with strategy.scope():
    # @tf.function
    def run_single_step(iterator):
      images, labels = next(iterator)
      features, labels = images, {'labels': labels}
      strategy.run(single_step, (features, labels))

    iterator = iter(ds)
    for i in range(eval_steps):
      run_single_step(iterator)
      logging.info('Completed eval for %d / %d steps', i + 1, eval_steps)
    logging.info('Finished eval for %s', ckpt)

  # Write summaries
  cur_step = global_step.numpy()
  logging.info('Writing summaries for %d step', cur_step)
  with summary_writer.as_default():
    metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
    summary_writer.flush()

  # Record results as JSON.
  result_json_path = os.path.join(FLAGS.model_dir, 'result.json')
  result = {metric.name: metric.result().numpy() for metric in all_metrics}
  result['global_step'] = global_step.numpy()
  logging.info(result)
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  result_json_path = os.path.join(FLAGS.model_dir, 'result_%d.json'%result['global_step'])
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')
  with tf.io.gfile.GFile(flag_json_path, 'w') as f:
    serializable_flags = {}
    for key, val in FLAGS.flag_values_dict().items():
      # Some flag value types e.g. datetime.timedelta are not json serializable, filter those out.
      if json_serializable(val):
        serializable_flags[key] = val
    json.dump(serializable_flags, f)

  # Export as SavedModel for finetuning and inference.
  save(model, global_step=result['global_step'])
  return result

def _restore_latest_or_from_pretrain(checkpoint_manager):
  """Restores the latest ckpt if training already.
  Or restores from FLAGS.checkpoint if in finetune mode.
  Args:
    checkpoint_manager: tf.traiin.CheckpointManager.
  """
  latest_ckpt = checkpoint_manager.latest_checkpoint
  if latest_ckpt:
    # The model is not build yet so some variables may not be available in
    # the object graph. Those are lazily initialized. To suppress the warning
    # in that case we specify `expect_partial`.
    logging.info('Restoring from %s', latest_ckpt)
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
  elif FLAGS.train_mode == 'finetune':
    # Restore from pretrain checkpoint.
    assert FLAGS.checkpoint, 'Missing pretrain checkpoint.'
    logging.info('Restoring from %s', FLAGS.checkpoint)
    checkpoint_manager.checkpoint.restore(FLAGS.checkpoint).expect_partial()
    # TODO(iamtingchen): Can we instead use a zeros initializer for the
    # supervised head?
    if FLAGS.zero_init_logits_layer:
      model = checkpoint_manager.checkpoint.model
      output_layer_parameters = model.supervised_head.trainable_weights
      logging.info('Initializing output layer parameters %s to zero',
                   [x.op.name for x in output_layer_parameters])
      for x in output_layer_parameters:
        x.assign(tf.zeros_like(x))


def model_summary(mdl):
  model_vars = mdl.trainable_variables
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
  builder.download_and_prepare()
  num_train_examples = builder.info.splits[FLAGS.train_split].num_examples
  num_eval_examples = builder.info.splits[FLAGS.eval_split].num_examples
  num_classes = builder.info.features['label'].num_classes
  eval_steps = FLAGS.eval_steps or int(
      math.ceil(num_eval_examples / FLAGS.eval_batch_size))

  train_steps_1 = model_lib.get_train_steps(num_train_examples)
  epoch_steps_1 = int(round(num_train_examples / FLAGS.train_batch_size))
  logging.info('# train examples M1: %d', num_train_examples)
  logging.info('# train_steps M1: %d', train_steps_1)
  logging.info('# epoch_steps M1: %d', epoch_steps_1)
  logging.info('# eval examples M1: %d', num_eval_examples)
  logging.info('# eval steps M1: %d', eval_steps)
  checkpoint_steps_1 = (FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * epoch_steps_1))
  topology = None
  strategy = tf.distribute.MirroredStrategy()
  logging.info('Running using MirroredStrategy on %d replicas',strategy.num_replicas_in_sync)

  with strategy.scope():
    model = model_lib.Model(num_classes)

  if FLAGS.mode == 'eval':
    for ckpt in tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs=15):
      result = perform_evaluation(model, builder, eval_steps, ckpt, strategy, topology)
      if result['global_step'] >= train_steps_1:
        logging.info('Eval complete. Exiting...')
        return
  else:
    summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)
    with strategy.scope():
      # Build input pipeline.
      ds = data_lib.build_distributed_dataset(dataset0, FLAGS.train_batch_size, True, strategy, topology)
      # Build LR schedule and optimizer.
      learning_rate = model_lib.WarmUpAndCosineDecay(FLAGS.learning_rate, num_train_examples)
      FLAGS.optimizer='adam'
      optimizer_1 = model_lib.build_optimizer(0.001)
      FLAGS.optimizer='lars'
      optimizer = model_lib.build_optimizer(learning_rate)
      optimizer_2 = model_lib.build_optimizer(learning_rate)

      # Build metrics.
      all_metrics = []  # For summaries.
      weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
      total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
      all_metrics.extend([weight_decay_metric, total_loss_metric])
      if FLAGS.train_mode == 'pretrain':
        contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
        contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
        contrast_entropy_metric = tf.keras.metrics.Mean('train/contrast_entropy')
        all_metrics.extend([contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric])
      if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
        supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
        supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')
        all_metrics.extend([supervised_loss_metric, supervised_acc_metric])

      # Restore checkpoint if available.
      checkpoint_manager = try_restore_from_checkpoint(model, optimizer.iterations, optimizer)

    def single_step(features, labels):
      with tf.GradientTape() as tape:
        should_record = tf.equal((optimizer.iterations + 1) % checkpoint_steps_1, 0)
        with tf.summary.record_if(should_record):
          # Only log augmented images for the first tower.
          tf.summary.image('image', features[:, :, :, :3], step=optimizer.iterations + 1)

        projection_head_outputs, supervised_head_outputs, msh = model(features, training=True)
        tape.watch(features)
        loss = None
        if projection_head_outputs is not None:
          outputs = projection_head_outputs
          con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
              outputs,hidden_norm=FLAGS.hidden_norm,temperature=FLAGS.temperature,strategy=strategy)
          if loss is None:
            loss = con_loss
          else:
            loss += con_loss
          metrics.update_pretrain_metrics_train(contrast_loss_metric,contrast_acc_metric,
                                                contrast_entropy_metric,con_loss, logits_con,labels_con)
        if supervised_head_outputs is not None:
          outputs = supervised_head_outputs
          l = labels['labels']
          if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            l = tf.concat([l, l], 0)
          sup_loss = obj_lib.add_supervised_loss(labels=l, logits=outputs)
          if loss is None:
            loss = sup_loss
          else:
            loss += sup_loss
          metrics.update_finetune_metrics_train(supervised_loss_metric,supervised_acc_metric, sup_loss,l, outputs)
        weight_decay = model_lib.add_weight_decay(model, adjust_per_optimizer=True)
        weight_decay_metric.update_state(weight_decay)
        loss += weight_decay
        total_loss_metric.update_state(loss)
        # The default behavior of `apply_gradients` is to sum gradients from all
        # replicas so we divide the loss by the number of replicas so that the mean gradient is applied.
        loss = loss / strategy.num_replicas_in_sync
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    with strategy.scope():
      @tf.function
      def train_multiple_steps(iterator):
        # `tf.range` is needed so that this runs in a `tf.while_loop` and is not unrolled.
        for _ in tf.range(checkpoint_steps_1):
          # Drop the "while" prefix created by tf.while_loop which otherwise gets prefixed to every variable name.
          # This does not affect training but does affect the checkpoint conversion script. TODO(b/161712658): Remove this.
          with tf.name_scope(''):
            images, labels = next(iterator)
            features, labels = images, {'labels': labels}
            strategy.run(single_step, (features, labels))

      global_step = optimizer.iterations
      cur_step_1 = global_step.numpy()
      iterator = iter(ds)
      while cur_step_1 < train_steps_1:
        # Calls to tf.summary.xyz lookup the summary writer resource which is
        # set by the summary writer's context manager.
        with summary_writer.as_default():
          train_multiple_steps(iterator)
          cur_step_1 = global_step.numpy()
          checkpoint_manager.save(cur_step_1)
          logging.info('Completed: %d / %d steps', cur_step_1, train_steps_1)
          metrics.log_and_write_metrics_to_summary(all_metrics, cur_step_1)
          tf.summary.scalar('learning_rate',learning_rate(tf.cast(global_step, dtype=tf.float32)),global_step)
          summary_writer.flush()
        for metric in all_metrics:
          metric.reset_states()
      logging.info('Training 1 complete...')

    if FLAGS.mode == 'train_then_eval':
      perform_evaluation(model, dataset2, eval_steps, checkpoint_manager.latest_checkpoint, strategy,topology)

if __name__ == '__main__':

  (xt, yt), (xe, ye) = keras.datasets.cifar10.load_data() 
  idx0 = (yt == 0).reshape(xt.shape[0]);idx1 = (yt == 1).reshape(xt.shape[0])
  idx2 = (yt == 2).reshape(xt.shape[0]);idx3 = (yt == 3).reshape(xt.shape[0])
  idx4 = (yt == 4).reshape(xt.shape[0]);idx5 = (yt == 5).reshape(xt.shape[0])
  idx6 = (yt == 6).reshape(xt.shape[0]);idx7 = (yt == 7).reshape(xt.shape[0])
  idx8 = (yt == 8).reshape(xt.shape[0]);idx9 = (yt == 9).reshape(xt.shape[0])

  ti0 = xt[idx0]; ti1 = xt[idx1]; ti2 = xt[idx2]; ti3 = xt[idx3]; ti4 = xt[idx4]; ti5 = xt[idx5]; ti6 = xt[idx6]; ti7 = xt[idx7] 
  ti8 = xt[idx8]; ti9 = xt[idx9]
  tl0=yt[idx0];tl1=yt[idx1];tl2=yt[idx2]; tl3=yt[idx3];tl4=yt[idx4];tl5=yt[idx5]; tl6=yt[idx6];tl7=yt[idx7];tl8=yt[idx8]; tl9=yt[idx9]

  jdx0 = (ye == 0).reshape(xe.shape[0]);jdx1 = (ye == 1).reshape(xe.shape[0])
  jdx2 = (ye == 2).reshape(xe.shape[0]);jdx3 = (ye == 3).reshape(xe.shape[0])
  jdx4 = (ye == 4).reshape(xe.shape[0]);jdx5 = (ye == 5).reshape(xe.shape[0])
  jdx6 = (ye == 6).reshape(xe.shape[0]);jdx7 = (ye == 7).reshape(xe.shape[0])
  jdx8 = (ye == 8).reshape(xe.shape[0]);jdx9 = (ye == 9).reshape(xe.shape[0])

  ei0 = xe[jdx0]; ei1 = xe[jdx1]; ei2 = xe[jdx2]; ei3 = xe[jdx3]; ei4 = xe[jdx4]; ei5 = xe[jdx5]
  ei6 = xe[jdx6]; ei7 = xe[jdx7]; ei8 = xe[jdx8]; ei9 = xe[jdx9]
  el0=ye[jdx0];el1=ye[jdx1];el2=ye[jdx2]; el3=ye[jdx3];el4=ye[jdx4];el5=ye[jdx5]; el6=ye[jdx6];el7=ye[jdx7];el8=ye[jdx8]; el9=ye[jdx9]

    
  (xtm, ytm), (xem, yem) = keras.datasets.mnist.load_data()
  idx00 = (ytm == 0).reshape(xtm.shape[0]);idx11 = (ytm == 1).reshape(xtm.shape[0])
  idx22 = (ytm == 2).reshape(xtm.shape[0]);idx33 = (ytm == 3).reshape(xtm.shape[0])
  idx44 = (ytm == 4).reshape(xtm.shape[0]);idx55 = (ytm == 5).reshape(xtm.shape[0])
  idx66 = (ytm == 6).reshape(xtm.shape[0]);idx77 = (ytm == 7).reshape(xtm.shape[0])
  idx88 = (ytm == 8).reshape(xtm.shape[0]);idx99 = (ytm == 9).reshape(xtm.shape[0])

  tim0 = xtm[idx00]; tim1 = xtm[idx11]; tim2 = xtm[idx22]; tim3 = xtm[idx33]; tim4 = xtm[idx44]; tim5 = xtm[idx55]
  tim6 = xtm[idx66]; tim7 = xtm[idx77]; tim8 = xtm[idx88]; tim9 = xtm[idx99]
  tlm0=ytm[idx00];tlm1=ytm[idx11];tlm2=ytm[idx22]; tlm3=ytm[idx33];tlm4=ytm[idx44];tlm5=ytm[idx55]
  tlm6=ytm[idx66];tlm7=ytm[idx77];tlm8=ytm[idx88]; tlm9=ytm[idx99]

  jdx00 = (yem == 0).reshape(xem.shape[0]);jdx11 = (yem == 1).reshape(xem.shape[0])
  jdx22 = (yem == 2).reshape(xem.shape[0]);jdx33 = (yem == 3).reshape(xem.shape[0])
  jdx44 = (yem == 4).reshape(xem.shape[0]);jdx55 = (yem == 5).reshape(xem.shape[0])
  jdx66 = (yem == 6).reshape(xem.shape[0]);jdx77 = (yem == 7).reshape(xem.shape[0])
  jdx88 = (yem == 8).reshape(xem.shape[0]);jdx99 = (yem == 9).reshape(xem.shape[0])

  eim0 = xem[jdx00]; eim1 = xem[jdx11]; eim2 = xem[jdx22]; eim3 = xem[jdx33]; eim4 = xem[jdx44]; eim5 = xem[jdx55]
  eim6 = xem[jdx66]; eim7 = xem[jdx77]; eim8 = xem[jdx88]; eim9 = xem[jdx99]
  elm0=yem[jdx00];elm1=yem[jdx11];elm2=yem[jdx22]; elm3=yem[jdx33];elm4=yem[jdx44];elm5=yem[jdx55]
  elm6=yem[jdx66];elm7=yem[jdx77];elm8=yem[jdx88]; elm9=yem[jdx99]
 
  height1=32; width1=32; ch1=3; height2=28; width2=28; m=0; abj=5000
  
  im0=np.zeros((abj, 32+28+10, width1, ch1), np.uint8); im1=np.zeros((abj, 32+28+10, width1, ch1), np.uint8)
  im2=np.zeros((abj, 32+28+10, width1, ch1), np.uint8); im3=np.zeros((abj, 32+28+10, width1, ch1), np.uint8)
  im4=np.zeros((abj, 32+28+10, width1, ch1), np.uint8); im5=np.zeros((abj, 32+28+10, width1, ch1), np.uint8)
  im6=np.zeros((abj, 32+28+10, width1, ch1), np.uint8); im7=np.zeros((abj, 32+28+10, width1, ch1), np.uint8)
  im8=np.zeros((abj, 32+28+10, width1, ch1), np.uint8); im9=np.zeros((abj, 32+28+10, width1, ch1), np.uint8)

  tm0=np.zeros((1000, 32+28+10, width1, ch1), np.uint8); tm1=np.zeros((1000, 32+28+10, width1, ch1), np.uint8)
  tm2=np.zeros((1000, 32+28+10, width1, ch1), np.uint8); tm3=np.zeros((1000, 32+28+10, width1, ch1), np.uint8)
  tm4=np.zeros((1000, 32+28+10, width1, ch1), np.uint8); tm5=np.zeros((1000, 32+28+10, width1, ch1), np.uint8)
  tm6=np.zeros((1000, 32+28+10, width1, ch1), np.uint8); tm7=np.zeros((1000, 32+28+10, width1, ch1), np.uint8)
  tm8=np.zeros((1000, 32+28+10, width1, ch1), np.uint8); tm9=np.zeros((1000, 32+28+10, width1, ch1), np.uint8)

  for j in range (abj):
    for x in range(0, height2):
      for y in range(0, width2):
        im0[j][x+height1, y]=tim0[j][x, y] 
        if abj<1000:
          tm0[j][x+height1, y]=eim0[j][x, y]  
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im0[j][x, y, c]=ti0[j][x, y, c]
          if abj<1000: 
            tm0[j][x, y, c]=el0[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im1[j][x+height1, y]=tim1[j][x, y]
        if abj<1000:
          tm1[j][x+height1, y]=eim1[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im1[j][x, y, c]=ti1[j][x, y, c] 
          if abj<1000:
            tm1[j][x, y, c]=el1[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im2[j][x+height1, y]=tim2[j][x, y] 
        if abj<1000:
          tm2[j][x+height1, y]=eim2[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im2[j][x, y, c]=ti2[j][x, y, c] 
          if abj<1000:
            tm2[j][x, y, c]=el2[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im3[j][x+height1, y]=tim3[j][x, y] 
        if abj<1000:
          tm3[j][x+height1, y]=eim3[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im3[j][x, y, c]=ti3[j][x, y, c] 
          if abj<1000:
            tm3[j][x, y, c]=el3[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im4[j][x+height1, y]=tim4[j][x, y] 
        if abj<1000:
          tm4[j][x+height1, y]=eim4[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im4[j][x, y, c]=ti4[j][x, y, c] 
          if abj<1000:
            tm4[j][x, y, c]=el4[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im5[j][x+height1, y]=tim5[j][x, y] 
        if abj<1000:
          tm5[j][x+height1, y]=eim5[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im5[j][x, y, c]=ti5[j][x, y, c] 
          if abj<1000:
            tm5[j][x, y, c]=el5[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im6[j][x+height1, y]=tim6[j][x, y] 
        if abj<1000:
          tm6[j][x+height1, y]=eim6[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im6[j][x, y, c]=ti6[j][x, y, c]
          if abj<1000:
            tm6[j][x, y, c]=el6[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im7[j][x+height1, y]=tim7[j][x, y] 
        if abj<1000:
          tm7[j][x+height1, y]=eim7[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im7[j][x, y, c]=ti7[j][x, y, c] 
          if abj<1000:
            tm7[j][x, y, c]=el7[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im8[j][x+height1, y]=tim8[j][x, y] 
        if abj<1000:
          tm8[j][x+height1, y]=eim8[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im8[j][x, y, c]=ti8[j][x, y, c] 
          if abj<1000:
            tm8[j][x, y, c]=el8[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im9[j][x+height1, y]=tim9[j][x, y]
        if abj<1000:
          tm9[j][x+height1, y]=eim9[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im9[j][x, y, c]=ti9[j][x, y, c] 
          if abj<1000:
            tm9[j][x, y, c]=el9[j][x, y, c]

  big=np.concatenate((im0[0:abj], im1[0:abj], im2[0:abj], im3[0:abj], im4[0:abj], im5[0:abj], im6[0:abj], im7[0:abj], im8[0:abj], im9[0:abj]), axis=0)
  lbig=np.concatenate((tl0[0:abj], tl1[0:abj], tl2[0:abj], tl3[0:abj], tl4[0:abj], tl5[0:abj], tl6[0:abj], tl7[0:abj], tl8[0:abj], tl9[0:abj]), axis=0)

  x_nn01, y_nn01 = shuffle(np.array(big), np.array(lbig))
  y_nn01=y_nn01.reshape(abj*10,)
  Mydatasetx01 = tf.data.Dataset.from_tensor_slices(x_nn01)
  Mydatasety01 = tf.data.Dataset.from_tensor_slices(y_nn01)
  dataset0 = tf.data.Dataset.zip((Mydatasetx01, Mydatasety01))

  big2=np.concatenate((tm0, tm1, tm2, tm3, tm4, tm5, tm6, tm7, tm8, tm9), axis=0)
  lbig2=np.concatenate((el0, el1, el2, el3, el4, el5, el6, el7, el8, el9), axis=0)

  x_nn02, y_nn02 = shuffle(np.array(big2), np.array(lbig2))
  y_nn02=y_nn02.reshape(1000*10,)
  Mydatasetx02 = tf.data.Dataset.from_tensor_slices(x_nn02)
  Mydatasety02 = tf.data.Dataset.from_tensor_slices(y_nn02)
  dataset2 = tf.data.Dataset.zip((Mydatasetx02, Mydatasety02))

  tf.compat.v1.enable_v2_behavior()
  # For outside compilation of summaries on TPU.
  tf.config.set_soft_device_placement(True)
  app.run(main)
