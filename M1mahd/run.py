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

  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  (x_train1, y_train1), (x_test1, y_test1) = keras.datasets.mnist.load_data()
  
  idx0 = (y_train == 0).reshape(x_train.shape[0]);idx1 = (y_train == 1).reshape(x_train.shape[0])
  idx2 = (y_train == 2).reshape(x_train.shape[0]);idx3 = (y_train == 3).reshape(x_train.shape[0])
  idx4 = (y_train == 4).reshape(x_train.shape[0]);idx5 = (y_train == 5).reshape(x_train.shape[0])
  idx6 = (y_train == 6).reshape(x_train.shape[0]);idx7 = (y_train == 7).reshape(x_train.shape[0])
  idx8 = (y_train == 8).reshape(x_train.shape[0]);idx9 = (y_train == 9).reshape(x_train.shape[0])

  jdx0 = (y_test == 0).reshape(x_test.shape[0]);jdx1 = (y_test == 1).reshape(x_test.shape[0])
  jdx2 = (y_test == 2).reshape(x_test.shape[0]);jdx3 = (y_test == 3).reshape(x_test.shape[0])
  jdx4 = (y_test == 4).reshape(x_test.shape[0]);jdx5 = (y_test == 5).reshape(x_test.shape[0])
  jdx6 = (y_test == 6).reshape(x_test.shape[0]);jdx7 = (y_test == 7).reshape(x_test.shape[0])
  jdx8 = (y_test == 8).reshape(x_test.shape[0]);jdx9 = (y_test == 9).reshape(x_test.shape[0])

  filtered_images0 = x_train[idx0]; filtered_images1 = x_train[idx1]; filtered_images2 = x_train[idx2]
  filtered_images3 = x_train[idx3]; filtered_images4 = x_train[idx4]; filtered_images5 = x_train[idx5]
  filtered_images6 = x_train[idx6]; filtered_images7 = x_train[idx7]; filtered_images8 = x_train[idx8]
  filtered_images9 = x_train[idx9]

  filtered_lable0=y_train[idx0];filtered_lable1=y_train[idx1];filtered_lable2=y_train[idx2]
  filtered_lable3=y_train[idx3];filtered_lable4=y_train[idx4];filtered_lable5=y_train[idx5]
  filtered_lable6=y_train[idx6];filtered_lable7=y_train[idx7];filtered_lable8=y_train[idx8]
  filtered_lable9=y_train[idx9]

  test_images0 = x_test[jdx0]; test_images1 = x_test[jdx1]; test_images2 = x_test[jdx2]
  test_images3 = x_test[jdx3]; test_images4 = x_test[jdx4]; test_images5 = x_test[jdx5]
  test_images6 = x_test[jdx6]; test_images7 = x_test[jdx7]; test_images8 = x_test[jdx8]
  test_images9 = x_test[jdx9]

  test_lable0=y_test[jdx0];test_lable1=y_test[jdx1];test_lable2=y_test[jdx2]
  test_lable3=y_test[jdx3];test_lable4=y_test[jdx4];test_lable5=y_test[jdx5]
  test_lable6=y_test[jdx6];test_lable7=y_test[jdx7];test_lable8=y_test[jdx8]
  test_lable9=y_test[jdx9]

  idx00 = (y_train1 == 0).reshape(x_train1.shape[0]);idx11 = (y_train1 == 1).reshape(x_train1.shape[0])
  idx22 = (y_train1 == 2).reshape(x_train1.shape[0]);idx33 = (y_train1 == 3).reshape(x_train1.shape[0])
  idx44 = (y_train1 == 4).reshape(x_train1.shape[0]);idx55 = (y_train1 == 5).reshape(x_train1.shape[0])
  idx66 = (y_train1 == 6).reshape(x_train1.shape[0]);idx77 = (y_train1 == 7).reshape(x_train1.shape[0])
  idx88 = (y_train1 == 8).reshape(x_train1.shape[0]);idx99 = (y_train1 == 9).reshape(x_train1.shape[0])

  jdx00 = (y_test1 == 0).reshape(x_test1.shape[0]);jdx11 = (y_test1 == 1).reshape(x_test1.shape[0])
  jdx22 = (y_test1 == 2).reshape(x_test1.shape[0]);jdx33 = (y_test1 == 3).reshape(x_test1.shape[0])
  jdx44 = (y_test1 == 4).reshape(x_test1.shape[0]);jdx55 = (y_test1 == 5).reshape(x_test1.shape[0])
  jdx66 = (y_test1 == 6).reshape(x_test1.shape[0]);jdx77 = (y_test1 == 7).reshape(x_test1.shape[0])
  jdx88 = (y_test1 == 8).reshape(x_test1.shape[0]);jdx99 = (y_test1 == 9).reshape(x_test1.shape[0])

  filtered_images00 = x_train1[idx00]; filtered_images11 = x_train1[idx11]; filtered_images22 = x_train1[idx22]
  filtered_images33 = x_train1[idx33]; filtered_images44 = x_train1[idx44]; filtered_images55 = x_train1[idx55]
  filtered_images66 = x_train1[idx66]; filtered_images77 = x_train1[idx77]; filtered_images88 = x_train1[idx88]
  filtered_images99 = x_train1[idx99]

  # filtered_lable00=y_train1[idx00];filtered_lable11=y_train1[idx11];filtered_lable22=y_train1[idx22]
  # filtered_lable33=y_train1[idx33];filtered_lable44=y_train1[idx44];filtered_lable55=y_train1[idx55]
  # filtered_lable66=y_train1[idx66];filtered_lable77=y_train1[idx77];filtered_lable88=y_train1[idx88]
  # filtered_lable99=y_train1[idx99]

  test_images00 = x_test1[jdx00]; test_images11 = x_test1[jdx11]; test_images22 = x_test1[jdx22]
  test_images33 = x_test1[jdx33]; test_images44 = x_test1[jdx44]; test_images55 = x_test1[jdx55]
  test_images66 = x_test1[jdx66]; test_images77 = x_test1[jdx77]; test_images88 = x_test1[jdx88]
  test_images99 = x_test1[jdx99]

  # test_lable00=y_test[jdx00];test_lable11=y_test[jdx11];test_lable22=y_test[jdx22]
  # test_lable33=y_test[jdx33];test_lable44=y_test[jdx44];test_lable55=y_test[jdx55]
  # test_lable66=y_test[jdx66];test_lable77=y_test[jdx77];test_lable88=y_test[jdx88]
  # test_lable99=y_test[jdx99]
 
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
        im0[j][x+height1, y]=filtered_images00[j][x, y] 
        if abj<1000:
          tm0[j][x+height1, y]=test_images00[j][x, y]  
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im0[j][x, y, c]=filtered_images0[j][x, y, c];
          if abj<1000: 
            tm0[j][x, y, c]=test_images0[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im1[j][x+height1, y]=filtered_images11[j][x, y]
        if abj<1000:
          tm1[j][x+height1, y]=test_images11[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im1[j][x, y, c]=filtered_images1[j][x, y, c]; 
          if abj<1000:
            tm1[j][x, y, c]=test_images1[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im2[j][x+height1, y]=filtered_images22[j][x, y]; 
        if abj<1000:
          tm2[j][x+height1, y]=test_images22[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im2[j][x, y, c]=filtered_images2[j][x, y, c]; 
          if abj<1000:
            tm2[j][x, y, c]=test_images2[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im3[j][x+height1, y]=filtered_images33[j][x, y]; 
        if abj<1000:
          tm3[j][x+height1, y]=test_images33[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im3[j][x, y, c]=filtered_images3[j][x, y, c]; 
          if abj<1000:
            tm3[j][x, y, c]=test_images3[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im4[j][x+height1, y]=filtered_images44[j][x, y]; 
        if abj<1000:
          tm4[j][x+height1, y]=test_images44[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im4[j][x, y, c]=filtered_images4[j][x, y, c]; 
          if abj<1000:
            tm4[j][x, y, c]=test_images4[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im5[j][x+height1, y]=filtered_images55[j][x, y]; 
        if abj<1000:
          tm5[j][x+height1, y]=test_images55[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im5[j][x, y, c]=filtered_images5[j][x, y, c]; 
          if abj<1000:
            tm5[j][x, y, c]=test_images5[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im6[j][x+height1, y]=filtered_images66[j][x, y]; 
        if abj<1000:
          tm6[j][x+height1, y]=test_images66[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im6[j][x, y, c]=filtered_images6[j][x, y, c]; 
          if abj<1000:
            tm6[j][x, y, c]=test_images6[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im7[j][x+height1, y]=filtered_images77[j][x, y]; 
        if abj<1000:
          tm7[j][x+height1, y]=test_images77[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im7[j][x, y, c]=filtered_images7[j][x, y, c]; 
          if abj<1000:
            tm7[j][x, y, c]=test_images7[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im8[j][x+height1, y]=filtered_images88[j][x, y]; 
        if abj<1000:
          tm8[j][x+height1, y]=test_images88[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im8[j][x, y, c]=filtered_images8[j][x, y, c]; 
          if abj<1000:
            tm8[j][x, y, c]=test_images8[j][x, y, c]

  for j in range (0, abj):
    for x in range(0, height2):
      for y in range (0, width2):
        im9[j][x+height1, y]=filtered_images99[j][x, y]; 
        if abj<1000:
          tm9[j][x+height1, y]=test_images99[j][x, y]
    for x in range(0, height1):
      for y in range (0, width1):
        for c in range(ch1):
          im9[j][x, y, c]=filtered_images9[j][x, y, c]; 
          if abj<1000:
            tm9[j][x, y, c]=test_images9[j][x, y, c]

  big=np.concatenate((im0[0:abj], im1[0:abj], im2[0:abj], im3[0:abj], im4[0:abj], im5[0:abj], im6[0:abj], im7[0:abj], im8[0:abj], im9[0:abj]), axis=0)
  lbig=np.concatenate((filtered_lable0[0:abj], filtered_lable1[0:abj], filtered_lable2[0:abj], 
              filtered_lable3[0:abj], filtered_lable4[0:abj], filtered_lable5[0:abj], 
              filtered_lable6[0:abj], filtered_lable7[0:abj], filtered_lable8[0:abj], filtered_lable9[0:abj]), axis=0)

  x_nn01, y_nn01 = shuffle(np.array(big), np.array(lbig))
  y_nn01=y_nn01.reshape(abj*10,)
  Mydatasetx01 = tf.data.Dataset.from_tensor_slices(x_nn01)
  Mydatasety01 = tf.data.Dataset.from_tensor_slices(y_nn01)
  dataset0 = tf.data.Dataset.zip((Mydatasetx01, Mydatasety01))

  big2=np.concatenate((tm0[0:abj], tm1[0:abj], tm2[0:abj], tm3[0:abj], tm4[0:abj], tm5[0:abj], tm6[0:abj], tm7[0:abj], 
              tm8[0:abj], tm9[0:abj]), axis=0)
  lbig2=np.concatenate((test_lable0[0:abj], test_lable1[0:abj], test_lable2[0:abj], 
              test_lable3[0:abj], test_lable4[0:abj], test_lable5[0:abj], 
              test_lable6[0:abj], test_lable7[0:abj], test_lable8[0:abj], test_lable9[0:abj]), axis=0)

  x_nn02, y_nn02 = shuffle(np.array(big2), np.array(lbig2))
  y_nn02=y_nn02.reshape(1000*10,)
  Mydatasetx02 = tf.data.Dataset.from_tensor_slices(x_nn02)
  Mydatasety02 = tf.data.Dataset.from_tensor_slices(y_nn02)
  dataset2 = tf.data.Dataset.zip((Mydatasetx02, Mydatasety02))

  tf.compat.v1.enable_v2_behavior()
  # For outside compilation of summaries on TPU.
  tf.config.set_soft_device_placement(True)
  app.run(main)
