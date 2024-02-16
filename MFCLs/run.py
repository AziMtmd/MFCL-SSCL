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
# from model_profiler import model_profiler
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

FLAGS = flags.FLAGS

flags.DEFINE_integer('train_batch_size', 512, 'Batch size for training.')

flags.DEFINE_bool('module1_train', True, 'Training the first module')

flags.DEFINE_integer('train_epochs', 20, 'Number of epochs to train for.')

flags.DEFINE_integer('numofclients', 10, 'Number of epochs to train for.')

flags.DEFINE_integer('m2_epoch', 50, 'Number of epochs to train for.')

flags.DEFINE_float('warmup_epochs', 10, 'Number of epochs of warmup.')

flags.DEFINE_string('dataset', 'imagenet_resized', 'Name of a dataset.')

flags.DEFINE_integer('proj_out_dim', 128,'Number of head projection dimension.')

flags.DEFINE_integer('num_proj_layers', 3,'Number of non-linear head layers.')

flags.DEFINE_integer('resnet_depth', 18,'Depth of ResNet.') 

flags.DEFINE_integer('image_size', 32, 'Input image size.')

flags.DEFINE_integer('Beta', 200, 'Input image size.')

flags.DEFINE_float('learning_rate', 1.5, 'Initial learning rate per batch size of 256.')
flags.DEFINE_enum('learning_rate_scaling', 'linear', ['linear', 'sqrt'],'How to scale the learning rate as a function of batch size.')
flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')
flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')
flags.DEFINE_string('train_split', 'train', 'Split for training.')
flags.DEFINE_integer('train_steps', 0, 'Number of steps to train for. If provided, overrides train_epochs.')
flags.DEFINE_integer('eval_steps', 0, 'Number of steps to eval for. If not provided, evals over entire dataset.')
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
flags.DEFINE_integer('keep_checkpoint_max', 1,'Maximum number of checkpoints to keep.')
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


# def flops(mrd):
#   session = tf.compat.v1.Session()
#   graph = tf.compat.v1.get_default_graph()
#   with graph.as_default():
#     with session.as_default():
#       run_meta = tf.compat.v1.RunMetadata()
#       opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#       flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
#   return flops.total_float_ops


def build_saved_model(model, include_projection_head=True):
  """Returns a tf.Module for saving to SavedModel."""
  class SimCLRModel(tf.Module):
    """Saved model for exporting to hub."""
    def __init__(self, model):
      self.model = model
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
    logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
  elif FLAGS.checkpoint:
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


def perform_evaluation(modelG, model_0, builder, eval_steps, ckpt, strategy, topology):
  """Perform evaluation."""
  if FLAGS.train_mode == 'pretrain' and not FLAGS.lineareval_while_pretraining:
    logging.info('Skipping eval during pretraining without linear eval.')
    return
  ds = data_lib.build_distributed_dataset(builder, FLAGS.eval_batch_size, False, strategy, topology)
  summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)
  with strategy.scope():
    regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
    label_top_1_accuracy = tf.keras.metrics.Accuracy('eval/label_top_1_accuracy')
    label_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(5, 'eval/label_top_5_accuracy')
    all_metrics = [regularization_loss, label_top_1_accuracy, label_top_5_accuracy]
    logging.info('Restoring from %s', ckpt)
    checkpoint = tf.train.Checkpoint(model=modelG, global_step=tf.Variable(0, dtype=tf.int64))
    checkpoint.restore(ckpt).expect_partial()
    global_step = checkpoint.global_step
    logging.info('Performing eval at step %d', global_step.numpy())

  def single_step(features, labels):
    rep0 = model_0(features, training=False);      

    # rep=tf.concat([rep0, rep1,rep2,rep3, rep4, rep5,rep6,rep7, rep8, rep9], 0)
    _, supervised_head_outputs = modelG(rep0, training=False)

    assert supervised_head_outputs is not None
    outputs = supervised_head_outputs
    l = labels['labels']

    metrics.update_finetune_metrics_eval(label_top_1_accuracy,label_top_5_accuracy, outputs, l)
    reg_loss = model_lib.add_weight_decay(modelG, adjust_per_optimizer=True)
    regularization_loss.update_state(reg_loss)

  with strategy.scope():
    @tf.function
    def run_single_step(iterator):
      images, labels = next(iterator); features, labels = images, {'labels': labels}
      strategy.run(single_step, (features, labels))

    iterator = iter(ds)
    for i in range(eval_steps):
      run_single_step(iterator)
      logging.info('Completed eval for %d / %d steps', i + 1, eval_steps)
    logging.info('Finished eval for %s', ckpt)

  cur_step = global_step.numpy()
  logging.info('Writing summaries for %d step', cur_step)
  with summary_writer.as_default():
    metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
    summary_writer.flush()

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
      if json_serializable(val):
        serializable_flags[key] = val
    json.dump(serializable_flags, f)

  # save(model, global_step=result['global_step'])

  return result

def _restore_latest_or_from_pretrain(checkpoint_manager):
  """Restores the latest ckpt if training already.
  Or restores from FLAGS.checkpoint if in finetune mode.
  Args:
    checkpoint_manager: tf.traiin.CheckpointManager.
  """
  latest_ckpt = checkpoint_manager.latest_checkpoint
  if latest_ckpt:
    logging.info('Restoring from %s', latest_ckpt)
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
  elif FLAGS.train_mode == 'finetune':
    assert FLAGS.checkpoint, 'Missing pretrain checkpoint.'
    logging.info('Restoring from %s', FLAGS.checkpoint)
    checkpoint_manager.checkpoint.restore(FLAGS.checkpoint).expect_partial()
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
  # num_train_examples = builder.info.splits[FLAGS.train_split].num_examples
  num_eval_examples = builder.info.splits[FLAGS.eval_split].num_examples
  num_classes = builder.info.features['label'].num_classes
  eval_steps = FLAGS.eval_steps or int(math.ceil(num_eval_examples / FLAGS.eval_batch_size))
  num_train_examples=5000
  kept=FLAGS.train_batch_size; 
  FLAGS.train_batch_size=64
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
    model_0=model_lib.Module_1(num_classes); model_1=model_lib.Module_1(num_classes)
    model_2=model_lib.Module_1(num_classes); model_3=model_lib.Module_1(num_classes)
    model_4=model_lib.Module_1(num_classes); model_5=model_lib.Module_1(num_classes)
    model_6=model_lib.Module_1(num_classes); model_7=model_lib.Module_1(num_classes)
    model_8=model_lib.Module_1(num_classes); model_9=model_lib.Module_1(num_classes)
  
    modelG = model_lib.Model(num_classes)

  if FLAGS.mode == 'eval':
    for ckpt in tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs=15):
      result = perform_evaluation(modelG, model_0, builder, eval_steps, ckpt, strategy, topology)
      if result['global_step'] >= train_steps_3:
        logging.info('Eval complete. Exiting...')
        return
  else:
    summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)
    
    with strategy.scope():
      FLAGS.train_split='train[0:5000]'
      ds1=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology)
      dsc1=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology)
      FLAGS.train_split='train[5000:10000]'
      ds2=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology)
      dsc2=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology)
      FLAGS.train_split='train[10000:15000]'
      ds3=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology)
      dsc3=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology)
      FLAGS.train_split='train[15000:20000]'
      ds4=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology)  
      dsc4=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology)
      FLAGS.train_split='train[20000:25000]'
      ds5=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology)
      dsc5=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology)  
      FLAGS.train_split='train[25000:30000]'
      ds6=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology)  
      dsc6=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology)
      FLAGS.train_split='train[30000:35000]'
      ds7=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology)
      dsc7=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology) 
      FLAGS.train_split='train[35000:40000]'
      ds8=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology) 
      dsc8=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology)  
      FLAGS.train_split='train[40000:45000]'
      ds9=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology) 
      dsc9=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology)   
      FLAGS.train_split='train[45000:50000]'
      ds0=data_lib.build_distributed_dataset(builder, 64, True, strategy, topology)  
      dsc0=data_lib.build_distributed_dataset(builder, 52, False, strategy, topology)   

      learning_rate = model_lib.WarmUpAndCosineDecay(FLAGS.learning_rate, num_train_examples)
      FLAGS.optimizer='adam'
      optimizer_0=model_lib.build_optimizer(0.001); optimizer_1=model_lib.build_optimizer(0.001)
      optimizer_2=model_lib.build_optimizer(0.001); optimizer_3=model_lib.build_optimizer(0.001)
      optimizer_4=model_lib.build_optimizer(0.001); optimizer_5=model_lib.build_optimizer(0.001)
      optimizer_6=model_lib.build_optimizer(0.001); optimizer_7=model_lib.build_optimizer(0.001)
      optimizer_8=model_lib.build_optimizer(0.001); optimizer_9=model_lib.build_optimizer(0.001)

      FLAGS.optimizer='lars'
      optimizerG = model_lib.build_optimizer(learning_rate)
      
      all_metrics = []  # For summaries.
      weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
      total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
      all_metrics.extend([weight_decay_metric, total_loss_metric])
      if FLAGS.train_mode == 'pretrain':
        unsupervised_loss_metric = tf.keras.metrics.Mean('train/unsupervised_loss')
        unsupervised_acc_metric = tf.keras.metrics.Mean('train/unsupervised_acc')        
        contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
        contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
        contrast_entropy_metric = tf.keras.metrics.Mean('train/contrast_entropy')
        all_metrics.extend([unsupervised_loss_metric, unsupervised_acc_metric,
            contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric])
      if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
        supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
        supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')
        all_metrics.extend([supervised_loss_metric, supervised_acc_metric])

      checkpoint_manager_0=try_restore_from_checkpoint(model_0, optimizer_0.iterations, optimizer_0)
      checkpoint_manager_1=try_restore_from_checkpoint(model_1, optimizer_1.iterations, optimizer_1)
      checkpoint_manager_2=try_restore_from_checkpoint(model_2, optimizer_2.iterations, optimizer_2)
      checkpoint_manager_3=try_restore_from_checkpoint(model_3, optimizer_3.iterations, optimizer_3)
      checkpoint_manager_4=try_restore_from_checkpoint(model_4, optimizer_4.iterations, optimizer_4)
      checkpoint_manager_5=try_restore_from_checkpoint(model_5, optimizer_5.iterations, optimizer_5)
      checkpoint_manager_6=try_restore_from_checkpoint(model_6, optimizer_6.iterations, optimizer_6)
      checkpoint_manager_7=try_restore_from_checkpoint(model_7, optimizer_7.iterations, optimizer_7)
      checkpoint_manager_8=try_restore_from_checkpoint(model_8, optimizer_8.iterations, optimizer_8)
      checkpoint_manager_9=try_restore_from_checkpoint(model_9, optimizer_9.iterations, optimizer_9)  

      checkpoint_managerG = try_restore_from_checkpoint(modelG, optimizerG.iterations, optimizerG)

    steps_per_loop_1 = checkpoint_steps_1
    def single_step_1(features, labels):
      with tf.GradientTape() as tape:
        should_record = tf.equal((optimizer.iterations + 1) % steps_per_loop_1, 0) 
        FLAGS.module1_train=True       
        hdd, fea = model(features, training=True)
        # flops(mod1)
        loss = None
        if hdd is not None:
          outputs = hdd          
          unsup_loss = obj_lib.add_usupervised_loss(fea, outputs)
          if loss is None:
            loss = unsup_loss
          else:
            loss += unsup_loss          
          metrics.update_finetune_metrics_train(unsupervised_loss_metric,
                                                unsupervised_acc_metric, loss, fea, outputs)
        total_loss_metric.update_state(loss)
        loss = loss / strategy.num_replicas_in_sync
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    

    def single_step(features0, labels0, features1, labels1, features2, labels2,
    features3, labels3,features4, labels4, features5, labels5, features6, labels6, features7, labels7, features8, labels8,
                features9, labels9):
      FLAGS.module1_train=False
      with tf.GradientTape() as tape:
        should_record = tf.equal((optimizerG.iterations + 1) % steps_per_loop_3, 0)
        with tf.summary.record_if(should_record):
          tf.summary.image('image', features0[:, :, :, :3], step=optimizerG.iterations + 1)

        rep0 = model_0(features0, training=False); rep1 = model_1(features1, training=False)
        rep2 = model_2(features2, training=False); 
        rep3 = model_3(features3, training=False)
        rep4 = model_4(features4, training=False); rep5 = model_5(features5, training=False)
        rep6 = model_6(features6, training=False); rep7 = model_7(features7, training=False)
        rep8 = model_8(features8, training=False); rep9 = model_9(features9, training=False)        
        rep=tf.concat([rep0, rep1,rep2,rep3, rep4, rep5,rep6,rep7, rep8, rep9], 0)

        projection_head_outputs, supervised_head_outputs = modelG(rep, training=True)
        # flops(model)
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
          l0 = labels0['labels'];l1 = labels1['labels'];l2 = labels2['labels'];l3 = labels3['labels']
          l4 = labels4['labels'];l5 = labels5['labels'];l6 = labels6['labels'];l7 = labels7['labels']
          l8 = labels8['labels'];l9 = labels9['labels']
          l = tf.concat([l0, l1, l2, l3, l4, l5, l6, l7, l8, l9], 0)
          if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            b=0
          sup_loss = obj_lib.add_supervised_loss(labels=l, logits=outputs)
          if loss is None:
            loss = sup_loss
          else:
            loss += sup_loss
          metrics.update_finetune_metrics_train(supervised_loss_metric,supervised_acc_metric, sup_loss,l, outputs)
        weight_decay = model_lib.add_weight_decay(modelG, adjust_per_optimizer=True)
        weight_decay_metric.update_state(weight_decay)
        loss += weight_decay
        total_loss_metric.update_state(loss)
        loss = loss / strategy.num_replicas_in_sync
        # logging.info('Trainable variables:')
        # for var in model.trainable_variables:
        #   logging.info(var.name)
        grads = tape.gradient(loss, modelG.trainable_variables)
        optimizerG.apply_gradients(zip(grads, modelG.trainable_variables))
    
  # baray Module sevom
    FLAGS.train_epochs=FLAGS.m2_epoch;FLAGS.train_batch_size=kept
    train_steps_3 = model_lib.get_train_steps(50000) 
    epoch_steps_3 = int(round(num_train_examples / FLAGS.train_batch_size))
    logging.info('# epoch_steps M3: %d', epoch_steps_3)
    logging.info('# train_steps M3: %d', train_steps_3)
    checkpoint_steps_3 = (FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * epoch_steps_3))    
    steps_per_loop_3 = checkpoint_steps_3

    for tek in range(1):
      avg=[]
      for m in range(FLAGS.numofclients):
        if m==0:
          iterator = iter(ds0);model=model_0; optimizer=optimizer_0; checkpoint_manager=checkpoint_manager_0
        if m==1:
          iterator = iter(ds1);model=model_1; optimizer=optimizer_1; checkpoint_manager=checkpoint_manager_1
        if m==2:
          iterator = iter(ds2);model=model_2; optimizer=optimizer_2; checkpoint_manager=checkpoint_manager_2
        if m==3:
          iterator = iter(ds3);model=model_3; optimizer=optimizer_3; checkpoint_manager=checkpoint_manager_3
        if m==4:
          iterator = iter(ds4);model=model_4; optimizer=optimizer_4; checkpoint_manager=checkpoint_manager_4
        if m==5:
          iterator = iter(ds5);model=model_5; optimizer=optimizer_5; checkpoint_manager=checkpoint_manager_5
        if m==6:
          iterator = iter(ds6);model=model_6; optimizer=optimizer_6; checkpoint_manager=checkpoint_manager_6
        if m==7:
          iterator = iter(ds7);model=model_7; optimizer=optimizer_7; checkpoint_manager=checkpoint_manager_7
        if m==8:
          iterator = iter(ds8);model=model_8; optimizer=optimizer_8; checkpoint_manager=checkpoint_manager_8
        if m==9:
          iterator = iter(ds9);model=model_9; optimizer=optimizer_9; checkpoint_manager=checkpoint_manager_9

        with strategy.scope():
          @tf.function
          def train_multiple_steps(iterator):
            # `tf.range` is needed so that this runs in a `tf.while_loop` and is not unrolled.
            for _ in tf.range(50):
              # Drop the "while" prefix created by tf.while_loop which otherwise gets prefixed to every variable name. 
              # This does not affect training but does affect the checkpoint conversion script. TODO(b/161712658): Remove this.
              with tf.name_scope(''):
                images, labels = next(iterator)
                features, labels = images, {'labels': labels}
                strategy.run(single_step_1, (features, labels))
          
          global_step = optimizer.iterations
          cur_step_3 = global_step.numpy()        
          # while cur_step_3 < train_steps_3:
          # Calls to tf.summary.xyz lookup the summary writer resource which is
          # set by the summary writer's context manager.
          with summary_writer.as_default():
            train_multiple_steps(iterator)
            cur_step_3 = global_step.numpy()
            checkpoint_manager.save(cur_step_3+m)
            logging.info('Completed: %d / %d steps', cur_step_3, train_steps_3)
            metrics.log_and_write_metrics_to_summary(all_metrics, cur_step_3)
            tf.summary.scalar('learning_rate',learning_rate(tf.cast(global_step, dtype=tf.float32)),global_step)
            summary_writer.flush()
          for metric in all_metrics:
            metric.reset_states()
        # logging.info('Training 1 complete...')
        avg=model.get_weights()  
      for m in range(FLAGS.numofclients):
        if m==0:
          fer=avg
        else:
          for rr in range(len(avg)):
            fer[rr]=tf.add(avg[rr],fer[rr])
      for lr in range(len(avg)):
        fer[lr]=(1/FLAGS.numofclients)*fer[lr]
      
      model_0.set_weights(fer);model_1.set_weights(fer);
      model_2.set_weights(fer);model_3.set_weights(fer);
      model_4.set_weights(fer);model_5.set_weights(fer);
      model_6.set_weights(fer);model_7.set_weights(fer); 
      model_8.set_weights(fer);model_9.set_weights(fer)


    with strategy.scope():
      @tf.function
      def train_multiple_steps(iterator0, iterator1, iterator2, 
      iterator3, iterator4, iterator5, iterator6,
      iterator7, iterator8, iterator9):
        images, labels = next(iterator0); features0, labels0 = images, {'labels': labels}
        images, labels = next(iterator1); features1, labels1 = images, {'labels': labels}
        images, labels = next(iterator2); features2, labels2 = images, {'labels': labels}
        images, labels = next(iterator3); features3, labels3 = images, {'labels': labels}
        images, labels = next(iterator4); features4, labels4 = images, {'labels': labels}
        images, labels = next(iterator5); features5, labels5 = images, {'labels': labels}
        images, labels = next(iterator6); features6, labels6 = images, {'labels': labels}
        images, labels = next(iterator7); features7, labels7 = images, {'labels': labels}
        images, labels = next(iterator8); features8, labels8 = images, {'labels': labels}
        images, labels = next(iterator9); features9, labels9 = images, {'labels': labels}
        for _ in tf.range(steps_per_loop_3):
          with tf.name_scope(''):
            strategy.run(single_step, (features0, labels0, features1, labels1, features2, labels2,
            features3, labels3,features4, labels4, features5, labels5, features6, labels6, features7, labels7, features8, 
            labels8,features9, labels9))
      
      global_step = optimizerG.iterations
      cur_step_3 = global_step.numpy()
      iterator0 = iter(dsc0); iterator1 = iter(dsc1);iterator2 = iter(dsc2)
      iterator3 = iter(dsc3); iterator4 = iter(dsc4); iterator5 = iter(dsc5);iterator6 = iter(dsc6); 
      iterator7 = iter(dsc7); iterator8 = iter(dsc8); iterator9 = iter(dsc9)
      
      while cur_step_3 < train_steps_3:
        with summary_writer.as_default():
          train_multiple_steps(iterator0, iterator1, iterator2, iterator3, iterator4, iterator5, iterator6,
            iterator7, iterator8, iterator9)
          cur_step_3 = global_step.numpy()
          checkpoint_manager.save(cur_step_3)
          logging.info('Completed: %d / %d steps', cur_step_3, train_steps_3)
          metrics.log_and_write_metrics_to_summary(all_metrics, cur_step_3)
          tf.summary.scalar('learning_rate',learning_rate(tf.cast(global_step, dtype=tf.float32)),global_step)
          summary_writer.flush()
        for metric in all_metrics:
          metric.reset_states()
      logging.info('Training 2 complete...')


    if FLAGS.mode == 'train_then_eval':
      perform_evaluation(modelG, model_1, builder, eval_steps,
                        checkpoint_manager.latest_checkpoint, strategy,topology)

if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.config.set_soft_device_placement(True)
  app.run(main)

