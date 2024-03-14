import json
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

from Wtrain import autokhodamClass
from Strain import ClassikhodamClass
from Server import Ser

FLAGS = flags.FLAGS

# flags.DEFINE_integer('num_cli', 10,'Number of clients.')
flags.DEFINE_string('ModeAuto', 'TrainAuto', ['TrainAuto','TestAuto'])
flags.DEFINE_string('auo', '1', ['1','2'])

flags.DEFINE_float('learning_rate', 0.3,'Initial learning rate per batch size of 256.')
flags.DEFINE_enum('learning_rate_scaling', 'linear', ['linear', 'sqrt'],
    'How to scale the learning rate as a function of batch size.')
flags.DEFINE_integer('serBatch', 32,'Server Batch')
flags.DEFINE_integer('tsneBatch', 400,'TSNE Batch')
flags.DEFINE_integer('AutoBatch', 64,'Auto Batch')
flags.DEFINE_integer('testBatch', 32,'Test Batch')
flags.DEFINE_string('norm', 'PN','Test Batch')
flags.DEFINE_integer('CPC', 1,'Test Batch')

flags.DEFINE_float('warmup_epochs', 10,'Number of epochs of warmup.')
flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')
flags.DEFINE_float('batch_norm_decay', 0.9,'Batch norm decay parameter.')
flags.DEFINE_integer('train_batch_size', 256,'Batch size for training.')
flags.DEFINE_string('train_split', 'train','Split for training.')
flags.DEFINE_integer('train_epochs', 100,'Number of epochs to train for.')
flags.DEFINE_integer('train_steps', 0,'Number of steps to train for. If provided, overrides train_epochs.')
flags.DEFINE_integer('eval_steps', 0,'Number of steps to eval for. If not provided, evals over entire dataset.')
flags.DEFINE_integer('eval_batch_size', 256,'Batch size for eval.')
flags.DEFINE_integer('checkpoint_epochs', 1,'Number of epochs between checkpoints/summaries.')
flags.DEFINE_integer('checkpoint_steps', 0,'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_string('eval_split', 'validation','Split for evaluation.')
flags.DEFINE_string('dataset', 'imagenet2012','Name of a dataset.')

flags.DEFINE_bool('cache_dataset', False,'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can improve performance.')

flags.DEFINE_enum('mode', 'train_then_eval', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum('train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_bool('lineareval_while_pretraining', True, 'Whether to finetune supervised head while pretraining.')

flags.DEFINE_string('checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')

flags.DEFINE_bool('zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer('fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning just the linear head.')

flags.DEFINE_string('model_dir', None,'Model directory for training.')
flags.DEFINE_string('data_dir', None,'Directory where dataset is stored.')
flags.DEFINE_enum('optimizer', 'lars', ['momentum', 'adam', 'lars'],'Optimizer to use.')
flags.DEFINE_float('momentum', 0.9,'Momentum parameter.')
flags.DEFINE_string('eval_name', None,'Name for eval.')
flags.DEFINE_integer('keep_checkpoint_max', 5,'Maximum number of checkpoints to keep.')
flags.DEFINE_integer('keep_hub_module_max', 1,'Maximum number of Hub modules to keep.')
flags.DEFINE_float('temperature', 0.1,'Temperature parameter for contrastive loss.')
flags.DEFINE_boolean('hidden_norm', True,'Temperature parameter for contrastive loss.')

flags.DEFINE_enum('proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer('proj_out_dim', 128,'Number of head projection dimension.')
flags.DEFINE_integer('num_proj_layers', 3,'Number of non-linear head layers.')

flags.DEFINE_integer('ft_proj_selector', 0,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')

flags.DEFINE_boolean('global_bn', True,'Whether to aggregate BN statistics across distributed cores.')
flags.DEFINE_integer('width_multiplier', 1,'Multiplier to change width of network.')
flags.DEFINE_integer('resnet_depth', 50,'Depth of ResNet.')
flags.DEFINE_float('sk_ratio', 0.,'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')
flags.DEFINE_float('se_ratio', 0.,'If it is bigger than 0, it will enable SE.')
flags.DEFINE_integer('image_size', 224,'Input image size.')
flags.DEFINE_float('color_jitter_strength', 1.0,'The strength of color jittering.')
flags.DEFINE_boolean('use_blur', True,'Whether or not to use Gaussian blur for augmentation during pretraining.')

flags.DEFINE_integer('tekrar', 1,'Number of itirarion on main loop.')
flags.DEFINE_integer('NumofWorkers', 1,'Number of workers.')
flags.DEFINE_integer('epochsm', 2000,'Number of iterationsw to train Autoencoder.')

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  WOA=[]
  S=Ser()
  for m in range(FLAGS.NumofWorkers):
    WOA.append(autokhodamClass(m))  
  iti=0; m=0
  SOA = ClassikhodamClass(m)

  for iti in range(FLAGS.tekrar):
    for m in range(FLAGS.NumofWorkers):
      WOA[m].encoderkhodam(iti)
      S.addList(WOA[m], iti) 
    S.devid()
    for m in range(FLAGS.NumofWorkers):
      S.broad(WOA[m])
  SOA.classiKhodam(WOA, iti)

if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.config.set_soft_device_placement(True)
  app.run(main)


