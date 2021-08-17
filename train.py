'''
# Author: Shaoran Lu
# Date: 2021/08/11
# Email: lushaoran92@gmail.com
# Description: 公式识别训练脚本

'''
# coding: utf-8

# In[1]:

import datetime
import time
import os
import tensorflow as tf

from loguru import logger

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

from model import Image2toLatex_Transformer

from datasetpipline import DataSetPipline

from tools import im2latex_input, get_optimizer, loss_function, create_masks
from cfg import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# In[2]:

logger.add('runtime.log')


root = './dataset/data_v1'

# latex100k数据集
train_img_dir, train_label_file, train_file_num = im2latex_input(root=root, split='train')
test_img_dir, test_label_file, test_file_num = im2latex_input(root=root, split='test')


logger.info(train_img_dir, train_label_file)
logger.info(test_img_dir, test_label_file)

voc_file = os.path.join(root, 'vocab.txt')

voc2id = {}
voc2id['START_TOKEN'] = 0
voc2id['PAD_TOKEN'] = 1
voc2id['END_TOKEN'] = 2
voc2id['UNK_TOKEN'] = 3

with open(voc_file, mode='r') as f:
    ff = f.readlines()
    for i, voc in enumerate(ff):
        voc = voc.strip()
        voc2id[voc] = i+4 
        
voc_size = len(voc2id)
logger.info(voc_size)

id2voc = {i:j for i,j in enumerate(voc2id)}


train_data = DataSetPipline(train_img_dir, train_label_file, voc_file, max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
                            image_size=IMAGE_SIZE, filter_size=FILTER_SIZE, resize=RESIZE, is_training=True, shuffle=False)

test_data = DataSetPipline(test_img_dir, test_label_file, voc_file, max_length=MAX_LENGTH, batch_size=1,
                            image_size=IMAGE_SIZE, filter_size=FILTER_SIZE, resize=RESIZE, is_training=False, shuffle=False)


decay_steps = EPOCHS * train_file_num // BATCH_SIZE

optimizer = get_optimizer(optimizer_name='Adam', lr=lr, lr_scheduler='CosineDecay', decay_steps=decay_steps)


if Finetune:
    checkpoint_dir = './checkpoints/2021-07-07'
else:
    checkpoint_dir = f'./checkpoints/{datetime.date.today()}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

# In[4]:

latex_model = Image2toLatex_Transformer(enc_units=UNITS, decoder_num_layers=decoder_num_layers, voc_size=voc_size,
                          max_length=MAX_LENGTH, d_model=DECODING_UNITS, num_heads=8, dff=1024, rate=0.2, use2dpe=True)

step = tf.Variable(1)
epoch = tf.Variable(1)
checkpoint = tf.train.Checkpoint(transformer=latex_model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                directory=checkpoint_dir,
                                                max_to_keep=55,
                                                step_counter=step,
                                                checkpoint_name='ckpt')

if Finetune:
    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
    status.expect_partial()
    logger.info("Restored from latest checkpoint {}".format(checkpoint_manager.latest_checkpoint))


@tf.function
def train_step(inp, targ):
    decoding_input = targ[ : , :-1 ] # Ignore <end> token
    real = targ[ : , 1: ]         # Ignore <start> token

    decoder_mask = create_masks(decoding_input)

    encoder_decoder_padding_mask = None

    with tf.GradientTape() as tape:
        logits, _ = latex_model(inp, decoding_input, decoder_mask, True,
                                encoder_decoder_padding_mask)
        
        loss = loss_function(real, logits)
        
    gradients = tape.gradient(loss, latex_model.trainable_variables)
#     gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
#     gradients = [gradient if gradient is not None else tf.zeros_like(var) for var, gradient in zip(variables, gradients)]
    optimizer.apply_gradients(zip(gradients, latex_model.trainable_variables))

    return loss

for ep in range(EPOCHS):
    start = time.time()
    total_loss = 0.
    visual_loss = 0.

    for (batch, (inp, targ)) in enumerate(train_data):
        batch_loss = train_step(inp, targ)
        total_loss += batch_loss
        visual_loss += batch_loss
        step.assign_add(1)
        
        if batch % 100 == 0 and batch != 0:
            
            logger.info('Epoch {} Batch {} Loss {:.4f}'.format(ep + 1, batch, visual_loss.numpy() / 100))
            visual_loss = 0.

    logger.info('Epoch {} Loss {:.4f}'.format(ep + 1, total_loss.numpy() / (batch+1)))
    saved_path = checkpoint_manager.save()
    if saved_path:
        logger.info("Saved checkpoint for {}: {}".format(int(step.numpy()), saved_path))

    epoch.assign_add(1)


