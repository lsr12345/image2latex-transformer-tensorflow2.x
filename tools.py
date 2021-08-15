
# coding: utf-8

import os
import tensorflow as tf
from loguru import logger

def im2latex_input(root, split='train'):
    assert split in ['train', 'test', 'val'], 'check split name'
    norm_txt = split + '.formulas.norm.txt'
    match_txt = split + '.matching.txt'
    images_dir = os.path.join(root, 'images_'+split)
    if not os.path.exists(os.path.join(root, split + '.matching.formulas.norm.txt')):
        img_list = os.listdir(images_dir)

        norm_labels = []
        with open (os.path.join(root, norm_txt), mode='r') as fr:
            ff = fr.readlines()
            for line in ff:
                line = line.strip()
                norm_labels.append(line)

        match_labels = []
        with open (os.path.join(root, match_txt), mode='r') as fr:
            ff = fr.readlines()
            for line in ff:
                line = line.strip()
                name, index = line.split(' ')
                if name in img_list:
                    match_labels.append([name, norm_labels[int(index)]])
        logger.info('{} dataset nums: {}'.format(split, str(len(match_labels))))
        with open(os.path.join(root, split + '.matching.formulas.norm.txt'), mode='w',encoding='utf-8') as fw:
            for i, label in enumerate(match_labels):
                if i < len(match_labels)-1:
                    fw.write(' '.join(label)+'\n')
                else:
                    fw.write(' '.join(label))
        
        return images_dir, os.path.join(root, split + '.matching.formulas.norm.txt'), i+1
        
    else:
        with open(os.path.join(root, split + '.matching.formulas.norm.txt'), mode='r',encoding='utf-8') as fr:
            ff = fr.readlines()
            i = len(ff)
        return images_dir, os.path.join(root, split + '.matching.formulas.norm.txt'), i


def get_optimizer(optimizer_name='Adam', lr=0.001, lr_scheduler='CosineDecay', decay_steps=1000):

    if lr_scheduler == 'ExponentialDecay':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=0.96)
            
    elif lr_scheduler == 'CosineDecay':
        lr = tf.keras.experimental.CosineDecay(
                initial_learning_rate=lr, decay_steps=decay_steps)
    else:
        lr = lr

    if optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                             epsilon=1e-9,
                                             beta_1=0.9,
                                             beta_2=0.98)
    elif optimizer_name == 'SGD':
        optimizer = tf.optimizers.SGD(learning_rate=lr, momentum=0.9)

    else:
        raise RuntimeError('Optimizer {} is not supported.'.format(optimizer_name))

    return optimizer


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 1))
    loss_ = loss_object(real, pred)
    loss_ = tf.boolean_mask(loss_, mask)
    loss_ = tf.reduce_mean(loss_)
    
    return loss_

# batch_data.shape: [batch_size, seq_len]
def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 1), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]
    
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask # (seq_len, seq_len)
    
def create_masks(tar):
    """
    Encoder:
      - encoder_padding_mask (self attention of EncoderLayer)
    Decoder:
      - look_ahead_mask (self attention of DecoderLayer)
      - encoder_decoder_padding_mask (encoder-decoder attention of DecoderLayer)
      - decoder_padding_mask (self attention of DecoderLayer)
    """    
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decoder_padding_mask = create_padding_mask(tar)
    decoder_mask = tf.maximum(decoder_padding_mask,
                              look_ahead_mask)
    
    return decoder_mask
