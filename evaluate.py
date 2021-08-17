'''
# Author: Shaoran Lu
# Date: 2021/08/11
# Email: lushaoran92@gmail.com
# Description: demo

'''

import tensorflow as tf
import cv2
from model import Image2toLatex_Transformer
from datasetpipline import process_resize
from cfg import *
from tools import create_masks

checkpoint_dir = './ckpt/ckpt-74'

voc_file = './dataset/vocab.txt'

voc2id = {}
voc2id['START_TOKEN'] = 0
voc2id['PAD_TOKEN'] = 1
voc2id['END_TOKEN'] = 2
voc2id['UNK_TOKEN'] = 3

with open(voc_file, mode='r') as f:
    ff = f.readlines()
    for i, voc in enumerate(ff):
        voc = voc.strip()
        voc2id[voc] = i + 4

voc_size = len(voc2id)

id2voc = {i: j for i, j in enumerate(voc2id)}


transformer = Image2toLatex_Transformer(enc_units=UNITS, decoder_num_layers=decoder_num_layers, voc_size=voc_size,
                          max_length=MAX_LENGTH, d_model=DECODING_UNITS, num_heads=8, dff=1024, rate=0.2, use2dpe=False)

checkpoint = tf.train.Checkpoint(transformer=transformer)
checkpoint.restore(checkpoint_dir)


def evaluate(inp_img, max_length=160):
    encoder_input = tf.expand_dims(inp_img, 0)

    decoder_input = tf.expand_dims([voc2id['START_TOKEN']], 0)

    for i in range(max_length):
        training = True if i == 0 else False
        decoder_mask = create_masks(decoder_input)
        encoder_decoder_padding_mask = None

        predictions, attention_weights, encoder_input = transformer(encoder_input, decoder_input, decoder_mask,
                                                                    training, encoder_decoder_padding_mask)

        predictions = predictions[:, -1, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1),
                               tf.int32)

        if tf.equal(predicted_id, voc2id['END_TOKEN']):
            return tf.squeeze(decoder_input, axis=0), attention_weights

        decoder_input = tf.concat([decoder_input, [predicted_id]],
                                  axis=-1)
    return tf.squeeze(decoder_input, axis=0), attention_weights


demo_path = './demo.png'
img = cv2.imread(demo_path)
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
image, flag = process_resize(img, target_size=IMAGE_SIZE)
pred, _ = evaluate(image, max_length=160)
res = ' '.join([id2voc[i] for i in pred.numpy()][1:])
print(res)
