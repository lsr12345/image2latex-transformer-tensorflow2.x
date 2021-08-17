# coding: utf-8
'''
# Author: Shaoran Lu
# Date: 2021/08/11
# Email: lushaoran92@gmail.com
# Description: 数据pipline

'''

import tensorflow as tf
from tensorflow.data import Dataset as tfd

import numpy as np
import cv2
import os 

from imgaug import augmenters as iaa

aug = iaa.SomeOf((0, 6),
    [
    iaa.Affine(scale=[0.9, 1.0], cval=255, mode='constant'),  # mode='edge'
    iaa.Affine(rotate=(-1,1), cval=255, mode='constant'),  # 旋转增强器
    iaa.GaussianBlur(sigma=(0.0,0.5)),  # 高斯模糊增强器
    iaa.AdditiveGaussianNoise(scale=(0, 0.001*255)),  # 高斯噪声增强器
    iaa.JpegCompression(compression=[0, 10]),
    iaa.PiecewiseAffine(scale=(0.004, 0.006))  # 扭曲增强器
])

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3

STD = 1.0
MEAN = 0.0

def augment_func(image):
    image = image.astype(np.uint8)
    image = aug.augment_image(image)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=-1)
    return image

def tf_augment_func(image, label):
    [image,] = tf.numpy_function(augment_func, [image], [tf.float32])
    return image, label


def process_resize(img, target_size=(56, 512)):
    h, w = img.shape
    scale = target_size[0] / h
    t_img = cv2.resize(img, (int(w * scale), int(h * scale)))
    if t_img.shape[1] > target_size[1]:
        return img, False
    else:
        img_mask = np.full((target_size[0], target_size[1]), fill_value=255, dtype=np.float32)
        img_mask[:t_img.shape[0], :t_img.shape[1]] = t_img
        return img_mask, True

def process_resize_(img, target_size=(56, 512)):
    h, w = img.shape
    scale = target_size[0 ] / h
    t_img = cv2.resize(img, (int(w*scale//64*64),int( h*scale)))
    return t_img, True

# In[4]:

def yield_func(img_dir, label_file, voc_file, max_length=160, image_size=(56, 512), filter_size=True, resize=True, is_training=True, shuffle=True, padsize=True):
    
    img_dir = bytes.decode(img_dir, encoding='utf-8')
    label_file = bytes.decode(label_file, encoding='utf-8')
    
    img_lists = []
    label_lists = []
    with open(label_file, mode='r') as fr:
        ff  = fr.readlines()
        for line in ff:
            line = line.strip()
            line = line.split(' ')
            img_name = line[0]
            if not img_name.endswith('g'):
                img_name = img_name[:-1]
            img_lists.append(os.path.join(str(img_dir), img_name))
    #         print(line)
            label_lists.append(line[1:])
    
    print('imgs num & labels num: {}-{}'.format(len(img_lists), len(label_lists)))
    assert len(img_lists) == len(label_lists), 'train_labels != train_images'
    assert len(img_lists) != 0, 'No file exists'

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
    index = 0
    assert len(img_lists)==len(label_lists), 'check the inputs length!'
    stop_nums = len(img_lists)
    while index < stop_nums:
        image = cv2.imread(img_lists[index], cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        if filter_size and not resize:
            if w > image_size[1] or h > image_size[0]:
                index += 1
                continue

        if resize:
            if padsize:
                image, flag = process_resize(image, target_size=image_size)
                if not flag:
                    index += 1
                    continue
            else:
                try:
                    image, _  = process_resize_(image, target_size=image_size)
                    h, w = image.shape
                except:
                    index += 1
                    continue

        image = np.rot90(image, 3)
        # image = (image/255. - MEAN) / STD
            
        label_mask = np.full(shape=(max_length), fill_value=voc2id['PAD_TOKEN'], dtype=int)
        label = label_lists[index]
        label = [voc2id.get(i, voc2id['UNK_TOKEN']) for i in label]
        label.insert(0,voc2id['START_TOKEN'])
        label.append(voc2id['END_TOKEN'])
        label_len = len(label) if len(label) < max_length-1 else max_length-1
        label_mask[:label_len] = label[:label_len]
        index += 1
        
        yield image, label_mask
        
def DataSetPipline(img_dir, label_file, voc_file, max_length=160, batch_size=1, image_size=(56,512), filter_size=True, resize=True,
                   is_training=True, shuffle=True, padsize=True):
    dataset = tfd.from_generator(yield_func,
                                 output_types=(tf.float32, tf.int16),
                                 output_shapes=((tf.TensorShape([None, None]),
                                                 tf.TensorShape([max_length]))),
                                 args=(img_dir, label_file, voc_file, max_length, image_size, filter_size, resize,
                                       is_training, shuffle, padsize)
                                )
    
    if is_training:
        dataset = dataset.map(tf_augment_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
        
    return dataset.batch(batch_size, drop_remainder=True)


# In[ ]:


if __name__ == '__main__':
    voc_file = '/data/small_vocab.txt'

    img_dir = '/data/images_train/'
    label_file = '/data/train.formulas.norm.txt'


    dataset = DataSetPipline(img_dir, label_file, voc_file, max_length=160, batch_size=1, image_size=(56,336),
                             resize=False, is_training=True, shuffle=True)

    dataset_test = dataset.take(5)
    print(len(list(dataset_test.as_numpy_iterator())))
    print(np.array(list(dataset_test.as_numpy_iterator())[0][0]).shape)
    print(np.array(list(dataset_test.as_numpy_iterator())[0][1]).shape)
    print('...')
    print(np.array(list(dataset_test.as_numpy_iterator())[1][0]).shape)
    print(np.array(list(dataset_test.as_numpy_iterator())[1][1]).shape)
    print('...')
    print(np.array(list(dataset_test.as_numpy_iterator())[2][0]).shape)
    print(np.array(list(dataset_test.as_numpy_iterator())[2][1]).shape)
    print('...')
    print(np.array(list(dataset_test.as_numpy_iterator())[3][0]).shape)
    print(np.array(list(dataset_test.as_numpy_iterator())[3][1]).shape)

