
# coding: utf-8


BATCH_SIZE = 4
EPOCHS = 100
lr = 0.0001
MAX_LENGTH = 160
Finetune = False

UNITS = 512
DECODING_UNITS = UNITS

EM = 512
decoder_num_layers = 2

IMAGE_SIZE = (42, 640)
RESIZE = True
FILTER_SIZE = True

print('*** lr ', lr)
print('*** batchsize ', BATCH_SIZE)
print('*** EPOCHS ', EPOCHS)
print('*** MAX_LENGTH ', MAX_LENGTH)
print('*** Finetune ', Finetune)
print('*** UNITS ', UNITS)
print('*** DECODING_UNITS ', DECODING_UNITS)
print('*** EM ', EM)
print('*** IMAGE_SIZE ', IMAGE_SIZE)
print('*** RESIZE ', RESIZE)
print('*** FILTER_SIZE ', FILTER_SIZE)