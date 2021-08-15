
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras


def get_angles(pos, i, d_model):
    angle_rates = tf.cast(1, tf.float32) / tf.pow(tf.cast(10000, tf.float32), tf.cast((2 * (i // 2)) / d_model, tf.float32))
    return tf.cast(pos, tf.float32) * angle_rates

def get_position_embedding(sentence_length, d_model, bs):
    angle_rads = get_angles(tf.expand_dims(tf.range(sentence_length), axis=-1),
                            tf.expand_dims(tf.range(d_model), axis=0),
                            d_model)

    sines = tf.math.sin(angle_rads[:, 0::2])

    sines = tf.reshape(sines, shape=(sines.shape[0], -1, 1))
    cosines = tf.math.cos(angle_rads[:, 1::2])

    cosines = tf.reshape(cosines, shape=(cosines.shape[0], -1, 1))
    position_embedding = tf.concat([sines, cosines], axis=-1)
    position_embedding = tf.reshape(position_embedding, shape=(position_embedding.shape[0], -1))
    position_embedding = tf.expand_dims(position_embedding, axis=0)
    position_embedding = tf.repeat(position_embedding, repeats=[bs], axis=0)
    return tf.cast(position_embedding, dtype=tf.float32)

def get_2Dposition_embedding(d_model, bs, max_h, max_w):
    # b, max_h, d_model//2
    pe_h = get_position_embedding(max_h, d_model // 2, bs)
    # b, max_h, 1, d_model//2
    pe_h = tf.expand_dims(pe_h, axis=-2)
    # b, max_h, max_w, d_model//2
    pe_h = tf.repeat(pe_h, repeats=[max_w], axis=-2)
    
    # b, max_w, d_model//2
    pe_w = get_position_embedding(max_w, d_model // 2, bs)
    # b, 1, max_w, d_model//2
    pe_w = tf.expand_dims(pe_w, axis=-3)
    # b, max_h, max_w, d_model//2
    pe_w = tf.repeat(pe_w, repeats=[max_h], axis=-3)
    
    position_embedding2d = tf.concat([pe_h, pe_w], axis=-1)
    
    return position_embedding2d
    

class GolbalContextBlock(tf.keras.layers.Layer):

    def __init__(self,
                 inplanes,
                 ratio,
                 headers,
                 pooling_type='att',
                 att_scale=False,
                 fusion_type='channel_add',
                 **kwargs):
        """

        Args:
            inplanes:
            ratio:
            headers:
            pooling_type:
            att_scale:
            fusion_type:
            **kwargs:
        """

        super().__init__(name='GCB', **kwargs)
        assert pooling_type in ['att', 'avg']
        assert fusion_type in ['channel_add', 'channel_concat', 'channel_mul']
        assert inplanes % headers == 0 and inplanes >= 8

        self.headers = headers
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = att_scale

        self.single_header_inplanes = int(inplanes / headers)

        if self.pooling_type == 'att':
            self.conv_mask = tf.keras.layers.Conv2D(1,
                                                    kernel_size=1,
                                                    kernel_initializer=tf.initializers.he_normal())
        else:
            self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=1)

        if self.fusion_type == 'channel_add':
            self.channel_add_conv = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(self.planes, kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                    tf.keras.layers.LayerNormalization([1, 2, 3]),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(self.inplanes, kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                ],
                name='channel_add_conv'
            )
        elif self.fusion_type == 'channel_concat':
            self.channel_concat_conv = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(self.planes,
                                           kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                    tf.keras.layers.LayerNormalization([1,2,3]),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(self.inplanes,
                                           kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                ],
                name='channel_concat_conv'
            )
            self.cat_conv = tf.keras.layers.Conv2D(self.inplanes,
                                                   kernel_size=1,
                                                   kernel_initializer=tf.initializers.he_normal())
            self.layer_norm = tf.keras.layers.LayerNormalization(axis=[1,2,3])
        else:
            self.channel_mul_conv = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(self.planes,
                                           kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                    tf.keras.layers.LayerNormalization([1, 2, 3]),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(self.inplanes,
                                           kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                ],
                name='channel_mul_conv'
            )

#     @tf.function
    def spatial_pool(self, inputs: tf.Tensor):
        B = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        C = tf.shape(inputs)[3]

        if self.pooling_type == 'att':

            # B, H, W, h, C/h
            x = tf.reshape(inputs, shape=(B, H, W, self.headers, self.single_header_inplanes))
            # B, h, H, W, C/h
            x = tf.transpose(x, perm=(0, 3, 1, 2, 4))

            # B*h, H, W, C/h
            x = tf.reshape(x, shape=(B*self.headers, H, W, self.single_header_inplanes))

            input_x = x

            # B*h, 1, H*W, C/h
            input_x = tf.reshape(input_x, shape=(B*self.headers, 1, H*W, self.single_header_inplanes))
            # B*h, 1, C/h, H*W
            input_x = tf.transpose(input_x, perm=[0, 1, 3, 2])

            # B*h, H, W, 1,
            context_mask = self.conv_mask(x)

            # B*h, 1, H*W, 1
            context_mask = tf.reshape(context_mask, shape=(B*self.headers, 1, H*W, 1))

            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask = context_mask / tf.sqrt(self.single_header_inplanes)

            # B*h, 1, H*W, 1
            context_mask = tf.keras.activations.softmax(context_mask, axis=2)

            # B*h, 1, C/h, 1
            context = tf.matmul(input_x, context_mask)
            context = tf.reshape(context, shape=(B, 1, C, 1))

            # B, 1, 1, C
            context = tf.transpose(context, perm=(0, 1, 3, 2))
        else:
            # B, 1, 1, C
            context = self.avg_pool(inputs)

        return context


    def call(self, inputs, **kwargs):
        # B, 1, 1, C
        context = self.spatial_pool(inputs)
        # print('context: ', context.shape)
        out = inputs
        if self.fusion_type == 'channel_mul':
            channel_mul_term = tf.sigmoid(self.channel_mul_conv(context))
            out = channel_mul_term * out
        elif self.fusion_type == 'channel_add':
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # B, 1, 1, C
            channel_concat_term = self.channel_concat_conv(context)


            B = tf.shape(out)[0]
            H = tf.shape(out)[1]
            W = tf.shape(out)[2]
            C = tf.shape(out)[3]
            out = tf.concat([out, tf.broadcast_to(channel_concat_term, shape=(B, H, W, C))], axis=-1)
            out = self.cat_conv(out)
            out = self.layer_norm(out)
            out = tf.keras.activations.relu(out)

        return out
    
def conv33(out_planes, stride=1):
    return keras.layers.Conv2D(out_planes, kernel_size=3, strides=stride, padding='same', use_bias=False)

class BasicBlock(keras.layers.Layer):
    expansion = 1

    def __init__(self,
                 planes,
                 stride=1,
                 downsample=None,
                 gcb_config=None,
                 use_gcb=None,  **kwargs):
        super().__init__(**kwargs)   # name='BasciBlock',
        self.conv1 = conv33(planes, stride)
        self.bn1 = keras.layers.BatchNormalization(momentum=0.1,
                                                   epsilon=1e-5)
        self.relu = keras.layers.ReLU()
        self.conv2 = conv33(planes, stride)
        self.bn2 = keras.layers.BatchNormalization(momentum=0.1,
                                                   epsilon=1e-5)
        if downsample:
            self.downsample = downsample
        else:
            self.downsample = tf.identity

        self.stride = stride

        if use_gcb:
            self.gcb = GolbalContextBlock(
                inplanes=planes,
                ratio=gcb_config['ratio'],
                headers=gcb_config['headers'],
                pooling_type=gcb_config['pooling_type'],
                fusion_type=gcb_config['fusion_type'],
                att_scale=gcb_config['att_scale']
            )
        else:
            self.gcb = tf.identity

    def call(self, inputs, **kwargs):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out = self.gcb(out)
        out = out + self.downsample(inputs)

        out = self.relu(out)
        return out
    
class Encoder_Resnet31(keras.layers.Layer):
    def __init__(self, block=BasicBlock, enc_units=512, backbone_config= {'gcb':{"ratio": 0.0625, "headers": 1, "att_scale": True,
                                                       "fusion_type": 'channel_add', "pooling_type": 'att', "layers":[False,True,True,True]}}, use2dpe=True):
        super(Encoder_Resnet31, self).__init__(name='Encoder_ResNet31')
        layers = [1, 2, 5, 3]
        gcb_config = backbone_config['gcb']
        gcb_enabling = gcb_config['layers']
        self.use2dpe = use2dpe
        self.enc_units = enc_units
        
        self.inplanes = 128
        self.conv1 = keras.layers.Conv2D(64,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu1 = keras.layers.ReLU()


        self.conv2 = keras.layers.Conv2D(128,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu2 = keras.layers.ReLU()
        self.maxpool1 = keras.layers.MaxPool2D(strides=2)
        self.layer1 = self._make_layer(block,
                                       256,
                                       layers[0],
                                       stride=1,
                                       use_gcb=gcb_enabling[0],
                                       gcb_config=gcb_config)

        self.conv3 = keras.layers.Conv2D(256,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn3 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu3 = keras.layers.ReLU()
        self.maxpool2 = keras.layers.MaxPool2D(strides=2)
        self.layer2 = self._make_layer(block,
                                       256,
                                       layers[1],
                                       stride=1,
                                       use_gcb=gcb_enabling[1],
                                       gcb_config=gcb_config)


        self.conv4 = keras.layers.Conv2D(256,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn4 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu4 = keras.layers.ReLU()
        self.maxpool3 = keras.layers.MaxPool2D(pool_size=(2,1), strides=(2,1))
        self.layer3 = self._make_layer(block,
                                       512,
                                       layers[2],
                                       stride=1,
                                       use_gcb=gcb_enabling[2],
                                       gcb_config=gcb_config)


        self.conv5 = keras.layers.Conv2D(512,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn5 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu5 = keras.layers.ReLU()
        self.maxpool4 = keras.layers.MaxPool2D(pool_size=(1,2), strides=(1,2))
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=1,
                                       use_gcb=gcb_enabling[3],
                                       gcb_config=gcb_config)


        self.conv6 = keras.layers.Conv2D(512,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn6 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu6 = keras.layers.ReLU()
        self.maxpool5 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        self.reshape = tf.keras.layers.Reshape((-1, 512))
        self.pe = tf.keras.layers.Lambda(lambda x: get_position_embedding(x[0], x[1], x[2]))
        self.pe2d = tf.keras.layers.Lambda(lambda x: get_2Dposition_embedding(x[0], x[1], x[2], x[3]))
        self.add = tf.keras.layers.Add()
        
    def _make_layer(self, block, planes, blocks, stride=1, gcb_config=None, use_gcb=False):
        downsample =None
        if stride!=1 or self.inplanes != planes * block.expansion:

            downsample = keras.Sequential(
                [keras.layers.Conv2D(planes * block.expansion,
                                    kernel_size=(1,1),
                                    strides=stride,
                                    use_bias=False,
                                    kernel_initializer=keras.initializers.he_normal()),
                keras.layers.BatchNormalization(momentum=0.9,
                                                epsilon=1e-5)]
                # name='downsample'
            )
        layers = []
        layers.append(block(planes, stride, downsample, gcb_config, use_gcb))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(planes))

        return keras.Sequential(layers) # , name='make_layer'


    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.layer3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.maxpool4(x)
        x = self.layer4(x)
        
        
        x = self.maxpool5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        if not self.use2dpe:
            x = self.reshape(x)
            # print("x.shape ", x.shape)
            pe = self.pe([x.shape[-2], self.enc_units, x.shape[0]])
            x = self.add([pe,x])
        else:
            pe = self.pe2d([self.enc_units, x.shape[0], x.shape[1], x.shape[2]])
            x = self.add([pe,x])
            x = self.reshape(x)
        return x


def scaled_dot_product_attention(q, k, v, mask):

    # matmul_qk.shape: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b = True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        # 使得在softmax后值趋近于0
        scaled_attention_logits += (mask * -1e9)
    
    # attention_weights.shape: (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis = -1)
    
    # output.shape: (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights

class MultiHeadAttention(keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0
        
        self.depth = self.d_model // self.num_heads
        
        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)
        
        self.dense = keras.layers.Dense(self.d_model)
    
    def split_heads(self, x, batch_size):
        # x.shape: (batch_size, seq_len, d_model)
        # d_model = num_heads * depth
        # x -> (batch_size, num_heads, seq_len, depth)
        
        x = tf.reshape(x,
                       (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.WQ(q) # q.shape: (batch_size, seq_len_q, d_model)
        k = self.WK(k) # k.shape: (batch_size, seq_len_k, d_model)
        v = self.WV(v) # v.shape: (batch_size, seq_len_v, d_model)
        
        # q.shape: (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # k.shape: (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # v.shape: (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)
        
        # scaled_attention_outputs.shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_outputs, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # scaled_attention_outputs.shape: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs, perm = [0, 2, 1, 3])
        # concat_attention.shape: (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention_outputs,
                                      (batch_size, -1, self.d_model))
        
        # output.shape : (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
    
def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])

class DecoderLayer(keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate = 0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = feed_forward_network(d_model, dff)
        
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        self.layer_norm3 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)
    
    
    def call(self, x, encoding_outputs, training,
             decoder_mask, encoder_decoder_padding_mask):
        # decoder_mask: 由look_ahead_mask和decoder_padding_mask合并而来
        
        # x.shape: (batch_size, target_seq_len, d_model)
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)
        
        # attn1, out1.shape : (batch_size, target_seq_len, d_model)
        attn1, attn_weights1 = self.mha1(x, x, x, decoder_mask)
        attn1 = self.dropout1(attn1, training = training)
        out1 = self.layer_norm1(attn1 + x)
        
        # attn2, out2.shape : (batch_size, target_seq_len, d_model)
        attn2, attn_weights2 = self.mha2(
            out1, encoding_outputs, encoding_outputs,
            encoder_decoder_padding_mask)
        attn2 = self.dropout2(attn2, training = training)
        out2 = self.layer_norm2(attn2 + out1)
        
        # ffn_output, out3.shape: (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layer_norm3(ffn_output + out2)
        
        return out3, attn_weights1, attn_weights2
    
    
class Decoder_Transformer(keras.layers.Layer):
    def __init__(self, num_layers, target_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(Decoder_Transformer, self).__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model
        
        self.embedding = keras.layers.Embedding(target_vocab_size,
                                                d_model)
        # self.position_embedding = get_position_embedding(max_length,
        #                                                  d_model)
        
        self.position_embedding = tf.keras.layers.Lambda(lambda x: get_position_embedding(x[0], x[1], x[2]))
        self.add = tf.keras.layers.Add()
        
        # self.position_embedding = keras.layers.Lambda(lambda x: get_position_embedding(x[0], x[1], x[2]))
        
        self.dropout = keras.layers.Dropout(rate)
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.num_layers)]
        
    
    def call(self, x, encoding_outputs,
             decoder_mask, encoder_decoder_padding_mask, training):
        # x.shape: (batch_size, output_seq_len)
        output_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            output_seq_len, self.max_length,
            "output_seq_len should be less or equal to self.max_length")
        
        attention_weights = {}
        
        # x.shape: (batch_size, output_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        pe = self.position_embedding([x.shape[-2], self.d_model, x.shape[0]])
        x = self.add([pe,x])

        x = self.dropout(x, training = training)
        
        for i in range(self.num_layers):
            x, attn1, attn2 = self.decoder_layers[i](
                x, encoding_outputs, training,
                decoder_mask, encoder_decoder_padding_mask)
            attention_weights[
                'decoder_layer{}_att1'.format(i+1)] = attn1
            attention_weights[
                'decoder_layer{}_att2'.format(i+1)] = attn2
        # x.shape: (batch_size, output_seq_len, d_model)
        return x, attention_weights


class Image2toLatex_Transformer(tf.keras.Model):
    def __init__(self, enc_units, decoder_num_layers,
                 voc_size, max_length, d_model, num_heads=8, dff=1024, rate=0.1, use2dpe=True):
        super(Image2toLatex_Transformer, self).__init__()

        self.encoder_model = Encoder_Resnet31(enc_units=enc_units, use2dpe=use2dpe)

        self.decoder_model = Decoder_Transformer(num_layers=decoder_num_layers,
                                                 target_vocab_size=voc_size, max_length=max_length,
                                                 d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)

        self.final_layer = tf.keras.layers.Dense(voc_size)

    def call(self, inp, tar, decoder_mask, training, encoder_decoder_padding_mask=None):
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)
        encoding_outputs = self.encoder_model(inp)

        # decoding_outputs.shape: (batch_size, output_seq_len, d_model)
        decoding_outputs, attention_weights = self.decoder_model(
            tar, encoding_outputs,
            decoder_mask, encoder_decoder_padding_mask, training)

        # predictions.shape: (batch_size, output_seq_len, target_vocab_size)
        predictions = self.final_layer(decoding_outputs)

        return predictions, attention_weights

