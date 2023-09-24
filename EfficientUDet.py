import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend
from keras import datasets, layers, models, initializers, activations
from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate
from functools import reduce


from skimage import measure
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter

print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.compat.v1.config.experimental.list_physical_devices("GPU") else "OFF" }')


import numpy as np
import matplotlib, random
import matplotlib.pyplot as plt
import pandas as pd 
import cv2 as cv
import os

from imutils import paths
import pathlib
import time
import glob
from natsort import natsorted
import tarfile
import tqdm


############ Backbone EfficientNet-B7 ############ 

from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
# from _EfficientNet_B7 import EfficientNetB7 

img_size = 640
#EfficientNet0, EfficientNet1, EfficientNet2, EfficientNet3, EfficientNet4, EfficientNet5, EfficientNet6, EfficientNet7
w_bifpns = [64, 88, 112, 160, 224, 288, 384, 384]
d_bifpns = [3, 4, 5, 6, 7, 7, 8, 8]
d_heads = [3, 3, 3, 4, 4, 4, 5, 5]
image_sizes = [img_size, 640, 768, 896, 1024, 1280, 1280, 640] #256
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7]

MOMENTUM = 0.9998
EPSILON = 1e-4

def get_efnet(w_bifpns, d_bifpns, d_heads,image_sizes, backbones,phi):
    input_size = image_sizes[phi]
    input_shape = (384, input_size, 3)
    image_input = layers.Input(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = d_bifpns[phi]
    w_head = w_bifpn
    d_head = d_heads[phi]
    backbone_cls = backbones[phi]
    encoder = backbone_cls(input_tensor=image_input)
    i=0
    while False:
        s1 = encoder.get_layer(str("input_"+str(i))).output       ## 1536
        
    s2 = encoder.get_layer("block2a_expand_activation").output    ## 768
    s3 = encoder.get_layer("block3a_expand_activation").output    ## 384
    s4 = encoder.get_layer("block4a_expand_activation").output    ## 192
    s5 = encoder.get_layer("block5a_expand_activation").output    ## 96
#     s6 = encoder.get_layer("block6a_expand_activation").output  ## 96
    s6 = encoder.get_layer("block6a_bn").output                   ## 48
    s7 = encoder.get_layer("block7a_expand_activation").output    ## 48
    fpn_features = (s3,s4,s5,s6,s7)
    return image_input, w_bifpn, d_bifpn, w_head,  d_head, fpn_features

class wBiFPNAdd(layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config

## in - input node (feature from 3-7 levls of Backbone | 2d part of layer)
class input_node():
    def get_p3_inp_node(feature, num_channels, id):
        P3_in = feature
        P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        return P3_in
    def get_p4_inp_node(feature, num_channels, id):
        P4_in = feature
        P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        return P4_in_1, P4_in_2
    def get_p5_inp_node(feature, num_channels, id):
        P5_in = feature
        P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        return P5_in_1, P5_in_2
    def get_p6_inp_node(feature, num_channels, id):
        P6_in = feature
        P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(P6_in)
        P6_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        # P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        return P6_in
    def get_p7_inp_node(feature, num_channels, id):
        P7_in = feature
        P7_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p7/conv2d')(P7_in)
        P7_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p7/bn')(P7_in)
        P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P7_in)
        return P7_in

## td - top-down node (feature on the top-down pathway | 2d part of layer)
class top_down_node():
    def get_p6_td_node(feature, upsample_block, num_channels, id):
        P6_in = feature
        P7_U = upsample_block
        P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        return P6_td

    def get_p5_td_node(feature, upsample_block, num_channels, id):
        P5_in_1 = feature
        P6_U = upsample_block
        P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
        P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)   
        return P5_td
    
    def get_p4_td_node(feature, upsample_block, num_channels, id):
        P4_in_1 = feature
        P5_U = upsample_block
        P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
        P4_td = layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        return P4_td
## out - output node (feature on the bottom-up pathway | 3d part of layer)
class output_node():
    def get_p3_out_node(feature, upsample_block, num_channels, id):
        P3_in = feature
        P4_U = upsample_block
        P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        return P3_out

    def get_p4_out_node(feature, top_down_node, downsample_block, num_channels, id):
        P4_in_2 = feature
        P4_td = top_down_node
        P3_D = downsample_block
        P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)
        return P4_out
    
    def get_p5_out_node(feature, top_down_node, downsample_block, num_channels, id):
        P5_in_2 = feature
        P5_td = top_down_node
        P4_D = downsample_block
        P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
        # print(x == P5_out)
        P5_out = layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)
        return P5_out
    
    def get_p6_out_node(feature, top_down_node, downsample_block, num_channels, id):
        P6_in = feature
        P6_td = top_down_node
        P5_D = downsample_block
        P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)
        return P6_out
    
    def get_p7_out_node(feature, downsample_block, num_channels, id):
        P7_in = feature
        P6_D = downsample_block
        P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
        return P7_out

def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=f'{name}/conv')
    f2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

def build_wBiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        C1, C2, C3, C4, C5 = features

    #Get input node
        P3_in = input_node.get_p3_inp_node(C1, num_channels, id)
        P4_in_1, P4_in_2 = input_node.get_p4_inp_node(C2, num_channels, id)
        P5_in_1, P5_in_2 = input_node.get_p5_inp_node(C3, num_channels, id)
        P6_in = input_node.get_p6_inp_node(C4, num_channels, id)
        P7_in = input_node.get_p7_inp_node(C5, num_channels, id)
        # P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(C5)

    #Get top-down node
        #upsampling
        P7_U = layers.UpSampling2D()(P7_in)
        P6_td = top_down_node.get_p6_td_node(P6_in, P7_U, num_channels, id)
        #upsampling
        P6_U = layers.UpSampling2D()(P6_td)
        P5_td = top_down_node.get_p5_td_node(P5_in_1, P6_U, num_channels, id)
        #upsampling
        P5_U = layers.UpSampling2D()(P5_td)
        P4_td = top_down_node.get_p4_td_node(P4_in_1, P5_U, num_channels, id)
        
    #Get output node
        #upsampling
        P4_U = layers.UpSampling2D()(P4_td)    
        P3_out = output_node.get_p3_out_node(P3_in, P4_U, num_channels, id)
        #downsampling
        P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = output_node.get_p4_out_node(P4_in_2, P4_td, P3_D, num_channels, id)
        #downsampling
        P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = output_node.get_p5_out_node(P5_in_2, P5_td, P4_D, num_channels, id)
        #downsampling
        P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = output_node.get_p6_out_node(P6_in, P6_td, P5_D, num_channels, id)
        #downsampling
        P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = output_node.get_p7_out_node(P7_in, P6_D, num_channels, id)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
    
    #Get top-down node
        P7_U = layers.UpSampling2D()(P7_in)
        P6_td = top_down_node.get_p6_td_node(P6_in, P7_U, num_channels, id)

        P6_U = layers.UpSampling2D()(P6_td)
        P5_td = top_down_node.get_p5_td_node(P5_in, P6_U, num_channels, id)

        P5_U = layers.UpSampling2D()(P5_td)
        P4_td = top_down_node.get_p4_td_node(P4_in, P5_U, num_channels, id)
 
    #Get output node  
        P4_U = layers.UpSampling2D()(P4_td)
        P3_out = output_node.get_p3_out_node(P3_in, P4_U, num_channels, id)

        P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = output_node.get_p4_out_node(P4_in, P4_td, P3_D, num_channels, id)
     
        P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = output_node.get_p5_out_node(P5_in, P5_td, P4_D, num_channels, id)
       
        P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = output_node.get_p6_out_node(P6_in, P6_td, P5_D, num_channels, id)
        
        P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = output_node.get_p7_out_node(P7_in, P6_D, num_channels, id)
    return P3_out, P4_td, P5_td, P6_td, P7_out

class UNet ():
    def decoder_block(inputs, skip, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = Concatenate()([x, skip])
        x = UNet.downsample_block(x, num_filters)
        return x

    def downsample_block(inputs, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("LeakyReLU")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("LeakyReLU")(x)

        return x

    def build_unet(fpn_features, num_classes):

        b1 = fpn_features[4]
        b1 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(b1) ## 16
        d1 = UNet.decoder_block(b1, fpn_features[4], 384)
        d2 = UNet.decoder_block(d1, fpn_features[3], 192)
        d3 = UNet.decoder_block(d2, fpn_features[2], 96)
        d4 = UNet.decoder_block(d3, fpn_features[1], 48)
        d5 = UNet.decoder_block(d4, fpn_features[0], num_classes)
        u_net = layers.UpSampling2D(size=4, name = 'UNet' )(d5)
        
        return u_net
    
def Semantic_model(phi, num_classes=20, num_anchors=9, freeze_bn=False,
                 score_threshold=0.01, detect_quadrangle=False, anchor_parameters=None, separable_conv=True):
    
    ############ get features from previous NN ############
    assert phi in range(8)
    image_input, w_bifpn, d_bifpn, w_head, d_head, fpn_features = get_efnet(w_bifpns, d_bifpns, d_heads,image_sizes, backbones,phi)
    

    ############ build wBiFPN ############
    for i in range(d_bifpn):
        fpn_features = build_wBiFPN(fpn_features, w_bifpn, i, freeze_bn=freeze_bn)

    ########### Create Heads ############
    u_net = UNet.build_unet(fpn_features, num_classes)

    ############ Assemble Model ############
    model = models.Model(image_input,u_net, name='EfficientUDet') #[classification, regression] 
    return model