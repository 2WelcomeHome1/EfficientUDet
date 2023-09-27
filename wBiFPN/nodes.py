from keras import layers
from functools import reduce
import tensorflow as tf
from .wBiFPNAdd import *


def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    MOMENTUM, EPSILON = 0.9998,  1e-4
    f1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=f'{name}/conv')
    f2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


class input_node(): ## in - input node (feature from 3-7 levls of Backbone | 2d part of layer)
    def __init__(self) -> None:
        self.MOMENTUM = 0.9998
        self.EPSILON = 1e-4
        pass
    
    def get_p3_inp_node(self, feature, num_channels, id):
        P3_in = feature
        P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                          name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        return P3_in
    
    def get_p4_inp_node(self, feature, num_channels, id):
        P4_in = feature
        P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        return P4_in_1, P4_in_2
    
    def get_p5_inp_node(self, feature, num_channels, id):
        P5_in = feature
        P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        return P5_in_1, P5_in_2
    
    def get_p6_inp_node(self, feature, num_channels, id):
        P6_in = feature
        P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(P6_in)
        P6_in = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON, name='resample_p6/bn')(P6_in)
        # P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        return P6_in
    
    def get_p7_inp_node(self, feature, num_channels, id):
        P7_in = feature
        P7_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p7/conv2d')(P7_in)
        P7_in = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON, name='resample_p7/bn')(P7_in)
        P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P7_in)
        return P7_in

class top_down_node(): ## td - top-down node (feature on the top-down pathway | 2d part of layer)

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

class output_node(): ## out - output node (feature on the bottom-up pathway | 3d part of layer)

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

