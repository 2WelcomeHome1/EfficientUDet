import copy
import math

from keras import layers
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

import tensorflow as tf

from tensorflow.python.util.tf_export import keras_export 

print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.compat.v1.config.experimental.list_physical_devices("GPU") else "OFF" }')


# EfficientNet-B0 baseline network

default_model = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}


DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def load_weights (weights, include_top, WEIGHTS_HASHES, BASE_WEIGHTS_PATH, model_name, model):
    if weights == 'imagenet':
        if include_top:
            file_suffix = '.h5'
            file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
        else:
            file_suffix = '_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
        file_name = model_name + file_suffix
        weights_path = data_utils.get_file(file_name, BASE_WEIGHTS_PATH + file_name,
                                           cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)


class EfficientNet ():
    
     # Round number of filters based on depth multiplier
    def round_filters(filters, width_coefficient, divisor=8):
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    # Round number of repeats based on depth multiplier.
    def round_repeats(depth_coefficient, repeats):
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    def build_stem (img_input, bn_axis, activation, width_coefficient):
        x = img_input
        x = layers.Rescaling(1. / 255.)(x)
        x = layers.Normalization(axis=bn_axis)(x)

        x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, 3),name='stem_conv_pad')(x)
        x = layers.Conv2D(EfficientNet.round_filters(32, width_coefficient), 3, strides=2, 
                          padding='valid', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name='stem_conv')(x)

        x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
        x = layers.Activation(activation, name='stem_activation')(x)
        return x
    
    # Build blocks
    def build_blocks(x, blocks_args, depth_coefficient, width_coefficient, activation, drop_connect_rate):
        blocks_args = copy.deepcopy(blocks_args)

        b = 0
        blocks = float(sum(EfficientNet.round_repeats(depth_coefficient, args['repeats']) for args in blocks_args))
        for (i, args) in enumerate(blocks_args):
            assert args['repeats'] > 0

            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = EfficientNet.round_filters(args['filters_in'], width_coefficient)
            args['filters_out'] = EfficientNet.round_filters(args['filters_out'], width_coefficient)

            for j in range(EfficientNet.round_repeats(depth_coefficient, args.pop('repeats'))):
              # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = model_assemble.block(x, activation, drop_connect_rate * b / blocks,
                                         name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
                b += 1
        return x, b
    
    # Build top
    def build_top(x, width_coefficient, bn_axis, activation, include_top, dropout_rate, classifier_activation, weights,classes,pooling, input_tensor, img_input ):
        x = layers.Conv2D(EfficientNet.round_filters(1280, width_coefficient), 1, padding='same', 
                          use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER,name='top_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
        x = layers.Activation(activation, name='top_activation')(x)
        if include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name='top_dropout')(x)
            imagenet_utils.validate_activation(classifier_activation, weights)
            x = layers.Dense(classes, activation=classifier_activation, 
                             kernel_initializer=DENSE_KERNEL_INITIALIZER,name='predictions')(x)
        else:
            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D(name='max_pool')(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = layer_utils.get_source_inputs(input_tensor)
        else:
            inputs = img_input
        return x, inputs


class block_create():
    
    # Expansion phase
    def expansion(filters_in, expand_ratio, name, inputs, bn_axis,activation, strides, kernel_size):
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            x = layers.Conv2D(filters, 1, padding='same', use_bias=False, 
                              kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'expand_conv')(inputs)
            x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
            x = layers.Activation(activation, name=name + 'expand_activation')(x)
        else:
            x = inputs
        return filters, x

    # Depthwise Convolution
    def  Depthwise_Convolution(x, strides,kernel_size,name, bn_axis, activation):
        if strides == 2:
            x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, kernel_size),
                                        name=name + 'dwconv_pad')(x)
            conv_pad = 'valid'
        else:
            conv_pad = 'same'
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=conv_pad,
                                    use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                    name=name + 'dwconv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
        x = layers.Activation(activation, name=name + 'activation')(x)
        return x
    
    # Squeeze and Excitation phase
    def  SqueezeExcitation(x, se_ratio,filters_in, bn_axis, filters, name, activation):
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
            if bn_axis == 1:
                se_shape = (filters, 1, 1)
            else:
                se_shape = (1, 1, filters)
            se = layers.Reshape(se_shape, name=name + 'se_reshape')(se)
            se = layers.Conv2D(filters_se, 1, padding='same', 
                               activation=activation, kernel_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'se_reduce')(se)
            se = layers.Conv2D(filters,1,padding='same',activation='sigmoid',
                               kernel_initializer=CONV_KERNEL_INITIALIZER,name=name + 'se_expand')(se)
            x = layers.multiply([x, se], name=name + 'se_excite')
        return x
   
    def Output_phase (x, filters_out, filters_in, bn_axis, name, id_skip, strides, drop_rate, inputs):
        x = layers.Conv2D(filters_out,1, padding='same', use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'project_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
            x = layers.add([x, inputs], name=name + 'add')
        return x

class model_assemble():
    def block(inputs, activation='swish', drop_rate=0., name='',filters_in=32, filters_out=16, 
            kernel_size=3, strides=1, expand_ratio=1, se_ratio=0., id_skip=True):
        
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        filters, x = block_create.expansion(filters_in, expand_ratio, name, inputs, bn_axis,activation, strides, kernel_size)
        x = block_create.Depthwise_Convolution(x, strides,kernel_size,name, bn_axis, activation)
        x = block_create.SqueezeExcitation (x, se_ratio,filters_in, bn_axis, filters, name, activation)
        x = block_create.Output_phase (x, filters_out, filters_in, bn_axis, name, id_skip, strides, drop_rate, inputs)
        return x
    
    def EfficientNet_model(width_coefficient, depth_coefficient, default_size, dropout_rate=0.2, drop_connect_rate=0.2,
                 depth_divisor=8, activation='swish',blocks_args='default', model_name='efficientnet',
                 include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                 pooling=None, classes=7, classifier_activation='softmax'):
    
        if blocks_args == 'default':
            blocks_args = default_model
            
        # Determine proper input shape
    
        input_shape = imagenet_utils.obtain_input_shape(input_shape,default_size=default_size, min_size=32, 
                                                        data_format=backend.image_data_format(),
                                                        require_flatten=include_top,weights=weights)
        
        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
            
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
                
            else:
                img_input = input_tensor
               
        
    

        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        
        x = EfficientNet.build_stem(img_input, bn_axis, activation, width_coefficient)
        x, b = EfficientNet.build_blocks(x, blocks_args, depth_coefficient, width_coefficient, activation, drop_connect_rate)
        x, inputs = EfficientNet.build_top(x, width_coefficient, bn_axis, activation, include_top, dropout_rate, classifier_activation, weights,classes,pooling, input_tensor, img_input )

        # Create model.
        model = training.Model(inputs, x, name=model_name)
        
        return model

def EfficientNetB7(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                   classes=1000, classifier_activation='softmax', **kwargs):
    return model_assemble.EfficientNet_model(2.0, 3.1, 600, 0.5, model_name='efficientnetb7', include_top=include_top,
                        weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling,
                        classes=classes, classifier_activation=classifier_activation, **kwargs)

# input_shape = (1536, 1536, 3)
# image_input = layers.Input(input_shape)
# EfficientNetB7(input_tensor=image_input)