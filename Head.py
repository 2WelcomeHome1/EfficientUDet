from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate
from keras import layers

class UNet ():
    def __init__(self):
        pass
    
    def decoder_block(inputs, skip, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = Concatenate()([x, skip])
        x = UNet.downsample_block(x, num_filters)
        return x

    def downsample_block(inputs, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("ReLU")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("ReLU")(x)

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
