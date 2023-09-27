from keras import layers
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,\
                                 EfficientNetB4, EfficientNetB5, EfficientNetB6#, EfficientNetB7
from EfficientNet_B7 import * 



class EfficientNet():
    def __init__(self, img_size:int = 640) -> None:
        self.img_size = img_size
        
        #EfficientNet0, EfficientNet1, EfficientNet2, EfficientNet3, EfficientNet4, EfficientNet5, EfficientNet6, EfficientNet7
        self.w_bifpns = [64, 88, 112, 160, 224, 288, 384, 384]
        self.d_bifpns = [3, 4, 5, 6, 7, 7, 8, 8]
        self.d_heads = [3, 3, 3, 4, 4, 4, 5, 5]
        self.image_sizes = [img_size, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
                    EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7]


    def get_efnet(self, phi):
        input_size = self.image_sizes[phi]
        input_shape = (384, input_size, 3)
        image_input = layers.Input(input_shape)
        w_bifpn = self.w_bifpns[phi]
        d_bifpn = self.d_bifpns[phi]
        w_head = w_bifpn
        d_head = self.d_heads[phi]
        backbone_cls = self.backbones[phi]
        encoder = backbone_cls(input_tensor=image_input)

        s1 = encoder.get_layer(f"input_{1}").output                   ## 1536 
        s2 = encoder.get_layer("block2a_expand_activation").output    ## 768
        s3 = encoder.get_layer("block3a_expand_activation").output    ## 384
        s4 = encoder.get_layer("block4a_expand_activation").output    ## 192
        s5 = encoder.get_layer("block5a_expand_activation").output    ## 96 
        s6 = encoder.get_layer("block6a_bn").output    ## 48     #   block6a_expand_activation
        s7 = encoder.get_layer("block7a_expand_activation").output    ## 48
        fpn_features = (s3,s4,s5,s6,s7)
        
        return image_input, w_bifpn, d_bifpn, w_head,  d_head, fpn_features
