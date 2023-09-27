from Head import *
from Neck import *
from Backbone import *
from keras import models
from keras.optimizers import SGD, Adam
from keras import metrics

class EfficientUDet():
    def __init__(self, num_classes) -> None:
        self.num_classes=num_classes
        pass
    
    def Semantic_model(self, phi, num_anchors=9, freeze_bn=False,
                    score_threshold=0.01, detect_quadrangle=False, anchor_parameters=None, separable_conv=True):
        
        ############ build Backbone ############
        print(phi)
        assert phi in range(8)
        image_input, w_bifpn, d_bifpn, w_head, d_head, fpn_features = EfficientNet(img_size = 640).get_efnet(phi)

        ############ build wBiFPN ############
        for i in range(d_bifpn):
            fpn_features = wBiFPN().build_wBiFPN(fpn_features, w_bifpn, i, freeze_bn=freeze_bn)

        ########### Create Heads ############
        u_net = UNet.build_unet(fpn_features, self.num_classes)

        ############ Assemble Model ############
        model = models.Model(image_input,u_net, name='EfficientUDet') 
        return model
    
if __name__ == "__main__":
    num_classes = 6
    Semantic_model = EfficientUDet(num_classes).Semantic_model(phi=7)
    Semantic_model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', \
                           metrics=[metrics.OneHotIoU(num_classes,[i for i in range(0,num_classes)])])
    Semantic_model.summary()
