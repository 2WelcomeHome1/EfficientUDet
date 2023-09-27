from keras import layers
from wBiFPN.nodes import *
from wBiFPN.nodes import *

class wBiFPN():
    def __init__(self):
        pass
    
    def build_wBiFPN(self, features, num_channels, id, freeze_bn=False):
        if id == 0:
            C1, C2, C3, C4, C5 = features

        #Get input node
            P3_in = input_node().get_p3_inp_node(C1, num_channels, id)
            P4_in_1, P4_in_2 = input_node().get_p4_inp_node(C2, num_channels, id)
            P5_in_1, P5_in_2 = input_node().get_p5_inp_node(C3, num_channels, id)
            P6_in = input_node().get_p6_inp_node(C4, num_channels, id)
            P7_in = input_node().get_p7_inp_node(C5, num_channels, id)
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
