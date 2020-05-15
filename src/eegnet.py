from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

# Modified from ARL code
def EEGNet(nb_classes = 2, Chans = 16, Samples = 125, 
             dropoutRate = 0.25, kernLength = 32, F1 = 16, 
             D = 4, F2 = 16, norm_rate = 0.25, dropoutType = 'SpatialDropout2D', softmax=True):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    inputs = Input(shape = (1, Chans, Samples))
        
    layers = []
    layers.append(Conv2D(F1, (1, kernLength), padding = 'same', input_shape = (1, Chans, Samples), use_bias = False))
    
    layers.append(Permute((2, 3, 1)))
    #layers.append(BatchNormalization(axis = -1))
    layers.append(DepthwiseConv2D((Chans, 1), use_bias = False, depth_multiplier = D, depthwise_constraint = max_norm(1.), data_format="channels_last"))
    #layers.append(BatchNormalization(axis = -1))
    layers.append(Permute((3, 1, 2)))
    
    layers.append(Activation('elu'))
    
    layers.append(Permute((2, 3, 1)))
    layers.append(AveragePooling2D((1, 4), data_format="channels_last"))
    layers.append(Permute((3, 1, 2)))
    
    layers.append(dropoutType(dropoutRate))
    
    layers.append(Permute((2, 3, 1)))
    layers.append(SeparableConv2D(F2, (1, 16), use_bias = False, padding = 'same', data_format="channels_last"))
    #layers.append(BatchNormalization(axis = -1))
    layers.append(Permute((3, 1, 2)))
    
    layers.append(Activation('elu'))
    
    layers.append(Permute((2, 3, 1)))
    layers.append(AveragePooling2D((1, 8), data_format="channels_last"))
    layers.append(Permute((3, 1, 2)))
    
    layers.append(dropoutType(dropoutRate))
    layers.append(Flatten())
    layers.append(Dense(nb_classes, kernel_constraint = max_norm(norm_rate)))
    if softmax:
        layers.append(Activation('softmax'))
    else:
        layers.append(Activation('relu'))
        layers.append(Dense(nb_classes))

    
    x = inputs
    for l in layers:
        x = l(x)
    outputs = x
    
    return Model(inputs=inputs, outputs=outputs)