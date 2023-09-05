from keras.layers import Input, Activation, Add, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Concatenate, Multiply
from keras.layers import GlobalAveragePooling2D, Reshape, Permute


from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid


from keras.layers import MaxPooling2D
from keras.layers import Dropout, concatenate


img_size = 512

image_shape = (img_size, img_size, 4)
image_d_shape = (img_size, img_size, 3)


# Define Num. of Channels:
num_filter_1st = 64
ch_mul = 0.25


def convolution_2d(x, num_filter=32, k_size=3, act_type="mish"):
            
    x = Conv2D(num_filter, k_size, padding='same', kernel_initializer = 'he_normal')(x)
    # x = BatchNormalization()(x)
    
    if act_type=="mish": 
        softplus_x = Activation('softplus')(x)
        tanh_softplus_x = Activation('tanh')(softplus_x)
        x = multiply([x, tanh_softplus_x])

    elif act_type=="swish":
        sigmoid_x = Activation('sigmoid')(x)
        x = multiply([x, sigmoid_x])
        
    elif act_type=="leakyrelu": x = LeakyReLU(alpha=0.1)(x)
    elif act_type=="tanh": x = Activation('tanh')(x)
    
    return x


def gan_model(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image, guided_fm, generated_image_2 = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, guided_fm, generated_image_2, outputs])
    return model


### SDDN
def unet_spp_swish_generator_model():
       
    inputs = Input(image_shape)     # H
    
    conv1 = convolution_2d(inputs, int(num_filter_1st*ch_mul), 3,  act_type="swish")
    conv1 = convolution_2d(conv1, int(num_filter_1st*ch_mul), 3,  act_type="swish")      # H
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                # H/2

    conv2 = convolution_2d(pool1, int(2*num_filter_1st*ch_mul), 3,  act_type="swish")
    conv2 = convolution_2d(conv2, int(2*num_filter_1st*ch_mul), 3,  act_type="swish")     # H/2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                # H/4
    
    conv3 = convolution_2d(pool2, int(4*num_filter_1st*ch_mul), 3,  act_type="swish")
    conv3 = convolution_2d(conv3, int(4*num_filter_1st*ch_mul), 3,  act_type="swish")     # H/4
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)                # H/8
    
    conv4 = convolution_2d(pool3, int(8*num_filter_1st*ch_mul), 3,  act_type="swish")
    conv4 = convolution_2d(conv4, int(8*num_filter_1st*ch_mul), 3,  act_type="swish")     # H/8
    drop4 = Dropout(0.5)(conv4) 
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)                # H/16
    
    conv5 = convolution_2d(pool4, int(16*num_filter_1st*ch_mul), 3,  act_type="swish")
    conv5 = convolution_2d(conv5, int(16*num_filter_1st*ch_mul), 3,  act_type="swish")
    
    # SPP #
    conv5 = convolution_2d(conv5, int(8*num_filter_1st*ch_mul), 1, act_type="swish")


    # Guided Features
    guided_fm = convolution_2d(conv5, int(8*num_filter_1st), 1, act_type="leakyrelu")
    

    conv5 = concatenate([conv5,
                         MaxPooling2D(pool_size=(13, 13), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(conv5)], axis = 3)

    conv5 = convolution_2d(conv5, int(16*num_filter_1st*ch_mul), 1, act_type="swish")     # H/16
    drop5 = Dropout(0.5)(conv5)

    up6 = convolution_2d((UpSampling2D(size = (2,2))(drop5)), int(8*num_filter_1st*ch_mul), 2, act_type="swish")
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = convolution_2d(merge6, int(8*num_filter_1st*ch_mul), 3,  act_type="swish")
    conv6 = convolution_2d(conv6, int(8*num_filter_1st*ch_mul), 3,  act_type="swish")
    
    up7 = convolution_2d((UpSampling2D(size = (2,2))(conv6)), int(4*num_filter_1st*ch_mul), 2, act_type="swish")
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = convolution_2d(merge7, int(4*num_filter_1st*ch_mul), 3,  act_type="swish")
    conv7 = convolution_2d(conv7, int(4*num_filter_1st*ch_mul), 3,  act_type="swish")
   
    up8 = convolution_2d((UpSampling2D(size = (2,2))(conv7)), int(2*num_filter_1st*ch_mul), 2, act_type="swish")
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = convolution_2d(merge8, int(2*num_filter_1st*ch_mul), 3,  act_type="swish")
    conv8 = convolution_2d(conv8, int(2*num_filter_1st*ch_mul), 3,  act_type="swish")

    up9 = convolution_2d((UpSampling2D(size = (2,2))(conv8)), int(num_filter_1st*ch_mul), 2, act_type="swish")
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = convolution_2d(merge9, int(num_filter_1st*ch_mul), 3,  act_type="swish")
    conv9 = convolution_2d(conv9, int(num_filter_1st*ch_mul), 3,  act_type="swish")
    
    conv10 = Conv2D(3, 1, activation = 'tanh')(conv9)

    model = Model(inputs=inputs, outputs=[conv10, guided_fm, conv10])
    # model = Model(inputs=inputs, outputs=[conv10, conv10])
    return model


## Critic
def unet_encoder_discriminator_model():
    
    inputs = Input(shape=image_d_shape)
    
    conv1 = Conv2D(int(num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(int(num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(int(2*num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(int(2*num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(int(4*num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(int(4*num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(int(8*num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(int(8*num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(int(16*num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(int(16*num_filter_1st*ch_mul), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    x = GlobalAveragePooling2D()(drop5)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model





### EDN-GTM Teacher Model
def unet_spp_large_swish_generator_model():
       
    inputs = Input(image_shape)
    
    conv1 = convolution_2d(inputs, 64, 3,  act_type="swish")
    conv1 = convolution_2d(conv1, 64, 3,  act_type="swish")
    conv1 = convolution_2d(conv1, 64, 3,  act_type="swish")
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = convolution_2d(pool1, 128, 3,  act_type="swish")
    conv2 = convolution_2d(conv2, 128, 3,  act_type="swish")
    conv2 = convolution_2d(conv2, 128, 3,  act_type="swish")
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = convolution_2d(pool2, 256, 3,  act_type="swish")
    conv3 = convolution_2d(conv3, 256, 3,  act_type="swish")
    conv3 = convolution_2d(conv3, 256, 3,  act_type="swish")
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = convolution_2d(pool3, 512, 3,  act_type="swish")
    conv4 = convolution_2d(conv4, 512, 3,  act_type="swish")
    conv4 = convolution_2d(conv4, 512, 3,  act_type="swish")
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = convolution_2d(pool4, 1024, 3,  act_type="swish")
    conv5 = convolution_2d(conv5, 1024, 3,  act_type="swish")
    conv5 = convolution_2d(conv5, 1024, 3,  act_type="swish")
    
    # SPP #
    conv5_hint = convolution_2d(conv5, 512, 1, act_type="swish")
    
    conv5 = concatenate([conv5_hint,
                         MaxPooling2D(pool_size=(13, 13), strides=1, padding='same')(conv5_hint),
                         MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(conv5_hint),
                         MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(conv5_hint)], axis = 3)

    conv5 = convolution_2d(conv5, 1024, 1, act_type="swish")
    drop5 = Dropout(0.5)(conv5)

    up6 = convolution_2d((UpSampling2D(size = (2,2))(drop5)), 512, 2, act_type="swish")
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = convolution_2d(merge6, 512, 3,  act_type="swish")
    conv6 = convolution_2d(conv6, 512, 3,  act_type="swish")
    conv6 = convolution_2d(conv6, 512, 3,  act_type="swish")
    
    up7 = convolution_2d((UpSampling2D(size = (2,2))(conv6)), 256, 2, act_type="swish")
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = convolution_2d(merge7, 256, 3,  act_type="swish")
    conv7 = convolution_2d(conv7, 256, 3,  act_type="swish")
    conv7 = convolution_2d(conv7, 256, 3,  act_type="swish")
   
    up8 = convolution_2d((UpSampling2D(size = (2,2))(conv7)), 128, 2, act_type="swish")
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = convolution_2d(merge8, 128, 3,  act_type="swish")
    conv8 = convolution_2d(conv8, 128, 3,  act_type="swish")
    conv8 = convolution_2d(conv8, 128, 3,  act_type="swish")

    up9 = convolution_2d((UpSampling2D(size = (2,2))(conv8)), 64, 2, act_type="swish")
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = convolution_2d(merge9, 64, 3,  act_type="swish")
    conv9 = convolution_2d(conv9, 64, 3,  act_type="swish")
    conv9 = convolution_2d(conv9, 64, 3,  act_type="swish")
    
    conv10 = Conv2D(3, 1, activation = 'tanh')(conv9)

    model = Model(inputs=inputs, outputs=[conv10, conv5_hint])
    return model



