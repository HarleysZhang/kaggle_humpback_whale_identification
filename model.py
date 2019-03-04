# coding:utf-8
# filename:model.py
# function:鲸鱼识别程序, Siamese网络架构


from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, GlobalAveragePooling2D, MaxPooling2D, Reshape, Dropout
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50

from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

img_shape = (512, 512, 3)  # 模型使用的图像形状


# shortcut connection structure
def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


# create a Siamese model function
def build_model(lr, l2, activation='sigmoid'):
    #################################
    # BRANCH MODEL,提取输入图像的特征向量
    #################################
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    # inp = Input(shape=img_shape, name='branch-input')
    # base_model = DenseNet121(include_top=False,
    #                          weights='imagenet',
    #                          input_shape=img_shape)
    # base_model.trainable = False
    # x = base_model(inp)
    # x = Conv2D(512, kernel_size=(1, 1), activation='relu', **kwargs)(x)
    # # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # x = GlobalMaxPooling2D()(x)

    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    # branch_model = Model(inp, x)

    inp = Input(shape=img_shape)  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
    	x = BatchNormalization()(x)
    	x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4):
    	x = subblock(x, 64, **kwargs)
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4):
    	x = subblock(x, 64, **kwargs)
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4):
    	x = subblock(x, 96, **kwargs)
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4):
    	x = subblock(x, 128, **kwargs)
    
    x = GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)

    #######################################
    # HEAD MODEL,比较来自branch_model的特征向量
    #######################################
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # 使用合适的步幅，让2D卷积实现具有共享权重的特征神经网络
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Dense layer的实现为加权和
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # 通过在每个输入图像上调用branch model来构建完整模型,
    # 然后是生成512个向量的head model.
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x)
    
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

    return model, branch_model, head_model


model, branch_model, head_model = build_model(64e-5, 0)
model.summary()
branch_model.summary()
head_model.summary()
