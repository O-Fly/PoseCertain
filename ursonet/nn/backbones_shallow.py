# ursonet/nn/backbones_shallow.py
import tensorflow.keras.layers as KL
from ursonet.nn.layers import BatchNorm

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name

def residual_basic_block(input_tensor, filters, stage, block, strides=(1, 1), cut='pre', use_bias=False, train_bn=True):

    # get names of layers
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

    # defining shortcut connection
    if cut == 'pre':
        shortcut = input_tensor
    elif cut == 'post':
        shortcut = KL.Conv2D(filters, (1, 1), name=sc_name, strides=strides, use_bias=use_bias)(input_tensor)
    else:
        raise ValueError('Cut type not in ["pre", "post"]')

    # Two 3x3 convolution layers
    x = KL.ZeroPadding2D(padding=(1, 1))(input_tensor)
    x = KL.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name + '2')(x, training=train_bn)
    x = KL.Activation('relu', name=relu_name + '1')(x)
    x = KL.ZeroPadding2D(padding=(1, 1))(x)
    x = KL.Conv2D(filters, (3, 3), name=conv_name + '2', use_bias=use_bias)(x)

    # add residual connection
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name=relu_name + '2')(x)
    return x

def resnet_shallow_graph(input_image, architecture, train_bn=True):
    '''

    N.b: Currently convolutions do not use the bias term (unlike the 'deeper' resnet_graph)
     to keep compatibility with pre-trained weights
    '''

    assert architecture in ["resnet18", "resnet34"]

    nr_init_filters = 64

    # Resnet bottom
    x = KL.ZeroPadding2D(padding=(3, 3))(input_image)
    x = KL.Conv2D(nr_init_filters, (7, 7), strides=(2, 2), name='conv0', use_bias=False)(x)
    x = BatchNorm(name='bn_conv0')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # TODO: Allow more architectures
    if architecture == 'resnet18':
        repetitions = [2, 2, 2, 2]
    else:
        # This is fo 34 layers
        repetitions = (3, 4, 6, 3)

    for stage, rep in enumerate(repetitions):
        for block in range(rep):

            nr_filters = nr_init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = residual_basic_block(x, nr_filters, stage, block, strides=(1, 1), cut='post', train_bn=train_bn)

            elif block == 0:
                x = residual_basic_block(x, nr_filters, stage, block, strides=(2, 2),cut='post',train_bn=train_bn)

            else:
                x = residual_basic_block(x, nr_filters, stage, block, strides=(1, 1), cut='pre', train_bn=train_bn)

    return x
