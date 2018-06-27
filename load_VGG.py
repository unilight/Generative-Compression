import tensorflow as tf
import numpy as np
import scipy.io as sio

def vgg_params():
    return sio.loadmat('imagenet-vgg-verydeep-19.mat')

def vgg19(input_image, vgg_pars):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4','pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )

    weights = vgg_pars['layers'][0]
    net = input_image
    network = {}
    for i,name in enumerate(layers):
        layer_type = name[:4]
        # for conv layers
        if layer_type == 'conv':
            kernels,bias = weights[i][0][0][0][0]
            # We need to permute the positions of the matrices
            # in `imagenet-vgg-verydeep-19.mat` since they are not 
            # consistent with those we defined
            kernels = np.transpose(kernels,(1,0,2,3))
            conv = tf.nn.conv2d(net, tf.constant(kernels),strides=(1,1,1,1),padding='SAME',name=name)
            net = tf.nn.bias_add(conv,bias.reshape(-1))
            net = tf.nn.relu(net)
        # for pooling layers
        elif layer_type == 'pool':
            net = tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
        # add the hidden layer or activation function in the set
        network[name] = net

    return network
