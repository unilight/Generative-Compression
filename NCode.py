import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from load_VGG import *
import collections

CONTENT_LAYERS = ['relu2_2']
EPS = 1e-12

Model = collections.namedtuple("Model", "outputs, outputs_using_pl, rand, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss, gen_grads_and_vars, enc_loss_L2, enc_loss_Perceptual, enc_grads_and_vars, enc_train, gen_train, dis_train, global_step, inc_step, latent_code, latent_pl")

def l2_loss(target_features,content_features):
    _,height,width,channel = map(lambda i:i.value,content_features.get_shape())
    content_size = height * width * channel
    return tf.nn.l2_loss(target_features - content_features) / content_size

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, 
            initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def create_generator(generator_inputs, generator_outputs_channels, args):
    z = tf.reshape(generator_inputs, [args.batch_size, 1, 1, args.z_dim])
    layers = [z]

    layer_specs = [
        (args.ngf * 8, 0.5),   # decoder_1: [batch, 1, 1, z_dim] => [batch, 2, 2, ngf * 8 * 2]
        (args.ngf * 8, 0.0),   # decoder_2: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (args.ngf * 4, 0.0),   # decoder_3: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 4 * 2]
        (args.ngf * 2, 0.0),   # decoder_4: [batch, 8, 8, ngf * 4 * 2] => [batch, 16, 16, ngf * 2 * 2]
        (args.ngf, 0.0),       # decoder_5: [batch, 16, 16, ngf * 2 * 2] => [batch, 32, 32, ngf * 2]
    ]

    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        with tf.variable_scope("decoder_%d" % decoder_layer):
            rectified = tf.nn.relu(layers[-1])
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            layers.append(output)

    # decoder_7: [batch, 32, 32, ngf * 2] => [batch, 64, 64, generator_outputs_channels]
    with tf.variable_scope("decoder_{}".format(len(layer_specs)+1)):
        rectified = tf.nn.relu(layers[-1])
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

def Perceptual_loss_function(content_image,target_image):
    vgg_pars = vgg_params()
    content_features = vgg19(content_image, vgg_pars)
    target_features = vgg19(target_image, vgg_pars)
    loss = 0.0
    for layer in CONTENT_LAYERS:
        loss += l2_loss(target_features[layer],content_features[layer])

    return loss

def create_model(inputs, args):

    # put these function here because they need args
    def create_encoder(inputs):
        apply_BN = True # if args.mode == 'train' else False
        layers = create_convolutor(inputs, apply_BN)
        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=args.z_dim,
            activation_fn=None):
                lat = slim.fully_connected(slim.flatten(layers[-1]))
        return lat

    def create_discriminator(inputs):
        layers = create_convolutor(inputs, False)
        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        # WGAN: No sigmoid for last layer in D
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(layers[-1], out_channels=1, stride=1)
            layers.append(convolved)
        return layers[-1]
    
    def create_convolutor(inputs, apply_BN):
        n_layers = 3
        layers = []

        # layer_1: [batch, 64, 64, in_channels * 2] => [batch, 32, 32, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(inputs, args.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 32, 32, ndf] => [batch, 16, 16, ndf * 2]
        # layer_3: [batch, 16, 16, ndf * 2] => [batch, 8, 8, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = args.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                if apply_BN:
                    convolved = batchnorm(convolved)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)
        return layers

    # Create networks
    
    # this is for video coding
    latent_pl = tf.placeholder(tf.float32, [None, args.z_dim])

    with tf.variable_scope("encoder"):
       latent = create_encoder(inputs)
    
    z = tf.random_uniform(tf.shape(latent), minval=-1, maxval=1,name='z')

    with tf.name_scope("sample_generator"):
        with tf.variable_scope("generator"):
            out_channels = int(inputs.get_shape()[-1])
            random_sample = create_generator(z, out_channels, args)

    with tf.name_scope("compress_generator"):
        with tf.variable_scope("generator", reuse=True):
            out_channels = int(inputs.get_shape()[-1])
            recon_img = create_generator(latent, out_channels, args)
            recon_img_using_pl = create_generator(latent_pl, out_channels, args)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # [batch, height, width, channels] => [batch, 6, 6, 1]
            predict_real = create_discriminator(inputs)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # [batch, height, width, channels] => [batch, 6, 6, 1]
            predict_fake = create_discriminator(random_sample)

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(predict_fake - predict_real)
        
        # GP
        alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
        alpha = alpha_dist.sample((args.batch_size, 1, 1, 1))
        interpolated = inputs + alpha*(random_sample-inputs)
        with tf.variable_scope("discriminator", reuse=True):
            inte_logit = create_discriminator(interpolated)
        gradients = tf.gradients(inte_logit, [interpolated,])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
        gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
        gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
        grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
        tf.summary.scalar("W_dis", discrim_loss)
        discrim_loss += args.gp_weight * gradient_penalty

    with tf.name_scope("generator_loss"):
        gen_loss = tf.reduce_mean(-predict_fake)

    with tf.name_scope("encoder_loss"):
        enc_loss_L2 = l2_loss(recon_img, inputs)
        enc_loss_Perceptual = Perceptual_loss_function(recon_img, inputs)
        enc_loss = enc_loss_L2 * args.lambda1_weight + enc_loss_Perceptual * args.lambda2_weight
        
    with tf.name_scope("encoder_train"):
        enc_tvars = [var for var in tf.trainable_variables() if var.name.startswith("encoder")]
        enc_optim = tf.train.AdamOptimizer(args.lr, args.beta1, args.beta2)
        enc_grads_and_vars = enc_optim.compute_gradients(enc_loss, var_list=enc_tvars)
        enc_train = enc_optim.apply_gradients(enc_grads_and_vars)

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(args.lr, args.beta1, args.beta2)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer(args.lr, args.beta1, args.beta2)
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=discrim_loss,
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss=gen_loss,
        gen_grads_and_vars=gen_grads_and_vars,
        enc_loss_L2=enc_loss_L2,
        enc_loss_Perceptual=enc_loss_Perceptual,
        enc_grads_and_vars=enc_grads_and_vars,
        rand=random_sample,
        outputs=recon_img,
        outputs_using_pl=recon_img_using_pl,
        global_step=global_step,
        dis_train=discrim_train,
        gen_train=tf.group(incr_global_step, gen_train),
        enc_train=tf.group(incr_global_step, enc_train),
        inc_step = incr_global_step,
        latent_code = latent,
        latent_pl = latent_pl,
    )
