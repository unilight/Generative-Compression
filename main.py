import tensorflow as tf
import numpy as np
import argparse
import scipy.io as sio
import os
import json
import glob
import random
import collections
import math
import time

from load_VGG import *
from utils import *
from NCode import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", default='output', help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--first_steps", type=int, default=500000, help="number of training epochs for first stage")
parser.add_argument("--second_steps", type=int, default=200000, help="number of training epochs for second stage")
parser.add_argument("--summary_freq", type=int, default=50, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=500, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=500, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--crop_leftx", type=int, default=0, help="crop to bounding box left x coord")
parser.add_argument("--crop_lefty", type=int, default=20, help="crop to bounding box left y coord")
parser.add_argument("--crop_rightx", type=int, default=178, help="crop to bounding box right x coord")
parser.add_argument("--crop_righty", type=int, default=198, help="crop to bounding box right y coord")
parser.add_argument("--batch_size", type=int, default=32, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=64, help="scale images to this size")

parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--beta2", type=float, default=0.9, help="momentum term of adam")
parser.add_argument("--lambda1_weight", type=float, default=1, help="weight on L1 term for encoder gradient")
parser.add_argument("--lambda2_weight", type=float, default=0.02, help="weight on perceptual term for encoder gradient")
parser.add_argument("--gp_weight", type=float, default=0.25, help="GP weight")
parser.add_argument("--z_dim", type=int, default=100, help="dimension of the latent code")

a = parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"ngf", "ndf", "crop_leftx", "crop_lefty", "crop_rightx", "crop_righty"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    print("Loading data...")
    examples = load_examples(a)
    print("Data count = %d" % examples.count)

    print("Building Model...")
    model = create_model(examples.inputs, a)
    print("Done building Model")

    inputs = deprocess(examples.inputs)
    outputs = deprocess(model.outputs)
    random_outputs = deprocess(model.rand)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
    
    with tf.name_scope("conver_random_outputs"):
        converted_random_outputs = convert(random_outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            "random_outputs": tf.map_fn(tf.image.encode_png, converted_random_outputs, dtype=tf.string, name="random_output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("random_outputs_summary"):
        tf.summary.image("random_outputs", converted_random_outputs)
   
    # This shows patched Wasserstain distance by the discriminator (since we use PatchGAN)
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("generator_loss_", model.gen_loss)
    tf.summary.scalar("encoder_loss_L2", model.enc_loss_L2)
    tf.summary.scalar("encoder_loss_Perceptual", model.enc_loss_Perceptual)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    saver = tf.train.Saver(max_to_keep=100)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0,
        saver=None, global_step = model.global_step)
    with sv.managed_session() as sess:

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            print(checkpoint)
            saver.restore(sess, checkpoint)
        max_steps = a.first_steps + a.second_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(a.output_dir, results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(a.output_dir, filesets)
            print("wrote index at", index_path)
        
        else:
            # training
            start = time.time()
                
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {}
                fetches["global_step"] = sv.global_step
                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss"] = model.gen_loss
                    fetches["enc_loss_L2"] = model.enc_loss_L2
                    fetches["enc_loss_Perceptual"] = model.enc_loss_Perceptual

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                if step <= a.first_steps:
                    # Train D first
                    results = sess.run(model.dis_train)
                    # Then train G
                    fetches["train"] = model.gen_train
                    results = sess.run(fetches, options=options, run_metadata=run_metadata)
                else:
                    # Train E
                    fetches["train"] = model.enc_train
                    results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(a.output_dir, results["display"], step=results["global_step"])
                    append_index(a.output_dir, filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("[progress] global step %d epoch %d  step %d  image/sec %0.1f  remaining %dm" 
                        % (results["global_step"], train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss"])
                    print("enc_loss_L2", results["enc_loss_L2"])
                    print("enc_loss_Perceptual", results["enc_loss_Perceptual"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
