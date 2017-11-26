from __future__ import print_function

import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Deconv2D, Flatten, Dense, Concatenate
import scipy

import pickle
import os


sess = tf.Session()
K.set_session(sess)

# batch_size = tf.constant(2, dtype=tf.float32)
batch_size = 8


def pickle_dump(obj, path):
    filename = path + '.pickle'
    pickle_out = open(filename, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()

    return None


def pickle_load(filename):
    pickle_in = open(filename, 'rb')

    return pickle.load(pickle_in)


def apply_square_patch(images, h, w):
    patched_images = images
    patched_images[:, h:h+64, w:w+64, 0] = np.mean(images)
    patched_images[:, h:h+64, w:w+64, 1] = np.mean(images)
    patched_images[:, h:h+64, w:w+64, 2] = np.mean(images)

    return patched_images


# def completion_function(patched_batch, generated_batch, h, w):
#     generated_batch = generated_batch  # # to prevent values getting reduced to infinitesimal because of
#     completed_batch = np.copy(patched_batch)
#     completed_batch[:,h:h+64,w:w+64,0] = generated_batch[:,h:h+64,w:w+64,0]
#     completed_batch[:,h:h+64,w:w+64,1] = generated_batch[:,h:h+64,w:w+64,1]
#     completed_batch[:,h:h+64,w:w+64,2] = generated_batch[:,h:h+64,w:w+64,2]
#
#     return completed_batch


def completion_function(patched_batch, generated_batch, mask, batch_size):
    with tf.name_scope('Ops_Completion'):
        completed_batch_patch_area = tf.multiply(mask, generated_batch)
        negative_mask = 1 - mask
        completed_batch_outside_patch_area = tf.multiply(negative_mask, patched_batch)
        completed_batch = tf.add(completed_batch_patch_area, completed_batch_outside_patch_area)
        completed_batch = tf.reshape(completed_batch, shape=(batch_size, 256, 256, 3))

    return completed_batch


def surrounding_patch(completed_batch, mask_local_dis, batch_size):
    with tf.name_scope('Ops_Surrounding_Patch'):
        completed_batch_surrounding_patch = tf.boolean_mask(completed_batch, mask_local_dis)
        completed_batch_surrounding_patch = tf.reshape(completed_batch_surrounding_patch, shape=(batch_size,128,128,3))

    return completed_batch_surrounding_patch


with tf.name_scope('Generator_Model'):
    generation = Sequential(name='Generation_Model')
    generation.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu',strides=(1,1), input_shape=(256,256,3), padding='same', name='Conv_1'))
    generation.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=(2,2), name='Conv_2', padding='same'))
    generation.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_3', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(2,2), name='Conv_4', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_5', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_6', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Dilated_Conv_1', dilation_rate=2, padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Dilated_Conv_2', dilation_rate=4, padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Dilated_Conv_3', dilation_rate=8, padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Dilated_Conv_4', dilation_rate=16, padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_7', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_8', padding='same'))
    generation.add(Deconv2D(filters=128, kernel_size=(4,4), activation='relu', strides=(2,2), name='DeConv_1', padding='same'))
    generation.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_9', padding='same'))
    generation.add(Deconv2D(filters=64, kernel_size=(4,4), activation='relu', strides=(2,2), name='DeConv_2', padding='same'))
    generation.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_10', padding='same'))
    generation.add(Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', name='Output'))


# with tf.name_scope('Input_placeholders'):
with tf.name_scope('ph_patched_batch'):
    patched_batch_placeholder = K.placeholder(shape=(None, 256, 256, 3), dtype=tf.float32, name='ph_patched_batch')
with tf.name_scope('ph_original_batch'):
    original_batch_placeholder = K.placeholder(shape=(batch_size, 256, 256, 3), dtype=tf.float32, name='ph_original_batch')
with tf.name_scope('ph_mask_'):
    mask_placeholder = K.placeholder(shape=(None, 256, 256, 3), dtype=tf.float32, name='ph_mask')
with tf.name_scope('ph_mask_local_dis'):
    mask_local_dis_placeholder = K.placeholder(shape=(None, 256, 256, 3), dtype=tf.bool, name='ph_mask_local_dis')

with tf.name_scope('Generator_Network'):
    generated_batch = generation(patched_batch_placeholder)
    with tf.name_scope('Generated_batchX1000'):
        generated_batch = generated_batch*1

completed_batch = completion_function(patched_batch_placeholder, generated_batch, mask_placeholder, batch_size)

completed_batch_surrounding_patch = surrounding_patch(completed_batch, mask_local_dis_placeholder, batch_size)


image_gen = ImageDataGenerator(featurewise_center=False,
                               samplewise_center=False,
                               featurewise_std_normalization=False,
                               samplewise_std_normalization=False,
                               zca_whitening=False,
                               zca_epsilon=1e-6,
                               rotation_range=0.,
                               width_shift_range=0.,
                               height_shift_range=0.,
                               shear_range=0.,
                               zoom_range=0.,
                               channel_shift_range=0.,
                               fill_mode=None,
                               cval=0.,
                               horizontal_flip=False,
                               vertical_flip=False,
                               rescale=1.0/255.0,
                               preprocessing_function=None,
                               data_format=K.image_data_format())

# import sys
# sys.path.append('S:/InPainting')
image_batch = image_gen.flow_from_directory('data/validation',   # # Restore to da
                                            batch_size=batch_size,
                                            target_size=(256, 256), shuffle=False)


def generator_loss(generated_batch, patched_batch, mask):
    diff = tf.subtract(generated_batch, patched_batch)
    diff = mask*diff
    loss = tf.norm(diff)

    return loss


# # Discriminator


with tf.name_scope('Discriminator_Model'):
    global_discriminator = Sequential(name='Global_Dicriminator_Model')
    global_discriminator.add(Conv2D(64,  kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_1', padding='same', input_shape=(256,256,3)))
    global_discriminator.add(Conv2D(128, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_2', padding='same'))
    global_discriminator.add(Conv2D(256, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_3', padding='same'))
    global_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_4', padding='same'))
    global_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_5', padding='same'))
    global_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_6', padding='same'))
    global_discriminator.add(Flatten())
    global_discriminator.add(Dense(1024, activation='relu'))

    local_discriminator = Sequential(name='Local_Discriminator_Model')
    local_discriminator.add(Conv2D(64,  kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_1', padding='same', input_shape=(128,128,3)))
    local_discriminator.add(Conv2D(128, kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_2', padding='same'))
    local_discriminator.add(Conv2D(256, kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_3', padding='same'))
    local_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_4', padding='same'))
    local_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_5', padding='same'))
    local_discriminator.add(Flatten())
    local_discriminator.add(Dense(1024, activation='relu'))


# local_discriminator_fc = local_discriminator(completed_batch_surrounding_patch)
# global_discriminator_fc = global_discriminator(completed_batch)

concat_layer = Concatenate(axis=-1)
# concat_value = concat_layer([ global_discriminator_fc, local_discriminator_fc])
concat_dense_layer = Dense(1, activation=None)
# probability_real = concat_dense_layer(concat_value)


def discriminator_function(image_batch, mask, batch_size):
    with tf.name_scope('Discriminator_Network'):
        image_batch_surrounding_patch = surrounding_patch(image_batch, mask, batch_size)
        local_discriminator_fc_2 = local_discriminator(image_batch_surrounding_patch)
        global_discriminator_fc_2 = global_discriminator(image_batch)
        concat_value_2 = concat_layer([local_discriminator_fc_2, global_discriminator_fc_2])
        probability_real = concat_dense_layer(concat_value_2)

    return probability_real


# probability_real = discriminator_function(completed_batch, mask_local_dis_placeholder, batch_size)

D_logit_original = discriminator_function(original_batch_placeholder, mask_local_dis_placeholder, batch_size)
D_logit_completed = discriminator_function(completed_batch, mask_local_dis_placeholder, batch_size)

trainable_vars = tf.trainable_variables()

Gen_trainable_vars = []

[Gen_trainable_vars.append(var) for var in trainable_vars if 'Gen' in var.name]

Dis_trainable_vars = []

[Dis_trainable_vars.append(var) for var in trainable_vars if 'Dis' in var.name]


with tf.name_scope('loss_Gen_Euclidean_completed'):
    loss_Gen_Euclidean_completed = generator_loss(generated_batch, patched_batch_placeholder, mask_placeholder)


with tf.name_scope('loss_Gen_cross_entropy_completed'):
    loss_Gen_cross_entropy_completed = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_completed),
                                                                                              logits=D_logit_completed))

with tf.name_scope('loss_Dis'):
    with tf.name_scope('loss_Dis_cross_entropy_original'):
        loss_Dis_cross_entropy_original = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_original),
                                                                                               logits=D_logit_original))

    with tf.name_scope('loss_Dis_cross_entropy_completed'):
        loss_Dis_cross_entropy_completed = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_completed),
                                                                                                logits=D_logit_completed))

    loss_Dis = loss_Dis_cross_entropy_original + loss_Dis_cross_entropy_completed


with tf.name_scope('loss_Gen'):
    loss_Gen = loss_Gen_cross_entropy_completed + loss_Gen_Euclidean_completed


with tf.name_scope('loss_Gen_Dis_Combined'):
    loss_Gen_Dis_Combined = loss_Dis + loss_Gen

with tf.name_scope('train_op_Gen_Euclidean'):
    G_Optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    train_op_Gen_Euclidean = G_Optimizer.minimize(loss_Gen_Euclidean_completed,
                                                  var_list=Gen_trainable_vars)


with tf.name_scope('train_op_Dis'):
    Dis_Optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    train_op_Dis = Dis_Optimizer.minimize(loss_Dis, var_list=Dis_trainable_vars)


with tf.name_scope('train_op_Gen_Dis_Combined'):
    Gen_Dis_Combined_Optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    train_op_Gen_Dis_Combined = Gen_Dis_Combined_Optimizer.minimize(loss_Gen_Dis_Combined)

h1 = tf.trainable_variables()[0]

tf.summary.histogram('Conv_1/Kernel_0', h1)
tf.summary.scalar('loss_Gen_Euclidean_completed', loss_Gen_Euclidean_completed)
tf.summary.scalar('loss_Dis', loss_Dis)  # # Causing Error
# tf.summary.scalar('loss_Gen_cross_entropy_completed', loss_Gen_cross_entropy_completed)
tf.summary.scalar('loss_Gen_Dis_Combined', loss_Gen_Dis_Combined)


merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('logs/', sess.graph)

init_op = tf.global_variables_initializer()
sess.run(init_op)

no_of_iterations = 50001
# no_of_iterations_per_training_op = no_of_iterations/3
#

saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.1)

# # Restoring the main Session if it exists


if os.path.isfile('saved_sessions/main_session/session.index'):
    saver.restore(sess, 'saved_sessions/main_session/session')
    pickle_in = open('saved_sessions/main_session/curr_iteration.pickle', 'rb')
    curr_iteration = pickle.load(pickle_in)

else:
    curr_iteration = 0

# sess = saver.restore()

with sess.as_default():

    D_loss_list = []
    for i in range(no_of_iterations):

        i = curr_iteration

        if i < 10000:
            print('Training only Generator with Euclidean Loss -', ' iter:', i, sep='')
            h = np.random.randint(64, 128)
            w = np.random.randint(64, 128)
            mask = np.zeros(shape=(1, 256, 256, 3), dtype='float32')
            mask[:, h:h + 64, w:w + 64, 0:3] = 1.0
            mask_local_dis = np.zeros(shape=(batch_size, 256, 256, 3), dtype='float32')
            mask_local_dis[:, h-32:h+96, w-32:w+96, 0:3] = 1.0
            mask_local_dis = mask_local_dis.astype(dtype='bool')

            original_batch = image_batch.next()[0]
            patched_batch_value = apply_square_patch(np.copy(original_batch), h, w)

            # G_loss_value = loss__Gen_Euclidean.eval(feed_dict={
            #                      patched_batch_placeholder:patched_batch_value,
            #                            mask_placeholder:mask})
            # print(G_loss_value)

            # # The commented code below is for generating a test batch at the beginning of the training but it is
            # # better to the load the batch manually and dump it using pickle as batch generated by default may not
            # # be good enough for testing

            if i == 0:
                # test_mask = np.copy(mask)
                # test_mask_local_dis = np.copy(mask_local_dis)
                # test_batch_original = np.copy(original_batch)
                # test_batch_patched = np.copy(patched_batch_value)

                test_mask = pickle_load('test_batch/test_mask.pickle')

                test_mask_local_dis = pickle_load('test_batch/test_mask_local_dis.pickle')

                test_batch_original = pickle_load('test_batch/test_batch_original.pickle')

                test_batch_patched = pickle_load('test_batch/test_batch_patched.pickle')

                for j in range(batch_size):
                    filename = 'test_batch/' + 'image_' + str(j) + '_original_to_be_patched.jpg'
                    scipy.misc.imsave(filename, test_batch_patched[j])


                # scipy.misc.imsave('test_batch/original_to_be_patched.jpg', test_batch_patched[0])

                test_batch_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_patched,
                                                                   mask_placeholder: test_mask})

                for j in range(batch_size):
                    filename = 'test_batch/' + 'image_' + str(j) + '_no_training.jpg'
                    scipy.misc.imsave(filename, test_batch_value[j])

            #
            #
            #     # scipy.misc.imsave('results/no_training.jpg', test_batch_value[0])
            #
            #     # # Saving the first test batch in pickle format so that it can always be loaded from the disk
            #
            #     pickle_dump(test_mask, 'test_batch/test_mask')
            #
            #     pickle_dump(test_mask_local_dis, 'test_batch/test_mask_local_dis')
            #
            #     pickle_dump(test_batch_original, 'test_batch/test_batch_original')
            #
            #     pickle_dump(test_batch_patched, 'test_batch/test_batch_patched')
            #

            train_op_Gen_Euclidean.run(feed_dict={
                                 patched_batch_placeholder: patched_batch_value,
                                       mask_placeholder: mask})

            if i % 50 == 0:

                test_mask = pickle_load('test_batch/test_mask.pickle')

                test_mask_local_dis = pickle_load('test_batch/test_mask_local_dis.pickle')

                test_batch_original = pickle_load('test_batch/test_batch_original.pickle')

                test_batch_patched = pickle_load('test_batch/test_batch_patched.pickle')

                test_batch_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_patched,
                                                                   mask_placeholder: test_mask})
                for j in range(batch_size):
                    filename = 'test_batch/' + 'image_' + str(j) + '_only_gen.jpg'
                    scipy.misc.imsave(filename, test_batch_value[j])

            # result = sess.run(merged, feed_dict={
            #                      patched_batch_placeholder:patched_batch_value,
            #                            mask_placeholder:mask} )
            # writer.add_summary(result, i)

        elif 10000 <= i < 15000:
            # continue
            print('Training only Discriminator with Cross Entropy Loss -', ' iter:', i,)
            h = np.random.randint(64, 128)
            w = np.random.randint(64, 128)
            mask = np.zeros(shape=(1, 256, 256, 3), dtype='float32')
            mask[:, h:h + 64, w:w + 64, 0:3] = 1.0
            mask_local_dis = np.zeros(shape=(batch_size, 256, 256, 3), dtype='float32')
            mask_local_dis[:, h - 32:h + 96, w - 32:w + 96, 0:3] = 1.0
            mask_local_dis = mask_local_dis.astype(dtype='bool')
            original_batch = image_batch.next()[0]
            patched_batch_value = apply_square_patch(np.copy(original_batch), h, w)

            train_op_Dis.run(feed_dict={patched_batch_placeholder:patched_batch_value,
                                        original_batch_placeholder:original_batch,
                                        mask_local_dis_placeholder:mask_local_dis,
                                        mask_placeholder:mask})

            if i % 50 == 0:

                test_mask = pickle_load('test_batch/test_mask.pickle')

                test_mask_local_dis = pickle_load('test_batch/test_mask_local_dis.pickle')

                test_batch_original = pickle_load('test_batch/test_batch_original.pickle')

                test_batch_patched = pickle_load('test_batch/test_batch_patched.pickle')

                test_batch_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_patched,
                                                                   mask_placeholder: test_mask})

                for j in range(batch_size):
                    filename = 'test_batch/' + 'image_' + str(j) + '_only_dis.jpg'
                    scipy.misc.imsave(filename, test_batch_value[j])

            # D_loss_value = D_loss.eval(feed_dict={patched_batch_placeholder:patched_batch_value,
            #                        original_batch_placeholder:original_batch,
            #                        mask_local_dis_placeholder:mask_local_dis,
            #                        mask_placeholder:mask})
            # D_loss_list.append(D_loss_value)

        elif 15000 <= i < no_of_iterations:

            print('Training both Gen, Dis -', ' iter:', i, sep='')
            h = np.random.randint(64, 128)
            w = np.random.randint(64, 128)
            mask = np.zeros(shape=(1, 256, 256, 3), dtype='float32')
            mask[:, h:h + 64, w:w + 64, 0:3] = 1.0
            mask_local_dis = np.zeros(shape=(batch_size, 256, 256, 3), dtype='float32')
            mask_local_dis[:, h - 32:h + 96, w - 32:w + 96, 0:3] = 1.0
            mask_local_dis = mask_local_dis.astype(dtype='bool')
            original_batch = image_batch.next()[0]
            patched_batch_value = apply_square_patch(np.copy(original_batch), h, w)

            train_op_Gen_Dis_Combined.run(feed_dict={patched_batch_placeholder: patched_batch_value,
                                                     original_batch_placeholder: original_batch,
                                                     mask_local_dis_placeholder: mask_local_dis,
                                                     mask_placeholder: mask})

            if i % 50 == 0:

                test_mask = pickle_load('test_batch/test_mask.pickle')

                test_mask_local_dis = pickle_load('test_batch/test_mask_local_dis.pickle')

                test_batch_original = pickle_load('test_batch/test_batch_original.pickle')

                test_batch_patched = pickle_load('test_batch/test_batch_patched.pickle')

                test_batch_value = completed_batch.eval(feed_dict={patched_batch_placeholder:test_batch_patched,
                                                                   mask_placeholder: test_mask})

                for j in range(batch_size):
                    filename = 'test_batch/' + 'image_' + str(j) + '_both_gen_dis.jpg'
                    scipy.misc.imsave(filename, test_batch_value[j])

        else:
            pass

        if i % 10 == 0:

            saver.save(sess, 'saved_sessions/main_session/session')  # # Saving Session
            pickle_out = open('saved_sessions/main_session/curr_iteration.pickle', 'wb')
            pickle.dump(curr_iteration, pickle_out)
            pickle_out.close()

            test_mask = pickle_load('test_batch/test_mask.pickle')  # # Loading test batch for
            #  calculating loss for Tensorboard

            test_mask_local_dis = pickle_load('test_batch/test_mask_local_dis.pickle')

            test_batch_original = pickle_load('test_batch/test_batch_original.pickle')

            test_batch_patched = pickle_load('test_batch/test_batch_patched.pickle')

            result = sess.run(merged, feed_dict={patched_batch_placeholder: test_batch_patched,
                                                 mask_placeholder: test_mask,
                                                  original_batch_placeholder: np.copy(test_batch_original),
                                                 mask_local_dis_placeholder: test_mask_local_dis})
            writer.add_summary(result, i)

        i = i + 1
        curr_iteration = i

writer.close()


# Evaluate

with sess.as_default():
    generated_batch_value = generated_batch.eval(feed_dict={patched_batch_placeholder:patched_batch_value})
    completed_batch_value = completed_batch.eval(feed_dict={patched_batch_placeholder:patched_batch_value,
                                                            mask_placeholder: mask})
    completed_batch_surrounding_patch_value = completed_batch_surrounding_patch.eval(feed_dict={
                                                            patched_batch_placeholder:patched_batch_value,
                                                            mask_placeholder: mask,
                                                            mask_local_dis_placeholder:mask_local_dis})

with sess.as_default():
    test_batch_value = completed_batch.eval(feed_dict={patched_batch_placeholder:test_batch_patched,
                                                        mask_placeholder:test_mask})

# plt.imshow(completed_batch_value[0])
# plt.figure()
# plt.imshow(patched_batch_value[0])
# plt.imshow(generated_batch_value[0])
# plt.imshow(completed_batch_surrounding_patch_value[0])
# plt.imshow(test_batch_value[0])
# plt.figure()
