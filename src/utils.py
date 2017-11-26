
import numpy as np


def apply_square_patch(images, h, w):
    patched_images = images
    patched_images[:, h:h+64, w:w+64, 0] = np.mean(images)
    patched_images[:, h:h+64, w:w+64, 1] = np.mean(images)
    patched_images[:, h:h+64, w:w+64, 2] = np.mean(images)
    return patched_images


def completion_function(patched_batch, generated_batch, h, w):
    generated_batch = generated_batch#*1000  # # to prevent values getting reduced to infinitesimal because of
    completed_batch = patched_batch
    completed_batch[:,h:h+64,w:w+64,0] = generated_batch[:,h:h+64,w:w+64,0]
    completed_batch[:,h:h+64,w:w+64,1] = generated_batch[:,h:h+64,w:w+64,1]
    completed_batch[:,h:h+64,w:w+64,2] = generated_batch[:,h:h+64,w:w+64,2]
    # plt.imshow(completed_batch[0])
    return completed_batch