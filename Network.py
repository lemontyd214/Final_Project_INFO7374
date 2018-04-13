from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from scipy.misc import imread, imresize, imsave, fromimage, toimage
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
import argparse
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model

"""
Neural Style Transfer with Keras 2.0.5

Based on:
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

-----------------------------------------------------------------------------------------------------------------------
"""

THEANO_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

TH_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
TF_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')

parser.add_argument('syle_image_paths', metavar='ref', nargs='+', type=str,
                    help='Path to the style reference image.')

parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

parser.add_argument("--style_masks", type=str, default=None, nargs='+',
                    help='Masks for style images')

parser.add_argument("--content_mask", type=str, default=None,
                    help='Masks for the content image')

parser.add_argument("--color_mask", type=str, default=None,
                    help='Mask for color preservation')

parser.add_argument("--image_size", dest="img_size", default=400, type=int,
                    help='Minimum image size')

parser.add_argument("--content_weight", dest="content_weight", default=0.025, type=float,
                    help="Weight of content")

parser.add_argument("--style_weight", dest="style_weight", nargs='+', default=[1], type=float,
                    help="Weight of style, can be multiple for multiple styles")

parser.add_argument("--style_scale", dest="style_scale", default=1.0, type=float,
                    help="Scale the weighing of the style")

parser.add_argument("--total_variation_weight", dest="tv_weight", default=8.5e-5, type=float,
                    help="Total Variation weight")

parser.add_argument("--num_iter", dest="num_iter", default=10, type=int,
                    help="Number of iterations")

parser.add_argument("--model", default="vgg16", type=str,
                    help="Choices are 'vgg16' and 'vgg19'")

parser.add_argument("--content_loss_type", default=0, type=int,
                    help='Can be one of 0, 1 or 2. Readme contains the required information of each mode.')

parser.add_argument("--rescale_image", dest="rescale_image", default="False", type=str,
                    help="Rescale image after execution to original dimentions")

parser.add_argument("--rescale_method", dest="rescale_method", default="bilinear", type=str,
                    help="Rescale image algorithm")

parser.add_argument("--maintain_aspect_ratio", dest="maintain_aspect_ratio", default="True", type=str,
                    help="Maintain aspect ratio of loaded images")

parser.add_argument("--content_layer", dest="content_layer", default="conv5_2", type=str,
                    help="Content layer used for content loss.")

parser.add_argument("--init_image", dest="init_image", default="content", type=str,
                    help="Initial image used to generate the final image. Options are 'content', 'noise', or 'gray'")

parser.add_argument("--pool_type", dest="pool", default="max", type=str,
                    help='Pooling type. Can be "ave" for average pooling or "max" for max pooling')

parser.add_argument('--preserve_color', dest='color', default="False", type=str,
                    help='Preserve original color in image')

parser.add_argument('--min_improvement', default=0.0, type=float,
                    help='Defines minimum improvement required to continue script')


def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1")

''' Arguments '''

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_paths = args.syle_image_paths
result_prefix = args.result_prefix

style_image_paths = []
for style_image_path in style_reference_image_paths:
    style_image_paths.append(style_image_path)

style_masks_present = args.style_masks is not None
mask_paths = []

if style_masks_present:
    for mask_path in args.style_masks:
        mask_paths.append(mask_path)

if style_masks_present:
    assert len(style_image_paths) == len(mask_paths), "Wrong number of style masks provided.\n" \
                                                      "Number of style images = %d, \n" \
                                                      "Number of style mask paths = %d." % \
                                                      (len(style_image_paths), len(style_masks_present))

content_mask_present = args.content_mask is not None
content_mask_path = args.content_mask


color_mask_present = args.color_mask is not None

rescale_image = str_to_bool(args.rescale_image)
maintain_aspect_ratio = str_to_bool(args.maintain_aspect_ratio)
preserve_color = str_to_bool(args.color)

# these are the weights of the different loss components
content_weight = args.content_weight
total_variation_weight = args.tv_weight

style_weights = []

if len(style_image_paths) != len(args.style_weight):
    print("Mismatch in number of style images provided and number of style weights provided. \n"
          "Found %d style images and %d style weights. \n"
          "Equally distributing weights to all other styles." % (len(style_image_paths), len(args.style_weight)))

    weight_sum = sum(args.style_weight) * args.style_scale
    count = len(style_image_paths)

    for i in range(len(style_image_paths)):
        style_weights.append(weight_sum / count)
else:
    for style_weight in args.style_weight:
        style_weights.append(style_weight * args.style_scale)

# Decide pooling function
pooltype = str(args.pool).lower()
assert pooltype in ["ave", "max"], 'Pooling argument is wrong. Needs to be either "ave" or "max".'

pooltype = 1 if pooltype == "ave" else 0

read_mode = "gray" if args.init_image == "gray" else "color"

# dimensions of the generated picture.
img_width = img_height = 0

img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0

assert args.content_loss_type in [0, 1, 2], "Content Loss Type must be one of 0, 1 or 2"

