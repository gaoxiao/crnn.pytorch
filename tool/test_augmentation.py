import argparse

import Augmentor
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--img_path', type=str, default='/home/xiao/Pictures/a.png')

opt = parser.parse_args()

img_path = opt.img_path
image = Image.open(img_path)
# plt.imshow(image, cmap='gray')
# plt.show()

resize = transforms.Compose([transforms.Resize((32, 100), Image.BILINEAR)])
image = resize(image)
plt.imshow(image, cmap='gray')
plt.show()

# p = Augmentor.Pipeline()
# p.random_distortion(probability=1, grid_width=2, grid_height=2, magnitude=4)
# transform_train = transforms.Compose([p.torch_transform()])
# plt.imshow(transform_train(image), cmap='gray')
# plt.show()
#
# p = Augmentor.Pipeline()
# p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
# transform_train = transforms.Compose([p.torch_transform()])
# plt.imshow(transform_train(image), cmap='gray')
# plt.show()

# p = Augmentor.Pipeline()
# p.random_erasing(probability=1, rectangle_area=0.11)
# transform_train = transforms.Compose([p.torch_transform()])
# plt.imshow(transform_train(image), cmap='gray')
# plt.show()
#
# p = Augmentor.Pipeline()
# p.random_erasing(probability=1, rectangle_area=0.2)
# transform_train = transforms.Compose([p.torch_transform()])
# plt.imshow(transform_train(image), cmap='gray')
# plt.show()
#

# p = Augmentor.Pipeline()
# # p.random_erasing(probability=1, rectangle_area=0.5)
# # p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=4)
# # p.shear(probability=1, max_shear_left=10, max_shear_right=10)
# p.skew_tilt(probability=1, magnitude=0.5)
# transform_train = transforms.Compose([p.torch_transform()])
# plt.imshow(transform_train(image), cmap='gray')
# plt.show()


p = Augmentor.Pipeline()
p.invert(probability=0.5)
p.random_color(1, 0, 1)
p.random_contrast(1, 0, 1)
p.random_brightness(1, 0, 1)
transform_train = transforms.Compose([
    # lambda img: F.adjust_contrast(img, 5),
    p.torch_transform(),
])
plt.imshow(transform_train(image), cmap='gray')
plt.show()
