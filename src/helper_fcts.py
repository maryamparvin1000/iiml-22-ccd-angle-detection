
import torch

import numpy as np
from dataclasses import dataclass


@dataclass
class Pad:
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0


def get_padding(img: torch.Tensor, size: tuple = (512, 512)) -> torch.nn.ZeroPad2d:
    """
    Returns a torch padding object which scales an input image to the desired size:


    :param img: input image of arbitrary size, torch.Tensor
    :param size: target size to which input in padded, tuple

    :return: padding object which pads input image to desired size

    """
    pad = Pad()

    # get duplicate squeezed img to remove potential channels
    img_no_channels = torch.squeeze(img)

    diff_width = (size[1] - img_no_channels.size()[1])
    diff_height = (size[0] - img_no_channels.size()[0])

    if diff_width > 0:
        pad.left = int(np.ceil(diff_width / 2))
        pad.right = int(np.floor(diff_width / 2))

    if diff_height > 0:
        pad.top = int(np.ceil(diff_height / 2))
        pad.bottom = int(np.floor(diff_height / 2))

    return torch.nn.ZeroPad2d((pad.left, pad.right, pad.top, pad.bottom))


# raw moments
def M(i, j, I):
    w, h = I.shape
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    x, y = np.meshgrid(x, y)
    return np.sum((x**i)*(y**j)*I)


def orientation_angle_heatmap_line(vec):
    x_bar = M(1, 0, vec)/M(0, 0, vec)
    y_bar = M(0, 1, vec)/M(0, 0, vec)
    u20 = M(2, 0, vec)/M(0, 0, vec) - (x_bar**2)
    u02 = M(0, 2, vec)/M(0, 0, vec) - (y_bar**2)
    u11 = M(1, 1, vec)/M(0, 0, vec) - (x_bar*y_bar)

    theta = 0.5 * np.arctan(2*u11*(1/(u20-u02)))
    return x_bar, y_bar, theta


def extract_line(pred_heatmap):
    #TODO add a threshold instead of using the max
    indices = np.where(pred_heatmap.numpy() == pred_heatmap.numpy().max())
    start_point = indices[0][0], indices[1][0]
    end_point = indices[0][-1], indices[1][-1]
    return start_point, end_point

def resize_img(img, image_size):
    """Resize and pad image to have unified image size. Also convert uint8 values to float."""
    # save original image size to adaptively resize image
    orig_height = img.shape[1]
    orig_width = img.shape[2]


    # calculate ceiled resize factor to ensure together with zero padding unified image sizes
    resize_factor = np.ceil(max(orig_height / image_size[0], orig_width / image_size[1]))

    # resize image for performance issues
    img = torchvision.transforms.Resize([int(orig_height / resize_factor), int(orig_width / resize_factor)])(img)

    # perform zero padding to ensure desired image size
    pad = get_padding(img, size=image_size)
    img = pad(img)

    # changes uint8 values to float, as uint8 values threw error in training loop
    img = img.type(torch.float)

    #normalization
    img = (img - torch.median(img)) / torch.std(img)
    if img.min() < 0:
        img = img + torch.abs(img.min())
    return img

def getSlope(points):
    x1, x2, y1, y2 = points[0][0], points[1][0], points[0][1], points[1][1]
    if x2 - x1 == 0:
        # TODO return something different?
        return np.Inf
    return (y2 - y1) / (x2 - x1)

def getAngle(m1, m2):
    theta = np.arctan((m1 - m2) / (1 + m1 * m2))
    return theta * (180 / np.pi)
