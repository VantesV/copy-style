"""
Optimizer
"""
from torch import optim
learning_rate = 0.0001

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    # optimizer = optim.LBFGS([input_img.requires_grad_()])
    optimizer = optim.Adam([input_img.requires_grad_()], learning_rate)
    return optimizer
