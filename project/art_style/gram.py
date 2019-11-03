import torch
from torch import nn


def gram_matrix(mat):
    batch_size, num_feature_maps, width, height = mat.size()

    # resize the matrix
    features = mat.view(batch_size * num_feature_maps, width * height)
    # pylint: disable=E1101
    gram_product = torch.mm(features, features.t())
    # pylint: enable=E1101

    # normalize the matrix
    return gram_product.div(batch_size * num_feature_maps * width * height)
