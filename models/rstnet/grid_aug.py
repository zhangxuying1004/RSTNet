import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    # returns an 2D embedding for visual feature
    # follow implementation of https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/position_encoding.py

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        not_mask = (mask == False)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        pos = pos.flatten(1, 2)
        return pos


# scale the pos to (0 ~ 1)
def get_relative_pos(x, batch_size, norm_len):
    x = x.view(1, -1, 1).expand(batch_size, -1, -1)
    return  x / norm_len


def get_grids_pos(batch_size, seq_len, grid_size=(7, 7)):
    assert seq_len == grid_size[0] * grid_size[1]

    # record the pos of each grid according to the form of region box
    x = torch.arange(0, grid_size[0]).float().cuda()
    y = torch.arange(0, grid_size[1]).float().cuda()

    px_min = x.view(-1, 1).expand(-1, grid_size[0]).contiguous().view(-1)
    py_min = y.view(1, -1).expand(grid_size[1], -1).contiguous().view(-1)

    px_max = px_min + 1
    py_max = py_min + 1

    # scale pos
    rpx_min = get_relative_pos(px_min, batch_size, grid_size[0])
    rpy_min = get_relative_pos(py_min, batch_size, grid_size[1])

    rpx_max = get_relative_pos(px_max, batch_size, grid_size[0])
    rpy_max = get_relative_pos(py_max, batch_size, grid_size[1])

    return rpx_min, rpy_min, rpx_max, rpy_max


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size, seq_len = f_g.size(0), f_g.size(1)
    x_min, y_min, x_max, y_max = get_grids_pos(batch_size, seq_len)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)
