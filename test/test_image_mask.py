__author__ = 'aymgal'

from imd.image_mask import ImageMask

import numpy as np
import numpy.testing as npt
import pytest
import unittest


def test_image_mask():
    mask_shape = (10, 10)
    delta_pix = 1
    center_list = [None, (4, 8), (1, 1), (3, 2)]
    margin = 3
    radius_list = [3, 2, 3.5, 1]
    axis_ratio_list = [None, 0.9, 0.4, 0.2]
    angle_list = [None, 10, 20, 40]
    operation_list = ['union', 'inter', 'subtract']
    inverted_list = [True, False, False, True]
    kwargs_square = {
        'mask_type': 'square',
        'margin': margin,
        'operation_list': operation_list,
        'inverted_list': operation_list,
    }
    kwargs_circle = {
        'mask_type': 'circle',
        'radius_list': radius_list,
        'center_list': center_list,
        'operation_list': operation_list,
        'inverted_list': inverted_list,
    }
    kwargs_ellipse = {
        'mask_type': 'ellipse',
        'radius_list': radius_list,
        'center_list': center_list,
        'axis_ratio_list': axis_ratio_list,
        'angle_list': angle_list,
    }
    mask_cls_sq = ImageMask(mask_shape, delta_pix, **kwargs_square)
    mask_sq = mask_cls_sq.get_mask()
    npt.assert_equal(mask_sq, mask_sq.astype(bool))  # test it's only 0s and 1s
    assert mask_sq[0, 0] == 0
    assert mask_sq[5, 5] == 1

    mask_cls_c  = ImageMask(mask_shape, delta_pix, **kwargs_circle)
    mask_c = mask_cls_c.get_mask()
    mask_c_inv = mask_cls_c.get_mask(inverted=True)
    npt.assert_equal(mask_c, 1 - mask_c_inv)

    mask_cls_e  = ImageMask(mask_shape, delta_pix, **kwargs_ellipse)
    mask_e_smo = mask_cls_e.get_mask(smoothed=True)
    mask_e_bool = mask_cls_e.get_mask(convert_to_bool=True)
    assert mask_e_bool.dtype == bool

    # extreme case
    mask_cls_big_sq = ImageMask(mask_shape, delta_pix, mask_type='square', margin=100)
    mask = mask_cls_big_sq.get_mask()
    npt.assert_equal(mask, np.zeros(mask_shape))
    mask_cls_big_c = ImageMask(mask_shape, delta_pix, mask_type='circle', radius_list=[100], center_list=[None])
    mask = mask_cls_big_c.get_mask(inverted=True)
    npt.assert_equal(mask, np.zeros(mask_shape))


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            mask_shape = (10,)
            delta_pix = 1
            mask_class = ImageMask(mask_shape, delta_pix)
        

if __name__ == '__main__':
    pytest.main()
