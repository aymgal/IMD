__author__ = 'aymgal'

from imd.image_mask import ImageMask

import numpy as np
import numpy.testing as npt
import pytest
import unittest


def test_image_mask():
    verbose = True

    mask_shape = (10, 10)
    delta_pix = 1
    center_list = [None, (4, 8), (1, 1), (3, 2)]
    margin_list = [3]
    radius_list = [3, 2, 3.5, 1]
    axis_ratio_list = [None, 0.9, 0.4, 0.2]
    angle_list = [None, 10, 20, 40]
    operation_list = ['union', 'inter', 'subtract']
    inverted_list = [True, False, False, True]
    kwargs_square = {
        'mask_type_list': ['square'],
        'margin_list': margin_list,
        'inverted_list': [False],
    }
    kwargs_circle = {
        'mask_type_list': ['circle', 'circle', 'circle', 'circle'],
        'radius_list': radius_list,
        'center_list': center_list,
        'operation_list': operation_list,
        'inverted_list': inverted_list,
    }
    kwargs_ellipse = {
        'mask_type_list': ['ellipse', 'ellipse', 'ellipse', 'ellipse'],
        'radius_list': radius_list,
        'center_list': center_list,
        'axis_ratio_list': axis_ratio_list,
        'angle_list': angle_list,
    }
    kwargs_mixed = {
        'mask_type_list': ['square', 'circle', 'ellipse'],
        'margin_list': [3, None, None],
        'radius_list': [None, 2, 3],
        'center_list': [None, None, (4, 8)],
        'axis_ratio_list': [None, 0.8, 0.7],
        'angle_list': [None, 10, 30],
        'operation_list': operation_list[:2],
        'inverted_list': inverted_list[:3],
    }
    mask_cls_sq = ImageMask(mask_shape, delta_pix, verbose=verbose, **kwargs_square)
    mask_sq = mask_cls_sq.get_mask()
    npt.assert_equal(mask_sq, mask_sq.astype(bool))  # test it's only 0s and 1s
    assert mask_sq[0, 0] == 0
    assert mask_sq[5, 5] == 1

    mask_cls_c  = ImageMask(mask_shape, delta_pix, verbose=verbose, **kwargs_circle)
    mask_c = mask_cls_c.get_mask()
    mask_c_inv = mask_cls_c.get_mask(inverted=True)
    npt.assert_equal(mask_c, 1 - mask_c_inv)

    mask_cls_e  = ImageMask(mask_shape, delta_pix, verbose=verbose, **kwargs_ellipse)
    mask_e_smo = mask_cls_e.get_mask(smoothed=True)
    mask_e_bool = mask_cls_e.get_mask(convert_to_bool=True)
    assert mask_e_bool.dtype == bool

    mask_cls_m  = ImageMask(mask_shape, delta_pix, verbose=verbose, **kwargs_mixed)
    mask_m_smo = mask_cls_m.get_mask(smoothed=True)
    mask_m_bool = mask_cls_m.get_mask(convert_to_bool=True)
    assert mask_m_bool.dtype == bool

    # extreme case
    mask_cls_big_sq = ImageMask(mask_shape, delta_pix, verbose=verbose, mask_type_list=['square'], margin_list=[100])
    mask = mask_cls_big_sq.get_mask()
    npt.assert_equal(mask, np.zeros(mask_shape))
    mask_cls_big_c = ImageMask(mask_shape, delta_pix, verbose=verbose, mask_type_list='circle', radius_list=[100], center_list=[None])
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
