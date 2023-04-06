__author__ = 'aymgal', 'martin-millon'

import numpy as np
from scipy import ndimage
from skimage import morphology
import matplotlib.pyplot as plt


class ImageMask(object):
    """
    Generates a pixel mask to a specific image shape.

    Sizes privided by the user (margin, radius, center, ...) shoud be in arcsec.

    Code borrowed from TDLMCpipeline package (credits: A. Galan & M. Millon).
    """

    def __init__(self, mask_shape, delta_pix, mask_type_list=['square'], 
                 margin_list=[10], radius_list=[], center_list=[], angle_list=[], axis_ratio_list=[], 
                 operation_list=[], inverted_list=[False], verbose=False):
        """
        :param mask_shape: 2D numpy array shape
        :param delta_pix: size of a pixel in physical units
        :param mask_type_list: list of string among 'square', 'circle', 'ellipse'
        :param margin_list: only for 'square' mask (ignored with others), the width of margins on all sides (in arcsec)
        :param radius_list: only for 'circle' and 'ellipse' masks (ignored with others), list of radii of each circle
        :param center_list: only for 'circle' and 'ellipse' masks (ignored with others), list of centers [x, y] of each circle, 
        :param (None are replaced by the center of the image)
        :param angle_list: only for 'ellipse' mask, list of ellipse orientation angles in degrees
        :param axis_ratio_list: only for 'ellipse' mask, list of axis ratio b/a
        :param operation_list: only supported for 'circle' masks, to control how to 
        combine successive component masks in the above list: 'union', 'inter', 'subtract'
        :param inverted_list: only supported for 'circle' and 'ellipse' masks, invert (True) or not (False) each component mask
        """
        if len(mask_shape) != 2:
            raise ValueError("Only 2D arrays are supported for mask creation (for now)")

        self.mask_type_list = mask_type_list
        self.num_components = len(mask_type_list)
        self._mask_shape = mask_shape
        self._delta_pix = delta_pix

        # translate physical units to pixels
        dp = self._delta_pix
        self._margin_list = [m / dp if m is not None else 0. for m in margin_list]
        self._radius_list = [r / dp if r is not None else 0. for r in radius_list]
        self._center_list = [(c[0] / dp, c[1] / dp) if c is not None else None for c in center_list]

        # translate degrees to radians
        self._angle_list = [a * np.pi / 180. if a is not None else None for a in angle_list]
        self._axis_ratio_list = axis_ratio_list

        self._operation_list = operation_list
        self._inverted_list  = inverted_list
        self._fill_list_gaps()

        if any([mt not in ['square', 'circle', 'ellipse'] for mt in mask_type_list]):
            self.num_components = 0
            if len(operation_list) > 0 and verbose:
                print("WARNING : operations on are not supported on 'margin' masks")
            elif len(inverted_list) > 0 and verbose:
                print("WARNING : mask-by-mask inversions are not supported on 'margin' masks")

    def get_mask(self, inverted=False, smoothed=False, show_details=False, convert_to_bool=False):
        """
        inverted : if True, invert the whole mask
        show_details : if True, show a plot of the mask, and possibly the steps followed to build it 
        """
        if self.num_components > 0:
            mask, mask_list = None, []
            for i, mask_type in enumerate(self.mask_type_list):
                inv = self._inverted_list[i]

                if mask_type == 'circle':
                    c = self._center_list[i]
                    r = self._radius_list[i]
                    mask_tmp = self._create_circular_mask(radius=r, center=c,
                                                          inverted=inv)
                elif mask_type == 'ellipse':
                    c = self._center_list[i]
                    r = self._radius_list[i]
                    phi = self._angle_list[i]
                    q = self._axis_ratio_list[i]
                    mask_tmp = self._create_elliptical_mask(radius=r, center=c,
                                                            phi=phi, q=q,
                                                            inverted=inv)
                elif mask_type == 'square':
                    m = self._margin_list[i]
                    mask_tmp = self._create_margin_mask(margin=m, inverted=inv)

                # save mask as it is for showing details
                mask_list.append(mask_tmp)

                # combine the new mask with current one
                if i > 0:
                    op = self._operation_list[i-1]
                    mask = self.combine_masks(mask, mask_tmp, operation=op)
                else:
                    mask = mask_tmp

        else:
            mask = np.ones(self._mask_shape)

        if inverted:
            mask = self._invert(mask)

        if smoothed is True:
            # dilation followed by gaussian filtering with sigma = 1 pixel
            mask_s = morphology.binary_dilation(mask).astype(float)
            mask_s = ndimage.gaussian_filter(mask_s, 1, mode='nearest')
            # re-normalize so max is 1
            mask = mask_s / mask_s.max()

        if show_details:
            self._plot_details(mask, mask_list)

        if convert_to_bool and not smoothed:
            mask = mask.astype(bool)

        return mask

    def _fill_list_gaps(self):
        num_inv = len(self._inverted_list)
        if num_inv < self.num_components:
            gap = self.num_components - num_inv
            self._inverted_list += [False] * gap

        num_op = len(self._operation_list)
        if num_op < self.num_components - 1:
            gap = self.num_components - num_op
            self._operation_list += ['union'] * gap

    def _create_margin_mask(self, margin=5, inverted=False):
        if margin < self._delta_pix:
            return np.ones(self._mask_shape)
        else:
            mask = np.zeros(self._mask_shape)
            m = int(margin)
            mask[m:-m, m:-m] = 1
            return self._invert(mask) if inverted else mask

    def _create_circular_mask(self, radius=5, center=None, inverted=False):
        if radius < self._delta_pix:
            return np.ones(self._mask_shape)
        else:
            r = radius
            nx, ny = self._mask_shape
            if center is None:
                cx, cy = int(nx/2), int(ny/2)  # by default : center of the image
            else:
                cx, cy = center
            mask = np.zeros(self._mask_shape)
            y, x = np.ogrid[-cx:nx-cx, -cy:ny-cy]
            disc = x*x + y*y < r*r
            mask[disc] = 1
            return self._invert(mask) if inverted else mask

    def _create_elliptical_mask(self, radius=5, center=None, phi=None, q=None, inverted=False):
        if radius < self._delta_pix:
            return np.ones(self._mask_shape)
        else:
            r = radius
            nx, ny = self._mask_shape
            if center is None:
                cx, cy = int(nx/2), int(ny/2)  # by default : center of the image
            else:
                cx, cy = center
            if phi is None:
                phi = 0
            if q is None:
                q = 1
            mask = np.zeros(self._mask_shape)
            y, x = np.ogrid[-cx:nx-cx, -cy:ny-cy]
            xp =  np.cos(phi)*x + np.sin(phi)*y
            yp = -np.sin(phi)*x + np.cos(phi)*y
            ellipse = xp*xp + yp*yp/(q*q) < r*r
            mask[ellipse] = 1
            return self._invert(mask) if inverted else mask

    def _invert(self, mask):
        ones  = np.where(mask == 1)
        zeros = np.where(mask == 0)
        mask[ones]  = 0
        mask[zeros] = 1
        return mask

    def _plot_details(self, mask, mask_list):
        if len(mask_list) == 1:
            fig, axes = plt.subplots(1, 1, figsize=(4, 3))
            ax = axes
        else:
            fig, axes = plt.subplots(1, self.num_components+1, 
                                     figsize=(4*self.num_components, 3))
            for i in range(self.num_components):
                ax = axes[i]
                ax.imshow(mask_list[i], cmap='gray', vmin=0, vmax=1, origin='lower')
                ax.set_title("Mask {}".format(i+1))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            ax = axes[-1]
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1, origin='lower')
        ax.set_title("Final mask")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()

    def combine_masks(self, mask1, mask2, operation='inter'):
        assert mask1.shape == mask2.shape, "Both masks should have same shape to be combined"
        if operation == 'union':
            mask_combined = np.zeros_like(mask1)
            mask_combined[(mask1 == 1) | (mask2 == 1)] = 1
        elif operation == 'inter':
            mask_combined = np.zeros_like(mask1)
            mask_combined[(mask1 == 1) & (mask2 == 1)] = 1
        elif operation == 'subtract':
            mask_combined = mask1 - mask2
            mask_combined[mask_combined < 0] = 0
        return mask_combined
