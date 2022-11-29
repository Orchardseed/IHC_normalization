import numpy as np
import spams
import copy
import cv2 as cv
from PIL import Image
import tifffile

"""
Original author: https://github.com/sebastianffx/stainlib
"""

class ExtractiveStainNormalizer(object):
    def __init__(self, method):
        if method.lower() == 'macenko':
            self.extractor = MacenkoStainExtractor
        else:
            raise Exception('Method not recognized.')

    def fit(self, target):
        """
        Fit to a target image.
        :param target: Image RGB uint8.
        :return:
        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))

    # def test(self, target):
    #     stain_matrix_target = self.extractor.get_stain_matrix(target)
    #     target_concentrations = get_concentrations(target, self.stain_matrix_target)
    #     maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
    #     return stain_matrix_target, target_concentrations, maxC_target


    def transform(self, I):
        """
        Transform an image.
        :param I: Image RGB uint8.
        :return:
        """
        stain_matrix_source = self.extractor.get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))

        source_concentrations *= (self.maxC_target / maxC_source)
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)

    def transform_tile(self, I, I_matrix, I_concentrations):
        """
        Transform an image.
        :param I: Image RGB uint8.
        :param I_matrix: Image RGB uint8.
        :param I_concentrations: Image RGB uint8.
        :return:
        """
        # stain_matrix_source_sample = I_matrix
        # source_concentrations_sample = I_concentrations
        source_concentrations = get_concentrations(I, I_matrix)
        maxC_source = np.percentile(I_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (self.maxC_target / maxC_source)
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)


class LuminosityStandardizer(object):
    @staticmethod
    def standardize(I, percentile=95):
        """
        Transform image I to standard brightness.
        Modifies the luminosity channel such that a fixed percentile is saturated.
        :param I: Image uint8 RGB.
        :param percentile: Percentile for luminosity saturation. At least (100 - percentile)% of pixels should be fully luminous (white).
        :return: Image uint8 RGB with standardized brightness.
        """
        assert is_uint8_image(I), "Image should be RGB uint8."
        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L_float = I_LAB[:, :, 0].astype(float)
        p = np.percentile(L_float, percentile)
        I_LAB[:, :, 0] = np.clip(255 * L_float / p, 0, 255).astype(np.uint8)
        I = cv.cvtColor(I_LAB, cv.COLOR_LAB2RGB)
        return I


class LuminosityThresholdTissueLocator(object):  #ABCTissueLocator
    @staticmethod
    def get_tissue_mask(I, luminosity_threshold=0.8):
        """
        Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
        Typically we use to identify tissue in the image and exclude the bright white background.
        :param I: RGB uint 8 image.
        :param luminosity_threshold: Luminosity threshold.
        :return: Binary mask.
        """
        assert is_uint8_image(I), "Image should be RGB uint8."
        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
        mask = L < luminosity_threshold

        # Check it's not empty
        if mask.sum() == 0:
            raise TissueMaskException("Empty tissue mask computed")
        return mask

class TissueMaskException(Exception):
    pass


class MacenkoStainExtractor(object):  #ABCStainExtractor
    @staticmethod
    def get_stain_matrix(I, luminosity_threshold=0.8, angular_percentile=99):
        """
        Stain matrix estimation via method of:
        M. Macenko et al. 'A method for normalizing histology slides for quantitative analysis'
        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param angular_percentile:
        :return:
        """
        assert is_uint8_image(I), "Image should be RGB uint8."
        # Convert to OD and ignore background
        tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        OD = OD[tissue_mask]
        # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
        # The two principle eigenvectors
        V = V[:, [2, 1]]
        # Make sure vectors are pointing the right way
        if V[0, 0] < 0: V[:, 0] *= -1
        if V[0, 1] < 0: V[:, 1] *= -1
        # Project on this basis.
        That = np.dot(OD, V)
        # Angular coordinates with repect to the principle, orthogonal eigenvectors
        phi = np.arctan2(That[:, 1], That[:, 0])
        # Min and max angles
        minPhi = np.percentile(phi, 100 - angular_percentile)
        maxPhi = np.percentile(phi, angular_percentile)
        # the two principle colors
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        # Order of H and E.
        # H first row.
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])
        return normalize_matrix_rows(HE)


def convert_RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).
    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = (I == 0)
    I_masked = copy.deepcopy(I)
    I_masked[mask] = 1
    #I[mask] = 1
    return np.maximum(-1 * np.log(I_masked / 255), 1e-6)

def get_concentrations(I, stain_matrix, regularizer=0.01):
    """
    Estimate concentration matrix given an image and stain matrix.
    :param I:
    :param stain_matrix:
    :param regularizer:
    :return:
    """
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T

def normalize_matrix_rows(A):
    """
    Normalize the rows of an array.
    :param A: An array.
    :return: Array with rows normalized.
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def is_image(I):
    """
    Is I an image.
    """
    if not isinstance(I, np.ndarray):
        return False
    if not I.ndim == 3:
        return False
    return True

def is_uint8_image(I):
    """
    Is I a uint8 image.
    """
    if not is_image(I):
        return False
    if I.dtype != np.uint8:
        return False
    return True

def image_reader(path, method='pillow'):
    """
    Open image with different format.
    """
    if method.lower() == 'pillow':
        image = Image.open(path)
        I = np.asarray(image)
        assert is_uint8_image(I), "Image should be RGB uint8."
        return I

    elif method.lower() == 'tifffile':
        I = tifffile.imread(path)
        assert is_uint8_image(I), "Image should be RGB uint8."
        return I

    elif method.lower() == 'openslide':
        raise Exception(
            """If you want to use openslide in windows, please use this structure:
        import os
        OPENSLIDE_PATH = Path(r'path\\to\\openslide-win64-20221111\\bin')
        if hasattr(os, 'add_dll_directory'):
            # Python >= 3.8 on Windows
            with os.add_dll_directory(OPENSLIDE_PATH):
                import openslide
        else:
            import openslide
    
        pth = Path(r'path\\to\\image')
        image = openslide.OpenSlide(pth)
        I = np.asarray(image)
        """)

    else:
        raise Exception('Method not recognized.')

