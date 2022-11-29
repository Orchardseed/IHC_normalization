from __future__ import annotations
from pathlib import Path
from typing import Tuple
from PIL import Image
import tifffile
import torch
import torch.nn as nn
from torchvision import transforms



def is_image(image_tensor):
    """
    Is I an image.
    """
    if not isinstance(image_tensor, torch.Tensor):
        return False
    return True

def image_reader(path: Path, method='pillow'):
    """
    Open image with different format.
    """
    if method.lower() == 'pillow':
        image = Image.open(path)
        image_ndarray = torch.asarray(image)
        image_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])(image_ndarray)
        assert is_image(image_tensor), "Image should be RGB."
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    elif method.lower() == 'tifffile':
        image_ndarray = tifffile.imread(path)
        image_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])(image_ndarray)
        assert is_image(image_tensor), "Image should be RGB."
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

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
        image_ndarray = torch.asarray(image)
        image_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])(image_ndarray)
        """)
    else:
        raise Exception('Method not recognized.')



class MacenkoNormalizer(nn.Module):
    HE_REFERENCE = torch.Tensor([[0.5042, 0.1788], [0.7723, 0.8635], [0.3865, 0.4716]])
    MAX_CON_REFERENCE = torch.Tensor([1.3484, 1.0886])
    """
            Normalize staining appearence of hematoxylin & eosin stained images. Based on [1].
            Parameters
            ----------
            alpha : float
                Percentile
            beta : float
                Transparency threshold
            transmitted_intensity : int
                Transmitted light intensity
            learnable : bool
                If true, the normalization matrix will be learned during training.
            References
            ----------
            [1] A method for normalizing histology slides for quantitative analysis. M. Macenko et al., ISBI 2009
            """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.20,
        transmitted_intensity: int = 255,
        learnable: bool = True,
    ):
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._transmitted_intensity = transmitted_intensity
        self._learnable = learnable
        self._target_he_matrix = nn.Parameter(self.HE_REFERENCE, requires_grad=self._learnable)
        self._target_max_concentration = nn.Parameter(self.MAX_CON_REFERENCE, requires_grad=self._learnable)


    def convert_rgb_to_OD(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        # calculate optical density
        optical_density = -torch.log(
            (image_tensor.reshape((-1, image_tensor.shape[-1])).float() + 1) / self._transmitted_intensity
        )
        # remove transparent pixels
        optical_density_hat = optical_density[~torch.any(optical_density < self._beta, dim=1)]
        if optical_density_hat.numel() == 0:
            raise RuntimeError(f"The batch contains tiles with only transparent pixels.")
        return optical_density, optical_density_hat

    def percentile(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        """
        Original author: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
        Parameters
        ----------
        tensor: torch.Tensor
            input tensor for which the percentile must be calculated.
        value: float
            The percentile value
        Returns
        -------
        ``value``-th percentile of the input tensor's data.
        Notes
        -----
         Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         Values are not interpolated, which corresponds to
           ``numpy.percentile(..., interpolation="nearest")``
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if value is a np.float32.
        k = 1 + round(0.01 * float(value) * (tensor.numel() - 1))
        return tensor.view(-1).kthvalue(k).values

    def find_he_components(self, optical_density_hat: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        """
        This function -
        1. Computes the H&E staining vectors by projecting the OD values of the image pixels on the plane
        spanned by the eigenvectors corresponding to their two largest eigenvalues.
        2. Normalizes the staining vectors to unit length.
        3. Calculates the angle between each of the projected points and the first principal direction.
        Parameters:
        ----------
        optical_density_hat: torch.Tensor
            Optical density of the image
        eigvecs: torch.Tensor
            Eigenvectors of the covariance matrix
        Returns:
        -------
        he_components: torch.Tensor
            The H&E staining vectors
        """
        t_hat = torch.matmul(optical_density_hat, eigvecs)
        phi = torch.atan2(t_hat[:, 1], t_hat[:, 0])

        min_phi = self.percentile(phi, self._alpha)
        max_phi = self.percentile(phi, 100 - self._alpha)
        v_min = torch.matmul(eigvecs, torch.stack((torch.cos(min_phi), torch.sin(min_phi)))).unsqueeze(1)
        v_max = torch.matmul(eigvecs, torch.stack((torch.cos(max_phi), torch.sin(max_phi)))).unsqueeze(1)

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        he_vector = torch.where(v_min[0] > v_max[0], torch.cat((v_min, v_max), dim=1), torch.cat((v_max, v_min), dim=1))
        he_vector_normalized = self.normalize_matrix_rows(he_vector)
        return he_vector_normalized

    def normalize_matrix_rows(self, A):
        return A / torch.linalg.norm(A, axis=1)[:, None]

    def __compute_matrices(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the H&E staining vectors and their concentration values for every pixel in the image tensor.
        """
        # Convert RGB values in the image to optical density values following the Beer-Lambert's law.
        optical_density, optical_density_hat = self.convert_rgb_to_OD(image_tensor)
        # Calculate the eigenvectors of optical density matrix thresholded to remove transparent pixels.
        _, eigvecs = torch.linalg.eigh(torch.cov(optical_density_hat.T), UPLO="U")
        # choose the first two eigenvectors corresponding to the two largest eigenvalues.
        eigvecs = eigvecs[:, [1, 2]]
        # Find the H&E staining vectors and their concentration values for every pixel in the image tensor.
        he = self.find_he_components(optical_density_hat, eigvecs)
        concentration = torch.linalg.lstsq(he, optical_density.T).solution  # question here
        max_concentration = torch.stack([self.percentile(concentration[0, :], 99), self.percentile(concentration[1, :], 99)])
        return he, concentration, max_concentration


    def __normalize_concentrations(self, concentrations: torch.Tensor, maximum_concentration: torch.Tensor) -> torch.Tensor:
        concentrations *= (self._target_max_concentration / maximum_concentration).unsqueeze(-1)
        return concentrations

    def transpose_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.transpose(tensor, 1, 3)
        tensor = torch.transpose(tensor, 2, 3)
        return tensor

    def __create_normalized_images(self, normalized_concentrations: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
        batch, classes, height, width = image_tensor.shape
        # recreate the image using reference mixing matrix
        normalised_image_tensor = self._transmitted_intensity * torch.exp(
            -torch.matmul(self._target_he_matrix, normalized_concentrations)
        )
        normalised_image_tensor[normalised_image_tensor > 255] = 255
        normalised_image_tensor = normalised_image_tensor.T.reshape(batch, height, width, classes)
        normalised_image_tensor = self.transpose_channels(normalised_image_tensor)
        return normalised_image_tensor


    def fit(self, image_tensor: torch.Tensor) -> None:
        he_matrix, concentration, maximum_concentration = self.__compute_matrices(image_tensor)
        self._target_he_matrix = nn.Parameter(he_matrix, requires_grad=self._learnable)
        self._target_max_concentration = nn.Parameter(maximum_concentration, requires_grad=self._learnable)
        # print(he_matrix, maximum_concentration)

    def transform(self, image_tensor: torch.Tensor) -> torch.Tensor:
        he_matrix, concentrations, maximum_concentration = self.__compute_matrices(image_tensor)
        concentrations = self.__normalize_concentrations(concentrations, maximum_concentration)
        tmp = self.__create_normalized_images(concentrations, image_tensor)
        return tmp

