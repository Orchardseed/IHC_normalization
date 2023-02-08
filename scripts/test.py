from pathlib import Path
import tifffile
from tifffile import TiffWriter
import numpy as np
template_pth = Path(r'/Data/Image_normalized')
template1_pth = Path(template_pth, 'template_3_normalized_gpu1.tif')
image_ndarray = tifffile.imread(template1_pth)
image = (image_ndarray * 255).round().astype(np.uint8)
with TiffWriter(Path(template_pth, 'template_3_normalized_gpu1s.tif'), bigtiff=True) as tif:
    tif.write(image, photometric='rgb')

