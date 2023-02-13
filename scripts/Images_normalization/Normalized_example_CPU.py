import sys
from pathlib import Path
pth = Path(Path.cwd().parent.parent)
sys.path.append(str(pth))
print(pth)
import pickle
import numpy as np
from tifffile import TiffWriter
from Macenko import NormalizationCPU



## Path
template_pth = Path(pth, 'Data', 'Image_templates')
template = Path(template_pth, 'template_1.tif')
original_image = NormalizationCPU.image_reader(template, method='tifffile')
pth_out = Path(pth, 'Data', 'Image_normalized')

## Fit model
model_pth = Path(pth, 'Data', 'Models')
model = open(Path(model_pth, "macenko1_CPU.pickle"), "rb")
normalizer = pickle.load(model)
model.close()


## Standardize brightness
## This step is optional but can improve the tissue mask calculation (sometimes good, sometimes bad)
# to_transform = NormalizationCPU.LuminosityStandardizer.standardize(original_image, percentile=95)
to_transform = original_image

## Normalization_1 recommend
transformed = normalizer.transform(to_transform)
to_transform_final = np.transpose(transformed, (2,0,1))
with TiffWriter(Path(pth_out, 'template_1_normalized_macenko1.tif'), bigtiff=True) as tif:  ## remember to change the saved image name
    tif.write(to_transform_final, photometric='rgb')


# ## Normalization_2 (use a small tile as sample for WSI)
# slide_tile_matrix = NormalizationCPU.MacenkoStainExtractor.get_stain_matrix(to_transform)
# slide_tile_concentrations = NormalizationCPU.get_concentrations(to_transform, slide_tile_matrix)
# transformed_tile = normalizer.transform_tile(to_transform, slide_tile_matrix, slide_tile_concentrations)
# to_transform_final_tile = np.transpose(transformed_tile, (2,0,1))
# with TiffWriter(Path(pth_out, 'normalized2.tif'), bigtiff=True) as tif:
#     tif.write(to_transform_final_tile, photometric='rgb')


print('Mocenko done')




