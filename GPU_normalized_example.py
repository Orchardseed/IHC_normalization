from Macenko import NormalizationGPU
from pathlib import Path
import torch
import pickle
from tifffile import TiffWriter

import numpy as np

## Path
template_pth = Path(r'D:\study\Ki67\Ki67_GPU\Data\Image_templates')
template = Path(template_pth, 'template_1.tif')
original_image = NormalizationGPU.image_reader(template, method='tifffile')
pth_out = Path(r'D:\study\Ki67\Ki67_GPU\Data\Image_normalized')

## Fit model
model_pth = Path(r'D:\study\Ki67\Ki67_GPU\Data\Models')
model = open(Path(model_pth, "macenko1_GPU3.pickle"), "rb")
normalizer = pickle.load(model)
model.close()


## Standardize brightness
to_transform = original_image

# if torch.cuda.is_available():
#     device = "cuda:0"
# else:
#     print('no')

## Normalization_1 recommend
device = "cuda:0" if torch.cuda.is_available() else "cpu"
to_transform_final = normalizer.transform(to_transform).to(device).squeeze(0)
to_transform_final = to_transform_final.detach().numpy()  # can't convert to np.uint8
with TiffWriter(Path(pth_out, 'template_1_normalized_gpu4.tif'), bigtiff=True) as tif:
    tif.write(to_transform_final, photometric='rgb')

print('Mocenko done')




