import sys
from pathlib import Path
pth = Path(Path.cwd().parent.parent)
sys.path.append(str(pth))
print(pth)
import torch
import pickle
import numpy as np
from tifffile import TiffWriter
from Macenko import NormalizationGPU


'''
Unfinished, NormalizationGPU.py still have error. We got wrong staining vector of HDAB.
'''


## Path
template_pth = Path(pth, 'Data', 'Image_templates')
template = Path(template_pth, 'template_1.tif')
original_image = NormalizationGPU.image_reader(template, method='tifffile')
pth_out = Path(pth, 'Data', 'Image_normalized')

## Fit model
model_pth = Path(pth, 'Data', 'Models')
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
to_transform_final = to_transform_final.detach().numpy()
with TiffWriter(Path(pth_out, 'template_1_normalized_gpu4.tif'), bigtiff=True) as tif:
    tif.write(to_transform_final, photometric='rgb')


print('Mocenko done')




