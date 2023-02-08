from Macenko import NormalizationGPU
from pathlib import Path
import torch
import pickle
from torchvision import transforms

'''
Unfinished, NormalizationGPU.py still have error. We got wrong staining vector of HDAB.
'''

## Path
model_pth = Path(r'/Data/Models')
template_pth = Path(r'/Data/Image_templates')
template1_pth = Path(template_pth, 'template_1.tif')
template1 = NormalizationGPU.image_reader(template1_pth, method='tifffile')
print(template1.size())

## Fit model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
normalizer = NormalizationGPU.MacenkoNormalizer()
target = transforms.ColorJitter(brightness=0.95)(template1)
normalizer.fit(target.to(device))

## Save model
model = open(Path(model_pth, "macenko1_GPU3.pickle"), "wb")
pickle.dump(normalizer, model)
model.close()
print('Model saved')
