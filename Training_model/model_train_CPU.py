from Macenko import NormalizationCPU
import pickle
from pathlib import Path

## Path
model_pth = Path(r'D:\study\Ki67\Ki67_GPU\Data\Models')
template_pth = Path(r'D:\study\Ki67\Ki67_GPU\Data\Image_templates')
template1_pth = Path(template_pth, 'template_1.tif')
template1 = NormalizationCPU.image_reader(template1_pth, method='tifffile')

## Fit model
normalizer = NormalizationCPU.ExtractiveStainNormalizer(method='macenko')
# target = template1
target = NormalizationCPU.LuminosityStandardizer.standardize(template1)
normalizer.fit(target)

## Save model
model = open(Path(model_pth, "macenko1_CPUx.pickle"), "wb")
pickle.dump(normalizer, model)
model.close()
print('Model saved')

# # Open model
# model = open(Path(model_pth, "macenko1.pickle"), "rb")
# normalizer = pickle.load(model)
# model.close()








