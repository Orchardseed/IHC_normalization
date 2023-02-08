from Macenko import NormalizationCPU
import pickle
from pathlib import Path

## Path
model_pth = Path(r'/Data/Models')
template_pth = Path(r'/Data/Image_templates')
template1_pth = Path(template_pth, 'template_1.tif')
template1 = NormalizationCPU.image_reader(template1_pth, method='tifffile')

## Fit model
normalizer = NormalizationCPU.ExtractiveStainNormalizer(method='macenko')
# target = template1
target = NormalizationCPU.LuminosityStandardizer.standardize(template1)
normalizer.fit(target) ## The staining vectors of H and DAB will be printed here

## Save model
model = open(Path(model_pth, "macenko1_CPUx.pickle"), "wb")  ## Remember to change the saved model name
pickle.dump(normalizer, model)
model.close()
print('Model saved')

# # Open model
# model = open(Path(model_pth, "macenko1.pickle"), "rb")
# normalizer = pickle.load(model)
# model.close()








