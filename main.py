import torch
import torchvision
import torchtext
import spacy
import jieba

print(f'torch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')
print(f'torchtext version: {torchtext.__version__}')
print(f'spacy version: {spacy.__version__}')
print(f'jieba version: {jieba.__version__}')


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")