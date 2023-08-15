'''
Description: 
Autor: 郭印林
Date: 2022-08-12 15:43:51
LastEditors: 郭印林
LastEditTime: 2022-08-18 19:14:50
'''
import torch
import torchaudio
import torchvision
import sys
import platform


print('System: ', platform.system())
print('Architecture: ', platform.architecture())
print('Platform version: ', platform.version())
print('Processor: ', platform.processor())

print("Python Version {}".format(str(sys.version).replace('\n', '')))
print('Torch Version: ', torch.__version__)
print('Torchaudio Version: ', torchaudio.__version__)
print("Torchvision Version {}".format(torchvision.__version__))

if torch.cuda.is_available():
    print('CUDA is available!')
    print('CUDA Version: ', torch.version.cuda)
    print('CUDA Device Count: ', torch.cuda.device_count())
    print('Current Device: ', torch.cuda.current_device())
else:
    print('CUDA is not available!')
