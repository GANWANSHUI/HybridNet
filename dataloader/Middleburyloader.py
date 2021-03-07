import os
import os.path
import numpy as np
import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def MD_dataloader():

  filepath = '/home/wsgan/HybridNet/1500*1000/1500MiddEval3-data-H1000/MiddEval3/trainingH/'
  disppath = '/home/wsgan/HybridNet/1500*1000/1500MiddEval3-GT0-H1000/MiddEval3/trainingH/'

  left_train = []
  right_train = []
  disp_train_L = []
  for item in os.listdir(filepath):
    #print(filepath + item)
    for img in os.listdir(filepath + item+'/'):
      #print(img)
      if img == 'im0.png':

        left_train.append(filepath + item+'/' + img)

      if img == 'im1.png':
        right_train.append(filepath + item+'/' + img)


    for disp in os.listdir(disppath + item + '/'):

      if disp == 'disp0GT.pfm':
        disp_train_L.append(disppath + item+'/' + disp)


  Testpath = '/home/wsgan/HybridNet/1500*1000/1500MiddEval3-data-H1000/MiddEval3/testH/'

  left_val = []
  right_val = []
  disp_val_L = []
  for item in os.listdir(Testpath):

    for img in os.listdir(Testpath + item + '/'):
      # print(img)
      if img == 'im0.png':
        left_val.append(Testpath + item + '/' + img)

      if img == 'im1.png':
        right_val.append(Testpath + item + '/' + img)


  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L


if __name__ == '__main__':
    MD_dataloader()
