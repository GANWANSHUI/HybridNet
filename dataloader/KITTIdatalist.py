import torch.utils.data as data
import random
from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader2012(filepath, split = False):

  left_fold  = 'colored_0/'
  right_fold = 'colored_1/'
  disp_noc   = 'disp_occ/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]


  if not split:

    np.random.seed(2)
    random.shuffle(image)
    train = image[:]
    val = image[160:]

  else:

    train = image[:160]
    val   = image[160:]



  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train = [filepath+disp_noc+img for img in train]


  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val = [filepath+disp_noc+img for img in val]

  return left_train, right_train, disp_train, left_val, right_val, disp_val




def dataloader2015(filepath, split = False ):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'
  disp_R = 'disp_occ_1/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]



  all_index = np.arange(200)
  np.random.seed(2)
  np.random.shuffle(all_index)
  vallist = all_index[:40]


  val = ['{:06d}_10.png'.format(x) for x in vallist]
  print("list val:", val)

  if split:
    train = [x for x in image if x not in val]
  # train = [x for x in image if x not in val]

  else:
    train = [x for x in image]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train_L = [filepath+disp_L+img for img in train]
  #disp_train_R = [filepath+disp_R+img for img in train]

  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val_L = [filepath+disp_L+img for img in val]
  #disp_val_R = [filepath+disp_R+img for img in val]

  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L

