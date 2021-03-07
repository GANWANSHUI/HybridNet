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


def dataloader(filepath, split = False ):

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

  # train = image[:160]
  # val   = image[160:]


  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train_L = [filepath+disp_L+img for img in train]
  #disp_train_R = [filepath+disp_R+img for img in train]

  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val_L = [filepath+disp_L+img for img in val]
  #disp_val_R = [filepath+disp_R+img for img in val]

  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L



def MD_dataloader():

  filepath = '/home/wsgan/HybridNet/1500*1000/1500MiddEval3-data-H1000/MiddEval3/trainingH/'
  disppath = '/home/wsgan/HybridNet/1500*1000/1500MiddEval3-GT0-H1000/MiddEval3/trainingH/'

  # filepath = '/home/wsgan/HybridNet/750*500/750MiddEval3-data-Q500/MiddEval3/trainingQ/'
  # disppath = '/home/wsgan/HybridNet/750*500/750MiddEval3-GT0-Q500/MiddEval3/trainingQ/'



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


  #pdb.set_trace()

  # Testpath = '/home/wsgan/HybridNet/750*500/750MiddEval3-data-Q500/MiddEval3/testQ/'
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
