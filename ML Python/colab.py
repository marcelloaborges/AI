from google.colab import drive

drive.mount('/content/gdrive')
base_dir = '/content/gdrive/My Drive'

import os
    
os.chdir( "/content/gdrive/My Drive/colab/Super_Mario_Bros_DDQN_LSTM/" )
dirpath = os.getcwd()
print("current directory is : " + dirpath)