import keras
from keras.utils import np_utils
import numpy as np

import os
import cv2
import random
from glob import glob
import keras
from tensorflow.keras.layers import Input, Convolution2D, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import hsv_to_rgb
import keras.utils as image
from keras.utils import load_img, img_to_array
import mahotas
from tkinter import filedialog

clas1 = [item[10:-1] for item in sorted(glob("./dataset/*/"))]

from keras.preprocessing import image
from tqdm import tqdm


# Note: modified these two functions, so that we can later also read the inception tensors which
# have a different format
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    print(img_path)
    img = load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)


# vilization_and_show()


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# from tkinter import filedialog
# filename = filedialog.askopenfilename(title='open')

# main_img = cv2.imread(filename)


filename = filedialog.askopenfilename(title='open')
img = cv2.imread(filename)
plt.imshow(img)
plt.show()
bins = 8

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(hsv_img)

lower_green = np.array([25, 0, 20])
upper_green = np.array([100, 255, 255])
mask = cv2.inRange(hsv_img, lower_green, upper_green)
result = cv2.bitwise_and(img, img, mask=mask)
plt.subplot(1, 1, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 1, 1)
plt.imshow(result)
plt.show()

lower_brown = np.array([10, 0, 10])
upper_brown = np.array([30, 255, 255])
disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
disease_result = cv2.bitwise_and(img, img, mask=disease_mask)
plt.subplot(1, 2, 1)
plt.imshow(disease_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(disease_result)
plt.show()

'''
final_mask = mask + disease_mask
final_result = cv2.bitwise_and(img, img, mask=final_mask)
plt.figure(figsize=(15,15))
plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()
   '''
from twilio.rest import Client

account_sid = 'ACcae318c2e605e865c905d00aebe5acc3'
auth_token = 'e6c4732b554f25234fb8937e06cb8db7'

client = Client(account_sid, auth_token)
from tensorflow.keras.models import load_model

model = load_model('trained_model_CNN.h5')

test_tensors = paths_to_tensor(filename) / 255
pred = model.predict(test_tensors)
print('Given Image Predicted is : ' + str(clas1[np.argmax(pred)]))

if "downy mildew" == clas1[np.argmax(pred)]:
    message = client.messages.create(from_='+15675871309',
                                     body='Affected grape plantation  disease:downy mildew   pesticide:chlorothalonil and mancozeb',
                                     to='+919686727370')
if "powdery mildew" == clas1[np.argmax(pred)]:
    message = client.messages.create(from_='+15675871309',
                                     body='Affected grape plantation  disease:powdery mildew    pesticide:potassium bicarbonate ',
                                     to='+919686727370')
if "healthy" == clas1[np.argmax(pred)]:
    message = client.messages.create(from_='+15675871309', body='healthy plant', to='+919686727370')

