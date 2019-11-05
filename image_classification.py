import numpy as np
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
from random import shuffle
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
import matplotlib.pyplot as plt 
num_train_imgs = 1064
num_test_imgs = 187
num_classes = 3
img_px = 22500  
IMG_SIZE=150
x = 'training_data\\images'
y = 'test_data\\images'
TRAIN_DIR=os.path.join(os.getcwd(),x)
TEST_DIR=os.path.join(os.getcwd(),y)
results_path = 'results'
MODEL_NAME = 'carbikeperson-{}-{}.model'.format(LR, '6conv-basic') 
def label_img(img): 
    word_label = [z for z in str(img).split('_')] 
    return word_label[0]
def create_train_data(): 
    training_data = [] 
    #training_label=[]
    for img in tqdm(os.listdir(TRAIN_DIR)): 
  
 
        label = label_img(img) 
  
        path = os.path.join(TRAIN_DIR, img) 

        if label=='bike' or label=='bikes':
            x=[1,0,0]
        elif label=='carsgraz':
            x=[0,1,0]
        else:
            x=[0,0,1]
  
        # loading the image from the path and then converting them into 
        # greyscale for easier covnet prob 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
  
        # resizing the image for processing them in the covnet 
        #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
  
        # final step-forming the training data list with numpy array of the images 
        training_data.append([np.array(img),np.array(x)])
        shuffle(training_data)
        #training_label.append(np.array(label))

  
    # saving our trained data for further uses if required 
    np.save('train_data.npy', training_data) 
    return training_data
def process_test_data(): 
    testing_data = [] 
    #testing_label=[]
    img_num=0
    for img in tqdm(os.listdir(TEST_DIR)): 
        path = os.path.join(TEST_DIR, img) 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
        testing_data.append([np.array(img),img_num])
        #testing_label.append(img_num)
        img_num+=1
        hell=np.array(img)
        shuffle(testing_data)
     
    np.save('test_data.npy', testing_data) 
    return testing_data

train_data = create_train_data() 
test_data = process_test_data()
LR = 1e-3
tf.reset_default_graph() 
convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.8) 
  
convnet = fully_connected(convnet, 3, activation ='softmax') 
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
      loss ='categorical_crossentropy', name ='targets') 
  
model = tflearn.DNN(convnet, tensorboard_dir ='log')


train = train_data[:-500] 
test = train_data[-500:] 
  
'''Setting up the features and lables'''
# X-Features & Y-Labels 
  
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
Y = [i[1] for i in train] 
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
test_y = [i[1] for i in test] 

model.fit({'input': X}, {'targets': Y}, n_epoch = 16,  
    validation_set =({'input': test_x}, {'targets': test_y}),  
    snapshot_step = 500, show_metric = True, run_id = MODEL_NAME) 
model.save(MODEL_NAME) 



#TESTING
ax=test_data[10][0]
ax.shape
y=ax.reshape(1,150,150,1)
zzz=Image.fromarray(ax)
zzz.show()
y.shape
model_out=model.predict(y)
if np.argmax(model_out)==0:
    print("Bike")
elif np.argmax(model_out)==1:
    print("car")
else:
    print("Person")


