import keras
import numpy as np
import os as os

import datetime
from datetime import timedelta
from PIL import Image
import requests
from io import BytesIO
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

#from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.imagenet_utils import decode_predictions

from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

def loadDataSet (fp,glbCnt):
    #read classes
    classes = []
    fclasses = open(r'Images/classes.txt','r')
    lines = fclasses.readlines()
    for line in lines:
        sep = line.split(' ')      
        classes.append( sep[0])
    fclasses.close()

    f = open(fp,'r')
    trainlist = [()]
    sn = []
    for i in range(glbCnt):      
        line = f.readline()
        sep = line.split('\t')
        category_sn = sep[0]
        temp = category_sn.split('_')
        category = temp[0]
        sn.append(temp[1])
        url = sep[1]            
        url = url[0:-1]
        trainlist.append((url,category))
    f.close()
    return trainlist[1:],sn

def getImage(url,sn):        
    try :
        response = requests.get(url[(0)],timeout = 5)
        img = Image.open(BytesIO(response.content))
    except (Exception):
        return None  

    imgnew = img.resize((224,224),Image.ANTIALIAS)

    numpy_image = img_to_array(imgnew) 

    image_batch = np.expand_dims(numpy_image, axis=0)
    if image_batch.shape[3] != 3: #not in color
        return None
    try:
        imgnew.save(r'Images/'+url[1]+'_'+sn+r'.jpg')
    except (Exception):
        pass #dummy
    return image_batch

def getImageLocal(path):        
    try :        
        img = Image.open(path)
    except (Exception):
        return None  
    numpy_image = img_to_array(img) 
    image_batch = np.expand_dims(numpy_image, axis=0)
    return image_batch

def getCategoryFromTalNet(talCat):
    if talCat == 0:
        return 'Egyptian_cat'
    if talCat == 1:
        return 'Persian_cat'
    if talCat == 2:
        return 'Siamese_cat'
    if talCat == 3:
        return 'tiger_cat'
    if talCat == 4:
        return 'tabby'
    return 'Madagascar_cat'

def getCategoryFromImgNet(imagenetCat):
    if imagenetCat == 'n02124075':
        return 0
    if imagenetCat == 'n02123394':
        return 1
    if imagenetCat == 'n02123597':
        return 2
    if imagenetCat == 'n02123159':
        return 3
    if imagenetCat == 'n02123045':
        return 4   
    return 5
         
def getImages(trainlist,sn,glbCnt):
    imgs = []
    classes = []
    for i in range(glbCnt):
        c= trainlist[glbCnt-1]
        c1 = c[1]
        c2=getCategoryFromImgNet(c1)
        path = 'Images/'+c1+'_'+sn[glbCnt-1]+'.jpg'
        if os.access(path,os.R_OK):
            img = getImageLocal(path)
        else:
            img = getImage(trainlist[glbCnt-1],sn[glbCnt-1])
        if img is not None:
            imgs.append(img)
            classes.append(np.array(c2))        
        glbCnt-=1
    sclasses = np_utils.to_categorical(classes,6)
    return np.vstack(imgs),sclasses,glbCnt

def definemodel():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(224, 224, 3)))
    model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())

    model.add(Dense(6, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
def predict (model,image,cat):
    #start = datetime.datetime.now()
    predictions = model.predict(image)    
    #end = datetime.datetime.now()
    #print ('Time for Forward pass: '+str((end-start).microseconds))
    t1 = getCategoryFromTalNet(np.argmax(cat))
    t2 = getCategoryFromTalNet(np.argmax(predictions))
    if t1 == t2:
        return True
    return False
    
def train (model,datagen,trainImages,fclasses,epochs,batch_size,pathToSaveWeights):
    checkpointer = ModelCheckpoint(filepath=pathToSaveWeights, 
                               verbose=1, save_best_only=False)

    model.fit_generator(datagen.flow(trainImages, fclasses, batch_size=batch_size),                        
                        steps_per_epoch=trainImages.shape[0] // batch_size,
                        epochs=epochs, callbacks=[checkpointer], verbose=1)
    return model

def generate(train_tensors):
    datagen = ImageDataGenerator(
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=True) 

    datagen.fit(train_tensors)
    return datagen


def training (pathToWeights,epochs,batch_size):
    glbCnt = 6000
    fp = 'Images/imagenet_fall11_urls/traincats.txt'
    trainlist,sn = loadDataSet(fp,glbCnt)
    images,trainclasses,glbCnt = getImages(trainlist,sn,glbCnt)
    model = definemodel()
    datagen = generate(images)
    if pathToWeights != None:
        model.load_weights(pathToWeights)
    train(model,datagen,images,trainclasses,epochs,batch_size,'saved_models/weights_model1.hdf5')

    # url2 = (r'https://cdn2-www.cattime.com/assets/uploads/gallery/siamese-cats-and-kittens-pictures/siamese-cat-kitten-picture-5.jpg',1)
    # testimage = getImage(url2,['test'])
    # predict(model,testimage,trainclasses)
    
def acc (pathToWeights):
    glbCnt = 605
    fp = 'Images/imagenet_fall11_urls/validatecats.txt'
    trainlist,sn = loadDataSet(fp,glbCnt)
    images,trainclasses,glbCnt = getImages(trainlist,sn,glbCnt)
    model = definemodel()

    model.load_weights(pathToWeights)
    good = 0
    ovrall = 0
    for img,cat in zip(images,trainclasses) :
        img = np.expand_dims(img,axis=0)
        ovrall +=1
        if (predict(model,img,cat)):
            good +=1

    per= good/ovrall
    print ('***Accuracy = '+str(per))

if __name__ == '__main__':
    raw = input('Welcome to Tal network\r\n'+
    'For Training press t\r\na'+
    'For Accuracy test press a\r\n'+
    'Any other key to quit\r\n')
    char = raw.split()
    char[0] = char[0].upper()
    if char[0] == 'T':
        training('saved_models/weights_model1.hdf5',50,20)
    elif char[0] == 'A':
        acc('saved_models/weights_model1.hdf5')



    

