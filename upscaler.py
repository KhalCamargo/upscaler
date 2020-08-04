from keras import layers
from keras import models
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import PIL
import imagesize
import tensorflow as tf
import os, shutil
from keras import *
import numpy as np

from keras.callbacks import CSVLogger

num_batches = 2
num_epochs = 80
num_layers = 20
num_lr = 0.01

print('batches: ' + str(num_batches) + ' epochs: ' + str(num_epochs) + ' layers: ' +  str(num_layers) + ' rate: ' + str(num_lr))

filename = str(num_epochs) + 'ep_' + str(num_layers) + 'ls_' + str(num_batches) + 'bt_' + str(num_lr) + 'lr.h5'
print("Filename will be: ", filename)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


extracted_dataset_dir = 'F:\Images Dataset\\0'
selected_dataset_dir = 'F:\Images Dataset\SelectedSizes'
original_dataset_dir = 'F:\Images Dataset\SelectedSizes'
base_dir = '.\\BaseDir'

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

FilterSizes = False

if FilterSizes:
    def findFilesInFolder(path, pathList, extension, subFolders = True):
        """  Recursive function to find all files of an extension type in a folder (and optionally in all subfolders too)

        path:        Base directory to find files
        pathList:    A list that stores all paths
        extension:   File extension to find
        subFolders:  Bool.  If True, find files in all subfolders under path. If False, only searches files in the specified folder
        """

        try:   # Trapping a OSError:  File permissions problem I believe
            for entry in os.scandir(path):
                if entry.is_file() and entry.path.endswith(extension):
                    w, h = imagesize.get(entry.path)
                    if (w >= 600) and (h >= 600):
                        pathList.append(entry.path)
                        dst = os.path.join(selected_dataset_dir, entry.name)
                        shutil.copyfile(entry.path,dst)
                elif entry.is_dir() and subFolders:   # if its a directory, then repeat process as a nested function
                    pathList = findFilesInFolder(entry.path, pathList, extension, subFolders)
        except OSError:
            print('Cannot access ' + path +'. Probably a permissions error')

        return pathList

    extension = ".jpg"

    pathList = []
    pathList = findFilesInFolder(extracted_dataset_dir, pathList, extension, True)

build_files = False

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
if build_files:  

    os.mkdir(base_dir)    
    os.mkdir(train_dir)    
    os.mkdir(validation_dir)

    
    os.mkdir(test_dir)

    for i,files in enumerate(os.listdir(original_dataset_dir)):
        if i < 640:
            #copia para train
            src = os.path.join(original_dataset_dir, files)
            dst = os.path.join(train_dir, files)
            shutil.copyfile(src, dst)
        elif i < 832:
            #copia para val
            src = os.path.join(original_dataset_dir, files)
            dst = os.path.join(validation_dir, files)
            shutil.copyfile(src, dst)
        elif i < 1024:
            #copia para teste
            src = os.path.join(original_dataset_dir, files)
            dst = os.path.join(test_dir, files)
            shutil.copyfile(src, dst)

scaleImages = False

if scaleImages:
    for file in os.listdir(train_dir):
        path = os.path.join(train_dir,file)
        im = PIL.Image.open(path)
        im_new = crop_center(im,600,600)
        im_new.save(path,quality=100)

    for file in os.listdir(validation_dir):
        path = os.path.join(validation_dir,file)
        im = PIL.Image.open(path)
        im_new = crop_center(im,600,600)
        im_new.save(path,quality=100)

    for file in os.listdir(test_dir):
        path = os.path.join(test_dir,file)
        im = PIL.Image.open(path)
        im_new = crop_center(im,600,600)
        im_new.save(path,quality=100)

makeRandomScalesInput = False
train_dir_scaled = os.path.join(base_dir, 'train_scaled')
validation_dir_scaled = os.path.join(base_dir, 'validation_scaled')
test_dir_scaled = os.path.join(base_dir, 'test_scaled')
if makeRandomScalesInput:
    
    os.mkdir(train_dir_scaled)    
    os.mkdir(validation_dir_scaled)    
    os.mkdir(test_dir_scaled)

    scales = [1.5,2,2.5,3,4]
    np.random.seed(0)
    for file in os.listdir(train_dir):
        scale = np.random.choice(scales)
        path = os.path.join(train_dir,file)
        path_save = os.path.join(train_dir_scaled,file)
        im = PIL.Image.open(path)
        im.thumbnail((int(600/scale),int(600/scale)))
        im.save(path_save,quality=100)

    for file in os.listdir(validation_dir):
        scale = np.random.choice(scales)
        path = os.path.join(validation_dir,file)
        path_save = os.path.join(validation_dir_scaled,file)
        im = PIL.Image.open(path)
        im.thumbnail((int(600/scale),int(600/scale)))
        im.save(path_save,quality=100)

    for file in os.listdir(test_dir):
        scale = np.random.choice(scales)
        path = os.path.join(test_dir,file)
        path_save = os.path.join(test_dir_scaled,file)
        im = PIL.Image.open(path)
        im.thumbnail((int(600/scale),int(600/scale)))
        im.save(path_save,quality=100)


#taxa de aprendizado
lr = num_lr

#Callback para reduzir a taxa de aprendizado
def adapt_learning_rate(epoch,lr):
    n = int(epoch/20)
    return lr/(10**n)

# Camada de entrada do modelo
inputLayer = layers.Input(shape=(600,600,3))

#adicionando as camadas
n_layers = num_layers
for i in range(n_layers):
    if i == 0:
        x = layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001),use_bias=False)(inputLayer)
    elif i == n_layers-1:
        x = layers.Conv2D(3,(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001),use_bias=False)(x)
    else:
        x = layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001),use_bias=False)(x)

#somando as camadas final e entrada para aprendizagem residual
layer_out = layers.add([x,inputLayer])
model = models.Model(inputs=inputLayer,outputs=layer_out)

'''
##model = models.Sequential()

###Definindo camada dummy para ter acesso à entrada
##model.add(layers.Lambda(lambda x: x,name='input',input_shape=(3840,2160,3))) 

###Definindo a camada de entrada
##model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001),use_bias=False))

###Adicionando as outras camadas
##n_layers = 5
##for i in range(n_layers):
##    model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001),use_bias=False))

###Camada de saída com único filtro
##model.add(layers.Conv2D(3,(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001),use_bias=False,name='output'))
##the_input = model.get_layer(index = 0)
##the_output = model.get_layer(index = len(model.layers) -1)
##layer_out = layers.add([the_input,the_output])


##model.add(layer_out)
'''

# Loss functtion
def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

#Compilando o modelo
opt = optimizers.SGD(learning_rate=lr,momentum=0.9,clipvalue=0.4)
model.compile(optimizer=opt, loss=ssim_loss)

model.summary()

# Conjunto treinamento

def load_data(data_path, target_path ,ids):
   X = []
   Y = []

   for i in ids:
    # read one or more samples from your storage, do pre-processing, etc.
    # for example:
    #Carrega Imagem
    path = os.path.join(data_path,i)
    x = PIL.Image.open(path)    
    #Rescala imagem com NN
    x = x.resize((600,600))
    #Transforma para numpy
    x = np.asarray(x)
    #Normaliza
    x = x/255.0

    #Carrega Imagem
    path = os.path.join(target_path,i)
    y = PIL.Image.open(path)    
    #Não escala imagem
    
    #Transforma para numpy
    y = np.asarray(y)
    #Normaliza
    y = y/255.0

    X.append(x)
    Y.append(y)

   return np.array(X), np.array(Y)

def batch_generator(data_path, target_path ,ids, batch_size = 64):
    batch=[]
    while True:
        for i in ids:
            batch.append(i)
            if len(batch)==batch_size:
                yield load_data(data_path, target_path,batch)
                batch=[]
a = 0
'''
#def dataset_loader(data_dir_scaled,data_target_dir):
#    X_train = list()
#    #Para todos arquivos no diretorio
#    for file in os.listdir(data_dir_scaled):
#        #Carrega Imagem
#        path = os.path.join(data_dir_scaled,file)
#        im = PIL.Image.open(path)    
#        #Rescala imagem com NN
#        im = im.resize((600,600))
#        #Transforma para numpy
#        im = np.asarray(im)
#        #Normaliza
#        im = im/255.0
#        #Adiciona no X_train
#        X_train.append(im)
#    X_train = np.array(X_train)

#    y_train = list()
#    for file in os.listdir(data_target_dir):
#        #Carrega Imagem
#        path = os.path.join(data_target_dir,file)
#        im = PIL.Image.open(path)    
#        #Não precisa dar rescale

#        #Transforma para numpy
#        im = np.asarray(im)
#        #Normaliza
#        im = im/255.0
#        #Adiciona no X_train
#        y_train.append(im)
#    y_train = np.array(y_train)

#    return X_train, y_train

#X_train, y_train = dataset_loader(train_dir_scaled,train_dir)
'''



image_names = os.listdir(train_dir_scaled)
train_generator = batch_generator(train_dir_scaled,train_dir,image_names,num_batches)

# Conjunto validação
image_names = os.listdir(validation_dir_scaled)
validation_generator = batch_generator(validation_dir_scaled,validation_dir,image_names,num_batches)

# Conjunto teste
#test_datagen = ImageDataGenerator(rescale=1./255)

#test_generator = test_datagen.flow_from_directory(
#    test_dir_scaled,target_size=(600, 600),
#    batch_size=64,
#    class_mode=None,
#    shuffle = False)
image_names = os.listdir(test_dir_scaled)
test_generator = batch_generator(test_dir_scaled,test_dir,image_names,num_batches)
callback = tf.keras.callbacks.LearningRateScheduler(adapt_learning_rate, verbose=1)

csv_logger = CSVLogger('training.log', separator=',', append=False)

history = model.fit_generator(train_generator,
    steps_per_epoch = 640//num_batches,
    epochs = num_epochs,
    callbacks=[callback, csv_logger],
    validation_data=validation_generator,
    validation_steps = 192//num_batches)

model.save(filename)

print("END")