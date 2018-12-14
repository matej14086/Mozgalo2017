from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras
import numpy as np
from PIL import Image
import os,sys
from keras.utils import plot_model


def fineTuneInceptionV3(source="clean",name="final.h5"):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(8, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


    train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
            './'+source,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical')


    checkpoint = ModelCheckpoint(name, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                                 period=1)
    callbacks_list = [checkpoint, TensorBoard(log_dir='/tmp/checks')]
    model.summary()


    model.fit_generator(train_generator,200,epochs=10,callbacks= callbacks_list)


    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)


    for layer in model.layers[:172]:
       layer.trainable = False
    for layer in model.layers[172:]:
       layer.trainable = True


    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')



    checkpoint = ModelCheckpoint("again.h5", monitor='val_loss', verbose=1, save_best_only=False,
                                 save_weights_only=False,
                                 period=1)
    callbacks_list = [checkpoint, TensorBoard(log_dir='/tmp/checks')]
    model.summary()

    model.fit_generator(train_generator, 200, epochs=10, callbacks=callbacks_list)

    return model

# im=Image.open("./resized/0001.jpg")
# im=np.asarray(im)
# im=np.asarray([im])
#
# read="./ret/5/"
# listing = os.listdir(read)
# for file in listing:
#     im=Image.open(read+file)
#     im = np.asarray(im)
#     im = np.asarray([im/255])
#     print(model.predict(im).argmax())

def fineTuneModel(model_path,source="clean",name="final.h5"):
    model=load_model(model_path)
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        './' + source,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')


    checkpoint = ModelCheckpoint(name, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                                 period=1)
    callbacks_list = [checkpoint, TensorBoard(log_dir='/tmp/checks')]
    model.summary()

    model.fit_generator(train_generator, 200, epochs=10, callbacks=callbacks_list)

    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')



    checkpoint = ModelCheckpoint("again.h5", monitor='val_loss', verbose=1, save_best_only=False,
                                 save_weights_only=False,
                                 period=1)
    callbacks_list = [checkpoint, TensorBoard(log_dir='/tmp/checks')]
    model.summary()

    model.fit_generator(train_generator, 200, epochs=10, callbacks=callbacks_list)

    return model

def predictCluster(model_path,source='converted',target='result'):
    model=load_model(model_path)
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        temp = np.asarray(im)
        temp = np.asarray([temp / 255])
        if not os.path.exists(write+str(model.predict(temp).argmax())+"\\"):
            os.makedirs(write+str(model.predict(temp).argmax())+"\\")
        im.save(write+str(model.predict(temp).argmax())+"\\"+file)



