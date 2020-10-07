import tensorflow as tf
import argparse
import numpy as np

from SNIPtf.prune import make_prune_callback
from SNIPtf.util import calc_weights_sparsity


parser 	= argparse.ArgumentParser(description="Get Sparsity , Epochs " )
parser.add_argument('sparsity', help="Sparsity.", type=float)
parser.add_argument('epochs', help='Epochs.', type=int)
args 	= parser.parse_args()

sparsity = args.sparsity
epochs   = args.epochs


model = tf.keras.applications.MobileNet(
    input_shape=(32,32,3),
    classes=10,
    weights=None
    )

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# subtract pixel mean
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

num_classes = 10

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

datagen.fit(x_train)



from tensorflow.keras.optimizers import Adam #TODO use SGD nestroev

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)



model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


pc = make_prune_callback(
    model,
    sparsity,
    tf.convert_to_tensor(x_train[1:10]),
    tf.convert_to_tensor(y_train[1:10])
    #tf.expand_dims(x_train[0],0),
    #tf.expand_dims(y_train[0],0)
    )



callbacks = [pc,lr_reducer, lr_scheduler]

print( "Initial Sparsity")
print( calc_weights_sparsity(model) )

model.fit(
	  datagen.flow(x_train, y_train, batch_size=512),
          epochs=epochs,
          workers=4,
	      steps_per_epoch = 50000//512,
	      validation_data = (x_test,y_test),
          callbacks=[pc]
          )


print( "FINAL Sparsity , Accuracy" )
print( calc_weights_sparsity(model) )
print( model.evaluate(x_test,y_test))
