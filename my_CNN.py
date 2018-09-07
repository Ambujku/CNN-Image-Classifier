import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# classifier for CNN

cnn_classifier = Sequential()

#Convolution ==>
cnn_classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Pooling ==>

cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding hidden convolutional layer ==>

cnn_classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening ==>

cnn_classifier.add(Flatten())

#Full connection ==>

cnn_classifier.add(Dense(output_dim = 128, activation = 'relu'))
cnn_classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Image Processing

train_dataset = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_dataset = ImageDataGenerator(rescale = 1./255)


training = train_dataset.flow_from_directory('Directory location for train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

testing = test_dataset.flow_from_directory('Directory location for test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# finally executing the model using keras fit method

cnn_classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 50,
                         validation_data = test_set,
                         nb_val_samples = 2000)



test_image = image.load_img('image_file_for_testing.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    print('dog')
else:
    print('cat')



