from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

inputShape = (224, 224, 3)
class VGG16:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()

        inputShape = (height, width, depth)
        conv2d_first = Conv2D(64, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_first)
        activation_first = Activation("relu")
        model.add(activation_first)

        conv2d_second = Conv2D(64, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_second)
        activation_second = Activation("relu")
        model.add(activation_second)

        pooling_first = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        model.add(pooling_first)

        conv2d_third = Conv2D(128, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_third)
        activation_third = Activation("relu")
        model.add(activation_third)

        conv2d_fourth = Conv2D(128, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_fourth)
        activation_fourth = Activation("relu")
        model.add(activation_fourth)

        pooling_second = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        model.add(pooling_second)

        conv2d_fifth = Conv2D(256, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_fifth)
        activation_fifth = Activation("relu")
        model.add(activation_fifth)

        conv2d_sixth = Conv2D(256, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_sixth)
        activation_sixth = Activation("relu")
        model.add(activation_sixth)

        conv2d_seventh = Conv2D(256, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_seventh)
        activation_seventh = Activation("relu")
        model.add(activation_seventh)

        pooling_third = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        model.add(pooling_third)

        conv2d_eighth = Conv2D(512, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_eighth)
        activation_eighth = Activation("relu")
        model.add(activation_eighth)

        conv2d_ninth = Conv2D(512, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_ninth)
        activation_ninth = Activation("relu")
        model.add(activation_ninth)

        conv2d_tenth = Conv2D(512, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_tenth)
        activation_tenth = Activation("relu")
        model.add(activation_tenth)

        pooling_fourth = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        model.add(pooling_fourth)

        conv2d_eleventh = Conv2D(512, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_eleventh)
        activation_eleventh = Activation("relu")
        model.add(activation_eleventh)

        conv2d_twelfth = Conv2D(512, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_twelfth)
        activation_twelfth = Activation("relu")
        model.add(activation_twelfth)

        conv2d_thirteenth = Conv2D(512, (3, 3), padding="same", input_shape=inputShape)
        model.add(conv2d_thirteenth)
        activation_thirteenth = Activation("relu")
        model.add(activation_thirteenth)

        pooling_fifth = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        model.add(pooling_fifth)


        flatten = Flatten()
        model.add(flatten)
        dense_first = Dense(4096)
        model.add(dense_first)
        activation_fourtheenth = Activation("relu")
        model.add(activation_fourtheenth)

        dense_second = Dense(4096)
        model.add(dense_second)
        activation_fifteenth = Activation("relu")
        model.add(activation_fifteenth)

        dense_third = Dense(classes)
        model.add(dense_third)
        activation_sixteenth = Activation("softmax")
        model.add(activation_sixteenth)

        return model