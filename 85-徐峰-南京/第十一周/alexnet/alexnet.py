from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam


#alexnet 网络结构
def Alexnet(input_shape=(224, 224, 3), output_shape=2):
    model = Sequential()

    #卷积
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )

    #
    model.add(BatchNormalization())
    #池化
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2, 2),
            padding='valid'
        )
    )

    #卷积
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5,5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    model.add(BatchNormalization())

    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2,2),
            padding='valid'
        )
    )

    #卷积
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    #全连接层
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_shape, activation='softmax'))

    return model