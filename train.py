from itertools import starmap, product, islice, groupby
from functools import reduce
from os.path import join
from operator import itemgetter
from random import shuffle
from numpy import asarray, zeros
from cv2 import imread, cvtColor, resize, COLOR_BGR2GRAY, INTER_CUBIC  # type: ignore
from keras.layers import (
    Input,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    Dropout,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
)
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils import plot_model, layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import keras.backend as K
from numpy import ndarray

from collections import Counter

K.set_image_data_format("channels_last")

X_train = []
X_test = []
Y_train = []
Y_test = []

FNT_PATH = join("assets", "English", "Fnt")


def convert_image(i, j, path):
    '''
    Преобразование изображения в черно-белое методом бикубической интерполяции
    '''
    return (
        i,
        j,
        resize(
            cvtColor(imread(path), COLOR_BGR2GRAY),
            None,
            fx=0.25,
            fy=0.25,
            interpolation=INTER_CUBIC,
        ),
    )


def construct_target_data(i, j):
    mat = zeros(36)
    mat[i] = 1
    return (i, j, mat)


def shuffled(collection):
    '''
    Случайная перестановка элементов коллекции
    '''
    copy = collection.copy()
    shuffle(copy)
    return copy


def separate_train_test(data, count=762):
    '''
    Разделение массива тренировочных данных на две части с равным представлением элементов всех классов.
    '''
    return list(
        map(
            asarray,
            reduce(
                lambda a, b: (a[0] + b[0], a[1] + b[1]),
                map(
                    lambda x: itemgetter(slice(0, count), slice(0, None))(
                        shuffled(list(map(itemgetter(2), x[1])))
                    ),
                    groupby(sorted(data, key=itemgetter(0)), key=itemgetter(0)),
                ),
            ),
        )
    )


# Загрузка изображений датасета с диска
input_data = starmap(
    convert_image,
    starmap(
        lambda i, j: (
            i,
            j,
            join(FNT_PATH, "Sample%03d" % i, "img%03d-%05d.png" % (i, j)),
        ),
        product(range(1, 37), range(1, 1017)),
    ),
)

# Разделение изображений на используемые для тренировки и тестовые
X_train: X_test = separate_train_test(input_data, count=762)

target_data = starmap(construct_target_data, product(range(36), range(1016)))
Y_train, Y_test = separate_train_test(target_data, count=762)

row_num, col_num = 32, 32
input_shape = (row_num, col_num, 1)

#Приведение обрабатываемых данных к вещественным числам от 0 до 255, нормирование данных
X_train = X_train.reshape(X_train.shape[0], *input_shape).astype("float32") / 255

X_test = X_test.reshape(X_test.shape[0], *input_shape).astype("float32") / 255

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# Всего необходимо классифицировать 36 объектов - буквы английского алфавита и цифры
num_classes = 36

# Входной слой соответствует форме входных данных
input_layer = Input(shape=input_shape)
#Три слоя свёртки с ядром 3x3
x = Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)(
    input_layer
)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = Conv2D(128, (3, 3), activation="relu")(x)
#Слой пуллинга
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
#Слой Flatten преобразует тензор к вектору
x = Flatten()(x)
# Соединение нейронов предыдущего слоя
x = Dense(128, activation="relu")(x)
# Слой пакетной нормализации на каждой итерации просматривает всю выборку изображений,
# после чего меняет веса модели.
x = BatchNormalization()(x)
x = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=[x])

epochs = 100
batch_size = 128

model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"])

model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, Y_test),
)

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
print("Потери в тесте:", score[0])
print("Точность в тесте", score[1])
model.save("Model100Epoch.h5")
