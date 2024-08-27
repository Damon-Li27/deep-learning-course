import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import utils.resnets_utils as rs_util

'''
    使用残差网络训练手指识别
'''

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = rs_util.load_dataset()

X_train = train_set_x_orig / 255.0
Y_train = tf.keras.utils.to_categorical(train_set_y_orig.T, num_classes=6)

X_test = test_set_x_orig / 255.0
Y_test = tf.keras.utils.to_categorical(test_set_y_orig.T, num_classes=6)

print(X_train.shape)  # (1080, 64, 64, 3)
print(Y_train.shape)  # (1080, 6)
print(X_test.shape)  # (120, 64, 64, 3)
print(Y_test.shape)  # (120, 6)

def identity_block(X, filters):
    '''
        实现恒等块（2层）
    :param X:
    :param filters:
    :return:
    '''
    X_shortcut = X

    # 第一层卷积
    X = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.ReLU()(X)

    # 第二层卷积
    X = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.001))(X)
    X = tf.keras.layers.BatchNormalization()(X)

    # 使用1*1卷积来修改通道数，使得维度一致
    X_shortcut = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.001))(X_shortcut)

    # 将输入加回输出（恒等映射）
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.ReLU()(X)
    return X


def built_resnet_model(input_shape=(64, 64, 3)):
    X_input = tf.keras.layers.Input(input_shape)

    # 初始卷积，使用较大的卷积核（如 7 * 7）提取低级别特征，同时步幅为 2 减小空间尺寸。  (30,30,8)
    X = tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), strides=2, padding="valid", kernel_regularizer=tf.keras.regularizers.L2(0.001))(X_input)
    # 标准化
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.ReLU()(X)
    # 池化，在初始卷积层后使用池化减少数据维度。 (15,15,8)
    X = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(X)

    # 第一个恒等块
    X = identity_block(X, filters=32)

    # 第二个恒等块
    X = identity_block(X, filters=64)

    # 第三个恒等块
    X = identity_block(X, filters=128)

    # 平均池化层，计算各通道的平均值。例如，输入张量的形状为 (batch_size, 6, 6, 256)，输出形状为 (batch_size, 256)。
    X = tf.keras.layers.GlobalAvgPool2D()(X)

    X = tf.keras.layers.Dense(units=32, activation="relu")(X)

    X = tf.keras.layers.Dropout(0.5)(X)  # 防止过拟合
    # (15,15,6)
    X = tf.keras.layers.Dense(units=6, activation="softmax", kernel_regularizer=tf.keras.regularizers.L2(0.001))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)
    return model


model = built_resnet_model((64, 64, 3))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()

history = model.fit(X_train, Y_train, batch_size=64, epochs=15, validation_data=(X_test, Y_test))

# 可视化训练过程
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 在验证集上评估模型
val_loss, val_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f'\nValidation accuracy: {val_acc}')


# def test_my_images(image_path):
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
#     plt.imshow(img)
#     x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     # 使用 np.expand_dims() 将数组的形状从 (64, 64, 3) 扩展为 (1, 64, 64, 3)。
#     # 这种扩展是因为模型通常期望输入是包含多个样本的批次（batch）。
#     # 这里只有一张图片，因此在第一个维度上增加一个大小为 1 的维度。
#     x = np.expand_dims(x, axis=0)
#     # preprocess_input(x) 对图像进行标准化或其他预处理操作，使其与训练模型时使用的数据格式一致。
#     # 具体的预处理方式取决于你使用的模型（如 VGG16、ResNet 等）。
#     x = tf.keras.applications.imagenet_utils.preprocess_input(x)
#     prediction = model.predict(x)
#     return "Smile" if prediction > 0.5 else "Not a Smile"

#
# image1_path = 'images/img.png'
# result1 = test_my_images(image1_path)
# print(result1)
# image2_path = 'images/img_1.png'
# print(test_my_images(image2_path))
#
# image3_path = 'images/img_2.png'
# print(test_my_images(image3_path))
# image4_path = 'images/img_3.png'
# print(test_my_images(image4_path))
