import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import utils.kt_utils as kt

'''
    使用卷积神经网络识别微笑笑脸
'''

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = kt.load_dataset()

X_train = train_set_x_orig / 255
X_test = test_set_x_orig / 255

Y_train = train_set_y_orig.T
Y_test = test_set_y_orig.T

print(train_set_x_orig.shape)  # (600, 64, 64, 3)
print(train_set_y_orig.shape)  # (1, 600)
print(test_set_x_orig.shape)
print(test_set_y_orig.shape)

model = tf.keras.Sequential([
    # 第一层卷积
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(64, 64, 3)),
    # 第一层池化
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    # 第二层卷积
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(31, 31, 32)),
    # 第二层池化
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    # 第三层卷积
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", input_shape=(14, 14, 64)),
    # 第三层池化
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),  # (600, 6, 6, 128)

    # 展开
    tf.keras.layers.Flatten(),

    # 全连接
    tf.keras.layers.Dense(units=120, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

model.summary()

history = model.fit(x=X_train,y=Y_train, batch_size=32, epochs=30, validation_data=(X_test, Y_test))

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

def test_my_images(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64,64))
    plt.imshow(img)
    x = tf.keras.preprocessing.image.img_to_array(img)
    # 使用 np.expand_dims() 将数组的形状从 (64, 64, 3) 扩展为 (1, 64, 64, 3)。
    # 这种扩展是因为模型通常期望输入是包含多个样本的批次（batch）。
    # 这里只有一张图片，因此在第一个维度上增加一个大小为 1 的维度。
    x = np.expand_dims(x, axis=0)
    # preprocess_input(x) 对图像进行标准化或其他预处理操作，使其与训练模型时使用的数据格式一致。
    # 具体的预处理方式取决于你使用的模型（如 VGG16、ResNet 等）。
    x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    return model.predict(x)

image1_path = 'images/img.png'
print(test_my_images(image1_path))
image2_path = 'images/img_1.png'
print(test_my_images(image2_path))

image3_path = 'images/img_2.png'
print(test_my_images(image3_path))
image4_path = 'images/img_3.png'
print(test_my_images(image4_path))