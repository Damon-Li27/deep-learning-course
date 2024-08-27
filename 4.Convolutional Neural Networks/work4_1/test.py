import matplotlib.pyplot as plt  # 用于数据可视化
import tensorflow as tf
import cnn_utils as cnn
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = cnn.load_dataset()
# 图像数据通常在[0,255]范围，缩放到[0,1]范围，能加快模型训练速度和提升性能
X_train = train_set_x_orig / 255
X_test = test_set_x_orig / 255


# Y_train = tf.keras.utils.to_categorical(train_set_y_orig, num_classes=6)
# Y_test = tf.keras.utils.to_categorical(test_set_y_orig, num_classes=6)
# 转换为独热编码
Y_train = cnn.convert_to_one_hot(train_set_y_orig, 6).T
Y_test = cnn.convert_to_one_hot(test_set_y_orig, 6).T


# 构建CNN模型，按照顺序添加各个层来定义模型的架构。
model = tf.keras.Sequential([
    # 第一层卷积+池化，strides默认值为 (1, 1)，padding默认值为 "valid" ，即不进行填充。32个滤波器，每个大小为(3,3)
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    # strides默认与pool_size一样，adding默认值为 "valid" ，即不进行填充。池化层减少特征图的大小，同时保留主要特征
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 第二层卷积+池化
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 第三层卷积+池化
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 展开层
    tf.keras.layers.Flatten(),

    # 全连接层，120个神经单元
    tf.keras.layers.Dense(units=120, activation='relu'),
    # Dropout，以50%的概率随机丢弃神经元，用于防止过拟合
    tf.keras.layers.Dropout(0.5),

    # 输出层，6个神经元，用于分6类
    tf.keras.layers.Dense(units=6, activation='softmax')
])

# 编译模型，指定训练模型时的优化器、损失函数和评估指标等参数。
# optimizer可取sgd、rmsprop、adam等，loss可取categorical_crossentropy、sparse_categorical_crossentropy、binary_crossentropy（二分类交叉熵损失）等
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型摘要，包括模型的结构和参数信息，方便查看每一层和总参数量
model.summary()

# 训练模型,使用训练集数据 X_train 和标签 Y_train 训练模型，训练 20 个轮次，每个批次大小为 32，并使用X_test和Y_test作为验证数据。
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))

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