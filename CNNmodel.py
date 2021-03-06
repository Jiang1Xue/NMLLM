# 导入需要依赖的包
import tensorflow as tf

# 定义一个cnnModel方法类
class cnnModel(object):
    # 初始化函数，将dropout的失效概率初始化
    def __init__(self, rate):
        self.rate = rate
    def createModel(self):
        # 使用tf.keras.Sequentia来逐层的搭建神经网络
        model = tf.keras.Sequential()
        # 使用双层卷积结构的方式提取图像特征，第一层双层卷积使用3*3的卷积核，输出维度是64，全局
        # 使用he_normal进行kernel_initializer,激活函数使用relu
        model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1,
                                         activation='relu', padding='same', input_shape=(78, 78, 1), name="conv1"))

       # model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1,
       #                                  activation='relu', padding='same', input_shape=(512, 512, 1), name="conv2"))

        # 使用tf.keras.layers.MaxPool2D搭建神经网络的池化层，使用最大值池化策略，将2*2局域的像素使用一个最大值代替，步幅为2，
        # padding 使用valid策略
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool1"))

        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d1"))

        # 使用batchnormalization对上一层的输出数据进行归一化
        model.add(tf.keras.layers.BatchNormalization())

        # 使用双层卷积结构的方式提取图像特征，第二层双层卷积使用3*3的卷积核，输出维度为128，
        # 全局使用he_normal进行kernel_initializer,激活函数使用relu
        model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1,
                                         activation='relu', padding='same', name="conv3"))

        #model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1,
        #                                 activation='relu', padding='same', name="conv4"))

        # 使用tf.keras.layers.MaxPool2D搭建神经网络的池化层，使用最大值池化策略，将2*2局域的像素
        # 使用一个最大值代替，步幅为2，padding使用valid策略
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding = 'valid', name = "pool2"))

        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d2"))
        # 使用batchNormalization对上一层的输出数据进行归一化
        model.add(tf.keras.layers.BatchNormalization())
        # 使用双层卷积结构的方式提取图像特征，第三层双卷积使用3*3的卷积核，输出维度是128，全局使用
        # he_normal 进行kernel_initializer,激活函数使用relu
        model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1,
                                         activation='relu', padding='same', name="conv5"))
        #model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1,
        #                                 activation='relu', padding='same', name="conv6"))

        # 使用tf.keras.layers.MaxPool2D搭建神经网络的池化层，使用最大值池化策略，将2*2局域的
        # 像素使用一个最大值代替，步幅为2， padding使用valid策略
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name= 'pool3'))

        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(tf.keras.layers.Dropout(rate = self.rate, name="d3"))
        # 使用batchnormalization对上一层的输出数据进行归一化
        model.add(tf.keras.layers.BatchNormalization())
        # 使用flatten将上层的输出数据压平
        model.add(tf.keras.layers.Flatten(name = "flatten"))
        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(tf.keras.layers.Dropout(self.rate))
        # 叠加一层全连接层，用于拟合最终结果，激活函数使用relu
        model.add(tf.keras.layers.Dense(128, activation = 'relu', kernel_initializer='he_normal'))
        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(tf.keras.layers.Dropout(self.rate))
        # 叠加一层全连接层作为输出层，激活函数使用softmax
        model.add(tf.keras.layers.Dense(6, activation = 'softmax', kernel_initializer='he_normal'))

        # 最后编译搭建完成的神经网络，使用categorical_crossentropy损失函数，adam优化器，模型的衡量指标是准确率
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 返回编译完成model
        return model






















