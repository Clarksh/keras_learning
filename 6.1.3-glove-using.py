import os
import pickle
from time import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt

"""

使用预训练词嵌入

"""

maxlen = 100#句子长截断为100
training_samples = 200#在 200 个样本上训练
validation_samples = 10000#在 10 000 个样本上验证
max_words = 10000#只考虑数据集中前 10 000 个最常见的单词

"""把文件数据放入数组中"""
def dataProcess():
    imdb_dir = 'data/aclImdb'#基本路径，经常要打开这个
    #处理训练集
    train_dir = os.path.join(imdb_dir, 'train')#添加子路径
    train_labels = []
    train_texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):#获取目录下所有文件名字
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname),'r',encoding='utf8')
                train_texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    train_labels.append(0)
                else:train_labels.append(1)
    #处理测试集
    test_dir = os.path.join(imdb_dir, 'test')
    test_labels = []
    test_texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(test_dir, label_type)
        for fname in sorted(os.listdir(dir_name)):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname),'r',encoding='utf8')
                test_texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    test_labels.append(0)
                else:
                    test_labels.append(1)

    #对数据进行分词和划分训练集和数据集
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_texts)#构建单词索引结构

    sequences = tokenizer.texts_to_sequences(train_texts)#整数索引的向量化模型
    word_index = tokenizer.word_index#索引字典
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=maxlen)
    train_labels = np.asarray(train_labels)#把列表转化为数组
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', train_labels.shape)
    indices = np.arange(data.shape[0])#评论顺序0，1，2，3
    np.random.shuffle(indices)#把评论顺序打乱3，1，2，0
    data = data[indices]
    train_labels = train_labels[indices]
    x_train = data[:training_samples]
    y_train = train_labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = train_labels[training_samples: training_samples + validation_samples]

    #同样需要将测试集向量化
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    x_test = pad_sequences(test_sequences, maxlen=maxlen)
    y_test = np.asarray(test_labels)

    return x_train,y_train,x_val,y_val,x_test,y_test,word_index


embedding_dim = 100#特征数设为100
"""将预训练的glove词嵌入文件，构建成可以加载到embedding层中的嵌入矩阵"""
def load_glove(word_index):#导入glove的词向量
    embedding_file='data/glove.6B'
    embeddings_index={}#定义字典
    f = open(os.path.join(embedding_file, 'glove.6B.100d.txt'),'r',encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    """转化为矩阵：构建可以加载到embedding层中的嵌入矩阵，形为(max_words（单词数）, embedding_dim（向量维数）) """
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():#字典里面的单词和索引
        if i >= max_words:continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

if __name__ == '__main__':
    t0 = time()
    """处理数据文件"""
    x_train, y_train, x_val, y_val,x_test,y_test, word_index = dataProcess()

    """处理glove词嵌入文件"""
    embedding_matrix=load_glove(word_index)
    #可以把得到的嵌入矩阵保存起来，方便后面fine-tune"""
    # #保存
    # with open('model/glove_embedding_matrix', 'wb') as fp:
    #     pickle.dump(embedding_matrix, fp)
    # # 读取
    # with open('model/glove_embedding_matrix', 'rb') as fp:
    #     embedding_matrix = pickle.load(fp)

    """构建神经网络"""
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())#压平，把多维输入一维化，
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    """将预训练的词嵌入矩阵放到embedding层,并冻结起来，训练神经网络时该层不会再变化"""
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    """训练和评估"""
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))
    model.save_weights('pre_trained_glove_model.h5')#保存结果

    """绘制结果"""
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    # 在测试集上评估模型
    model.load_weights('pre_trained_glove_model.h5')
    model.evaluate(x_test, y_test)

    print('Done in %.3fs...' % (time() - t0))