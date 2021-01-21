import pandas as pd
from tensorflow.python.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import matplotlib.pyplot as plt
import os
from konlpy.tag import Mecab

os.environ['KMP_DUPLICATE_LIB_OK']= 'True' #OMP error solution for MacOS
tokenizer = Mecab()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
vocab_size = 10000
word_vector_dim = 16  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)
# def make_vocab(num_words=vocab_size):

def show_graph(history_dict):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo"는 "파란색 점"입니다
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b는 "파란 실선"입니다
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()   # 그림을 초기화합니다

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def train_LSTM(save_file, dim, e_matrix=None):
    from tensorflow.keras.initializers import Constant

    # model 설계 - 딥러닝 모델 코드를 직접 작성해 주세요.
    model = keras.Sequential()
    if os.path.exists(save_file):
        model = load_model(save_file)
        return model, None

    # [[YOUR CODE]]
    if e_matrix is None:
        model.add(keras.layers.Embedding(vocab_size, dim, input_shape=(None,)))
    else:
        model.add(keras.layers.Embedding(vocab_size, dim,
                                         embeddings_initializer=Constant(e_matrix),
                                         input_shape=(None,)))

    model.add(keras.layers.LSTM(8))   # 가장 널리 쓰이는 RNN인 LSTM 레이어를 사용하였습니다. 이때 LSTM state 벡터의 차원수는 8로 하였습니다. (변경가능)
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    epochs=20  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다.

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    model.save(save_file)

    return model, history

def make_word_to_index(sentences, word_to_index):
    word_to_index["<PAD>"] = 0
    word_to_index["<BOS>"] = 1
    word_to_index["<UNK>"] = 2  # unknown
    word_to_index["<UNUSED>"] = 3
    start_index=4

    for st in sentences:
        if not isinstance(st, str):
            continue

        # tokens = tokenizer.nouns(st)
        tokens = tokenizer.morphs(st)

        for token in tokens:
            if token in stopwords:
                continue
            elif start_index > vocab_size:
                return
            else:
                word_to_index[token] = start_index
                start_index += 1

    return word_to_index

def load_data(train_data, test_data, num_words=vocab_size):
    # train_data, test_data 둘다 pandas 객체
    # train_data를 조사해보면 Nan값을 가진 결측치 data가 5개 존재함을 확인
    word_to_index={}
    train_data.info()
    reviews = train_data['document']
    make_word_to_index(reviews.tolist(), word_to_index)
    X_train=[]
    Y_train=[]


    for idx, s in enumerate(reviews):
        if not isinstance(s, str):# 결측치 제거
            continue
        enc = get_encoded_sentence(s, word_to_index)
        # print(enc)
# 문장 1개를 활용할 딕셔너리와 함께 주면, 단어 인덱스 리스트 벡터로 변환해 주는 함수입니다.
# 단, 모든 문장은 <BOS>로 시작하는 것으로 합니다.
def get_encoded_sentence(sentence, word_to_index):
    # return [word_to_index['<BOS>']]+[word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence.split()]
    return [word_to_index['<BOS>']]+[word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in tokenizer.nouns(sentence)]


# 여러 개의 문장 리스트를 한꺼번에 단어 인덱스 리스트 벡터로 encode해 주는 함수입니다.
def get_encoded_sentences(sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

# 숫자 벡터로 encode된 문장을 원래대로 decode하는 함수입니다.
def get_decoded_sentence(encoded_sentence, index_to_word):
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])  #[1:]를 통해 <BOS>를 제외

# 여러개의 숫자 벡터로 encode된 문장을 한꺼번에 원래대로 decode하는 함수입니다.
def get_decoded_sentences(encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]

    X_train.append(enc)
        Y_train.append(train_data['label'][idx])

    reviews = test_data['document']
    X_test=[]
    Y_test=[]

    for idx, s in enumerate(reviews):
        if not isinstance(s, str):# 결측치 제거
            continue
        enc = get_encoded_sentence(s, word_to_index)
        # print(enc)
        X_test.append(enc)
        Y_test.append(train_data['label'][idx])

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test, word_to_index




# 데이터를 읽어봅시다.
train_data = pd.read_table(os.getcwd()+'/ratings_train.txt')
test_data = pd.read_table(os.getcwd()+'/ratings_test.txt')

# train_data.head()

X_train, y_train, X_test, y_test, word_to_index = load_data(train_data, test_data, vocab_size)
index_to_word = {index:word for word, index in word_to_index.items()}


total_data_text = list(X_train) + list(X_test)
# 텍스트데이터 문장길이의 리스트를 생성한 후
num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)
# 문장길이의 평균값, 최대값, 표준편차를 계산해 본다.
print('문장길이 평균 : ', np.mean(num_tokens))
print('문장길이 최대 : ', np.max(num_tokens))
print('문장길이 표준편차 : ', np.std(num_tokens))
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)

print('pad_sequences maxlen : ', maxlen)
print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다. '.format(np.sum(num_tokens < max_tokens) / len(num_tokens)))

x_train = keras.preprocessing.sequence.pad_sequences(X_train,
                                                     value=word_to_index["<PAD>"],
                                                     padding='post', # 혹은 'pre'
                                                     maxlen=maxlen)

x_test = keras.preprocessing.sequence.pad_sequences(X_test,
                                                    value=word_to_index["<PAD>"],
                                                    padding='post', # 혹은 'pre'
                                                    maxlen=maxlen)


# validation set 5000건 분리
x_val = x_train[:5000]
y_val = y_train[:5000]

# validation set을 제외한 나머지 10000건
partial_x_train = x_train[5000:]
partial_y_train = y_train[5000:]

print(partial_x_train.shape)
print(partial_y_train.shape)



model, history = train_LSTM('naver_01.h5', dim=word_vector_dim)
results = model.evaluate(x_test,  y_test, verbose=2)
print(results)
if history is not None:
    show_graph(history.history)

