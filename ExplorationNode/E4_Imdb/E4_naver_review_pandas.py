import numpy as np
import pandas as pd
import os
from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from tensorflow.keras.initializers import Constant

os.environ['KMP_DUPLICATE_LIB_OK']= 'True' #OMP error solution for MacOS
tokenizer = Mecab()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
vocab_size = 10000
word_vector_dim = 16  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)

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

def preprocess_naver_review(data):
    data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
    # print('총 샘플의 수 :',len(data))
    # print(data.groupby('label').size().reset_index(name = 'count'))
    # print(data.isnull().values.any())
    data = data.dropna(how = 'any') # Null 값이 존재하는 행 제거
    # print(data.isnull().values.any()) # Null 값이 존재하는지 확인
    data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    # print(data[:5])
    data['document'].replace('', np.nan, inplace=True)
    # print(data.isnull().sum())
    # print(data.loc[data.document.isnull()][:5])
    data = data.dropna(how = 'any')
    return data

def tokenization_kor(data):
    X_train = []
    for sentence in data['document']:
        temp_X = []
        temp_X = tokenizer.morphs(sentence) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_train.append(temp_X)

    return X_train

def train_LSTM(X_train, y_train):

    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

    return model, history


def train_lstm_word2vec(x, y, tokenizer, max_len):
    word_vector_dim = 200  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)

    # /content/drive/MyDrive/DL_Study/AIFFEL/ko/ko.bin
    from gensim.models import Word2Vec
    # word2vec_path = '/content/drive/MyDrive/DL_Study/AIFFEL/ko/ko.bin'
    word2vec_path = os.getcwd() + '/ko/ko.bin'
    # word2vec = Word2Vec.load(word2vec_path, binary=True, limit=1000000)
    word2vec = Word2Vec.load(word2vec_path)

    embedding_matrix = np.random.rand(vocab_size, word_vector_dim)

    # embedding_matrix에 Word2Vec 워드벡터를 단어 하나씩마다 차례차례 카피한다.
    for i in range(2,vocab_size):
        if tokenizer.index_word[i] in word2vec:
            embedding_matrix[i] = word2vec[tokenizer.index_word[i]]

    model = Sequential()
    # 아래 embedding을 keras.layers.Embedding 이렇게 써주면 error가 나는데
    # keras, tensorflow type들을 혼용해서 써서 그러는게 아닐까?
    model.add(Embedding(vocab_size,
                                     word_vector_dim,
                                     embeddings_initializer=Constant(embedding_matrix),  # 카피한 임베딩을 여기서 활용
                                     input_length=max_len,
                                     trainable=True))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

    return model, history


def train_with_word2vec(x, y, tokenizer, max_len):
    word_vector_dim = 200  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)

    # /content/drive/MyDrive/DL_Study/AIFFEL/ko/ko.bin
    from gensim.models import Word2Vec
    # word2vec_path = '/content/drive/MyDrive/DL_Study/AIFFEL/ko/ko.bin'
    word2vec_path = os.getcwd() + '/ko/ko.bin'
    # word2vec = Word2Vec.load(word2vec_path, binary=True, limit=1000000)
    word2vec = Word2Vec.load(word2vec_path)

    embedding_matrix = np.random.rand(vocab_size, word_vector_dim)

    # embedding_matrix에 Word2Vec 워드벡터를 단어 하나씩마다 차례차례 카피한다.
    for i in range(2,vocab_size):
        if tokenizer.index_word[i] in word2vec:
            embedding_matrix[i] = word2vec[tokenizer.index_word[i]]

    # 모델 구성
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size,
                                     word_vector_dim,
                                     embeddings_initializer=Constant(embedding_matrix),  # 카피한 임베딩을 여기서 활용
                                     input_length=max_len,
                                     trainable=True))   # trainable을 True로 주면 Fine-tuning
    model.add(keras.layers.Conv1D(16, 7, activation='relu'))
    model.add(keras.layers.MaxPooling1D(5))
    model.add(keras.layers.Conv1D(16, 7, activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model_word2vec.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

    return model, history


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


# 데이터를 읽어봅시다.
train_data = pd.read_table(os.getcwd()+'/ratings_train.txt')
test_data = pd.read_table(os.getcwd()+'/ratings_test.txt')
# train_data = pd.read_table('/content/drive/MyDrive/DL_Study/AIFFEL/ratings_train.txt')
# test_data = pd.read_table('/content/drive/MyDrive/DL_Study/AIFFEL/ratings_test.txt')


train_data = preprocess_naver_review(train_data)
test_data = preprocess_naver_review(test_data)

print('전처리 후 훈련 샘플의 개수 :',len(train_data))
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

train_data_token = tokenization_kor(train_data)
test_data_token = tokenization_kor(test_data)

print(train_data_token[:3])

tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
tokenizer.fit_on_texts(train_data_token)
X_train = tokenizer.texts_to_sequences(train_data_token)
X_test = tokenizer.texts_to_sequences(test_data_token)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
max_len = 30
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
#
# model, history = train_LSTM(X_train, y_train)
#
# if history is not None:
#     show_graph(history.history)

model, history = train_lstm_word2vec(X_train, y_train, tokenizer, max_len)