
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

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

def save_embedding_file(filename, model):
    embedding_layer = model.layers[0]
    weights = embedding_layer.get_weights()[0]
    print(weights.shape)    # shape: (vocab_size, embedding_dim)

    # 학습한 Embedding 파라미터를 파일에 써서 저장합니다.
    word2vec_file_path = os.getcwd()+filename
    f = open(word2vec_file_path, 'w')
    f.write('{} {}\n'.format(vocab_size-4, word_vector_dim))  # 몇개의 벡터를 얼마 사이즈로 기재할지 타이틀을 씁니다.

    # 단어 개수(에서 특수문자 4개는 제외하고)만큼의 워드 벡터를 파일에 기록합니다.
    vectors = model.get_weights()[0]
    for i in range(4,vocab_size):
        f.write('{} {}\n'.format(index_to_word[i], ' '.join(map(str, list(vectors[i, :])))))
    f.close()

    return word2vec_file_path

def verify_gensim_word2vec(file_path):
    from gensim.models.keyedvectors import Word2VecKeyedVectors
    from gensim.models import KeyedVectors

    word_vectors = Word2VecKeyedVectors.load_word2vec_format(file_path, binary=False)
    vector = word_vectors['computer']
    print(vector)
    print(word_vectors.similar_by_word("love"))

    word2vec_path = os.getcwd()+'/GoogleNews-vectors-negative300.bin'
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=1000000)
    vector = word2vec['computer']
    print(vector)     # 무려 300dim의 워드 벡터입니다.

    return word2vec

imdb = keras.datasets.imdb

os.environ['KMP_DUPLICATE_LIB_OK']= 'True' #OMP error solution for MacOS


# IMDB 데이터셋 다운로드
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print("훈련 샘플 개수: {}, 테스트 개수: {}".format(len(x_train), len(x_test)))

vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 16  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)


print(x_train[0])  # 1번째 리뷰데이터
print('라벨: ', y_train[0])  # 1번째 리뷰데이터의 라벨
print('1번째 리뷰 문장 길이: ', len(x_train[0]))
print('2번째 리뷰 문장 길이: ', len(x_train[1]))

word_to_index = imdb.get_word_index()
index_to_word = {index:word for word, index in word_to_index.items()}
print(index_to_word[1])     # 'the' 가 출력됩니다.
print(word_to_index['the'])  # 1 이 출력됩니다.

#실제 인코딩 인덱스는 제공된 word_to_index에서 index 기준으로 3씩 뒤로 밀려 있습니다.
word_to_index = {k:(v+3) for k,v in word_to_index.items()}

# 처음 몇 개 인덱스는 사전에 정의되어 있습니다
word_to_index["<PAD>"] = 0
word_to_index["<BOS>"] = 1
word_to_index["<UNK>"] = 2  # unknown
word_to_index["<UNUSED>"] = 3

index_to_word = {index:word for word, index in word_to_index.items()}

print(index_to_word[1])     # '<BOS>' 가 출력됩니다.
print(word_to_index['the'])  # 4 이 출력됩니다.
print(index_to_word[4])     # 'the' 가 출력됩니다.

total_data_text = list(x_train) + list(x_test)
# 텍스트데이터 문장길이의 리스트를 생성한 후
num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)
# 문장길이의 평균값, 최대값, 표준편차를 계산해 본다.
print('문장길이 평균 : ', np.mean(num_tokens))
print('문장길이 최대 : ', np.max(num_tokens))
print('문장길이 표준편차 : ', np.std(num_tokens))

# 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)
print('pad_sequences maxlen : ', maxlen)
print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다. '.format(np.sum(num_tokens < max_tokens) / len(num_tokens)))

x_train = keras.preprocessing.sequence.pad_sequences(x_train,
                                                     value=word_to_index["<PAD>"],
                                                     padding='post', # 혹은 'pre'
                                                     maxlen=maxlen)

x_test = keras.preprocessing.sequence.pad_sequences(x_test,
                                                    value=word_to_index["<PAD>"],
                                                    padding='post', # 혹은 'pre'
                                                    maxlen=maxlen)

print(x_train.shape)
print(y_train.shape)

vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 16  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)



# validation set 10000건 분리
x_val = x_train[:10000]
y_val = y_train[:10000]

# validation set을 제외한 나머지 15000건
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

print(partial_x_train.shape)
print(partial_y_train.shape)

model, history = train_LSTM('imdb_01.h5', dim=word_vector_dim)
results = model.evaluate(x_test,  y_test, verbose=2)
print(results)
if history is not None:
    show_graph(history.history)

e_file_path = save_embedding_file('/word2vec.txt', model)
word2vec = verify_gensim_word2vec(e_file_path)

vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 300  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)

embedding_matrix = np.random.rand(vocab_size, word_vector_dim)

# embedding_matrix에 Word2Vec 워드벡터를 단어 하나씩마다 차례차례 카피한다.
for i in range(4,vocab_size):
    if index_to_word[i] in word2vec:
        embedding_matrix[i] = word2vec[index_to_word[i]]

model, history = train_LSTM('imdb_preW2V.h5', dim=word_vector_dim, e_matrix=embedding_matrix)
results = model.evaluate(x_test,  y_test, verbose=2)
print(results)
if history is not None:
    show_graph(history.history)
