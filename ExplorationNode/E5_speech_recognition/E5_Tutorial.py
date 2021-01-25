import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' #OMP error solution for MacOS

def residual_model():
    input_tensor = layers.Input(shape=(sr, 1))

    x = layers.Conv1D(32, 9, padding='same', activation='relu')(input_tensor)
    x = layers.Conv1D(32, 9, padding='same', activation='relu')(x)
    skip_1 = layers.MaxPool1D()(x)

    x = layers.Conv1D(64, 9, padding='same', activation='relu')(skip_1)
    x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
    x = tf.concat([x, skip_1], -1)
    skip_2 = layers.MaxPool1D()(x)

    x = layers.Conv1D(128, 9, padding='same', activation='relu')(skip_2)
    x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
    x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
    x = tf.concat([x, skip_2], -1)
    skip_3 = layers.MaxPool1D()(x)

    x = layers.Conv1D(256, 9, padding='same', activation='relu')(skip_3)
    x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
    x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
    x = tf.concat([x, skip_3], -1)
    x = layers.MaxPool1D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    output_tensor = layers.Dense(12)(x)

    model_wav_skip = tf.keras.Model(input_tensor, output_tensor)

    model_wav_skip.summary()

    return model_wav_skip

def Conv1D_model():

    input_tensor = layers.Input(shape=(sr, 1))

    x = layers.Conv1D(32, 9, padding='same', activation='relu')(input_tensor)
    x = layers.Conv1D(32, 9, padding='same', activation='relu')(x)
    x = layers.MaxPool1D()(x)

    x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
    x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
    x = layers.MaxPool1D()(x)

    x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
    x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
    x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
    x = layers.MaxPool1D()(x)

    x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
    x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
    x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
    x = layers.MaxPool1D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    output_tensor = layers.Dense(12)(x)

    model_wav = tf.keras.Model(input_tensor, output_tensor)

    model_wav.summary()

    return model_wav

def draw_graph():

    acc = history_wav.history['accuracy']
    val_acc = history_wav.history['val_accuracy']

    loss=history_wav.history['loss']
    val_loss=history_wav.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

data_path = os.getcwd()+'/data/speech_wav_8000.npz'
speech_data = np.load(data_path)

# 라벨 목록
# ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go' ]
# 이외 데이터들은 'unknown', 'silence'로 분류되어 있습니다.

target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

label_value = target_list
label_value.append('unknown')
label_value.append('silence')

print('LABEL : ', label_value)

new_label_value = dict()
for i, l in enumerate(label_value):
    new_label_value[l] = i
label_value = new_label_value

print('Indexed LABEL : ', new_label_value)

temp = []
for v in speech_data["label_vals"]:
    temp.append(label_value[v[0]])
label_data = np.array(temp)

sr = 8000
train_wav, test_wav, train_label, test_label = train_test_split(speech_data["wav_vals"],
                                                                label_data,
                                                                test_size=0.1,
                                                                shuffle=True)
print(train_wav)

train_wav = train_wav.reshape([-1, sr, 1]) # add channel for CNN
test_wav = test_wav.reshape([-1, sr, 1])

print("train data : ", train_wav.shape)
print("train labels : ", train_label.shape)
print("test data : ", test_wav.shape)
print("test labels : ", test_label.shape)
print("✅")

batch_size = 32
max_epochs = 10

# the save point
checkpoint_dir = os.getcwd()+'/models/wav'

# tf.data.Dataset.from_tensor_slices 함수에 return 받길 원하는 데이터를 튜플 (data, label)
# 형태로 넣어서 사용할 수 있습니다.
# map 함수는 dataset이 데이터를 불러올때마다 동작시킬 데이터 전처리 함수를 매핑해 주는 역할을 합니다. 첫번째 map 함수는 from_tensor_slice 에 입력한 튜플 형태로 데이터를 받으며 return 값으로 어떤 데이터를 반환할지 결정합니다.
# map 함수는 중첩해서 사용이 가능합니다.
def one_hot_label(wav, label):
    label = tf.one_hot(label, depth=12)
    return wav, label

# for train
train_dataset = tf.data.Dataset.from_tensor_slices((train_wav, train_label))
train_dataset = train_dataset.map(one_hot_label)
train_dataset = train_dataset.repeat().batch(batch_size=batch_size)
print(train_dataset)

# for test
test_dataset = tf.data.Dataset.from_tensor_slices((test_wav, test_label))
test_dataset = test_dataset.map(one_hot_label)
test_dataset = test_dataset.batch(batch_size=batch_size)
print(test_dataset)
print("✅")


model_wav = Conv1D_model()

optimizer=tf.keras.optimizers.Adam(1e-4)
model_wav.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 mode='auto',
                                                 save_best_only=True,
                                                 verbose=1)

history_wav = model_wav.fit(train_dataset, epochs=max_epochs,
                            steps_per_epoch=len(train_wav) // batch_size,
                            validation_data=test_dataset,
                            validation_steps=len(test_wav) // batch_size,
                            callbacks=[cp_callback]
                            )

draw_graph(history_wav)

model_wav.load_weights(checkpoint_dir)
results = model_wav.evaluate(test_dataset)
# loss
print("loss value: {:.3f}".format(results[0]))
# accuracy
print("accuracy value: {:.4f}%".format(results[1]*100))

inv_label_value = {v: k for k, v in label_value.items()}
batch_index = np.random.choice(len(test_wav), size=1, replace=False)

batch_xs = test_wav[batch_index]
batch_ys = test_label[batch_index]
y_pred_ = model_wav(batch_xs, training=False)

print("label : ", str(inv_label_value[batch_ys[0]]))

if np.argmax(y_pred_) == batch_ys[0]:
    print("y_pred: " + str(inv_label_value[np.argmax(y_pred_)]) + '(Correct!)')
else:
    print("y_pred: " + str(inv_label_value[np.argmax(y_pred_)]) + '(Incorrect!)')
print("✅")

model_wav = residual_model()

optimizer=tf.keras.optimizers.Adam(1e-4)
model_wav.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 mode='auto',
                                                 save_best_only=True,
                                                 verbose=1)

history_wav = model_wav.fit(train_dataset, epochs=max_epochs,
                            steps_per_epoch=len(train_wav) // batch_size,
                            validation_data=test_dataset,
                            validation_steps=len(test_wav) // batch_size,
                            callbacks=[cp_callback]
                            )

draw_graph(history_wav)