import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import librosa


def wav2spec(wav, fft_size=258): # spectrogram shape을 맞추기위해서 size 변형
    # for fft in wav:
    #     fft_list.append(np.abs(librosa.stft(wav, n_fft=fft_size)))
    D = np.abs(librosa.stft(wav, n_fft=fft_size))
    return D

# tf.data.Dataset.map함수로 등록되는 전처리 함수들은 모두 graph mode로 동작하기 때문에
# Note that irrespective of the context in which map_func is defined (eager vs. graph),
# tf.data traces the function and executes it as a graph.
# To use Python code inside of the function you have a few options:
# 1) Rely on AutoGraph to convert Python code into an equivalent graph computation.
# The downside of this approach is that AutoGraph can convert some but not all Python code.
# 2) Use tf.py_function, which allows you to write arbitrary Python code but
# will generally result in worse performance than 1)
def one_hot_label(wav, label):
    label = tf.one_hot(label, depth=12)
    # librosa의 경우 python code이므로 이렇게 tf.numpy_function으로 wrapping해주어야 한다.
    wav = tf.numpy_function(func = w2s, inp =[wav], Tout =tf.float32)

    return wav, label

def w2s(wav):
    # wav = tf.make_ndarray(wav)
    # fft = np.abs(librosa.stft(wav.numpy(), n_fft=258))
    fft = np.abs(librosa.stft(wav, n_fft=258))
    # fft = tf.audio.convert_audio_dtype(fft, 'float32')
    # fft = tf.convert_to_tensor(fft, dtype=tf.float32)
    fft = tf.reshape(fft, [130, 126, 1])

    return fft

def residual_model():
    # input_tensor = layers.Input(shape=(sr, 1))
    input_tensor = layers.Input(shape=(130, 126, 1))

    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    skip_1 = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(skip_1)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = tf.concat([x, skip_1], -1)
    skip_2 = layers.MaxPool2D()(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(skip_2)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = tf.concat([x, skip_2], -1)
    skip_3 = layers.MaxPool2D()(x)

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(skip_3)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = tf.concat([x, skip_3], -1)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    output_tensor = layers.Dense(12)(x)

    model_wav_skip = tf.keras.Model(input_tensor, output_tensor)

    model_wav_skip.summary()

    return model_wav_skip

def Conv2D_model():

    # input_tensor = layers.Input(shape=(sr, 1))
    input_tensor = layers.Input(shape=(130, 126, 1))

    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    output_tensor = layers.Dense(12)(x)

    model_wav = tf.keras.Model(input_tensor, output_tensor)

    model_wav.summary()

    return model_wav



def draw_graph(history_wav):

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

def train_2Dconv_n_show(model, checkpoint_dir, train_dataset, train_wav, test_dataset, test_wav):
    optimizer=tf.keras.optimizers.Adam(1e-4)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                     save_weights_only=True,
                                                     monitor='val_loss',
                                                     mode='auto',
                                                     save_best_only=True,
                                                     verbose=1)

    history_wav = model.fit(train_dataset, epochs=max_epochs,
                            steps_per_epoch=len(train_wav) // batch_size,
                            validation_data=test_dataset,
                            validation_steps=len(test_wav) // batch_size,
                            callbacks=[cp_callback]
                            )

    draw_graph(history_wav)

    model.load_weights(checkpoint_dir)
    results = model.evaluate(test_dataset)

    return results


sr = 8000
data_path = os.getcwd()+'/data/speech_wav_8000.npz'
checkpoint_dir = os.getcwd()+'/models/wav'
speech_data = np.load(data_path)

target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
label_value = target_list
label_value.append('unknown')
label_value.append('silence')

new_label_value = dict()
for i, l in enumerate(label_value):
    new_label_value[l] = i
label_value = new_label_value

temp = []
for v in speech_data["label_vals"]:
    temp.append(label_value[v[0]])
label_data = np.array(temp)

train_wav, test_wav, train_label, test_label = train_test_split(speech_data["wav_vals"],
                                                                label_data,
                                                                test_size=0.1,
                                                                shuffle=True)

import random
rand = random.randint(0, len(speech_data["wav_vals"]))

data = speech_data["wav_vals"][rand]
print("Wave data shape : ", data.shape)
print("label : ", speech_data["label_vals"][rand])
spec = wav2spec(data)
print("Waveform shape : ",data.shape)
print("Spectrogram shape : ",spec.shape)

batch_size = 32
max_epochs = 10

train_dataset = tf.data.Dataset.from_tensor_slices((train_wav, train_label))
train_dataset = train_dataset.map(one_hot_label)
train_dataset = train_dataset.repeat().batch(batch_size=batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_wav, test_label))
test_dataset = test_dataset.map(one_hot_label)
test_dataset = test_dataset.batch(batch_size=batch_size)


model_wav = Conv2D_model()
optimizer=tf.keras.optimizers.Adam(1e-4)
model_wav.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=optimizer,
              metrics=['accuracy'])
model_wav.load_weights(checkpoint_dir)
results = model_wav.evaluate(test_dataset)
# model_wav = residual_model()

# results = train_2Dconv_n_show(model_wav, checkpoint_dir, train_dataset, train_wav, test_dataset, test_wav)

# loss
print("loss value: {:.3f}".format(results[0]))
# accuracy
print("accuracy value: {:.4f}%".format(results[1]*100))
# inv_label_value = {v: k for k, v in label_value.items()}
# batch_index = np.random.choice(len(test_wav), size=1, replace=False)
# batch_xs = test_wav[batch_index]
# batch_ys = test_label[batch_index]
# y_pred_ = model_wav(batch_xs, training=False)
#
# print("label : ", str(inv_label_value[batch_ys[0]]))
#
# if np.argmax(y_pred_) == batch_ys[0]:
#     print("y_pred: " + str(inv_label_value[np.argmax(y_pred_)]) + '(Correct!)')
# else:
#     print("y_pred: " + str(inv_label_value[np.argmax(y_pred_)]) + '(Incorrect!)')
# print("✅")