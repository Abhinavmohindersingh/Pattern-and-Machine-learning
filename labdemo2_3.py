import imageio.v2 as imageio
import os
import numpy as np
import tensorflow as tf

from cifar import outputs

model = tf.keras.models.Model(...)

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K

IMG_SHAPE = (256, 256, 1)

# Directories
# Example using absolute paths
base_path = '/Users/abhinavsingh/Downloads/keras_png_slices_data'  # Replace 'path_to_datasets_folder' with the appropriate path
data_dirs = {
    "train": os.path.join(base_path, "keras_png_slices_train"),
    "validate": os.path.join(base_path, "keras_png_slices_validate"),
    "test": os.path.join(base_path, "keras_png_slices_test"),
    "seg_train": os.path.join(base_path, "keras_png_slices_seg_train"),
    "seg_validate": os.path.join(base_path, "keras_png_slices_seg_validate"),
    "seg_test": os.path.join(base_path, "keras_png_slices_seg_test")
}


datasets = {key: [imageio.imread(os.path.join(dir, img_name)) for img_name in os.listdir(dir)] for key, dir in data_dirs.items()}
for key in ["train", "validate", "test"]:
    datasets[key] = [(img - img.min()) / (img.max() - img.min()) for img in datasets[key]]
for key in datasets.keys():
    datasets[key] = [img[..., np.newaxis] for img in datasets[key]]
for key in datasets.keys():
    datasets[key] = np.array(datasets[key])

latent_dim = 2  # for visualization purpose

inputs = Input(shape=IMG_SHAPE)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z])

decoder_input = Input(shape=(latent_dim,))
x = Dense(1024, activation='relu')(decoder_input)
x = Dense(32 * 32 * 128, activation='relu')(x)
x = Reshape((32, 32, 128))(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = Model(decoder_input, decoded)
z_decoded = decoder(z)


class VAELossLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x, x_decoded_mean = inputs
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        return x



vae = Model(inputs, z_decoded)
vae.compile(optimizer='adam', loss=vae_loss)
y = VAELossLayer()([inputs, outputs])
vae = Model(inputs, y)

vae.fit(datasets['train'], datasets['train'], epochs=50, batch_size=32, shuffle=True)

vae.compile(optimizer='adam', loss=None)  # Loss is already handled in the VAELossLayer



import matplotlib.pyplot as plt

n = 20  # number of images per axis
digit_size = IMG_SHAPE[0]
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
