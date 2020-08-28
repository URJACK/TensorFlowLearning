import keras
import keras.layers as layers
import tensorflow as tf
import os
import numpy as np

import pydot
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pydot


def create_dataset(batch_size):
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    # all_digits = x_train[:100]
    all_digits = all_digits.astype("float32") / 255
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)
    return dataset


def create_discriminator():
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.GlobalMaxPooling2D(),
            layers.Dense(1),
        ],
        name="discriminator",
    )
    discriminator.summary()
    return discriminator


def create_generator(latent_dim):
    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            # We want to generate 128 coefficients to reshape into a 7x7x128 map
            layers.Dense(7 * 7 * 128),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

    generator.summary()
    return generator


class GAN(keras.Model):
    loss_fn: object
    g_optimizer: keras.optimizers.Optimizer
    d_optimizer: keras.optimizers.Optimizer

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass

    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]
        # 生成随机噪声
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)

        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        print(labels)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        print(labels)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}


class GANMonitor(keras.callbacks.Callback):
    gan: GAN

    def __init__(self, num_img, latent_dim, gan):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.gan = gan

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\gan_keras_g"
        self.model.generator.save(model_path)
        model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\gan_keras_d"
        self.model.discriminator.save(model_path)
        print("Save Success")
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("./samples/gan/img_{i}_{epoch}.png".format(i=i, epoch=epoch))


def train_model(latent_dim):
    batch_size = 64
    epochs = 1
    dataset = create_dataset(batch_size)
    gan = GAN(create_discriminator(), create_generator(latent_dim), latent_dim)
    gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
    gan.fit(dataset, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim, gan=gan)])


def use_model(latent_dim):
    batch_size = 3
    model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\gan_keras_g"
    generator = keras.models.load_model(model_path)
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    print(random_latent_vectors.numpy())
    generated_images = generator(random_latent_vectors)
    generated_images *= 255
    print(generated_images.numpy())
    for i in range(batch_size):
        img = keras.preprocessing.image.array_to_img(generated_images[i].numpy())
        img.save("./samples/gan/img_{i}.png".format(i=i))


def continue_train_model(latent_dim):
    batch_size = 64
    epochs = 5
    dataset = create_dataset(batch_size)

    model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\gan_keras_g"
    generator = keras.models.load_model(model_path)
    model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\gan_keras_d"
    discriminator = keras.models.load_model(model_path)

    gan = GAN(discriminator, generator, latent_dim)
    gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
    gan.fit(dataset, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim, gan=gan)])


def main():
    latent_dim = 128
    # train_model(latent_dim)
    # continue_train_model(latent_dim)
    use_model(latent_dim)


if __name__ == '__main__':
    main()
