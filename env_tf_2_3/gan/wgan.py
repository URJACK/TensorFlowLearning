import keras
import keras.layers as layers
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

import pydot
from keras.utils.vis_utils import model_to_dot

# draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\several_demo_model.png"
# keras.utils.plot_model(self._model, draw_path, show_shapes=True)

IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 512
NOISE_DIM = 128


# Define the loss functions to be used for discrimiator
# This should be (fake_loss - real_loss)
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions to be used for generator
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def conv_block(x,
               filters,
               activation,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding="same",
               use_bias=True,
               use_bn=False,
               use_dropout=False,
               drop_value=0.5):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def upsample_block(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        up_size=(2, 2),
        padding="same",
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        drop_value=0.3):
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)

    if use_dropout:
        x = layers.Dropout(drop_value)(x)

    return x


def create_discriminator():
    img_input = layers.Input(shape=IMG_SHAPE)
    x = layers.ZeroPadding2D((2, 2))(img_input)
    x = conv_block(x, 64, kernel_size=(5, 5), strides=(2, 2), use_bn=False, use_bias=True,
                   activation=layers.LeakyReLU(0.2), use_dropout=False, drop_value=0.3)

    x = conv_block(x, 128, kernel_size=(5, 5), strides=(2, 2), use_bn=False, use_bias=True,
                   activation=layers.LeakyReLU(0.2), use_dropout=True, drop_value=0.3)

    x = conv_block(x, 256, kernel_size=(5, 5), strides=(2, 2), use_bn=False, use_bias=True,
                   activation=layers.LeakyReLU(0.2), use_dropout=False, drop_value=0.3)

    x = conv_block(x, 512, kernel_size=(5, 5), strides=(2, 2), use_bn=False, use_bias=True,
                   activation=layers.LeakyReLU(0.2), use_dropout=False, drop_value=0.3)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1)(x)

    d_model = keras.Model(img_input, output, name="discriminator")
    return d_model


def create_generator():
    noise = layers.Input(shape=(NOISE_DIM,))
    x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((4, 4, 256))(x)
    x = upsample_block(x, 128, layers.LeakyReLU(0.2), strides=(1, 1),
                       use_bias=False, use_bn=True, padding="same", use_dropout=False)

    x = upsample_block(x, 64, layers.LeakyReLU(0.2), strides=(1, 1),
                       use_bias=False, use_bn=True, padding="same", use_dropout=False)

    x = upsample_block(x, 1, layers.Activation("tanh"), strides=(1, 1),
                       use_bias=False, use_bn=True)

    output = layers.Cropping2D((2, 2))(x)
    g_model = keras.Model(noise, output, name="generator")
    return g_model


class WGAN(keras.Model):
    d_optimizer: keras.optimizers.Optimizer
    g_optimizer: keras.optimizers.Optimizer

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass

    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=3, gp_weight=10.0):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.d_loss_fn = None
        self.g_loss_fn = None

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        print(d_loss_fn)

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated)
            grads = gp_tape.gradient(pred, [interpolated])[0]

        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images)
                # Get the logits for real images
                real_logits = self.discriminator(real_images)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        print("SAVE MODEL")
        model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\wgan_g"
        self.model.generator.save(model_path)
        model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\wgan_d"
        self.model.discriminator.save(model_path)
        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("./samples/wgan/img_{i}_{epoch}.png".format(i=i, epoch=epoch))


def train():
    (train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()
    # train_x = train_x[:10]
    print(f"Number of Examples: {len(train_x)}")
    print(f"Shape of Images: {train_x.shape[1:]}")

    train_x = train_x.reshape(train_x.shape[0], *IMG_SHAPE).astype("float32")
    train_x = (train_x - 127.5) / 127.5
    print(f"After Reshape, Shape of Images: {train_x.shape[1:]}")
    d_model = create_discriminator()
    # d_model.summary()
    g_model = create_generator()
    # g_model.summary()

    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )

    epochs = 20

    cbk = GANMonitor(num_img=3, latent_dim=NOISE_DIM)
    wgan = WGAN(discriminator=d_model, generator=g_model, latent_dim=NOISE_DIM, discriminator_extra_steps=3)
    wgan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer,
                 g_loss_fn=generator_loss, d_loss_fn=discriminator_loss)

    wgan.fit(train_x, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])


def continue_train():
    (train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()

    train_x = train_x.reshape(train_x.shape[0], *IMG_SHAPE).astype("float32")
    train_x = (train_x - 127.5) / 127.5

    model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\wgan_d"
    d_model = keras.models.load_model(model_path)

    model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\wgan_g"
    g_model = keras.models.load_model(model_path)

    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )

    epochs = 20

    cbk = GANMonitor(num_img=3, latent_dim=NOISE_DIM)
    wgan = WGAN(discriminator=d_model, generator=g_model, latent_dim=NOISE_DIM, discriminator_extra_steps=3)
    wgan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer,
                 g_loss_fn=generator_loss, d_loss_fn=discriminator_loss)

    wgan.fit(train_x, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])


if __name__ == '__main__':
    # train()
    continue_train()
