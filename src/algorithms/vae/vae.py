import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    Reshape,
    Conv2DTranspose,
    Layer,
)
from keras.models import Model
from keras import backend as ops
from keras import metrics


class Sampling(Layer):
    # Reparameterization trick
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = tf.random.set_seed(42)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = None
        self.batch_size = None
        self.activation = activation
        self.encoder = None
        self.decoder = None
        self._build_encoder()
        self._build_decoder()
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.val_total_loss_tracker = metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = metrics.Mean(name="val_kl_loss")

    def _build_encoder(self):
        encoder_inputs = Input(shape=(self.input_dim[0], self.input_dim[1], self.input_dim[2]))
        x = Conv2D(32, 3, activation=self.activation, strides=2, padding="same")(encoder_inputs)
        x = Conv2D(64, 3, activation=self.activation, strides=2, padding="same")(x)
        x = Conv2D(64, 3, activation=self.activation, strides=2, padding="same")(x)
        x = Conv2D(64, 3, activation=self.activation, strides=1, padding="same")(x)
        x = Flatten()(x)
        x = Dense(16, activation=self.activation)(x)

        z_mean = Dense(self.encoding_dim, name="z_mean")(x)
        z_log_var = Dense(self.encoding_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        latent_inputs = Input(shape=(self.encoding_dim, ))
        x = Dense(7 * 7 * 64, activation=self.activation)(latent_inputs)
        x = Reshape((7, 7, 64))(x)
        x = Conv2DTranspose(64, 3, activation=self.activation, strides=1, padding="same")(x)
        x = Conv2DTranspose(64, 3, activation=self.activation, strides=2, padding="same")(x)
        x = Conv2DTranspose(32, 3, activation=self.activation, strides=2, padding="same")(x)
        decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    def encode(self, data):
        return self.encoder.predict(data)

    def decode(self, encoded_data):
        return self.decoder.predict(encoded_data)

    def compress(self, data):
        z_mean, _, _ = self.encoder(data)

        return z_mean

    def decompress(self, encoded_data):
        return self.decode(encoded_data)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = ops.mean(
            ops.sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }

    def visualize_2d(self, data, labels, save_plot=False):
        if self.encoding_dim != 2:
            raise ValueError("This method is only for 2D visualization (encoding_dim must be 2)")

        os.makedirs("experiments", exist_ok=True)

        plot_path = (
            f"experiments/"
            f"{self.epochs}epochs_"
            f"{self.batch_size}batchsize_"
            f"{self.encoding_dim}encodingdim_"
            f"{self.activation}actfunc_"
            "2d.png"
        )

        encoded_data = self.compress(data)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap="tab10", alpha=0.5, s=10)
        plt.colorbar(scatter, label="Digit Label")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("2D Visualization of Encoded Data")
        plt.tight_layout()

        if save_plot:
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.show()

    def visualize_reconstruction(self, data, nb_images=10):
        encoded_data = self.compress(data)
        decoded_data = self.decompress(encoded_data)

        plt.figure(figsize=(20, 4))
        for i in range(nb_images):
            # Display original
            ax = plt.subplot(2, nb_images, i + 1)
            plt.imshow(data[i].reshape(self.input_dim[0], self.input_dim[1]), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, nb_images, i + 1 + nb_images)
            plt.imshow(decoded_data[i].reshape(self.input_dim[0], self.input_dim[1]), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def generate_images(self, nb_images=10):
        random_data = np.random.normal(size=(nb_images, self.encoding_dim))
        generated_images = self.decompress(random_data)

        plt.figure(figsize=(20, 4))
        for i in range(nb_images):
            ax = plt.subplot(1, nb_images, i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def visualize_latent_space_grid(self, data, labels, grid_size=15):
        grid_size = 15
        grid_scale = 1

        grid = []

        for y in scipy.stats.norm.ppf(np.linspace(0.99, 0.01, grid_size), scale=grid_scale):
            for x in scipy.stats.norm.ppf(np.linspace(0.01, 0.99, grid_size), scale=grid_scale):
                grid.append([x, y])

        grid = np.array(grid)

        encoded_data = self.compress(data)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap='tab10', alpha=0.5, s=10)
        plt.scatter(grid[:, 0], grid[:, 1], color='black', alpha=1, s=30, marker='x', linewidths=1)
        plt.colorbar(scatter, label="Digit Label")
        plt.show()

        images = self.decoder.predict(grid)

        _, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        for i, ax in enumerate(ax.flat):
            ax.imshow(images[i].reshape(self.input_dim[0], self.input_dim[1]), cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


class VAE_Base(Model):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = None
        self.batch_size = None
        self.activation = activation
        self.encoder = None
        self.decoder = None
        self._build_encoder()
        self._build_decoder()
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.val_total_loss_tracker = metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = metrics.Mean(name="val_kl_loss")

    def _build_encoder(self):
        encoder_inputs = Input(shape=(self.input_dim, ))
        x = Dense(1024, activation=self.activation)(encoder_inputs)
        x = Dense(512, activation=self.activation)(x)
        x = Dense(256, activation=self.activation)(x)
        x = Dense(128, activation=self.activation)(x)

        z_mean = Dense(self.encoding_dim, name="z_mean")(x)
        z_log_var = Dense(self.encoding_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        latent_inputs = Input(shape=(self.encoding_dim,))
        x = Dense(128, activation=self.activation)(latent_inputs)
        x = Dense(256, activation=self.activation)(x)
        x = Dense(512, activation=self.activation)(x)
        x = Dense(1024, activation=self.activation)(x)
        decoder_outputs = Dense(self.input_dim, activation="sigmoid")(x)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    def encode(self, data):
        return self.encoder.predict(data)

    def decode(self, encoded_data):
        return self.decoder.predict(encoded_data)

    def compress(self, data):
        z_mean, _, _ = self.encoder(data)

        return z_mean

    def decompress(self, encoded_data):
        return self.decode(encoded_data)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = ops.mean(
            ops.sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }

    def visualize_2d(self, data, labels, save_plot=False):
        if self.encoding_dim != 2:
            raise ValueError("This method is only for 2D visualization (encoding_dim must be 2)")

        os.makedirs("experiments", exist_ok=True)

        plot_path = (
            f"experiments/"
            f"{self.epochs}epochs_"
            f"{self.batch_size}batchsize_"
            f"{self.encoding_dim}encodingdim_"
            f"{self.activation}actfunc_"
            "2d.png"
        )

        encoded_data = self.compress(data)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap="tab10", alpha=0.5, s=10)
        plt.colorbar(scatter, label="Digit Label")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("2D Visualization of Encoded Data")
        plt.tight_layout()

        if save_plot:
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.show()

    def visualize_reconstruction(self, data, nb_images=10):
        encoded_data = self.compress(data)
        decoded_data = self.decompress(encoded_data)

        plt.figure(figsize=(20, 4))
        for i in range(nb_images):
            # Display original
            ax = plt.subplot(2, nb_images, i + 1)
            plt.imshow(data[i].reshape(self.input_dim[0], self.input_dim[1]), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, nb_images, i + 1 + nb_images)
            plt.imshow(decoded_data[i].reshape(self.input_dim[0], self.input_dim[1]), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def generate_images(self, nb_images=10):
        random_data = np.random.normal(size=(nb_images, self.encoding_dim))
        generated_images = self.decompress(random_data)

        plt.figure(figsize=(20, 4))
        for i in range(nb_images):
            ax = plt.subplot(1, nb_images, i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def visualize_latent_space_grid(self, data, labels, grid_size=15):
        grid_size = 15
        grid_scale = 1

        grid = []

        for y in scipy.stats.norm.ppf(np.linspace(0.99, 0.01, grid_size), scale=grid_scale):
            for x in scipy.stats.norm.ppf(np.linspace(0.01, 0.99, grid_size), scale=grid_scale):
                grid.append([x, y])

        grid = np.array(grid)

        encoded_data = self.compress(data)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap='tab10', alpha=0.5, s=10)
        plt.scatter(grid[:, 0], grid[:, 1], color='black', alpha=1, s=30, marker='x', linewidths=1)
        plt.colorbar(scatter, label="Digit Label")
        plt.show()

        images = self.decoder.predict(grid)

        _, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        for i, ax in enumerate(ax.flat):
            ax.imshow(images[i].reshape(self.input_dim[0], self.input_dim[1]), cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.show()



class VAE_LEGO(Model):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = None
        self.batch_size = None
        self.activation = activation
        self.encoder = None
        self.decoder = None
        self._build_encoder()
        self._build_decoder()
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.val_total_loss_tracker = metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = metrics.Mean(name="val_kl_loss")

    def _build_encoder(self):
        encoder_inputs = Input(shape=self.input_dim)
        x = Conv2D(32, 3, activation=self.activation, strides=2, padding="same")(encoder_inputs)
        x = Conv2D(64, 3, activation=self.activation, strides=2, padding="same")(x)
        x = Conv2D(64, 3, activation=self.activation, strides=2, padding="same")(x)
        x = Conv2D(64, 3, activation=self.activation, strides=1, padding="same")(x)
        x = Flatten()(x)
        x = Dense(16, activation=self.activation)(x)

        z_mean = Dense(self.encoding_dim, name="z_mean")(x)
        z_log_var = Dense(self.encoding_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        latent_inputs = Input(shape=(self.encoding_dim,))
        x = Dense(8 * 8 * 64, activation=self.activation)(latent_inputs)
        x = Reshape((8, 8, 64))(x)
        x = Conv2DTranspose(64, 3, activation=self.activation, strides=2, padding="same")(x)
        x = Conv2DTranspose(64, 3, activation=self.activation, strides=2, padding="same")(x)
        x = Conv2DTranspose(32, 3, activation=self.activation, strides=2, padding="same")(x)
        decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    def encode(self, data):
        return self.encoder.predict(data)

    def decode(self, encoded_data):
        return self.decoder.predict(encoded_data)

    def compress(self, data):
        z_mean, _, _ = self.encoder(data)

        return z_mean

    def decompress(self, encoded_data):
        return self.decode(encoded_data)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = ops.mean(
            ops.sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }

    def visualize_2d(self, data, labels, save_plot=False):
        if self.encoding_dim != 2:
            raise ValueError("This method is only for 2D visualization (encoding_dim must be 2)")

        os.makedirs("experiments", exist_ok=True)

        plot_path = (
            f"experiments/"
            f"{self.epochs}epochs_"
            f"{self.batch_size}batchsize_"
            f"{self.encoding_dim}encodingdim_"
            f"{self.activation}actfunc_"
            "2d.png"
        )

        encoded_data, _, _ = self.encode(data)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap="tab10", alpha=0.5, s=10)
        plt.colorbar(scatter, label="Digit Label")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("2D Visualization of Encoded Data")
        plt.tight_layout()

        if save_plot:
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.show()

    def visualize_reconstruction(self, data, nb_images=10):
        encoded_data = self.compress(data)
        decoded_data = self.decompress(encoded_data)

        plt.figure(figsize=(20, 4))
        for i in range(nb_images):
            # Display original
            ax = plt.subplot(2, nb_images, i + 1)
            plt.imshow(data[i].reshape(self.input_dim[0], self.input_dim[1]), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, nb_images, i + 1 + nb_images)
            plt.imshow(decoded_data[i].reshape(self.input_dim[0], self.input_dim[1]), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def generate_images(self, nb_images=10):
        random_data = np.random.normal(size=(nb_images, self.encoding_dim))
        generated_images = self.decompress(random_data)

        plt.figure(figsize=(20, 4))
        for i in range(nb_images):
            ax = plt.subplot(1, nb_images, i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def visualize_latent_space_grid(self, data, labels, grid_size=15):
        grid_size = 15
        grid_scale = 1

        grid = []

        for y in scipy.stats.norm.ppf(np.linspace(0.99, 0.01, grid_size), scale=grid_scale):
            for x in scipy.stats.norm.ppf(np.linspace(0.01, 0.99, grid_size), scale=grid_scale):
                grid.append([x, y])

        grid = np.array(grid)

        encoded_data, _, _ = self.encode(data)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap='tab10', alpha=0.5, s=10)
        plt.scatter(grid[:, 0], grid[:, 1], color='black', alpha=1, s=30, marker='x', linewidths=1)
        plt.colorbar(scatter, label="Digit Label")
        plt.show()

        images = self.decoder.predict(grid)

        _, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        for i, ax in enumerate(ax.flat):
            ax.imshow(images[i].reshape(self.input_dim[0], self.input_dim[1]), cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.show()