import matplotlib.pyplot as plt
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
        loss="binary_crossentropy",
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.loss = "binary_crossentropy"
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

    def _build_encoder(self):
        encoder_inputs = Input(shape=(self.input_dim[0], self.input_dim[1], 1))
        x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)
        z_mean = Dense(self.encoding_dim, name="z_mean")(x)
        z_log_var = Dense(self.encoding_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        latent_inputs = Input(shape=(self.encoding_dim,))
        x = Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = Reshape((7, 7, 64))(x)
        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
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

    def visualize_2d(self, data, labels, save_plot=False):
        plot_path = (
            f"experiments/"
            f"{self.epochs}epochs_"
            f"{self.batch_size}batchsize_"
            f"{self.encoding_dim}encodingdim_"
            f"{self.activation}actfunc_"
            "2d.png"
        )
        encoded_data = self.encode(data)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap="viridis", s=1
        )
        plt.colorbar(scatter, label="Digit Label")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("2D Visualization of MNIST with Autoencoder")

        if save_plot:
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        plt.show()
