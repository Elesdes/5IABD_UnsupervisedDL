import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential


class Autoencoder:
    def __init__(
        self,
        input_dim,
        encoding_dim,
        encoder_layers,
        decoder_layers,
        loss="binary_crossentropy",
        activation="relu"):

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.loss = loss
        self.epochs = None
        self.batch_size = None
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.activation = activation
        self._build_model()

    def _build_model(self):
        # Encoder
        self.encoder = Sequential(name='encoder')
        self.encoder.add(Input(shape=(self.input_dim,)))
        for units in self.encoder_layers:
            self.encoder.add(Dense(units, activation=self.activation))
        self.encoder.add(Dense(self.encoding_dim, activation=self.activation))

        # Decoder
        self.decoder = Sequential(name='decoder')
        self.decoder.add(Input(shape=(self.encoding_dim,)))
        for units in self.decoder_layers:
            self.decoder.add(Dense(units, activation=self.activation))
        self.decoder.add(Dense(self.input_dim, activation='sigmoid'))

        # Autoencoder
        self.autoencoder = Sequential([self.encoder, self.decoder], name='autoencoder')

        self.autoencoder.compile(optimizer='adam', loss=self.loss)

    def train(self, x_train, x_test, epochs=50, batch_size=256):
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, x_test),
        )

    def encode(self, data):
        return self.encoder.predict(data)

    def decode(self, encoded_data):
        return self.decoder.predict(encoded_data)

    def compress(self, data):
        return self.encode(data)

    def decompress(self, encoded_data):
        return self.decode(encoded_data)

    def visualize_2d(self, data, labels, save_plot=False):
        plot_path = (
            f"experiments/"
            f"{self.epochs}epochs_"
            f"{self.batch_size}batchsize_"
            f"{self.encoding_dim}encodingdim_"
            f"{self.activation}actfunc_"
            f"{self.encoder_layers}encoders_"
            f"{self.decoder_layers}decoders_"
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

    def visualize_3d(self, data, labels, save_plot=False):
        plot_path = (
            f"experiments/"
            f"{self.epochs}epochs_"
            f"{self.batch_size}batchsize_"
            f"{self.encoding_dim}encodingdim_"
            f"{self.loss}loss_"
            f"{self.activation}actfunc_"
            f"{self.encoder_layers}encoders_"
            f"{self.decoder_layers}decoders_"
            "3d.png"
        )
        encoded_data = self.encode(data)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            encoded_data[:, 0],
            encoded_data[:, 1],
            encoded_data[:, 2],
            c=labels,
            cmap="viridis",
            s=1,
        )
        ax.set_xlabel("Encoded Dimension 1")
        ax.set_ylabel("Encoded Dimension 2")
        ax.set_zlabel("Encoded Dimension 3")
        fig.colorbar(scatter, ax=ax, label="Digit Label")
        plt.title("3D Visualization of Data with Autoencoder")

        if save_plot:
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        plt.show()
