
from ....pytest_base import PytestBase
from keras.layers import Dense, Input
from keras.models import Model

from ...entities.autoencoder import autoencoder_model

class TestAutoencoder(PytestBase): 
    def test_autoencoder_has_proper_input_shape(self): 
        flattened_input_size = 12
        encoded_dimension = 3
        encoder_input = Input(shape=(flattened_input_size,))
        encoding_layer = Dense(encoded_dimension, activation='relu')(encoder_input)
        encoder = Model(inputs=encoder_input, outputs=[encoding_layer]) 

        decoder_input = Input(shape=(encoded_dimension,))
        decoding_layer = Dense(flattened_input_size, activation='sigmoid')(decoder_input)  
        decoder = Model(inputs=decoder_input, outputs=decoding_layer)

        autoencoder = autoencoder_model("test_autoencoder", encoder, decoder)
        assert isinstance(autoencoder, Model)
        assert autoencoder.input_shape == (None, flattened_input_size)
        assert autoencoder.output_shape == (None, flattened_input_size)
