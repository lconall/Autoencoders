
from ....pytest_base import PytestBase
from ...entities.decoder import decoder_model

from keras.models import Model


def TestDecoder(PytestBase): 

    def test_decoder_is_keras_model(self): 
        decoder = decoder_model("test_decoder", 12, 3)
        assert isinstance(decoder, Model)

    def test_decoder_has_proper_input_shape(self): 
        decoder = decoder_model("test_decoder", 12, 3)
        assert decoder.input_shape == (3,)
        assert decoder.output_shape == (12,)
