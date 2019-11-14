
from ....pytest_base import PytestBase
from ...entities.encoder import encoder_model

from keras.models import Model


def TestEncoder(PytestBase): 
    def test_encoder_is_keras_model(self): 
        encoder = encoder_model("test_encoder", 12, 3)
        assert isinstance(encoder, Model)

    def test_encoder_has_proper_input_shape(self): 
        encoder = encoder_model("test_encoder", 12, 3)
        assert encoder.input_shape == (12,)
        assert encoder.output_shape == (3,)
