
from keras.layers import Input, Dense
from keras.models import Model

def encoder_model(name, flattened_input_size, encoding_dimension):
    '''Creates a Keras encoder that maps an input to it's encoded representation.'''
    encoder_input = Input(shape=(flattened_input_size,))
    encoding_layer = Dense(encoding_dimension, activation='relu')(encoder_input)
    encoder = Model(inputs=encoder_input, outputs=[encoding_layer]) 
    encoder.name = name
    return encoder 