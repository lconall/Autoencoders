
from keras.layers import Input, Dense
from keras.models import Model

def decoder_model(name, encoded_dimension, flattened_img_size): 
    # "decoded" is the lossy reconstruction of the input
    encoded_input = Input(shape=(encoded_dimension,))
    decoding_layer = Dense(flattened_img_size, activation='sigmoid')(encoded_input)  
    decoder = Model(inputs=encoded_input, outputs=decoding_layer)
    decoder.name = name
    return decoder