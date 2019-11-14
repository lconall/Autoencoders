
from keras.layers import Input, Dense
from keras.models import Model


def autoencoder_model(name, encoder, decoder, **kwargs): 
    ''' This model uses an encoder and decoder model to map input to a reconstruction'''
    
    print("Encoder Input Shape: ", encoder.input_shape)
    input_data = Input(shape=(encoder.input_shape[1],))
    autoencoder = Model(inputs=input_data, outputs=decoder(encoder(input_data)))
    autoencoder.name = name
    optimizer = kwargs.get('optimizer', 'adadelta')
    loss = kwargs.get('loss', 'binary_crossentropy')
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder