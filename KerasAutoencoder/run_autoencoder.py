import sys, os
sys.path.append(os.path.join( os.path.dirname(__file__)))

from .entities.encoder import encoder_model
from .entities.decoder import decoder_model
from .entities.autoencoder import autoencoder_model


class RunAutoencoder(): 
    def __init__(self, *args, **kwargs):
        self.naming_prefix = kwargs.get('naming_prefix', "")
        self.encoder_name = str(self.naming_prefix) + kwargs.get('encoder_name', "encoder")
        self.decoder_name = str(self.naming_prefix) + kwargs.get('decoder_name', "decoder")
        self.autoencoder_name = str(self.naming_prefix) + kwargs.get('autoencoder_name', "autoencoder")
        self.data_loader = kwargs.get('data_loader', None)
    
    def run(self, encoding_dimension, **kwargs): 
        data = self.data_loader.load_data()
        encoder = encoder_model(self.encoder_name, 
                                self.data_loader.flattened_input_size, 
                                encoding_dimension)
        decoder = decoder_model(self.decoder_name, 
                                encoding_dimension,
                                self.data_loader.flattened_input_size)
        autoencoder = autoencoder_model(self.autoencoder_name, encoder, decoder)

        print(encoder.summary())
        print(decoder.summary())
        print(autoencoder.summary())


        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 256)
        model_history = autoencoder.fit(data['train_data'], 
                                        data['train_target'], 
                                        epochs=epochs, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        validation_data=(data['test_data'], data['test_target']))

        encoder_predictions = encoder.predict(data['test_data'])
        decoder_predictions = decoder.predict(encoder_predictions)

        return {'test_target': data['test_target'],
                'test_data': data['test_data'], 
                'encoded_data':encoder_predictions, 
                'decoded_data':decoder_predictions, 
                'model_history':model_history}
