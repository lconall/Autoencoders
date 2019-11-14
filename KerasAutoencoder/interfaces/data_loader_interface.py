import sys, os
sys.path.append(os.path.join( os.path.dirname(__file__)))

from abc import ABC, abstractmethod, abstractproperty

class DataLoaderInterface(ABC):
    def __init__(self, *args, **kwargs):
        self.flattened_input_size = None

    @abstractmethod
    def load_data(self):
        pass 

    @abstractmethod
    def flatten_input(self): 
        pass

    @abstractmethod
    def normalize_input(self): 
        pass
