
from ....pytest_base import PytestBase

from ....MINSTDataLoader.data_loader import MINSTDataLoader
from ....NoisyMINSTDataLoader.data_loader import NoisyMINSTDataLoader
from ...interactors.run_autoencoder import RunAutoencoder

class TestRunAutoencoder(PytestBase): 
    def test_run_autoencoder_with_minst_data_loader(self):
        minst_data_loader = MINSTDataLoader()
        run_autoencoder = RunAutoencoder(data_loader=minst_data_loader)
        results = run_autoencoder.run(3, epochs=5)
        print(results)

    def test_run_autoencoder_with_noisy_minst_data_loader(self):
        noisy_minst_data_loader = NoisyMINSTDataLoader()
        run_autoencoder = RunAutoencoder(data_loader=noisy_minst_data_loader)
        results = run_autoencoder.run(3, epochs=5)
        print(results)