import unittest
import yaml
from ABM.model import AirportModel


class TestModel(unittest.TestCase):
    def test_random_model(self):
        random_config = yaml.load(open('ABM/configs/default_config.yaml'), Loader=yaml.FullLoader)

        model = AirportModel()
        self.assertEqual(model.name, 'ABM')

    def test_image_view(self):
        image_config = yaml.load(open('ABM/configs/image_scenario.yaml'), Loader=yaml.FullLoader)
        model = AirportModel()
        self.assertEqual(model.name, 'ABM')
