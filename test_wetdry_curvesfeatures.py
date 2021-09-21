import unittest
import math
from wetdry_curves_features import linear
from wetdry_curves_features import exponential
from wetdry_curves_features import wetdry_features
from wetdry_curves_features import read_config


class TestWetDryCurvesFeatures(unittest.TestCase):
    def test_linear_1(self):
        self.assertEqual(linear(1, 1, 1), 2)

    def test_linear_2(self):
        self.assertEqual(linear(0, 1, 1), 1)

    def test_linear_3(self):
        self.assertEqual(linear(0, 0, 0), 0)

    def test_exponential_1(self):
        self.assertEqual(exponential(1, 1, 1), math.e)

    def test_exponential_2(self):
        self.assertEqual(linear(0, 1, 1), 1)

    def test_exponential_3(self):
        self.assertEqual(linear(0, 0, 0), 0)

    def test_data_file_not_exists(self):
        config_file_path = "config.json"
        data_file_path = "file_not_exists.csv"
        result_file_path = "."
        result = wetdry_features(config_file_path, data_file_path, result_file_path, 1)
        self.assertEqual(result.files, [])

    def test_one_group(self):
        config_file_path = "config.json"
        data_file_path = "test-data.csv"
        result_file_path = "."
        result = wetdry_features(config_file_path, data_file_path, result_file_path, 1)
        self.assertEqual(len(result.files), 1)

    def test_two_groups(self):
        config_file_path = "config.json"
        data_file_path = "test-data_2.csv"
        result_file_path = "."
        result = wetdry_features(config_file_path, data_file_path, result_file_path, 1)
        self.assertEqual(len(result.files), 2)

    def test_read_config(self):
        grower_id, iplant_id, plot_id, soil = read_config("config.json")
        self.assertEqual(grower_id, "grower_1")
        self.assertEqual(iplant_id, "iplant_1")
        self.assertEqual(plot_id, "plot_1")
        self.assertEqual(soil, "4025__VWC_Mineral_Soil")


if __name__ == '__main__':
    unittest.main()
