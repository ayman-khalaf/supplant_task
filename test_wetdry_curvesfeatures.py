import unittest
from wetdry_curves_features import linear


class TestWetDryCurvesFeatures(unittest.TestCase):
    def test_linear_1(self):
        self.assertEqual(linear(1, 1, 1), 2)

    def test_linear_2(self):
        self.assertEqual(linear(0, 1, 1), 1)

    def test_linear_3(self):
        self.assertEqual(linear(0, 0, 0), 0)


if __name__ == '__main__':
    unittest.main()
