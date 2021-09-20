
from wetdry_curves_features import wetdry_features

if __name__ == '__main__':
    config_file_path = "config.json"
    data_file_path = "test-data.csv"
    result_file_path = "."

    wetdry_features(config_file_path, data_file_path, result_file_path, 1)


