
from wetdry_curves_features import wetdry_features

if __name__ == '__main__':
    wetdry_features("""{
  "grower_id": "grower_1",
  "plot_id": "plot_1",
  "iplant_id": "iplant_1",
  "soil_moisture_sensor": "4025__VWC_Mineral_Soil"
    }""", "test-data.csv", ".", 1)


