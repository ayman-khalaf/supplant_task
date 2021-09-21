from typing import NamedTuple, List

import os
import numpy as np
import json
import pandas as pd
import s3fs
from scipy.optimize import curve_fit
from collections import namedtuple
import matplotlib.pyplot as plt


def exponential(x, a, k):
    """
    exponential function

    :param x
    :param a scale value of the function
    :param k degree of the exponential function

    :return a * np.exp(x * k)
    """
    return a * np.exp(x * k)


def linear(x, m, b):
    """
    linear function which is a polynomial function of degree zero or one. which is simply a straight line

    :param x
    :param m slope of the line
    :param b offset of the line

    :return m * x + b
    """
    return m * x + b


def compute(df, sensor_index, soil, event):
    """
    compute features for one irrigation interval which is applied to each dataframe
    of the grouped dataframe.

    :param df pandas dataframe which will be processed
    :param sensor_index string combined as the following from grower_id, plot_id, and iplant_id"
    :param soil mineral sensor name
    :param event irrigation start time


    :return irr_event_features which is dictionary containing the results of the calculations on the the given dataframe
    """
    irr_event_features = {}
    start_time = df.index.min()
    irr_event_features["start"] = start_time
    # irr_event_features["datetime"] = irr_event_features["start"].value // 10 ** 9
    irr_event_features["end"] = df.index.max()
    irr_event_features["sensor_index"] = sensor_index

    date_min = pd.to_datetime(start_time)
    irr_event_features["hour"] = date_min.hour
    irr_event_features["dayofyear"] = date_min.dayofyear
    irr_event_features["month"] = date_min.month
    irr_event_features["year"] = date_min.year
    on_time = 0
    if "dt_Irr_on" in df.columns:
        on_time = df.dt_Irr_on.max()
    irr_event_features["on_time"] = on_time
    off_time = 0
    if "dt_Irr_off" in df.columns:
        off_time = df.dt_Irr_off.max()
    irr_event_features["off_time"] = off_time
    number_of_rows = df.shape[0]
    start_sm_irr_on = pd.NA
    soil_in_df = soil in df.columns
    if number_of_rows > 0 and soil_in_df:
        start_sm_irr_on = df[soil].iloc[0]
    irr_event_features["start_sm_irr_on"] = start_sm_irr_on

    # since records in dataframe is calculated each half hour so the index of end time of irrigation
    # will be on_time * 2, however this will fail if we change the time of recording data
    # TODO(Ayman): make fix bug that happens when time of recording data is not each half hour
    end_sm_irr_on = int(on_time * 2)
    if number_of_rows > on_time * 2 and soil_in_df:
        irr_event_features["end_sm_irr_on"] = df[soil].iloc[end_sm_irr_on]
    else:
        irr_event_features["end_sm_irr_on"] = pd.NA

    if number_of_rows > on_time * 2 + 1 and soil_in_df:
        # start of irrigation off will equal to end of irrigation on plus one
        irr_event_features["start_sm_irr_off"] = df[soil].iloc[end_sm_irr_on + 1]
    else:
        irr_event_features["start_sm_irr_off"] = pd.NA

    if number_of_rows > 0 and soil_in_df:
        # end of irrigation off will be the last row
        irr_event_features["end_sm_irr_off"] = df[soil].iloc[number_of_rows - 1]
    else:
        irr_event_features["end_sm_irr_off"] = pd.NA

    try:
        irr_event_features["dsm_irr_on"] = (df[soil].iloc[end_sm_irr_on] - start_sm_irr_on)
    except:
        irr_event_features["dsm_irr_on"] = pd.NA

    try:
        irr_event_features["dsm_irr_off"] = (df[soil].iloc[number_of_rows - 1] - df[soil].iloc[end_sm_irr_on + 1])
    except:
        irr_event_features["dsm_irr_off"] = pd.NA

    try:
        irr_event_features["dsm"] = (
                df[soil].iloc[number_of_rows - 1] - start_sm_irr_on
        )
    except:
        irr_event_features["dsm"] = pd.NA

    try:
        irr_event_features["water_meter_avg_flow"] = df[
            df["Irrigation"] > 0
            ].Irrigation.mean()
    except:
        irr_event_features["water_meter_avg_flow"] = pd.NA

    # data when irrigation is off
    xoff = df.query("dt_Irr_off>=0")
    # data when irrigation is on
    xon = df.query("dt_Irr_off==0")
    if soil_in_df:
        irr_event_features["mean_off"] = xoff[soil].mean()
        irr_event_features["mean_on"] = xon[soil].mean()
        irr_event_features["std_off"] = xoff[soil].std()
        irr_event_features["std_on"] = xon[soil].std()
    x_array = df.query("dt_Irr_off>0")["dt_Irr_off"].values
    y_array = df.query("dt_Irr_off>0")[soil].values
    try:
        popt_linear, pcov_linear = curve_fit(linear, x_array, y_array, p0=[0, 0])
        irr_event_features["linslope_off_est"] = popt_linear[0]
        perr_linear = np.sqrt(np.diag(pcov_linear))
        irr_event_features["linslope_off_err_est"] = perr_linear[0]
    except Exception as e:
        print(f"error:wetdry_features:linear_curve_fit:{sensor_index}:event:{event}:{str(e)}")
    try:
        popt_exponential, pcov_exponential = curve_fit(
            exponential, x_array, y_array, p0=[0, 0]
        )
        irr_event_features["expdecay_off_est"] = popt_exponential[1]
        perr_exp = np.sqrt(np.diag(pcov_exponential))
        irr_event_features["expdecay_err_est"] = perr_exp[1]
    except Exception as e:
        irr_event_features["expdecay_off_est"] = pd.NA
        irr_event_features["expdecay_err_est"] = pd.NA

    try:
        irr_event_features["linslope_dsm_off"] = (
                                                         df[soil].iloc[number_of_rows - 1]
                                                         - df[soil].iloc[end_sm_irr_on + 1]
                                                 ) / off_time
    except:
        irr_event_features["linslope_dsm_off"] = pd.NA
    try:
        irr_event_features["linslope_dsm_on"] = (
                                                        df[soil].iloc[end_sm_irr_on] - df[soil].iloc[
                                                    0]
                                                ) / on_time
    except:
        irr_event_features["linslope_dsm_on"] = pd.NA

    ls = irr_event_features.get("linslope_off_est", 1.0)
    es = irr_event_features.get("expdecay_off_est", 1.0)
    irr_event_features["neg_slope"] = not ((ls > 0.0) or (es > 0.0))

    return irr_event_features


def save_wetdry_figure(df, soil, solar_radiation, accumulated_irrigation, hour):
    """
    function that plots the relation between soil, solar_radiation accumulated_irrigation and time.

    :param df: pandas dataframe which will be processed
    :param soil: soil mineral sensor name
    :param solar_radiation: solar radiation sensor name
    :param accumulated_irrigation: name of accumulated irrigation column in df
    :param hour: name of hour column in df

    :return: path of image plotted
    """
    font_size = 18
    fig, axs = plt.subplots(4)
    number_of_rows = df[solar_radiation].shape[0]
    time = np.linspace(0, number_of_rows, number_of_rows)
    axs[0].plot(time, df[solar_radiation], 'tab:red')
    axs[0].set_title('Solar Radiation', fontsize=font_size)
    axs[0].set_ylabel(ylabel='Watt', fontsize=font_size)
    axs[1].plot(time, df[soil], 'tab:brown')
    axs[1].set_title('VMC Mineral Soil', fontsize=font_size)
    axs[1].set_ylabel(ylabel='Mineral', fontsize=font_size)
    axs[2].plot(time, df[accumulated_irrigation], 'tab:blue')
    axs[2].set_title('ACC Irr ON', fontsize=font_size)
    axs[2].set_ylabel(ylabel='Meter Cube', fontsize=font_size)
    axs[3].plot(time, df["Hour"], 'tab:green')
    axs[3].set_title(hour, fontsize=font_size)
    axs[3].set_ylabel(ylabel='Hour', fontsize=font_size)
    axs[3].set_xlabel(xlabel='time', fontsize=font_size)
    figure = plt.gcf()
    figure.set_size_inches(32, 18)
    print("saving wet_dry_features.png")
    plt.savefig("wet_dry_features.png", bbox_inches='tight')

    return "saving wet_dry_features.png"


def read_config(config):
    """

    :param config: path of json config file
    :return: values read from json file as the following grower_id, iplant_id, plot_id, soil
    """
    config_file = open(config)
    config = json.load(config_file)
    config_file.close()

    grower_id = config["grower_id"]
    plot_id = config["plot_id"]
    iplant_id = config["iplant_id"]
    soil = config["soil_moisture_sensor"]
    return grower_id, iplant_id, plot_id, soil


def wetdry_features(config, features_file: str, bucket: str,
                    force: bool):  # -> NamedTuple["WetDryFeatures", ("summary", str), ("files", list[str])]:
    """
    Compute features for wetting-drying intervals
    A wetting-drying interval is the time period between each irrigation event,
    starting with and including the irrigation event itself and ending just as
    the next irrigation event begins.

    :param config  config from preceding steps, passed as a json string
    :param features_file: irrigation features_file, artifact cache filename. This is the file from which the features are derived.
    :param bucket root artifact folder
    :param force  force recompute even if output file already exists

    :return summary of each wetting-drying interval
    """
    if not os.path.isfile(config):
        raise Exception(f"file {config} is not found.")

    if not os.path.isdir(bucket):
        raise Exception(f"file {bucket} is not found.")

    grower_id, iplant_id, plot_id, soil = read_config(config)
    sensor_index = f"{grower_id}:{plot_id}:{iplant_id}"

    # s3 = s3fs.S3FileSystem()
    # list of columns used in grouping
    groupings = ["Irr_start"]
    output = namedtuple("WetDryFeatures", ["summary", "files"])
    wetdry_groups_summary_file = "wetdry-groups-summary.csv"
    result_dir = os.path.join(bucket, f"{grower_id}-{plot_id}", iplant_id)
    summary_file = os.path.join(result_dir, wetdry_groups_summary_file)

    if os.path.isfile(features_file):
        data = prepare_data(features_file)
        gdf = list(data.groupby(groupings))
        files = []
        ds = []
        i = 0
        for (ixi, dfi) in gdf:
            data_fn = os.path.join(result_dir, "wetdry", f"group_{i}.csv")
            i += 1

            # compute only if file didn't exist before, or force is equal to 1
            if not os.path.isfile(data_fn) or force:
                print(data_fn)
                dfi.to_csv(data_fn)
                files.append(data_fn)
                d = compute(dfi, sensor_index, soil, ixi)
                save_wetdry_figure(dfi, soil, "2611__Solar_Radiation", "acc_Irr_on", "Hour")
                ds.append(d)
        save_summary(ds, summary_file)
        return output(summary_file, files)
    else:
        return output(summary_file, [])


def save_summary(ds, summary_file):
    """
    saves the summary output of wetdry_features function as a csv file, if the file already exists it reads the file
    and combines it with the new summary ds
    :param ds: pandas dataframe contains summary output of wetdry_features function
    :param summary_file: path to summary file

    """
    summary = pd.DataFrame(ds)
    # remove Unnamed columns in the dataframe
    junkcols = [col for col in summary if "Unnamed" in col]
    if junkcols:
        print(f"info:wetdry_features:found junk columns:{len(junkcols)}")
        summary.drop(columns=junkcols, inplace=True)
    # if file summary already exists then read it and combine it with the new summary output
    if os.path.isfile(summary_file):
        summary_old = pd.read_csv(summary_file)
        junkcols = [col for col in summary_old if "Unnamed" in col]
        if junkcols:
            print(f"info:wetdry_features:found junk columns:{len(junkcols)}")
            summary_old.drop(columns=junkcols, inplace=True)
        summary = pd.concat([summary_old, summary], axis=0)
    summary.to_csv(summary_file)


def prepare_data(features_file):
    """
    read features_file as pandas dataframe and remove duplicated rows and if there is duplicated rows remove them and
    save new dataframe to the same file
    :param features_file: path to features_file
    :return: pandas dataframe of features
    """
    data = pd.read_csv(features_file)
    ishape = data.shape
    data.reset_index(drop=False, inplace=True)
    # remove duplicate rows depending on datetime column, keep the last value if duplicates exists
    data.drop_duplicates("datetime", keep="last", inplace=True)
    # change index of dataframe to be datetime column
    data.set_index("datetime", inplace=True)
    oshape = data.shape
    if ishape == oshape:
        print(f"info:wetdry_features:index duplicates:0:{features_file}")
    else:
        data.to_csv(features_file, index=True)
        print(
            f"info:wetdry_features:index duplicates:{ishape[0] - oshape[0]}:refeshing cache:{features_file}"
        )
    return data
