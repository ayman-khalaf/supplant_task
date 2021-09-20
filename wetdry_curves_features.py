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
    return a * np.exp(x * k)


def linear(x, m, b):
    return m * x + b


def compute(df, sensor_index, soil, ix_):
    """compute features for one irtrigation interval

        This is an internal function to be applied to each dataframe
        of the grouped dataframe.
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

    on_time = df.dt_Irr_on.max()
    try:
        irr_event_features["on_time"] = on_time
    except:
        irr_event_features["on_time"] = pd.NA

    off_time = df.dt_Irr_off.max()
    try:
        irr_event_features["off_time"] = off_time
    except:
        irr_event_features["off_time"] = pd.NA

    start_sm_irr_on = df[soil].iloc[0]
    try:
        irr_event_features["start_sm_irr_on"] = start_sm_irr_on
    except:
        irr_event_features["start_sm_irr_on"] = pd.NA

    try:
        irr_event_features["end_sm_irr_on"] = df[soil].iloc[
            int(on_time * 2)
        ]
    except:
        irr_event_features["end_sm_irr_on"] = pd.NA

    try:
        irr_event_features["start_sm_irr_off"] = df[soil].iloc[
            int(on_time * 2) + 1
            ]
    except:
        irr_event_features["start_sm_irr_off"] = pd.NA

    number_of_rows = df.shape[0]
    try:
        irr_event_features["end_sm_irr_off"] = df[soil].iloc[number_of_rows - 1]
    except:
        irr_event_features["end_sm_irr_off"] = pd.NA

    try:
        irr_event_features["dsm_irr_on"] = (
                df[soil].iloc[int(on_time * 2)] - start_sm_irr_on
        )
    except:
        irr_event_features["dsm_irr_on"] = pd.NA

    try:
        irr_event_features["dsm_irr_off"] = (
                df[soil].iloc[number_of_rows - 1]
                - df[soil].iloc[int(on_time * 2) + 1]
        )
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

    xoff = df.query("dt_Irr_off>=0")
    xon = df.query("dt_Irr_off==0")
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
        print(f"error:wetdry_features:linear_curve_fit:{sensor_index}:event:{ix_}:{str(e)}")
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
                                                         - df[soil].iloc[int(on_time * 2) + 1]
                                                 ) / off_time
    except:
        irr_event_features["linslope_dsm_off"] = pd.NA
    try:
        irr_event_features["linslope_dsm_on"] = (
                                                        df[soil].iloc[int(on_time * 2)] - df[soil].iloc[
                                                    0]
                                                ) / on_time
    except:
        irr_event_features["linslope_dsm_on"] = pd.NA

    ls = irr_event_features.get("linslope_off_est", 1.0)
    es = irr_event_features.get("expdecay_off_est", 1.0)
    if (ls > 0.0) or (es > 0.0):
        irr_event_features["neg_slope"] = False
    else:
        irr_event_features["neg_slope"] = True

    return irr_event_features


def save_wetdry_figure(df, soil, solar_radiation, accumulated_irrigation, hour):
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


def read_config(config):
    grower_id = config["grower_id"]
    plot_id = config["plot_id"]
    iplant_id = config["iplant_id"]
    soil = config["soil_moisture_sensor"]
    return grower_id, iplant_id, plot_id, soil


def wetdry_features(config, features_file: str, bucket: str, force: int):# -> NamedTuple["WetDryFeatures", ("summary", str), ("files", list[str])]:
    """Compute features for wetting-drying intervals

    A wetting-drying interval is the time period between each irrigation event,
    starting with and including the irrigation event itself and ending just as
    the next irrigation event begins.

    Args:
        config:  config from preceding steps, passed as a json string
        features_file: irrigation features_file, artifact cache filename.  This
                       is the file from which the features are derived.
        bucket: root artifact folder
        force:  force recompute even if output file already exists
    """
    if not os.path.isfile(config):
        raise Exception(f"file {config} is not found.")

    if not os.path.isdir(bucket):
        raise Exception(f"file {bucket} is not found.")

    config = json.load(open(config))
    grower_id, iplant_id, plot_id, soil = read_config(config)
    sensor_index = f"{grower_id}:{plot_id}:{iplant_id}"

    # s3 = s3fs.S3FileSystem()

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
        for (ixi, dfi) in gdf:
            ix_ = ixi  # .value // 10 ** 9
            data_fn = os.path.join(result_dir, "wetdry", "group.csv")
            if not os.path.isfile(data_fn) or force == 1:
                print(data_fn)
                dfi.to_csv(data_fn)
                files.append(data_fn)
                d = compute(dfi, sensor_index, soil, ix_)
                save_wetdry_figure(dfi, soil, "2611__Solar_Radiation", "acc_Irr_on", "Hour")
                ds.append(d)
        save_summary(ds, summary_file)
        return output(summary_file, files)
    else:
        return output(summary_file, [])


def save_summary(ds, summary_file):
    summary = pd.DataFrame(ds)
    junkcols = [col for col in summary if "Unnamed" in col]
    if junkcols:
        print(f"info:wetdry_features:found junk columns:{len(junkcols)}")
        summary.drop(columns=junkcols, inplace=True)
    if os.path.isfile(summary_file):
        summary_old = pd.read_csv(summary_file)
        junkcols = [col for col in summary_old if "Unnamed" in col]
        if junkcols:
            print(f"info:wetdry_features:found junk columns:{len(junkcols)}")
            summary_old.drop(columns=junkcols, inplace=True)
        summary = pd.concat([summary_old, summary], axis=0)
    summary.to_csv(summary_file)


def prepare_data(features_file):
    data = pd.read_csv(features_file)
    ishape = data.shape
    data.reset_index(drop=False, inplace=True)
    data.drop_duplicates("datetime", keep="last", inplace=True)
    data.set_index("datetime", inplace=True)
    oshape = data.shape
    if ishape == oshape:
        print(f"info:wetdry_features:index duplicates:0:{features_file}")
    else:
        data.to_parquet(features_file, index=True)
        print(
            f"info:wetdry_features:index duplicates:{ishape[0] - oshape[0]}:refeshing cache:{features_file}"
        )
    return data


