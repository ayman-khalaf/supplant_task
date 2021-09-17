from typing import NamedTuple


def wetdry_features(
    config, features_file: str, bucket: str, force: int
) -> NamedTuple("WetDryFeatures", [("summary", str), ("files", list[str])]):
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
    import numpy as np
    import json
    import pandas as pd
    import s3fs
    from scipy.optimize import curve_fit
    from pysupplant.models.estimators import RegBooster
    from cloudpickle import dump
    from collections import namedtuple


    def _exponential(x, a, k):
        return a * np.exp(x * k)

    def _linear(x, m, b):
        return m * x + b

    config = json.loads(config)
    grower_id = config["grower_id"]
    plot_id = config["plot_id"]
    iplant_id = config["iplant_id"]
    soil = config["soil_moisture_sensor"]
    sensor_index = f"{grower_id}:{plot_id}:{iplant_id}"

    s3 = s3fs.S3FileSystem()

    def _compute(df):
        """compute features for one irtrigation interval

        This is an internal function to be applied to each dataframe
        of the grouped dataframe.
        """
        irr_event_features = {}
        irr_event_features["start"] = df.index.min()
        irr_event_features["datetime"] = irr_event_features["start"].value // 10 ** 9
        irr_event_features["end"] = df.index.max()
        irr_event_features["six"] = sensor_index

        datemin = pd.to_datetime(df.index.min())
        irr_event_features["hour"] = pd.to_datetime(datemin).hour
        irr_event_features["dayofyear"] = pd.to_datetime(datemin).dayofyear
        irr_event_features["month"] = pd.to_datetime(datemin).month
        irr_event_features["year"] = pd.to_datetime(datemin).year

        try:
            irr_event_features["on_time"] = df.dt_Irr_on.max()
        except:
            irr_event_features["on_time"] = pd.NA

        try:
            irr_event_features["off_time"] = df.dt_Irr_off.max()
        except:
            irr_event_features["off_time"] = pd.NA

        try:
            irr_event_features["start_sm_irr_on"] = df[soil].iloc[0]
        except:
            irr_event_features["start_sm_irr_on"] = pd.NA

        try:
            irr_event_features["end_sm_irr_on"] = df[soil].iloc[
                int(df.dt_Irr_on.max() * 2)
            ]
        except:
            irr_event_features["end_sm_irr_on"] = pd.NA

        try:
            irr_event_features["start_sm_irr_off"] = df[soil].iloc[
                int(df.dt_Irr_on.max() * 2) + 1
            ]
        except:
            irr_event_features["start_sm_irr_off"] = pd.NA

        try:
            irr_event_features["end_sm_irr_off"] = df[soil].iloc[df.shape[0] - 1]
        except:
            irr_event_features["end_sm_irr_off"] = pd.NA

        try:
            irr_event_features["dsm_irr_on"] = (
                df[soil].iloc[int(df.dt_Irr_on.max() * 2)] - df[soil].iloc[0]
            )
        except:
            irr_event_features["dsm_irr_on"] = pd.NA

        try:
            irr_event_features["dsm_irr_off"] = (
                df[soil].iloc[df.shape[0] - 1]
                - df[soil].iloc[int(df.dt_Irr_on.max() * 2) + 1]
            )
        except:
            irr_event_features["dsm_irr_off"] = pd.NA

        try:
            irr_event_features["dsm"] = (
                df[soil].iloc[df.shape[0] - 1] - df[soil].iloc[0]
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
            popt_linear, pcov_linear = curve_fit(_linear, x_array, y_array, p0=[0, 0])
            irr_event_features["linslope_off_est"] = popt_linear[0]
            perr_linear = np.sqrt(np.diag(pcov_linear))
            irr_event_features["linslope_off_err_est"] = perr_linear[0]
        except Exception as e:
            print(f"error:wetdry_features:linear_curve_fit:{sensor_index}:event:{ix_}:{str(e)}")
        try:
            popt_exponential, pcov_exponential = curve_fit(
                _exponential, x_array, y_array, p0=[0, 0]
            )
            irr_event_features["expdecay_off_est"] = popt_exponential[1]
            perr_exp = np.sqrt(np.diag(pcov_exponential))
            irr_event_features["expdecay_err_est"] = perr_exp[1]
        except Exception as e:
            irr_event_features["expdecay_off_est"] = pd.NA
            irr_event_features["expdecay_err_est"] = pd.NA

        try:
            irr_event_features["linslope_dsm_off"] = (
                df[soil].iloc[df.shape[0] - 1]
                - df[soil].iloc[int(df.dt_Irr_on.max() * 2) + 1]
            ) / df.dt_Irr_off.max()
        except:
            irr_event_features["linslope_dsm_off"] = pd.NA
        try:
            irr_event_features["linslope_dsm_on"] = (
                df[soil].iloc[int(df.dt_Irr_on.max() * 2)] - df[soil].iloc[0]
            ) / df.dt_Irr_on.max()
        except:
            irr_event_features["linslope_dsm_on"] = pd.NA

        ls = irr_event_features.get("linslope_off_est", 1.0)
        es = irr_event_features.get("expdecay_off_est", 1.0)
        if (ls > 0.0) or (es > 0.0):
            irr_event_features["neg_slope"] = False
        else:
            irr_event_features["neg_slope"] = True

        return irr_event_features

    groupings = ["Irr_start"]
    output = namedtuple("WetDryFeatures", ["summary", "files"])
    summary_file = (
        f"{bucket}/{grower_id}-{plot_id}/{iplant_id}/wetdry-groups-summary.csv"
    )
    if s3.isfile(features_file):
        data = pd.read_parquet(features_file)
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
                f"info:wetdry_features:index duplicates:{ishape[0]-oshape[0]}:refeshing cache:{features_file}"
            )
        gdf = list(data.groupby(groupings))
        files = []
        ds = []
        for i in range(len(gdf)):
            ixi, dfi = gdf[i]
            ix_ = ixi.value // 10 ** 9
            path = f"{bucket}/{grower_id}-{plot_id}/{iplant_id}/wetdry/{ix_}"
            data_fn = f"{path}/group.parquet"
            if not s3.isfile(data_fn) or force == 1:
                ixi, dfi = gdf[i]
                dfi.to_parquet(data_fn)
                files.append(data_fn)
                d = _compute(dfi)
                ds.append(d)
        summary = pd.DataFrame(ds)
        junkcols = [col for col in summary if "Unnamed" in col]
        if junkcols:
            print(f"info:wetdry_features:found junk columns:{len(junkcols)}")
            summary.drop(columns=junkcols, inplace=True)
        if s3.isfile(summary_file):
            summary_old = pd.read_csv(summary_file)
            junkcols = [col for col in summary_old if "Unnamed" in col]
            if junkcols:
                print(f"info:wetdry_features:found junk columns:{len(junkcols)}")
                summary_old.drop(columns=junkcols, inplace=True)
            summary = pd.concat([summary_old, summary], axis=0)
        summary.to_csv(summary_file)
        return output(summary_file, files)
    else:
        return output(summary_file, [])
