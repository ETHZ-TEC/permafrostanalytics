import scipy
import stuett
from stuett.global_config import get_setting, setting_exists, set_setting
from sklearn import svm
import numpy as np
import pandas as pd
from skimage import io as imio
import io
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import anomaly_visualization
from dateutil import rrule
from datetime import date, timedelta
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.fftpack import fft
from sklearn.impute import SimpleImputer
import time
import os

account_name = (
    get_setting("azure")["account_name"]
    if setting_exists("azure")
    else "storageaccountperma8980"
)
account_key = (
    get_setting("azure")["account_key"] if setting_exists("azure") else None
)
store = stuett.ABSStore(
    container="hackathon-on-permafrost",
    prefix="seismic_data/4D/",
    account_name=account_name,
    account_key=account_key,
)
rock_temperature_file = "MH30_temperature_rock_2017.csv"
prec_file = "MH25_vaisalawxt520prec_2017.csv"
derived_store = stuett.ABSStore(
    container="hackathon-on-permafrost",
    prefix="timeseries_derived_data_products",
    account_name=account_name,
    account_key=account_key,
)
image_store = stuett.ABSStore(
    container="hackathon-on-permafrost",
    prefix="timelapse_images_fast",
    account_name=account_name,
    account_key=account_key,
)


# calculates entropy on the measurements
def calculate_entropy(v):
    counter_values = Counter(v).most_common()
    probabilities = [elem[1] / len(v) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


# extracts statistical features
def min_max_estractor(row):
    return [np.min(row), np.max(row), np.var(row), np.mean(row ** 2)]  # calculate_entropy(row),
    # np.percentile(row, 1), np.percentile(row, 5), np.percentile(row, 25),
    # np.percentile(row, 95), np.percentile(row, 95), np.percentile(row, 99)]


# computes fourier transform of the signal and extracts features
def fourier_extractor(x):
    sampling_freq = 250
    N = len(x)
    f_values = np.linspace(0.0, sampling_freq / 2, N // 2)
    fft_values_ = fft(x)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])

    coeff_0 = fft_values[0]  # coefficient at 0Hz
    peak_70 = 0  # coefficient around 70 Hz
    coeff = np.zeros(20)  # max coefficient from each 2 Hz interval (0-40)
    integral40 = 0  # integral from 0 to 40 Hz
    integral125 = np.mean(fft_values)  # integral over the whole transform
    for i in range(0, len(f_values)):
        if f_values[i] > 69 and f_values[i] < 72 and fft_values[i] > peak_70:
            peak_70 = fft_values[i]
        if f_values[i] < 40:
            integral40 += fft_values[i]
            if fft_values[i] > coeff[int(f_values[i] / 2)]:
                coeff[int(f_values[i] / 2)] = fft_values[i]
    return list(coeff) + [coeff_0, peak_70, integral40, integral125]


# extracts features from an hour worth of seismic data from three sensors
def transform_hour(data):
    data = np.array(data)
    # print(data)
    # print(data.shape)
    features = []
    for first_dimension in data:
        for row in first_dimension:
            # print(row.shape)
            for extractor in [min_max_estractor]:  # , fourier_extractor]:
                for element in extractor(row):
                    features.append(element)
    features = np.array(features)
    return features


def transform_minute(data):
    pass


def get_seismic_data(date):
    return np.array(stuett.data.SeismicSource(
        store=store,
        station="MH36",
        channel=["EHE", "EHN", "EHZ"],
        start_time=date,
        end_time=date + timedelta(hours=1),
    )())


# Load the data source
def load_seismic_source(start, end):
    output = []
    dates = []
    for date in pd.date_range(start, end, freq='1H'):
        try:
            seismic_node = stuett.data.SeismicSource(
                store=store,
                station="MH36",
                channel=["EHE", "EHN", "EHZ"],
                start_time=date,
                end_time=date + timedelta(hours=1),
            )
            print(date)
            dates.append(date)
            output.append(transform_hour(seismic_node()))
        except:
            pass
    return dates, output


def load_image_source():
    image_node = stuett.data.MHDSLRFilenames(
        store=store,
        force_write_to_remote=True,
        as_pandas=False,
    )
    return image_node, 3


import shutil

folder = 'data'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

rock_temperature_node = stuett.data.CsvSource(rock_temperature_file, store=derived_store)
rock_temperature = rock_temperature_node().to_dataframe()

rock_temperature = rock_temperature.reset_index('name').drop(["unit"], axis=1)

rock_temperature = rock_temperature.pivot(columns='name', values='CSV').drop(["position"], axis=1)
rock_temperature.index.rename("date")

prec_node = stuett.data.CsvSource(prec_file, store=derived_store)
prec = prec_node().to_dataframe()
prec = prec.reset_index('name').drop(["unit"], axis=1).pivot(columns='name', values='CSV').drop(["position"], axis=1)

dates, seismic_data = load_seismic_source(start=date(2017, 2, 2), end=date(2017, 2, 3))
seismic_data = np.array(seismic_data)
seismic_df = pd.DataFrame(seismic_data)
seismic_df["date"] = dates
seismic_df = seismic_df.set_index("date")

# dataset = seismic_df.join(rock_temperature).join(prec)
dataset = seismic_df.join(prec)
dataset = pd.DataFrame(SimpleImputer(strategy="constant").fit_transform(dataset), index=dataset.index,
                       columns=dataset.columns)

print(dataset.describe())
dataset.to_csv("seismic_prec_temp.csv")

n_samples = 300
outliers_fraction = 0.05
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
"""
anomaly_algorithms = [
    #("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    #("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
    #                                  gamma=0.1)),
    #("Isolation Forest", IsolationForest(behaviour='new',
    #                                     contamination=outliers_fraction,
    #                                     random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]
"""
anomaly_algorithms = [("Isolation Forest", IsolationForest(behaviour='new',
                                                           contamination=outliers_fraction,
                                                           random_state=42))]

for name, algorithm in anomaly_algorithms:
    y_pred = algorithm.fit_predict(dataset.values)

    os.makedirs("data/normal/", exist_ok=True)
    normals = dataset[y_pred > 0]
    prec.loc[normals.index].median(axis=0).to_csv("data/normal/precipitation_data.csv")
    normal_seismic = []
    for normal_data in normals.index:
        normal_seismic.append(get_seismic_data(normal_data)[0])
    normal_seismic = np.median(np.array(normal_seismic), axis=0)
    normal_seismic = pd.DataFrame(np.transpose(normal_seismic), columns=["EHE", "EHN", "EHZ"])
    normal_seismic.to_csv("data/normal/seismic_data.csv")

    scores = algorithm.decision_function(dataset[y_pred < 0].values)
    scores_min = scores.min()
    scores_max = scores.max()
    for date in dataset[y_pred < 0].index:

        os.makedirs("data/{}/images/".format(date), exist_ok=True)
        score = (algorithm.decision_function(
            dataset.loc[date].values.reshape((1, len(dataset.columns)))) - scores_min) * 5 / (scores_max - scores_min)
        with open("data/{}/score.txt".format(date), "w") as f:
            f.write(str(score[0]))

        print("event at {}".format(date))
        # print(dataset.loc[date])
        prec.loc[date].to_csv("data/{}/precipitation_data.csv".format(date))

        sism = pd.DataFrame(np.transpose(get_seismic_data(date)[0]), columns=["EHE", "EHN", "EHZ"])
        sism["date"] = np.array([d for d in pd.date_range(date, date + timedelta(hours=1), freq='1H')])
        sism.to_csv("data/{}/seismic_data.csv".format(date))

        # print(dataset.describe())
        start = str(date - timedelta(minutes=10))
        end = str(date + timedelta(minutes=60))

        images_df = anomaly_visualization.get_images_from_timestamps(image_store, start, end)()
        for key in images_df["filename"]:
            img = imio.imread(io.BytesIO(image_store[key]))
            imshow(img)
            print("data/{}/images/{}.png".format(date, key.split("/")[1]))
            imio.imsave("data/{}/images/{}.png".format(date, key.split("/")[1]), img)
            plt.show()
