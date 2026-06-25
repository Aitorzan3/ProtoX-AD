import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
import aeon

from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader

device = torch.device("mps")


class SelfDataset(Dataset):
    def __init__(self,feature):
        self.feature = feature
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        
        return item
    
def get_dataset(dataset, normal, reverse=False):

    train_data, train_labels, metadata = load_classification(dataset, split="train", load_equal_length=True, load_no_missing=True, return_metadata=True)
    test_data, test_labels = load_classification(dataset, split="test", load_equal_length=True, load_no_missing=True)


    real_labels = test_labels.copy()

    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    
        
    if normal not in real_labels:
        raise Exception("You need to choose an existing class value for parameter 'normal'. Possible values: ", np.unique(real_labels))

    normal = le.transform(np.array(normal).reshape(1,))[0]

    if not reverse:
        train_data = train_data[train_labels==normal]
        test_labels[test_labels!=normal]=-1
        test_labels[test_labels==normal]=1
        test_labels[test_labels==-1]=0
    else:
        train_data = train_data[train_labels!=normal]
        test_labels[test_labels==normal]=-1
        test_labels[test_labels!=-1]=1
        test_labels[test_labels==-1]=0

    print("Problem name: ", metadata['problemname'])
    print("Existing Classes: ", metadata['class_values'])
    print("Normal Class(es): ", np.unique(real_labels[np.where(test_labels==1)]))
    print("Anomalous Class(es): ", np.unique(real_labels[np.where(test_labels==0)]))

    return dataset, train_data.astype(np.float32) , test_data.astype(np.float32) , test_labels.astype(np.float32).astype(int), real_labels

def load_yorkshire(dma, day, percentile):

    # 1) Load data
    df = pd.read_csv('data/E1 2016_2017.csv', dayfirst=True)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    
    # 2) Ordered time index
    df = df.set_index('datetime').sort_index()
    
    # 3) Filter DMA
    df = df[df['DMA'] == dma]
    
    # 4) Filter valid
    df = df[df['Flow Validity Code'] == 'V']
    
    # 5) Convert time zone
    df = df.tz_localize('UTC').tz_convert('Europe/London')
    
    # 6) Sample every 15 minutes
    df = df.resample('15T').asfreq()

    # 7) Filter day
    df = df[df.index.weekday == day]
    
    # 8) Quick diagnostic: are there nights with 16 non-null readings?
    night = df.between_time('01:00', '05:00')  # range 01:00, 02:15, …, 04:45 → 12 puntos
    groups = night['Flow'].groupby(night.index.floor('D'))
    
    series = []
    fechas = []
    
    for fecha, flujo in groups:
        arr = flujo.values
        if len(arr) == 17 and not np.isnan(arr).any():
            series.append(arr)
            fechas.append(fecha)

    series = np.vstack(series)  # aquí ya no dará error

    mnfs  = series.min(axis=1)
    
    # 1) Compute percentiles
    pnorm = np.quantile(mnfs, percentile)
    panom = np.quantile(mnfs, percentile)
    
    normal_mask    = mnfs < pnorm
    anomaly_mask   = mnfs >= panom
    
    keep_mask = normal_mask | anomaly_mask
    
    series_clean    = series[keep_mask]           # shape = (n_keep, 12)
    labels_clean    = anomaly_mask[keep_mask].astype(int)  
    fechas_clean    = np.array(fechas)[keep_mask]
    
    series_normals    = series_clean[labels_clean == 0]
    series_anomalous  = series_clean[labels_clean == 1]
    
    # --- Split train and test
    test_ratio = 0.20   
    n_normals = len(series_normals)
    
    n_test_normals = max(1, int(np.floor(test_ratio * n_normals)))
    n_train_normals = n_normals - n_test_normals
    
    train_data = series_normals[:n_train_normals]
    test_normals = series_normals[n_train_normals:]
    test_anomalous = series_anomalous    
    
    test_data = np.vstack([test_normals, test_anomalous])
    
    labels_normals = np.ones(test_normals.shape[0], dtype=int)
    labels_anomalous = np.zeros(test_anomalous.shape[0], dtype=int)
    test_labels = np.concatenate([labels_normals, labels_anomalous])

    return train_data, test_data, test_labels


def load_temperatures(source
):

    # Load data
    df = pd.read_csv('data/temperatures_global.csv')
    normal_ratio=0.8   # 80% train, 20% test

    value_col = 'Mean'

    df['Year'] = pd.to_datetime(df['Year'], format='%Y-%m')
    
    df['year'] = df['Year'].dt.year
    df['month'] = df['Year'].dt.month
    
    # Filter source and year
    df = df[df.Source == source]
    df = df[df.year <= 2022]
    df = df.sort_values(['year', 'month'])

    years = np.sort(df['year'].unique())
    n_years = len(years)

    data = np.zeros((n_years, 12, 1), dtype=np.float32)
    labels = np.zeros(n_years, dtype=np.int64)

    # ---- Generate dataset ----
    for i, y in enumerate(years):
        year_df = df[df['year'] == y].sort_values('month')

        if len(year_df) != 12:
            raise ValueError(f"Año {y} no tiene 12 meses (tiene {len(year_df)})")

        values = year_df[value_col].values
        data[i, :, 0] = values

        annual_mean = values.mean()

        if -0.25 <= annual_mean <= 0.25:
            labels[i] = 0
        elif -0.75 <= annual_mean < -0.25:
            labels[i] = -1
        elif annual_mean < -0.75:
            labels[i] = -2
        elif 0.25 < annual_mean <= 0.75:
            labels[i] = 1
        else:
            labels[i] = 2

    # ---- Split train-test ----
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels != 0)[0]

    n_normals = len(normal_idx)
    n_train_normals = int(np.floor(normal_ratio * n_normals))

    train_idx = normal_idx[:n_train_normals]
    test_normal_idx = normal_idx[n_train_normals:]

    test_idx = np.concatenate([test_normal_idx, anomaly_idx])

    train_data = data[train_idx]
    train_years = years[train_idx]

    test_data = data[test_idx]
    test_years = years[test_idx]

    real_labels = labels[test_idx]
    test_labels = (real_labels == 0).astype(np.int64)

    return (
        train_data,
        test_data,
        test_labels,
        real_labels,
    )



