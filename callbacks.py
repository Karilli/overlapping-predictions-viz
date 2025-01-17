import pandas as pd


def read_data(id_: str) -> pd.DataFrame:
    """
    You should return a pandas dataframe with columns:
    - true: actual value.
    - pred: value predicted by the model.
    - base: value predicted by the baseline.
    - error: any metric of true/pred/base relationship.
    - Timestamp: uniquely identifies a single prediction vector (by its first timestamp)
    - ShiftAmountSeconds: unique identifies a single predicition within the vector (by its ofset in hours)
    - DeviceID: should be a constant 'id_' in the returned dataframe.
    """
    df = pd.read_pickle('data/model.pkl')
    df = df[df["DeviceID"] == id_]
    df['error'] = (df['pred'] - df['true']).abs()
    return df


def outlier_score(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    pivoted dataframe: 
      - x-axis: ShiftAmountHours
      - y-axis: PredictionFor (= Timestamp + ShiftAmountHours)
      - z-axis: error
      - obtained from user dataframe as
          df = read_data(...)
          df["PredictionFor"] = ...
          pivot_df = df.pivot(index='PredictionFor', columns='ShiftAmountHours', values='error')
    
    You can think of it as a 2D array, since z-axis always has only 1 value.
    The error may be None (consequence of PredictionFor = Timestamp + ShiftAmountHours).
    """
    # return pivot_df.max(axis=1, skipna=True)
    # return pivot_df.mean(axis=1, skipna=True)
    return pivot_df.max(axis=1, skipna=True) - pivot_df.min(axis=1, skipna=True)


def list_devices():
    """
    The dataset may contain predictions for multiple devices.
    But we can only visualize one at a time.
    """
    df = pd.read_pickle('data/model.pkl')
    return list(df["DeviceID"].unique())
