import pandas as pd
import pyarrow.parquet as pq
import os

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    # Convert to datetime and calculate duration in minutes
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Filter for durations between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Categorical columns (Location IDs)
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def main():
    print("Loading data...")
    df_jan = read_dataframe('green_tripdata_2021-01.parquet')
    df_feb = read_dataframe('green_tripdata_2021-02.parquet')
    df_march = read_dataframe('green_tripdata_2021-03.parquet')

    print(f"Jan rows: {len(df_jan)}, Feb rows: {len(df_feb)}")
    
    # Combine Jan and Feb for training
    df_train = pd.concat([df_jan, df_feb])
    print(f"Combined Training rows: {len(df_train)}")
    
    # March for testing (Task 2)
    df_test = df_march
    print(f"Test (March) rows: {len(df_test)}")

    # Save prepared data
    df_train.to_parquet('train.parquet')
    df_test.to_parquet('test.parquet')
    
    print("Data preparation complete. 'train.parquet' and 'test.parquet' created.")

if __name__ == "__main__":
    main()
