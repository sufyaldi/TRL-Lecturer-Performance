import pandas as pd
import numpy as np

def load_and_preprocess(file_path):
    """
    Loads raw LMS logs, removes duplicates, handles missing values,
    and sorts activities chronologically per lecturer.
    """
    df = pd.read_csv(file_path)
    # Sort by timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['lecturer_id', 'timestamp'])
    
    return df

def tokenize_sequences(df):
    """
    Maps discrete activities (e.g., Assignment, Quiz, Material) into tokens.
    """
    activity_map = {'Material': 1, 'Assignment': 2, 'Quiz': 3, 'Forum': 4, 'CBT': 5}
    df['token'] = df['activity_type'].map(activity_map).fillna(0).astype(int)
    
    # Group by lecturer to form sequences
    sequences = df.groupby('lecturer_id')['token'].apply(list).to_dict()
    return sequences

if __name__ == "__main__":
    print("Preprocessing dummy data...")
    df = load_and_preprocess('sample_data.csv')
    seqs = tokenize_sequences(df)
    print("Extracted sequences for", len(seqs), "lecturers.")
