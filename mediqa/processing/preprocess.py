import pandas as pd
from typing import Tuple

from mediqa.config.core import config, DataConfig, DATASET_DIR

def cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Removes offending examples from the dataset.
    Specifically, it removes samples with the following issues:
    - NaNs
    - Duplicate questions and/or answers

    Args:
        df (pd.DataFrame): Original dataset. `question` and `anser` columns are expected

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Remove NaNs
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates(subset=["question"])
    df = df.drop_duplicates(subset=["answer"])

    return df

def split(df: pd.DataFrame, frac=0.5, seed=0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into validation and test sets.

    Args:
        df (pd.DataFrame): Original dataset. `question` and `answer` columns are expected

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets.
    """
    # Split the dataset into val and test sets
    train_df = df.sample(frac=frac, random_state=seed)
    test_df = df.drop(train_df.index)

    return train_df, test_df

def preprocess(data_config: DataConfig) -> None:
    """Preprocess the input dataset and saves it as validation and test sets

    Args:
        input_path (str): 
        output_path (str): _description_
    """
    # Construct the relevant paths
    input_path = DATASET_DIR / data_config.eval_src_file
    val_path = DATASET_DIR / data_config.val_file
    test_path = DATASET_DIR / data_config.test_file

    # Load the dataset
    df = pd.read_csv(input_path)

    # Clean the dataset
    df = cleanup(df)

    # Split the dataset
    val_df, test_df = split(df, frac=data_config.split_frac)

    # Save the dataset
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

if __name__ == "__main__":
    # Load the configuration
    data_config = config.data_config

    # Preprocess the dataset
    preprocess(data_config)