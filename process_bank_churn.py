from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
from typing import Tuple, List


def split_data(raw_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Розділяє дані на навчальний та валідаційний набори.
    """
    if target_col not in raw_df.columns:
        raise ValueError(f"Цільова змінна '{target_col}' відсутня у DataFrame")

    X = raw_df.drop(columns=[target_col])
    y = raw_df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, numeric_cols: List[str]) -> Tuple[
    np.ndarray, np.ndarray, StandardScaler]:
    """
    Масштабує числові ознаки.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_val_scaled = scaler.transform(X_val[numeric_cols])
    return X_train_scaled, X_val_scaled, scaler


def encode_features(X_train: pd.DataFrame, X_val: pd.DataFrame, categorical_cols: List[str]) -> Tuple[
    np.ndarray, np.ndarray, OneHotEncoder]:
    """
    Кодує категоріальні ознаки.
    """
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_val_encoded = encoder.transform(X_val[categorical_cols])
    return X_train_encoded, X_val_encoded, encoder


def preprocess_data(raw_df: pd.DataFrame, target_col: str) -> Tuple[
    np.ndarray, pd.Series, np.ndarray, pd.Series, List[str], StandardScaler, OneHotEncoder]:
    """
    Основна функція для попередньої обробки даних.
    """
    X_train, X_val, train_targets, val_targets = split_data(raw_df, target_col)

    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
    input_cols = numeric_cols + categorical_cols

    X_train_numeric, X_val_numeric, scaler = scale_features(X_train, X_val, numeric_cols)
    X_train_categorical, X_val_categorical, encoder = encode_features(X_train, X_val, categorical_cols)

    X_train_preprocessed = np.hstack([X_train_numeric, X_train_categorical])
    X_val_preprocessed = np.hstack([X_val_numeric, X_val_categorical])

    return X_train_preprocessed, train_targets, X_val_preprocessed, val_targets, input_cols, scaler, encoder


def preprocess_new_data(new_data: pd.DataFrame, input_cols: List[str], scaler: StandardScaler,
                        encoder: OneHotEncoder) -> np.ndarray:
    """
    Обробляє нові дані перед передбаченням за допомогою навчених скейлера та енкодера.
    """
    numeric_cols = new_data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = new_data.select_dtypes(exclude=['number']).columns.tolist()

    X_numeric = scaler.transform(new_data[numeric_cols])
    X_categorical = encoder.transform(new_data[categorical_cols])

    return np.hstack([X_numeric, X_categorical])
