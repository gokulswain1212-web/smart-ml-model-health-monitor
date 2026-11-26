from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class HotelCancellationPreprocessor:
    def __init__(
        self,
        encoder: OrdinalEncoder,
        feature_names: Sequence[str],
        categorical_columns: Sequence[str],
        children_median: float,
        categorical_fill_values: Mapping[str, str],
    ) -> None:
        self.encoder = encoder
        self.feature_names = list(feature_names)
        self.categorical_columns = list(categorical_columns)
        self.children_median = children_median
        self.categorical_fill_values = dict(categorical_fill_values)

    def fit(self, X: pd.DataFrame, y: object = None) -> HotelCancellationPreprocessor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Hotel cancellation inputs must be a pandas DataFrame.")

        missing_columns = [name for name in self.feature_names if name not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")

        frame = X.loc[:, self.feature_names].copy()
        for column, value in self.categorical_fill_values.items():
            frame[column] = frame[column].fillna(value)

        frame["children"] = frame["children"].fillna(self.children_median)
        frame["agent"] = frame["agent"].fillna(0)

        mappings = {
            "hotel": {"Resort Hotel": 1, "City Hotel": 0},
            "arrival_date_month": {
                "January": 1,
                "February": 2,
                "March": 3,
                "April": 4,
                "May": 5,
                "June": 6,
                "July": 7,
                "August": 8,
                "September": 9,
                "October": 10,
                "November": 11,
                "December": 12,
            },
            "meal": {"BB": 1, "FB": 2, "HB": 3, "SC": 4, "Undefined": 0},
        }
        for column, mapping in mappings.items():
            encoded = frame[column].map(mapping)
            invalid = encoded.isna() & frame[column].notna()
            if invalid.any():
                values = frame.loc[invalid, column].drop_duplicates().tolist()
                raise ValueError(f"Unsupported values in {column}: {values}")
            frame[column] = encoded

        categorical_values = frame.loc[:, self.categorical_columns].astype(object)
        encoded = self.encoder.transform(categorical_values)
        for index, column in enumerate(self.categorical_columns):
            frame[column] = encoded[:, index]
        return frame.loc[:, self.feature_names]
