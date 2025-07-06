import pandas as pd
import pandera.pandas as pa
from encodings.punycode import selective_find
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .constants import SEED
from .predict_item import PredictItem


class Data:
    schema = pa.DataFrameSchema(
        columns={
            "age": pa.Column(int),
            "dataset": pa.Column(str),
            "sex": pa.Column(str, [pa.Check.isin(["Male", "Female"])]),
            "cp": pa.Column(str,
                            [pa.Check.isin(["typical angina", "atypical angina", "non-anginal", "asymptomatic"])]),
            "trestbps": pa.Column(float, nullable=True),
            "chol": pa.Column(float, nullable=True),
            "fbs": pa.Column(object, [pa.Check.isin([True, False])], nullable=True),
            "restecg": pa.Column(str, [pa.Check.isin(["normal", "st-t abnormality", "lv hypertrophy"])],
                                 nullable=True),
            "thalch": pa.Column(float, nullable=True),
            "exang": pa.Column(object, [pa.Check.isin([True, False])], nullable=True),
            "oldpeak": pa.Column(float, nullable=True),
            "slope": pa.Column(str, [pa.Check.isin(["upsloping", "flat", "downsloping"])], nullable=True),
            "num": pa.Column(int, [pa.Check.ge(0)]),
            # These columns are dropped in preprocessing
            "thal": pa.Column(str, [pa.Check.isin(["fixed defect", "reversable defect", "normal"])], nullable=True),
            "ca": pa.Column(float, nullable=True),
        }
    )
    
    predict_schema = pa.DataFrameSchema(
        columns={
            "age": pa.Column(int),
            "dataset": pa.Column(str),
            "sex": pa.Column(str, [pa.Check.isin(["Male", "Female"])]),
            "cp": pa.Column(str,
                            [pa.Check.isin(["typical angina", "atypical angina", "non-anginal", "asymptomatic"])]),
            "trestbps": pa.Column(float),
            "chol": pa.Column(float),
            "fbs": pa.Column(bool),
            "restecg": pa.Column(str, [pa.Check.isin(["normal", "st-t abnormality", "lv hypertrophy"])]),
            "thalch": pa.Column(float),
            "exang": pa.Column(bool),
            "oldpeak": pa.Column(float),
            "slope": pa.Column(str, [pa.Check.isin(["upsloping", "flat", "downsloping"])]),
        }
    )

    def __init__(self, path: str):
        self.df = self.load_data(path)
        self.is_first_run = True
        self.oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.preprocessed_df = self.preprocess_data(self.df)

    def load_data(self, path: str) -> pd.DataFrame:
        return self.schema.validate(pd.read_csv(path))

    def preprocess_data(self, df: pd.DataFrame):
        # Sort feature names
        df = df.reindex(sorted(df.columns), axis=1)
        # Drop the id column
        preprocessed_df = df.copy()
        if 'id' in preprocessed_df.columns:
            preprocessed_df = preprocessed_df.drop(columns=['id'])

        # Drop columns with a high-missing value rate
        if 'ca' in preprocessed_df.columns and 'thal' in preprocessed_df.columns:
            preprocessed_df = preprocessed_df.drop(columns=['ca', 'thal'])

        # Impute numeric columns with median values
        numeric_cols = preprocessed_df.select_dtypes(include='number').columns
        preprocessed_df[numeric_cols] = preprocessed_df[numeric_cols].fillna(preprocessed_df[numeric_cols].median())

        # Convert age into categorical
        if 'age' in preprocessed_df.columns:
            preprocessed_df['age_group'] = pd.cut(preprocessed_df['age'], bins=[29, 40, 55, 70, 100],
                                                  labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
            preprocessed_df = preprocessed_df.drop(columns=['age'])

        # Impute categorical columns with the modes
        cat_cols = preprocessed_df.select_dtypes(include=['object', 'string', 'category', 'boolean']).columns
        preprocessed_df[cat_cols] = preprocessed_df[cat_cols].fillna(preprocessed_df[cat_cols].mode().iloc[0])

        # One-hot encoding the categorical values
        if self.is_first_run:
            oh_encoded_df = pd.DataFrame(self.oh_encoder.fit_transform(preprocessed_df[cat_cols]),
                                         columns=self.oh_encoder.get_feature_names_out(cat_cols),
                                         index=preprocessed_df.index)
            self.is_first_run = False
        else:
            oh_encoded_df = pd.DataFrame(self.oh_encoder.transform(preprocessed_df[cat_cols]),
                                         columns=self.oh_encoder.get_feature_names_out(cat_cols),
                                         index=preprocessed_df.index)
        preprocessed_df = pd.concat([preprocessed_df.drop(columns=cat_cols), oh_encoded_df], axis=1)

        # Convert the stage to presence of heart disease (df['num'] > 0)
        if 'num' in preprocessed_df.columns:
            preprocessed_df['has_heart_disease'] = preprocessed_df['num'].apply(lambda x: 1 if x > 0 else 0)
            preprocessed_df = preprocessed_df.drop(columns=['num'])

        # Remove duplicated rows
        preprocessed_df = preprocessed_df.drop_duplicates()
        preprocessed_df = preprocessed_df.reset_index(drop=True)

        return preprocessed_df

    @staticmethod
    def separate_data(df: pd.DataFrame):
        return df.drop(columns=['has_heart_disease']), df['has_heart_disease']

    @staticmethod
    def split_data(X, y):
        # The ratio is 7:2:1 for train:test:val
        X_train, X_left, y_train, y_left = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        X_test, X_validate, y_test, y_validate = train_test_split(X_left, y_left, test_size=0.3, random_state=SEED,
                                                                  stratify=y_left)

        return X_train, y_train, X_test, y_test, X_validate, y_validate

    @staticmethod
    def oversample_data(X, y):
        smote = SMOTE(random_state=SEED, sampling_strategy='minority')
        X_oversampled, y_oversampled = smote.fit_resample(X, y)
        return X_oversampled, y_oversampled

    def preprocess_predict_input(self, predict_item: PredictItem):
        df = pd.DataFrame(predict_item.model_dump(), index=[0])
        df = self.predict_schema.validate(df)
        df = self.preprocess_data(df)
        return df
