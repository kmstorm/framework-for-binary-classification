import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

from utils.preprocess_utils import get_categorical_imputer, get_normalizer, get_real_imputer


class BioDataPreprocess:
    """
	A class that defines the preprocessing steps to be done on the input data and creates the training pipeline
	
	Attributes
	----------
	data : DataFrame
		The input data used for the training

	feature_group : list
        List of feature groups to used for the training

	target_column : str
		The name of the column in the dataset that is to be predicted by the model

	base_model 
		The machine learning model used for the last step of the pipeline

	drop_threshold: float
		The ratio of missing data above which the row or column will be dropped

	normalizer: str
		Name of the normalizer method

	categorical_impute: str
		Name of the impute method for categorical variables

	real_impute: str
		Name of the impute method for real variables

	random_state : int
		Random seed to make the iterative impute deterministic
	"""
    def __init__(self, data: pd.DataFrame,
                 augmented_data: pd.DataFrame,
                 base_model,
                 target_column: str,
                 feature_group: list = ['original', 'log', 'wavelet'],
                 smote: bool = False,
                 drop_threshold: int = 0.3,
                 normalizer: str = None,
                 categorical_impute: str = 'most_frequent',
                 real_impute: str = 'mean',
                 random_state=42):
        self.data = data
        self.augmentation = augmented_data
        self.base_model = base_model
        self.target_column = target_column
        self.feature_group = feature_group
        self.random_state = random_state
        self.smote = smote
        self.drop_threshold = drop_threshold
        self.normalizer = normalizer
        self.categorical_impute = categorical_impute
        self.real_impute = real_impute
        self.lof = LocalOutlierFactor()

    def detect_outliers(self):
        """
		Detects outliers in the dataset

		Returns
		-------
		outlier_detector: LocalOutlierFactor
		"""
        X = self.data
        real_columns = [col for col in X if len(X[col].dropna().unique()) > 10]
        categorical_columns = [col for col in X if len(X[col].dropna().unique()) <= 10]

        outlier_transformer = ColumnTransformer(
            [
                ('categorical', self.__preprocess_categorical_columns(),
                 lambda df: [c for c in df.columns if c in categorical_columns]),
                (
                'real', self.__preprocess_real_columns(False), lambda df: [c for c in df.columns if c in real_columns]),
            ]
        )
        outlier_detector = Pipeline(steps=[('imputer', outlier_transformer), ('lof', self.lof)])

        return X, outlier_detector

    def prerocess_and_create_pipeline(self):
        """
		Separates the data into input features and label, and creates the training pipeline
		
		Returns
		-------
		X : 2d array-like
			Array of features by chosen groups

		y : 1d array-like
			Array of labels
			
		pipeline : Pipeline
			the training pipeline
		"""
        X = self.data
        aug = self.augmentation
        bigX = pd.concat([X, aug], ignore_index=True)
        index_pairs = self.__create_pair(X, bigX)
        X = self.__preprocess_feature_groups(self.data)
        aug = self.__preprocess_feature_groups(self.augmentation)

        y = self.data[self.target_column]
        # X = X.loc[X.isna().mean(axis=1) < self.drop_threshold, X.isna().mean(axis=0) < self.drop_threshold]
        # aug = aug.loc[aug.isna().mean(axis=1) < self.drop_threshold, aug.isna().mean(axis=0) < self.drop_threshold]

        bigX = pd.concat([X, aug], ignore_index=True)
        bigY = pd.concat([y, pd.Series([1] * len(aug))], ignore_index=True)

        real_columns = [col for col in X if len(X[col].dropna().unique()) > 10]
        categorical_columns = [col for col in X if len(X[col].dropna().unique()) <= 10]

        col_transformer = ColumnTransformer(
            [
                ('categorical', self.__preprocess_categorical_columns(),
                 lambda df: [c for c in df.columns if c in categorical_columns]),
                ('real', self.__preprocess_real_columns(), lambda df: [c for c in df.columns if c in real_columns]),
            ]
        )
        preprc_steps = [
            ('preprocessor', col_transformer),
            ('classifier', self.base_model)
        ]
        if self.smote:
            preprc_steps.insert(1, ('smote', SMOTE(random_state=self.random_state)))

        pipeline = Pipeline(steps=preprc_steps)
        return X, y, bigX, bigY, index_pairs, pipeline

    def __preprocess_real_columns(self, normalize=True):
        imputer = get_real_imputer(self.real_impute, self.random_state)
        normalizer = get_normalizer(self.normalizer) if normalize else get_normalizer(None)
        return Pipeline(
            steps=[
                ('imputer', imputer),
                ('normalizer', normalizer)
            ]
        )

    def __preprocess_categorical_columns(self):
        imputer = get_categorical_imputer(self.categorical_impute)
        return Pipeline(steps=[('imputer', imputer)])

    def __preprocess_feature_groups(self, data):
        """
        Process the data according to the specified feature groups
        """
        # Identify the columns belonging to each feature group
        original_features = [col for col in data.columns if col.startswith('original')]
        log_features = [col for col in data.columns if col.startswith('log')]
        wavelet_features = [col for col in data.columns if col.startswith('wavelet')]

        # Create a dictionary to hold the feature groups
        feature_groups = {
            'original': original_features,
            'log': log_features,
            'wavelet': wavelet_features
        }

        # Filter the data according to the specified feature groups
        selected_features = []
        for group in self.feature_group:
            selected_features.extend(feature_groups.get(group, []))

        # Filter the data
        return data[selected_features]
    
    def __create_pair(self,X, bigX):
        pairs = []
        for original_index, row in X.iterrows():
            if row.Label == 0:
                continue
            original_id = row.Name
            augmented_id = f"{original_id}a"
            augmented_index = bigX[bigX['Name'] == augmented_id].index[0]
            pairs.append((original_index, augmented_index))
        return pairs