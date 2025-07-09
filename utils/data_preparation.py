from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
from typing import Optional,Tuple,Dict

# configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def encode_dataset(data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame],Optional[Dict]]:
    """"encode the categorical columns"""
    label_encoders = None
    try:
        categorical_columns = list(data.select_dtypes(include='object').columns)
        label_encoders = {}
        for col in categorical_columns:
            encoder = LabelEncoder()
            encoder.fit(data[col])
            data[col] = encoder.transform(data[col])
            label_encoders[col] = encoder
        logger.info(f"Completed encoding of columns: {categorical_columns}")
    except Exception as err:
        logger.error(f"An error occured: {err}")
    
    return data, label_encoders


def split_dataset(data:pd.DataFrame) -> Tuple[Optional[pd.DataFrame],
                                              Optional[pd.DataFrame],
                                              Optional[pd.DataFrame],
                                              Optional[pd.DataFrame],
                                              ]:
    """splits the dataset into train and test"""
    X_train,X_test,y_train,y_test = None, None, None, None
    X = data.drop(columns=['tip'])
    y = data['tip']
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=23)
        logger.info('Splitting successful')
    except Exception as err:
        logger.error(f'an erro occured: {err}')
    
    return X_train, X_test,y_train, y_test


def scale_dataset(X_train:pd.DataFrame,X_test:pd.DataFrame,
                  y_train:pd.DataFrame,y_test:pd.DataFrame) -> Tuple[Optional[pd.DataFrame],
                                              Optional[pd.DataFrame],
                                              Optional[pd.DataFrame],
                                              Optional[pd.DataFrame],
                                              ]:
    try:
        scaler = StandardScaler()
        columns = X_train.columns
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train = pd.DataFrame(data=X_train_scaled, columns=columns)
        X_test = pd.DataFrame(data=X_test_scaled, columns= columns)
    except Exception as err:
        X_train,X_test,y_train,y_test = None, None, None, None
        logger.error(f'An error occured: {err}')
        
    return X_train,X_test,y_train,y_test
        
        