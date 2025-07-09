from zenml import step
from zenml.logger import get_logger
from utils.data_preparation import scale_dataset,split_dataset,encode_dataset
from typing_extensions import Annotated
from typing import Optional, Tuple, Dict
import pandas as pd


logger = get_logger(__name__)


@step
def data_splitting(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:
    X_train, X_test, y_train, y_test = split_dataset(data)
    
    return X_train, X_test, y_train, y_test


@step
def data_encoding(data: pd.DataFrame) -> Tuple[Annotated[Optional[pd.DataFrame], "encoded data"],
                             Annotated[Optional[Dict],"label encoders"]]:
    
    data, label_encoders = encode_dataset(data=data)
    return data, label_encoders

@step
def scaling_dataset(X_train:pd.DataFrame,X_test:pd.DataFrame,
                  y_train:pd.Series,y_test:pd.Series) -> Tuple[
                      Annotated[Optional[pd.DataFrame], "X_train"],
                      Annotated[Optional[pd.DataFrame], "X_test"],
                      Annotated[Optional[pd.Series], "y_train"],
                      Annotated[Optional[pd.Series], "y_test"]]:
    X_train,X_test,y_train, y_test = scale_dataset(X_train, X_test, y_train, y_test)
    
    return X_train,X_test,y_train, y_test