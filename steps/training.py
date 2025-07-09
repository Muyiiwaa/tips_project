from zenml import step
from zenml.logger import get_logger
from utils.model_training import train_model
from typing_extensions import Annotated
from typing import Optional, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


logger = get_logger(__name__)


@step
def model_training(X_train:pd.DataFrame, y_train:pd.DataFrame) -> Annotated[
    Optional[RandomForestRegressor],"Trained Model"]:
    model = train_model(X_train,y_train)
    
    return model

