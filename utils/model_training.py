import pandas as pd
import logging
from typing import Optional
from sklearn.ensemble import RandomForestRegressor

# configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_model(X_train:pd.DataFrame, y_train:pd.DataFrame) -> Optional[RandomForestRegressor]:
    model = None
    try:
        model = RandomForestRegressor(n_estimators=23)
        model.fit(X_train, y_train)
        logger.info("MOdel trained successfully")
    except Exception as err:
        logger.error(f'An error occured: {err}')
        
    return model