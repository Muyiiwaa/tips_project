import plotly.express as px
import pandas as pd
import logging
from typing import Optional

# configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> Optional[pd.DataFrame]:
    """this function loads the dataframe from plotly library"""
    data = None
    try:
        data = px.data.tips()
        logger.info(msg=f"Loaded dataset successfully with shape {data.shape}")
    except Exception as err:
        logger.error(msg=f"Encountered error: {err}")
    return data
    