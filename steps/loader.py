from zenml import step
from zenml.logger import get_logger
from utils.data_loader import load_data
from typing_extensions import Annotated
from typing import Optional
import pandas as pd




logger = get_logger(__name__)

@step
def data_loading() -> Annotated[Optional[pd.DataFrame], "Load the dataset"]:
    data = load_data()
    return data