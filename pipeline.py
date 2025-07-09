from steps.loader import data_loading
from steps.preparation import data_encoding, data_splitting, scaling_dataset
from zenml import pipeline
from steps.training import model_training
from zenml.logger import get_logger


logger = get_logger(__name__)

@pipeline
def tips_pipeline():
    data = data_loading()
    data, label_encoders = data_encoding(data)
    X_train, X_test, y_train, y_test = data_splitting(data)
    X_train, X_test, y_train, y_test = scaling_dataset(X_train, X_test, y_train, y_test)
    model = model_training(X_train, y_train)
    
    
if __name__ == "__main__":
    tips_pipeline()
    



