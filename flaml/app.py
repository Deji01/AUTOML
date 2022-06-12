from flaml import AUTOML
from sklearn.datasets import load_iris
automl = AUTOML()
# Specify automl goal and constraint
automl_settings = {
"time_budget": 10, # in seconds
"metric": 'accuracy',
"task": 'classification',
}
X_train, y_train = load_iris(return_X_y=True)
# Train with labeled input data
automl.fit(X_train=X_train, y_train=y_train,
**automl_settings)
# Predict
print(automl.predict_proba(X_train))
# Export the best model
print(automl.model)