import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
sys.path.append("/home/nick/Natural-Language-Processing/src")
from keywords import xgboost


data = pd.read_csv("/home/nick/Natural-Language-Processing/test/drug review.csv")
data = data.replace(np.nan, "")

model = xgboost(
    X=data[[
        "benefitsReview", 
        "sideEffectsReview", 
        "commentsReview",
    ]], 
    grams=1,  # number of words to group together into phrases
    term_frac=0.01,  # fraction of terms to retain
    clusters=5,  # number of clusters to group the words into
)

print(model.metric)

predictions = model.predictions
labels = np.unique(predictions.to_numpy())
cmatrix = confusion_matrix(
    y_true=predictions["Actual"],   # rows
    y_pred=predictions["Predicted"],  # columns
    labels=labels,
)
cmatrix = pd.DataFrame(cmatrix, columns=labels, index=labels)
print(cmatrix)

print(model.keywords)
