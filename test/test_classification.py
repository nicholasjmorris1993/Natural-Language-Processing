import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("/home/nick/Natural-Language-Processing/src")
from classification import xgboost


data = pd.read_csv("/home/nick/Natural-Language-Processing/test/drug review.csv")
data = data.replace(np.nan, "")

# encode categories into integer labels
labeler = LabelEncoder()
data["effectiveness"] = labeler.fit_transform(data["effectiveness"])

model = xgboost(
    X=data[[
        "benefitsReview", 
        "sideEffectsReview", 
        "commentsReview",
    ]], 
    y=data[["effectiveness"]], 
    test_frac=0.5,
    grams=1,  # number of words to group together into phrases
    term_frac=0.01,  # fraction of terms to retain
)

print(model.metric)

# decode integer labels back to original labels
predictions = model.predictions
predictions["Actual"] = labeler.inverse_transform(predictions["Actual"])
predictions["Predicted"] = labeler.inverse_transform(predictions["Predicted"])

labels = np.unique(predictions.to_numpy())
cmatrix = confusion_matrix(
    y_true=predictions["Actual"],   # rows
    y_pred=predictions["Predicted"],  # columns
    labels=labels,
)
cmatrix = pd.DataFrame(cmatrix, columns=labels, index=labels)
print(cmatrix)
