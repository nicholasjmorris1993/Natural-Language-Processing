import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import sys
sys.path.append("/home/nick/Natural-Language-Processing/src")
from regression import xgboost


def parity(df, predict, actual, color=None, title=None, font_size=None):
    fig = px.scatter(df, x=actual, y=predict, color=color, title=title)
    fig.add_trace(go.Scatter(x=df[actual], y=df[actual], mode="lines", showlegend=False, name="Actual"))
    fig.update_layout(font=dict(size=font_size))
    plot(fig)


data = pd.read_csv("/home/nick/Natural-Language-Processing/test/drug review.csv")
data = data.replace(np.nan, "")

model = xgboost(
    X=data[[
        "benefitsReview", 
        "sideEffectsReview", 
        "commentsReview",
    ]], 
    y=data[["rating"]], 
    test_frac=0.5,
    grams=1,  # number of words to group together into phrases
    term_frac=0.01,  # fraction of terms to retain
)

parity(
    df=model.predictions,
    predict="Predicted",
    actual="Actual",
    font_size=16,
)

print(model.metric)
