import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.cluster import KMeans


def xgboost(X, grams, term_frac, clusters):
    model = XGBoost()
    model.numeric(X, grams, term_frac)
    model.cluster(clusters)
    model.train()
    model.predict()
    model.importance()

    return model


class XGBoost:
    def numeric(self, X, grams, term_frac):
        self.X = X

        # collect the words (and their uncommon) from each document
        # 'matrix' is a term (columns), document (rows) matrix
        matrix = pd.DataFrame()
        for c in X.columns:
            vector = TfidfVectorizer(ngram_range=(1, grams))
            matrix2 = vector.fit_transform(X[c].tolist())
            names = vector.get_feature_names_out()
            names = [f"{c}: {n}" for n in names]
            matrix2 = pd.DataFrame(matrix2.toarray(), columns=names)
            matrix = pd.concat([matrix, matrix2], axis="columns")
        
        # find the most uncommon terms
        uncommon = matrix.max(axis="index").reset_index()
        uncommon.columns = ["index", "score"]
        uncommon = uncommon.sort_values(by="score", ascending=False)

        # drop the common terms
        common = uncommon.tail(int(len(uncommon) * (1 - term_frac)))
        self.X = matrix.drop(columns=common["index"])

    def cluster(self, clusters):
        kmeans = KMeans(n_clusters=clusters, random_state=0)
        self.y = kmeans.fit_predict(self.X)
        self.y = pd.DataFrame(self.y, columns=["Cluster"])

    def train(self):
        self.model = XGBClassifier(
            booster="gbtree",
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(self.X, self.y)

    def predict(self):
        y_pred = self.model.predict(self.X)
        y_true = self.y.to_numpy().ravel()

        self.metric = accuracy_score(
            y_true=y_true, 
            y_pred=y_pred,
        )
        self.metric = f"Accuracy: {round(100 * self.metric, 2)}%"

        self.predictions = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred,
        })

    def importance(self):
        self.keywords = self.model.feature_importances_
        self.keywords = pd.DataFrame({
            "Word": self.X.columns,
            "Importance": self.keywords,
        })
        self.keywords = self.keywords.sort_values(
            by="Importance", 
            ascending=False,
        ).reset_index(drop=True)
