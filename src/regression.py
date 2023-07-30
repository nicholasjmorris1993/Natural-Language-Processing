import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from xgboost.sklearn import XGBRegressor


def xgboost(X, y, test_frac, grams, term_frac):
    model = XGBoost()
    model.numeric(X, y, test_frac, grams, term_frac)
    model.train()
    model.predict()

    return model


class XGBoost:
    def numeric(self, X, y, test_frac, grams, term_frac):
        self.X = X
        self.y = y
        self.test_frac = test_frac

        self.X = self.X.copy().sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
        self.y = self.y.copy().sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data

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

    def train(self):
        train_X = self.X.head(int(len(self.X)*(1 - self.test_frac)))
        train_y = self.y.head(int(len(self.y)*(1 - self.test_frac)))

        self.model = XGBRegressor(
            booster="gbtree",
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(train_X, train_y)

    def predict(self):
        test_X = self.X.tail(int(len(self.X)*self.test_frac))
        test_y = self.y.tail(int(len(self.y)*self.test_frac))

        y_pred = self.model.predict(test_X)
        y_true = test_y.to_numpy().ravel()

        self.metric = r2_score(
            y_true=y_true, 
            y_pred=y_pred,
        )
        self.metric = f"R2: {round(100 * self.metric, 2)}%"

        self.predictions = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred,
        })

