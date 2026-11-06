import numpy as np
import pandas as pd


class GroupEstimate(object):
    """
    Predict y for new rows using the mean or median of y in the
    matching group from the training data.
    """

    def __init__(self, estimate):
        if estimate not in ("mean", "median"):
            raise ValueError('estimate must be "mean" or "median".')
        self.estimate = estimate
        self.columns = None
        self.group_estimates = None

    def fit(self, X, y):
        """
        Work out the mean or median of y for each combination of
        categories in X. Keeps the column names and the per-group
        estimates, not X or y.
        """
        X = pd.DataFrame(X)
        y = pd.Series(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if y.isna().any():
            raise ValueError("y has missing values.")

        # Reset both indexes so y lines up with X by position, not by label.
        df = X.reset_index(drop=True)
        df["__y__"] = y.reset_index(drop=True)

        self.columns = list(X.columns)
        self.group_estimates = (
            df.groupby(self.columns, observed=True)["__y__"].agg(self.estimate)
        )
        return self

    def predict(self, X_):
        """
        Return the estimate for each row of X_ as a numpy array.
        Rows whose group never showed up in training get NaN, and a
        message says how many.
        """
        if self.group_estimates is None:
            raise ValueError("Call fit() before predict().")

        X_ = pd.DataFrame(X_)

        # A DataFrame with the training columns gets put back in training
        # order. A list of lists or numpy array has no real names, so it
        # takes the training names in order.
        if set(X_.columns) == set(self.columns):
            X_ = X_[self.columns]
        elif X_.shape[1] == len(self.columns):
            X_.columns = self.columns
        else:
            raise ValueError(
                f"X_ has {X_.shape[1]} columns, "
                f"but fit() saw {len(self.columns)}."
            )

        if len(self.columns) == 1:
            keys = X_[self.columns[0]].tolist()
        else:
            keys = list(X_.itertuples(index=False, name=None))

        predictions = self.group_estimates.reindex(keys).to_numpy()

        missing = int(np.isnan(predictions).sum())
        if missing:
            print(f"{missing} observation(s) had a group that was not in "
                  f"the training data, so their estimate is NaN.")

        return predictions
