import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


def print_regression_metrics(name: str, y_true, y_pred) -> None:
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"R²:  {r2:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")


def main() -> None:
    cali = fetch_california_housing()
    df = pd.DataFrame(cali.data, columns=cali.feature_names)
    df["MedHouseVal"] = cali.target

    print(df.describe())

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    lin_reg = LinearRegression()
    lin_reg.fit(x_train_scaled, y_train)
    y_pred_lin = lin_reg.predict(x_test_scaled)
    print_regression_metrics("Linear Regression", y_test, y_pred_lin)

    ridge = Ridge()
    ridge.fit(x_train_scaled, y_train)
    y_pred_ridge = ridge.predict(x_test_scaled)
    print_regression_metrics("Ridge (default)", y_test, y_pred_ridge)

    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge_grid = GridSearchCV(
        estimator=Ridge(),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    ridge_grid.fit(x_train_scaled, y_train)

    best_ridge = ridge_grid.best_estimator_
    y_pred_ridge_grid = best_ridge.predict(x_test_scaled)

    print(f"\nBest params for Ridge: {ridge_grid.best_params_}")
    print_regression_metrics("Ridge (GridSearchCV)", y_test, y_pred_ridge_grid)

    lasso = Lasso(alpha=0.01)
    lasso.fit(x_train_scaled, y_train)
    y_pred_lasso = lasso.predict(x_test_scaled)
    print_regression_metrics("Lasso (alpha=0.01)", y_test, y_pred_lasso)


if __name__ == "__main__":
    main()
