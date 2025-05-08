import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
import mlflow
from mlflow.models import infer_signature
import joblib

def preprocess_data(df):
    df = df.drop(columns=['car_ID', 'CarName'], errors='ignore')

    categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 
                       'drivewheel', 'enginelocation', 'enginetype', 
                       'cylindernumber', 'fuelsystem']
    numeric_cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 
                   'curbweight', 'enginesize', 'boreratio', 'stroke', 
                   'compressionratio', 'horsepower', 'peakrpm', 'citympg', 
                   'highwaympg']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor, df

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train():
    df = pd.read_csv("car_price_clean.csv")

    preprocessor, df = preprocess_data(df)
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    params = {
        'regressor__alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'regressor__l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "regressor__penalty": ["l1", "l2", "elasticnet"],
        "regressor__loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "regressor__fit_intercept": [False, True],
    }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', SGDRegressor(random_state=42))
    ])

    mlflow.set_experiment("car_price_prediction")
    
    with mlflow.start_run():
        # Поиск по сетке параметров
        clf = GridSearchCV(pipeline, params, cv=3, n_jobs=4, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)

        best_model = clf.best_estimator_

        y_pred = best_model.predict(X_val)

        rmse, mae, r2 = eval_metrics(y_val, y_pred)

        mlflow.log_params(clf.best_params_)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        joblib.dump(best_model, "best_car_price_model.pkl")
        
        print(f"Best model RMSE: {rmse:.2f}")
        print(f"Best model R2: {r2:.2f}")
        print(f"Best parameters: {clf.best_params_}")

if __name__ == "__main__":
    train()