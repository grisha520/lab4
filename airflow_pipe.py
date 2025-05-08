import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.impute import SimpleImputer
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta

def download_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv"
    df = pd.read_csv(url)
    df.to_csv("car_price.csv", index=False)
    print(f"Dataset downloaded with shape: {df.shape}")
    return df

def clean_data():
    df = pd.read_csv("car_price.csv")

    df = df.drop(columns=['car_ID', 'CarName'], errors='ignore')

    categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 
                       'drivewheel', 'enginelocation', 'enginetype', 
                       'cylindernumber', 'fuelsystem']

    numeric_cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 
                   'curbweight', 'enginesize', 'boreratio', 'stroke', 
                   'compressionratio', 'horsepower', 'peakrpm', 'citympg', 
                   'highwaympg']

    df = df[df['price'] < 40000]
    df = df[df['price'] > 1000]  
    df = df[df['horsepower'] < 250] 

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)

    df.to_csv('car_price_clean.csv', index=False)
    print(f"Cleaned dataset shape: {df.shape}")
    return True

def train_model():
    df = pd.read_csv('car_price_clean.csv')

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 
                       'drivewheel', 'enginelocation', 'enginetype', 
                       'cylindernumber', 'fuelsystem']
    numeric_cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 
                   'curbweight', 'enginesize', 'boreratio', 'stroke', 
                   'compressionratio', 'horsepower', 'peakrpm', 'citympg', 
                   'highwaympg']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
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

    models = {
        'sgd': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SGDRegressor(max_iter=1000, tol=1e-3))
        ]),
        'random_forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {'RMSE': rmse, 'MAE': mae}
        print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    return results

dag_car_price = DAG(
    dag_id="car_price_prediction",
    start_date=datetime(2025, 5, 8),
    schedule_interval=timedelta(hours=6),
    max_active_runs=1,
    catchup=False,
    default_args={
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }
)

download_task = PythonOperator(
    task_id="download_car_data",
    python_callable=download_data,
    dag=dag_car_price
)

clean_task = PythonOperator(
    task_id="clean_car_data",
    python_callable=clean_data,
    dag=dag_car_price
)

train_task = PythonOperator(
    task_id="train_car_model",
    python_callable=train_model,
    dag=dag_car_price
)

# Определение порядка выполнения задач
download_task >> clean_task >> train_task
