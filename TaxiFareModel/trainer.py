from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder, DistanceToCenter
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from memoized_property import memoized_property
from  mlflow.tracking import MlflowClient
import mlflow
import joblib
import numpy as np

class Trainer():
    def __init__(self, X, y, url=None):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.url = "https://mlflow.lewagon.co/" if url is None else url
        self.X = X
        self.y = y
        self.experiment_name = "[FR] [Nantes] [BaliMalo] Taxifare v1.0"

    def set_pipeline(self, model):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        dist_to_center_pipe = Pipeline([
            ('dist_trans', DistanceToCenter()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('distance_to_center', dist_to_center_pipe, ["pickup_latitude", "pickup_longitude"]),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', model)
        ])

    def run(self,model):
        """set and train the pipeline"""
        self.set_pipeline(model)
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.url)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        filename = 'model.joblib'
        joblib.dump(self.pipeline, filename)

if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

#define model

    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # n_estimators = [10,50,100,200,500]
    for i in range(1):
    # for kernel in kernels:
    # for estimators in n_estimators:
        model = LinearRegression()
        # model = SVR(kernel=kernel)
        # model = RandomForestRegressor(n_estimators=estimators)
        # model = RandomForestRegressor()
        
        eval_list = []

        for iteration in range(5):
            # hold out
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
            # train
            trainer = Trainer(X=X_train, y=y_train)    

            trainer.run(model)
            
            # evaluate
            eval_list.append(trainer.evaluate(X_val, y_val))

        eval= np.mean(eval_list)
        #log
        client = trainer.mlflow_client
        experiment_id = trainer.mlflow_experiment_id
        run = client.create_run(experiment_id)
        client.log_metric(run.info.run_id, "rmse", eval)
        client.log_param(run.info.run_id, "model", model)
        client.log_param(run.info.run_id, "distance_to_center", True)
    
    # trainer.save_model()
    
