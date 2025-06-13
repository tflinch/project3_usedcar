import os
import argparse
import mlflow
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.start_run()  # Start MLflow experiment

os.makedirs("./outputs", exist_ok=True)  # Ensure output dir exists

def select_first_file(path):
    """Returns first file from a folder."""
    files = os.listdir(path)
    return os.path.join(path, files[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--criterion", type=str, default="squared_error")
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--model_output", type=str, help="Path to save the trained model")
    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # Split into X/y
    y_train = train_df["price"].values
    X_train = train_df.drop("price", axis=1).values
    y_test = test_df["price"].values
    X_test = test_df.drop("price", axis=1).values

    # Train model
    model = DecisionTreeRegressor(criterion=args.criterion, max_depth=args.max_depth)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    # ✅ Log metrics with correct names
    mlflow.log_metric("mse", float(mse))
    mlflow.log_metric("r2_score", float(r2))  # Required for Azure ML

    # Save model
    mlflow.sklearn.save_model(model, args.model_output)

    mlflow.end_run()

if __name__ == "__main__":
    main()
