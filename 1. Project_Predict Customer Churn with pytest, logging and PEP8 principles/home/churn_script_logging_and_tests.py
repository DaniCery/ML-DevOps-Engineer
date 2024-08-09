# churn_script_logging_and_tests.py
"""
1) TESTING: Contains unit tests for the churn_library.py functions.
Instead of basic assert statements based on churn_library function results,
pytest fixtures are used to input churn_library.py returned arguments to test functions.
2) LOGGING: Logs errors and INFO messages in the .log file.
SUCCESS and ERROR tags facilitate readability.
Testing and logging are then called in the command line in advance of the respective functions.
"""

import os
import logging
import pytest

from churn_library import (
    import_data, perform_eda, encoder_helper,
    perform_feature_engineering, train_models,
    classification_report_image, feature_importance_plot
)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture
def data_frame_fixture():
    """Fixture to load data from CSV file."""
    return import_data(file_path="./data/bank_data.csv")


def test_import(data_frame_fixture):
    """
    Test data import. Checks that the data has rows and columns.
    """
    try:
        loaded_df = data_frame_fixture
        logging.info("SUCCESS: Testing import_data")
        assert loaded_df.shape[0] > 0
        assert loaded_df.shape[1] > 0
        logging.info("SUCCESS: dataframe has rows and columns")
    except AssertionError as err:
        logging.error(
            "ERROR: The file doesn't appear to have rows and columns")
        raise err
    except FileNotFoundError as err:
        logging.error("ERROR: The file wasn't found")
        raise err


def test_perform_eda(data_frame_fixture):
    """
    Test perform_eda function. Checks if charts are saved and if DataFrame has no null values.
    """
    df_eda = data_frame_fixture
    output_path = "/home/workspace/images/eda"
    filenames = [
        'histogram_churn.png', 'histogram_customer_age.png',
        'barplot_marital_status.png', 'distplot_total_trans_ct.png',
        'heatmap_correlations.png'
    ]
    perform_eda(df_eda, output_path)

    try:
        if df_eda.isnull().any().any():
            raise Warning("DataFrame has null values")
        logging.info("SUCCESS: DataFrame has no null values")
    except Warning as warn:
        logging.warning("WARNING: %s", warn)

    for filename in filenames:
        file_path = os.path.join(output_path, filename)
        try:
            assert os.path.exists(file_path), "File {} does not exist in {}".format(
                filename, output_path)
            logging.info(
                "SUCCESS: File %s exists in %s",
                filename,
                output_path)
        except AssertionError as err:
            logging.error("ERROR: %s", err)


@pytest.fixture
def encoded_data_fixture(data_frame_fixture):
    """
    Fixture that returns the DataFrame after encoding,
    along with category list and response variable.
    """
    df_to_encode = data_frame_fixture
    response_variable = 'Churn'
    category_columns = [
        'Gender', 'Education_Level', 'Marital_Status',
        'Income_Category', 'Card_Category'
    ]
    encoded_dataframe = encoder_helper(
        df_to_encode, category_columns, response_variable)
    return encoded_dataframe, category_columns, response_variable


def test_encoder_helper(encoded_data_fixture):
    """
    Test encoder_helper function. Checks if encoded columns are created correctly.
    """
    encoded_dataframe, category_columns, response_variable = encoded_data_fixture
    for column in category_columns:
        try:
            assert column in encoded_dataframe.columns, "Missing {} in DataFrame".format(
                column)
            logging.info("SUCCESS: Column %s is in DataFrame", column)
        except AssertionError as err:
            logging.error("ERROR: %s", err)

    for column in category_columns:
        encoded_column = "{}_{}".format(column, response_variable)
        if encoded_column in encoded_dataframe.columns:
            logging.info(
                "SUCCESS: Column %s created correctly.",
                encoded_column)
        else:
            logging.error(
                "ERROR: Column %s not created correctly.",
                encoded_column)


@pytest.fixture
def feature_engineered_data_fixture(encoded_data_fixture):
    """
    Fixture that returns the data after feature engineering.
    """
    encoded_dataframe, _, _ = encoded_data_fixture
    return perform_feature_engineering(encoded_dataframe)


def test_perform_feature_engineering(feature_engineered_data_fixture):
    """
    Test perform_feature_engineering function.
    Checks if data splits are correct and contains no null values.
    """
    x_data_train, x_data_test, y_data_train, y_data_test = feature_engineered_data_fixture
    try:
        assert x_data_train.shape[0] > 0 and x_data_train.shape[1] > 0
        assert x_data_test.shape[0] > 0 and x_data_test.shape[1] > 0
        assert y_data_train.shape[0] > 0
        assert y_data_test.shape[0] > 0
        logging.info(
            "SUCCESS: Train/test data splits have correct rows and columns")
    except AssertionError as err:
        logging.error("ERROR: %s", err)
        raise err

    datasets = [x_data_train, x_data_test, y_data_train, y_data_test]
    try:
        for dataset in datasets:
            if dataset.isnull().any().any():
                raise Warning("Null values found in datasets")
        logging.info("SUCCESS: No nulls in train, test datasets")
    except Warning as warn:
        logging.warning("WARNING: %s", warn)


def test_train_models():
    """
    Test train_models function. Checks if the model file exists.
    """
    try:
        model_path = './models/rfc_model.pkl'
        assert os.path.exists(model_path), "Model does not exist in the folder"
        logging.info("SUCCESS: Model saved successfully")
    except AssertionError as file_error:
        logging.error("ERROR: %s", file_error)


if __name__ == "__main__":
    print("Import function processing...")
    data_frame = import_data("./data/bank_data.csv")

    print("EDA function processing...")
    perform_eda(data_frame)

    print("Encoder function processing...")
    category_columns = [
        'Gender', 'Education_Level', 'Marital_Status',
        'Income_Category', 'Card_Category'
    ]
    encoded_df = encoder_helper(data_frame, category_columns)

    print("Feature engineering function processing...")
    x_train, x_test, y_train, y_test = perform_feature_engineering(encoded_df)

    print("Train function processing...")
    model, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = train_models(
        x_train, x_test, y_train)

    print("Report function processing...")
    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf
    )

    print("Feature importance function processing...")
    feature_importance_plot(model, x_test)
