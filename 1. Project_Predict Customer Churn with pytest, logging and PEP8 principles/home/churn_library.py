# churn_library.py
'''
Library of functions to find customers who are likely to churn.
They are then added to at if __name__ == "__main__" block,
this allows to run the code and produce the results
'''
import joblib
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

print(f"Matplotlib backend set to: {matplotlib.get_backend()}")
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(file_path):
    """
    Returns DataFrame for the CSV found at file_path.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    loaded_df = pd.read_csv(file_path)
    loaded_df['Churn'] = loaded_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return loaded_df


def perform_eda(data_frame, output_path='/home/workspace/images/eda'):
    """
    Performs EDA on the DataFrame and saves figures to the specified output path.

    Args:
        data_frame (pd.DataFrame): DataFrame to analyze.
        output_path (str): Path to save the figures.

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define plots and their filenames
    plots = [
        (data_frame['Churn'].hist,
         'histogram_churn.png'),
        (data_frame['Customer_Age'].hist,
         'histogram_customer_age.png'),
        (lambda: data_frame['Marital_Status'].value_counts(normalize=True)
         .plot(kind='bar'), 'barplot_marital_status.png'),
        (lambda: sns.histplot(data_frame['Total_Trans_Ct'],
                              stat='density', kde=True),
         'distplot_total_trans_ct.png'),
        (lambda: sns.heatmap(data_frame.corr(),
                             annot=False, cmap='Dark2_r',
                             linewidths=2),
         'heatmap_correlations.png')
    ]

    # Generate and save each plot
    for plot_func, filename in plots:
        plt.figure(figsize=(20, 10))
        plot_func()
        plt.savefig(os.path.join(output_path, filename))
        plt.close()


def encoder_helper(data_frame, category_list, response='Churn'):
    """
    Encodes categorical columns in the DataFrame with the
    proportion of churn for each category.

    Args:
        data_frame (pd.DataFrame): DataFrame to encode.
        category_list (list): List of columns containing categorical features.
        response (str): Name of the response variable.

    Returns:
        pd.DataFrame: DataFrame with new columns for encoded categorical features.
    """
    def encode_column(df_to_encode, column_name, target_column):
        """
        Encodes a categorical column based on the mean of a target column.

        Args:
            df_to_encode (pd.DataFrame): DataFrame to encode.
            column_name (str): Name of the column to encode.
            target_column (str): Name of the target column.

        Returns:
            pd.Series: Series with encoded values.
        """
        if column_name not in df_to_encode.columns:
            print(f"Column '{column_name}' is not in DataFrame")

        group_means = df_to_encode.groupby(column_name)[target_column].mean()
        return df_to_encode[column_name].map(group_means)

    if data_frame is None:
        print("DataFrame is None")
        return data_frame

    for column in category_list:
        encoded_column_name = f"{column}_{response}"
        data_frame[encoded_column_name] = encode_column(
            data_frame, column, response)

    return data_frame


def perform_feature_engineering(data_frame, response='Churn'):
    """
    Performs feature engineering on the DataFrame and splits it into train and test sets.

    Args:
        data_frame (pd.DataFrame): DataFrame to process.
        response (str): Name of the response variable.

    Returns:
        tuple: (x_train, x_test, y_train, y_test) where x_train and x_test are feature DataFrames,
               and y_train and y_test are response Series.
    """
    y_data = data_frame[response]
    x_data = data_frame[['Customer_Age',
                         'Dependent_count',
                         'Months_on_book',
                         'Total_Relationship_Count',
                         'Months_Inactive_12_mon',
                         'Contacts_Count_12_mon',
                         'Credit_Limit',
                         'Total_Revolving_Bal',
                         'Avg_Open_To_Buy',
                         'Total_Amt_Chng_Q4_Q1',
                         'Total_Trans_Amt',
                         'Total_Trans_Ct',
                         'Total_Ct_Chng_Q4_Q1',
                         'Avg_Utilization_Ratio',
                         'Gender_Churn',
                         'Education_Level_Churn',
                         'Marital_Status_Churn',
                         'Income_Category_Churn',
                         'Card_Category_Churn']]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
        output_path='/home/workspace/images/results'):
    """
    Produces and saves classification reports for training and testing results.

    Args:
        y_train (pd.Series): Training response values.
        y_test (pd.Series): Testing response values.
        y_train_preds_lr (np.ndarray): Training predictions from logistic regression.
        y_train_preds_rf (np.ndarray): Training predictions from random forest.
        y_test_preds_lr (np.ndarray): Testing predictions from logistic regression.
        y_test_preds_rf (np.ndarray): Testing predictions from random forest.
        output_path (str): Path to save the reports.

    Returns:
        None
    """
    reports = {
        'random_forest_test': classification_report(y_test, y_test_preds_rf),
        'random_forest_train': classification_report(y_train, y_train_preds_rf),
        'logistic_regression_test': classification_report(y_test, y_test_preds_lr),
        'logistic_regression_train': classification_report(y_train, y_train_preds_lr)
    }

    for name, report in reports.items():
        plt.figure(figsize=(12, 8))
        plt.text(
            0.1, 1.1, report, {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(
            os.path.join(
                output_path,
                f'classification_report_{name}.png'))
        plt.close()


def feature_importance_plot(
        model,
        x_data,
        output_path='/home/workspace/images/results'):
    """
    Creates and saves feature importance plots for the given model.

    Args:
        model (sklearn.base.ClassifierMixin): Model object containing feature_importances_.
        x_data (pd.DataFrame): DataFrame of feature values.
        output_path (str): Path to save the figure.

    Returns:
        None
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar")
    plt.savefig(os.path.join(output_path, 'shap_summary.png'))
    plt.close()

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), feature_names, rotation=90)
    plt.savefig(os.path.join(output_path, 'feature_importance.png'))
    plt.close()


def train_models(x_train, x_test, y_train):
    """
    Trains models, saves results and models.

    Args:
        x_train (pd.DataFrame): Training feature data.
        x_test (pd.DataFrame): Testing feature data.
        y_train (pd.Series): Training response data.
        y_test (pd.Series): Testing response data.

    Returns:
        tuple: (best_rf_model, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)
    """
    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
    cv_rf.fit(x_train, y_train)

    joblib.dump(cv_rf.best_estimator_, './models/random_forest_model.pkl')

    lr_model.fit(x_train, y_train)

    y_train_preds_rf = cv_rf.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rf.best_estimator_.predict(x_test)
    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)

    joblib.dump(lr_model, './models/logistic_regression_model.pkl')

    return (
        cv_rf.best_estimator_,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
