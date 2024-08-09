# churn_library.py
'''
library of functions to find customers who are likely to churn that are then added to at if __name__ == "__main__" block that allows you to run the code below and understand the results for each of the functions
'''

# import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
import shap
import joblib

os.environ['QT_QPA_PLATFORM']='offscreen'


def import_data(path):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(path)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df, output_pth = '/home/workspace/images/eda'):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Create the images directory if it doesn't exist
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)

    # Define plotting functions and filenames
    plots = [
        lambda: df['Churn'].hist(),
        lambda: df['Customer_Age'].hist(),
        lambda: df.Marital_Status.value_counts(normalize=True).plot(kind='bar'),
        lambda: sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True),
        lambda: sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    ]
    
    filenames = [
        'histogram_churn.png',
        'histogram_customer_age.png',
        'barplot_marital_status.png',
        'distplot_total_trans_ct.png',
        'heatmap_correlations.png'
    ]
    
    # Generate and save each plot
    for plot, filename in zip(plots, filenames):
        plt.figure(figsize=(20,10))
        plot()  # Call the plotting function
        plt.savefig(os.path.join(output_pth, filename))
        plt.close()
    return


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    def encode_column(df, column_name, target_column):
        """
        Encodes a categorical column based on the mean of a target column.
        """
        # Group by the categorical column and compute the mean of the target column
        group_means = df.groupby(column_name)[target_column].mean()

        # Map the mean values to the original DataFrame
        encoded_column = df[column_name].map(group_means)

        return encoded_column


    # Create new columns for encoded values
    for column in category_lst:
        encoded_column_name = f"{column}_{response}"
        df[encoded_column_name] = encode_column(df, column, 'Churn')
    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    
    y = df['Churn']
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    
    # This cell may take up to 15-20 minutes to run
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf, output_pth = '/home/workspace/images/results'):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores
    reports = {
        'Random Forest (Test)': classification_report(y_test, y_test_preds_rf),
        'Random Forest (Train)': classification_report(y_train, y_train_preds_rf),
        'Logistic Regression (Test)': classification_report(y_test, y_test_preds_lr),
        'Logistic Regression (Train)': classification_report(y_train, y_train_preds_lr)
    }
    
    for name, report in reports.items():
        plt.figure(figsize=(12, 8))
        plt.text(0.1, 1.1, report, {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(os.path.join(output_pth, f'classification_report_{name.replace(" ", "_")}.png'))
        plt.close()
    return


def feature_importance_plot(model, X_data, output_pth = '/home/workspace/images/results'):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # plots roc curve
    #plt.figure(figsize=(15, 8))
    #ax = plt.gca()
    #rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    #plt.show()

    # Shap summary
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.savefig(os.path.join(output_pth, 'shap_summary.png'))
    plt.close() 
    
    # Feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90);
    plt.savefig(os.path.join(output_pth, 'feature_importance.png'))
    plt.close() 
    return


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return cv_rfc.best_estimator_, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf