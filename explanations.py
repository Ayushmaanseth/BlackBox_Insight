import pandas as pd
import numpy as np
#import model_evaluation_utils as meu
import matplotlib.pyplot as plt
from collections import Counter
import shap
import eli5
import xgboost as xgb
import warnings
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
import os
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
shap.initjs()


# # Load the Census Income Dataset

# In[106]:
def run_explanations(csv_path,csv_columns,target_column,zero_value):
    # Read the dataset from the provided CSV and print out information about it.
    df=pd.read_csv(csv_path,names=csv_columns, skipinitialspace=True,skiprows=1)
    #df = df.drop('Target',axis=1)
    input_features = [name for name in csv_columns if name != target_column]
    #data, labels = shap.datasets.adult(display=True)
    if target_column not in csv_columns:
        print("target column error")
        return("target column error")
    elif zero_value not in df[target_column].tolist():
        if str.isdecimal(zero_value) and (np.int64(zero_value) in df[target_column].tolist() or np.float64(zero_value) in df[target_column].tolist()):
            print("happy")
            zero_value = np.int64(zero_value)
        else:
            print(zero_value,df[target_column].tolist(),df[target_column].dtype)
            return("zero value error")

    labels = df[target_column].tolist()
    #labels = np.array([int(label) for label in labels])
    labels2 = []
    for label in labels:
        if label == zero_value:
            labels2.append(0)
        else:
            labels2.append(1)
    labels = np.array(labels2)

    data = df[input_features]

    for feature in input_features:
        if data[feature].dtype is not np.dtype(np.int64) and data[feature].dtype is not np.dtype(np.float64) and data[feature].dtype is not np.dtype(np.float32):
            data[feature] = data[feature].astype('category')

    cat_cols = data.select_dtypes(['category']).columns
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    data_disp, labels_disp = shap.datasets.adult(display=True)
    X_train_disp, X_test_disp, y_train_disp, y_test_disp = train_test_split(data_disp, labels_disp, test_size=0.3, random_state=42)

    xgc = xgb.XGBClassifier(n_estimators=500, max_depth=5, base_score=0.5,
                                objective='binary:logistic', random_state=42)
    xgc.fit(X_train, y_train)
    predictions = xgc.predict(X_test)


    fig = plt.figure(figsize = (16, 12))
    title = fig.suptitle("Default Feature Importances from XGBoost", fontsize=14)

    ax1 = fig.add_subplot(2,2, 1)
    xgb.plot_importance(xgc, importance_type='weight', ax=ax1)
    t=ax1.set_title("Feature Importance - Feature Weight")

    ax2 = fig.add_subplot(2,2, 2)
    xgb.plot_importance(xgc, importance_type='gain', ax=ax2)
    t=ax2.set_title("Feature Importance - Split Mean Gain")

    ax3 = fig.add_subplot(2,2, 3)
    xgb.plot_importance(xgc, importance_type='cover', ax=ax3)
    t=ax3.set_title("Feature Importance - Sample Coverage")

    #plt.savefig('static/explanations.png')

    explanation = eli5.explain_weights(xgc.get_booster())
    explanation_html = eli5.formatters.html.format_as_html(explanation)
    print(explanation_html)

    with open("templates/explanation.html","a+") as file:
        file.write(explanation_html)


    doc_num = 0
    print('Actual Label:', y_test[doc_num])
    print('Predicted Label:', predictions[doc_num])
    #eli5.show_prediction(xgc.get_booster(), X_test.iloc[doc_num],
    #                     feature_names=list(data.columns) ,show_feature_values=True)
    explanation2 = eli5.explain_prediction(xgc.get_booster(), X_test.iloc[doc_num],
                         feature_names=list(data.columns))
    explanation_html2 = eli5.formatters.html.format_as_html(explanation2)
    with open("templates/explanation.html","a") as file:
        file.write(explanation_html2)

    doc_num = 2
    print('Actual Label:', y_test[doc_num])
    print('Predicted Label:', predictions[doc_num])
    #eli5.show_predicon(xgc.get_booster(), X_test.iloc[doc_num], feature_names=list(data.columns) ,show_feature_values=True)
    explanation3 = eli5.explain_prediction(xgc.get_booster(), X_test.iloc[doc_num], feature_names=list(data.columns))
    explanation_html3 = eli5.formatters.html.format_as_html(explanation3)
    with open("templates/explanation.html","a") as file:
        file.write(explanation_html3)


    #target_names = ['$50K or less', 'More than $50K']
    interpreter = Interpretation(training_data=X_test, training_labels=y_test, feature_names=list(data.columns))
    im_model = InMemoryModel(xgc.predict_proba, examples=X_train)



    plots = interpreter.feature_importance.plot_feature_importance(im_model, ascending=True, n_samples=23000)

    plots[0].savefig('skater.png')


    features_pdp = input_features


    xgc_np = xgb.XGBClassifier(n_estimators=500, max_depth=5, base_score=0.5,
                            objective='binary:logistic', random_state=42)
    xgc_np.fit(X_train.values, y_train)


    # In[ ]:


    from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer

    exp = LimeTabularExplainer(X_test.values, feature_names=list(data.columns),
                               discretize_continuous=True)



    doc_num = 0
    print('Actual Label:', y_test[doc_num])
    print('Predicted Label:', predictions[doc_num])
    instance = exp.explain_instance(X_test.iloc[doc_num].values, xgc_np.predict_proba)
    instance.save_to_file('templates/lime.html',show_all=False)



    doc_num = 2
    print('Actual Label:', y_test[doc_num])
    print('Predicted Label:', predictions[doc_num])
    instance2 = exp.explain_instance(X_test.iloc[doc_num].values, xgc_np.predict_proba)
    instance2.save_to_file('templates/lime2.html',show_all=False)






    explainer = shap.TreeExplainer(xgc)
    shap_values = explainer.shap_values(X_test)
    pd.DataFrame(shap_values).head()




    #shap.force_plot(explainer.expected_value, shap_values[:,], X_test_disp.iloc[:,],show=False,matplotlib=True)
    #plt.savefig("static/force_plot.png")

    shap.summary_plot(shap_values, X_test, plot_type="bar",show=False)
    plt.savefig("static/summary_plot.png")



    shap.summary_plot(shap_values, X_test,show=False)
    plt.savefig("static/summary_plot2.png")

    return "Everyone Happy"
