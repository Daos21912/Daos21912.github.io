---
layout: post
title: Build Week Project
subtitle: Using drivendata.org's competition data to predict earthquake damage
tags: [test]
comments: true
---

For this project I've taken up the challenge on drivendata.org about predicting damage to a building from an earthquake. The damage is categorized into minimal, substantial, and complete destruction. The competition can be found [here](https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/)

The data from the competition came already quite clean, and so there was minimal wrangling. There was a small issue converting the code into colab, but only because I tried to link directly to the data from my google drive. Once uploaded into the colab notebook the files worked just fine.

Once the data was imported sucessfully, I jumped right into model building. The classes for my target, damage_grade, weren't quite balanced. There were three of them, but the highest frequency class occured over 54% of the time. Even so, I believed this was good enough for model building.

**XGBoostClassifier**

I started with an XGBoostClassifier model, in a simple pipeline with just an ordinal encoder. I used this as there were several columns that were encoded as characters. I started by checking the permutation importance. There was a small increase in model performance, so I kept the newer model and moved on to hyperparameter tuning with the following code:
~~~
clf = make_pipeline(
            OrdinalEncoder(),
            XGBClassifier(random_state=42, n_jobs=-1)
)
params_grid = {
    'xgbclassifier__learning_rate':np.arange(0.1, 1.0, 0.1),
    'xgbclassifier__n_estimators':range(50, 150, 10)
}

model = RandomizedSearchCV(
    clf, 
    param_distributions=params_grid,
    n_jobs=-1,
    n_iter=30,
    cv=5,
    verbose=1
    )
model.fit(X_train.drop(columns=cols_to_remove), y_train)
~~~
This yielded a model with a fairly significant increase in the validation score, but it was still below 80%. I wanted a better model, but I also wanted more information, so I plotted a shap waterfall. This ended up not yielding much information.

**Random Forest**

I moved on to the RandomForestClassifier class, and fit a standard model with the same type of pipeline. This yielded a massively overfit model, but with a promising validation score, similar to what the XGBoost classifier yielded. I checked the permutation importances with the code below, but found no improvement with the model afterward.
~~~
perm_imp = permutation_importance(model_rf, X_val, y_val, random_state=42)
data = {'imp_mean':perm_imp['importances_mean'],
        'imp_std':perm_imp['importances_std']}
df_perm = pd.DataFrame(data, index=X_val.columns).sort_values('imp_mean')
df_perm

cols_to_remove = df_perm[df_perm['imp_mean'] <= 0].index
model_rfdc = make_pipeline(
    OrdinalEncoder(),
    RandomForestClassifier(random_state=42, n_jobs=-1)
)
model_rfdc.fit(X_train.drop(columns=cols_to_remove), y_train)
~~~

Afterward, I tried a similar method for hyperparameter tuning, using a randomized search CV for my random forest model, and came away with training and test accuracies of almost 93%. I also wanted to look at precision and recall, so I used the classification_report class. This yielded the results below:

| class | precision | recall | f1-score |
| :------ |:--- | :--- | :--- |
| 1 | 0.99 | 0.86 | 0.92 |
| 2 | 0.91 | 0.97 | 0.94 |
| 3 | 0.95 | 0.87 | 0.91 |

As seen in the chart above, the precision scores were good, but the recall scores could have been better. Overall though, I believed I had achieved the best model I could with the data presented.
