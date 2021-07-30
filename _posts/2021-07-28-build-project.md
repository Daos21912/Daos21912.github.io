---
layout: post
title: Earthquake damage
subtitle: Using drivendata.org's competition data to predict earthquake damage to a building
tags: [test]
comments: true
---

## Introduction

For this project I've taken up the challenge on drivendata.org about predicting damage to a building from an earthquake. The damage is categorized into minimal, substantial, and complete destruction. The competition can be found [here](https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/)

**Data Wrangling**

The data from the competition came already quite clean, and so there was minimal wrangling. All of the data was already encoded to some degree, the location data was split into three continuous columns, and even things like building materials were either character encoded, or one-hot encoded. The shape of the dataframe, once correctly imported was 260601 rows, 39 columns. Once the data was imported sucessfully, I plotted a heatmap, just to make sure there was no data leakage:

![heatmap](https://user-images.githubusercontent.com/84862112/127568185-0e8ebdf0-d185-4939-9eb6-07d675b9eb64.png)


After I confirmed there was no issues with the data, I jumped right into model building. The classes for my target, damage_grade, weren't quite balanced. There were three of them, but the highest frequency class occured over 54% of the time. Even so, I believed this was good enough for model building. The distribution of the target class is as follows:

| Class | Percent Dist |
| :------ | :--- |
| 1 |   0.095712 |
| 2 |   0.569705 |
| 3 |   0.334584 |

## Model Building

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

![Shap waterfall XGB](https://user-images.githubusercontent.com/84862112/127565017-7d994020-5125-497f-b37d-d9e63dc22182.PNG)

I also looked at a classification report for this model, and while precision was close to 0.70 for each class, recall was closer to 0.5 for classes 1 and 3. 

**Random Forest**

I moved on to the RandomForestClassifier class, and fit a standard model with the same type of pipeline. This yielded a massively overfit model, but with a promising validation score, similar to what the XGBoost classifier yielded. I checked the permutation importances with the code below, but found no improvement with the model afterward.
~~~
perm_imp = permutation_importance(model_rf, X_val, y_val, random_state=42)
data = {'imp_mean':perm_imp['importances_mean'],
        'imp_std':perm_imp['importances_std']}
df_perm = pd.DataFrame(data, index=X_val.columns).sort_values('imp_mean')

cols_to_remove = df_perm[df_perm['imp_mean'] <= 0].index
model_rfdc = make_pipeline(
    OrdinalEncoder(),
    RandomForestClassifier(random_state=42, n_jobs=-1)
)
model_rfdc.fit(X_train.drop(columns=cols_to_remove), y_train)
~~~

Afterward, I tried a similar method for hyperparameter tuning, using a randomized search CV for my random forest model. This time I used max_depth and n_estimators as my hyperparameters. I came away with training and test accuracies of almost 93%. I also wanted to look at precision and recall, so I used the classification_report class. This yielded the results below:


|         | precision  |  recall | f1-score | support |
| :------ | :--- | :--- | :--- | :--- |
|           1 |      0.99   |   0.86  |    0.92  |    5170 |
|           2 |      0.91   |   0.97  |    0.94  |   29487 |
|           3 |      0.95   |   0.87  |    0.91  |   17464 |
|    accuracy |             |         |    0.93  |   52121 |
|   macro avg |      0.95   |   0.90  |    0.92  |   52121 |
|weighted avg |      0.93   |   0.93  |    0.93  |   5212  |

As seen in the chart above, the precision scores were good, but the recall scores could have been better. Overall though, I believed I had achieved the best model I could with the data presented. I tried a shap waterfall plot, but this didn't give me any more information than the one for my XGBoost model did. I did find the ROC-AUC score, thought I wasn't able to plot a curve as my classification problem wasn't binary. The score was pretty good at 0.85. I also plotted a normalized confusion matrix, shown here:

![Confusion matrix RF](https://user-images.githubusercontent.com/84862112/127567599-0d761066-d736-4299-b5de-3b6f41383e64.PNG)

As shown, the classes 1 and 3 are fairly underrepresented, but that's the way it was in the recorded data as well.

## Conclusions

The final model produced good predictions. With average precision and recall over 90, and an accuracy of 93%. Looking to permutation importances, we can see our most impactful features here:

![RF Perm Imp](https://user-images.githubusercontent.com/84862112/127662667-1b814377-2ebd-43b5-87a5-749336ddcf0f.png)

Unfortunately, almost all of these are encoded, so we're not sure what exactly they mean. although the columns starting with 'has_superstructure' are one-hot encoded, the other top 8 are all encoded in a different way. The columns starting with 'geo_level' are the location data, but it's all encoded to protect the privacy of the building owners, and even foundation type is encoded as well. We can see, however, that these three data points are some of the most impactful in the model. To further improve the model, I would like to see data relating to the distance the building is from the epicenter, and the strength of the earthquake. While the structure of the building certainly plays a part in the level of damage it recieves, the encoded location data seems to play a bigger part. There's also no data about the strength of the earthquake or earthquakes involved. This is information I'd like to see.

The competition used the micro-averaged F1 score as their metric. For my best model, the tuned random forest, the result for this metric on my validation set was 91%. This puts my model in a good spot for the competition, though I haven't yet recieved my score for the submission.
