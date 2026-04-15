# Kaggle Homework - Sahil - sahilbainz

## 1. Helpful Notebooks & Discussion Posts

1. **[Irrigation HGBC, XGB, LGBM, CatB, RealMLP Baseline](https://www.kaggle.com/code/kospintr/irrigation-hgbc-xgb-lgbm-catb-realmlp-baseline)**
   - I chose this notebook because it provides a really good benchmark using multiple different gradient boosting frameworks (XGBoost, CatBoost, LightGBM). Having a clear baseline like this makes it easier to test my own models and see if I'm actually making progress.

2. **[PS6E4 XGB cuDF Pseudo Labels](https://www.kaggle.com/code/include4eto/ps6e4-xgb-cudf-pseudo-labels)**
   - I bookmarked this one because it leverages RAPIDS/cuDF to speed up pandas operations significantly. Also, using pseudo-labels from the test set is a strategy I haven't implemented yet, and I'm planning to try it in Phase 2 to get a competitive edge.

3. **[PS S6E4 CatBoost Pipeline](https://www.kaggle.com/code/rohit8527kmr7518/ps-s6e4-catboost-pipeline)**
   - I liked this notebook because CatBoost usually handles categorical features really well out of the box without needing extensive one-hot encoding. This will help simplify my data prep pipeline when I build an ensemble later.

---

## 2. Exploratory Data Analysis (EDA)

**Link to my EDA Notebook:** [hw2_eda.ipynb](./hw2_eda.ipynb)

**Insights from the EDA:**
- The biggest issue with this dataset is the severe class imbalance. The `Irrigation_Need` target variable is mostly 'Low' (~58.7%) with 'High' occurring very rarely (~3.3%). 
- Because of this imbalance, I realized I can't just rely on default accuracy. This explains why the competition uses Balanced Accuracy as the main metric. 
- Using Sweetviz (and plotting the distributions), I saw that some of our numeric features like Rainfall and Soil Moisture have interesting splits that our models can hopefully catch.

---

## 3. Modeling

**Link to my Modeling Notebook:** [hw2_analysis.ipynb](./hw2_analysis.ipynb)

**My Process:**
I followed the basic machine learning cycle we used in InClassAssignment 1 & 2. 
First, I dropped the ID columns and used `get_dummies()` for categorical variables and `LabelEncoder` for the target. Then I scaled the continuous features with `MinMaxScaler`.
I intentionally focused on applying weights through `compute_sample_weight` in sklearn because of the class imbalance, ensuring the models penalize mistakes on the rare 'High' class.

**Performance:**

| Model | CV Balanced Accuracy | Parameters Used | Leaderboard Score |
|-------|----------------------|-----------------|-------------------|
| Random Forest (Bagging) | 0.9566 | `n_estimators=100`, `class_weight='balanced'` | *[.96109]* |
| XGBoost (Boosting) | 0.9715 | `learning_rate=0.1`, `max_depth=5`, `sample_weight` | *[.96604]* |

**Bagging vs. Boosting Comparison:**
- The weights definitely helped both models recognize the 'High' instances instead of predicting 'Low' every time.
- **Random Forest (Bagging)** performed solidly and was very fast to train, but its recall for the minority 'High' class hovered around 90%. It missed some of the harder-to-classify edge cases.
- **XGBoost (Boosting)** outperformed the Random Forest significantly. Because boosting sequentially learns from its past errors (residuals), it managed to push the recall for the tricky 'High' instances up to 95%, pulling the overall balanced accuracy to 0.9715.

---

## 4. Phase 2 Plan

To improve my models in the next phase, my plan is:
1. Try out CatBoost, referring back to the pipeline notebook I saved, so that I can handle all the categorical features more natively.
2. Build an ensemble or stacking classifier that combines the predictions of XGBoost and CatBoost.
3. Incorporate pseudo-labeling on the test data observations to squeeze out a bit more predictive signal.
