# Kaggle Homework - Sahil - sahilbainz (HW4 - Ensembles & Feature Engineering)

## 1. Homework 4 Notebook
**Link to my Notebook:** [hw4.ipynb](./hw4.ipynb)

---

## 2. Modeling & Feature Engineering Approach

For this assignment, I focused on creating diverse, complementary models to power a robust ensemble, while explicitly generating new synthetic features to capture physical dynamics missed by the raw data.

**Models Used:**
1. **LightGBM:** Continued from my best results in HW3. Uses a leaf-wise growth strategy.
2. **CatBoost:** Introduced as an alternative gradient boosting algorithm that builds symmetric trees and naturally handles categoricals well, differing meaningfully from LightGBM.
3. **Gaussian Naive Bayes:** Chosen specifically because it relies on probabilistic conditional independence, providing a completely different mathematical approach compared to the tree-based models.

**Feature Engineering:**
1. **Interaction (`Temp_Humidity_Index`):** Multiplied Temperature by Humidity. Irrigation depends heavily on the combination of heat and air moisture (evapotranspiration).
2. **Transformations (`Aridity_Proxy`):** Created a ratio of Temperature to Rainfall. Fields that are hot with little rainfall will inherently face high water stress.
3. **Grouping (`Crop_Mean_Rainfall_Diff`):** Calculated the average rainfall across each specific crop type, then took the difference for the current row. This measures if a specific field is experiencing anomalous conditions relative to its peers.

---

## 3. Results and Ensemble Evaluation

I aggregated these models using two ensemble techniques: **Soft Voting (Probability Averaging)** and **Stacking** (using a Logistic Regression meta-model).

| Model Approach | Validation Accuracy | Estimated LB Score |
| :--- | :--- | :--- |
| **LightGBM (Base)** | ~0.983 | - |
| **CatBoost (Base)** | ~0.982 | - |
| **Naive Bayes (Base)** | ~0.760 | - |
| **Voting Ensemble (Soft)** | ~0.984 | - |
| **Stacking Classifier** | **~0.986** | **0.95812** |

### Discussion

* **What worked well:** The Stacking Classifier significantly outperformed the individual baseline models. Because Naive Bayes makes very different types of errors compared to the heavily correlated tree models (LightGBM/CatBoost), the Logistic Regression meta-model was able to learn *when* to trust the tree models and *when* to listen to the probabilistic baseline, resulting in a measurable performance bump. Furthermore, checking feature importances revealed that the custom `Temp_Humidity_Index` was consistently used by LightGBM to split upper nodes.
* **What didn't work well:** The Naive Bayes model on its own is quite weak compared to gradient boosting. Because of this, assigning it an equal weight in a standard probability average (Soft Voting) almost dragged the ensemble down. Furthermore, looking at the Public Leaderboard, our Stacking Classifier scored **0.95812**, which is actually slightly *lower* than the raw HW3 XGBoost and LightGBM baseline submissions (~0.965-0.966). This implies that stacking the models and including a weak learner like Naive Bayes may have led to slight overfitting on the training distribution, causing it to generalize less effectively to the unseen Kaggle test set.
* **Moving Forward:** Moving forward, I will exclusively use Stacking rather than manual Voting when integrating diverse models. The meta-learner is far superior at mathematically determining the optimal weights for weak vs. strong models compared to manual tuning. I will also continue engineering grouped statistical features (like `Crop_Mean_Rainfall_Diff`) as they provide powerful context to the algorithms.
