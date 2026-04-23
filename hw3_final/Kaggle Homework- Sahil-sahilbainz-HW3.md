# Kaggle Homework - Sahil - sahilbainz (HW3 - Gradient Boosting)

## 1. Gradient Boosting Notebook
**Link to my Notebook:** [hw3.ipynb](./hw3.ipynb)

---

## 2. Modeling Approach

For this assignment, I focused on deep-tuning two powerhouse Gradient Boosting algorithms: **XGBoost** and **LightGBM**. Given the ordinal nature of the irrigation targets (Low < Medium < High), I explored two distinct strategies:

1. **Ordinal Regression Attempt:** I initially tried using the Regressor versions of both models, optimizing for Mean Squared Error to respect the ordering of the classes. While this provided an interesting mathematical perspective, I found it caused a slight drop in raw accuracy (around 0.95) when rounding the continuous outputs.
2. **Multi-class Classification:** I ultimately found that using the standard Classifier approach with a `multiclass` objective was the most effective strategy for the Kaggle leaderboards, reaching a validation accuracy of **0.98+**.

To handle the severe class imbalance, I utilized `compute_sample_weight('balanced')` across every trial, which ensured that the models did not ignore the rare 'High' irrigation need cases.

---

## 3. Hyperparameter Tuning Trials

I explored four distinct sets of hyperparameters for each model to see how they influenced the results:

| Model | Tuning Method | Validation Accuracy | Leaderboard Score |
| :--- | :--- | :--- | :--- |
| **XGBoost** | Baseline (Default) | 0.981 | [Pending] |
| **XGBoost** | GridSearchCV | 0.982 | [Pending] |
| **XGBoost** | Optuna | 0.984 | [Pending] |
| **LightGBM** | Baseline (Default) | 0.980 | [Pending] |
| **LightGBM** | GridSearchCV | 0.981 | [Pending] |
| **LightGBM** | Optuna | 0.983 | [Pending] |

**What worked well:**
- **Optuna Tuning:** This was the most meaningful improvement. By fixing a learning rate bug I had in my first run (switching it from an integer search to a float search), the optimizer was finally able to find the "sweet spot" rather than stagnating at 0.
- **Floating-point Learning Rates:** Moving the learning rate from 0.1 to smaller values with more estimators significantly reduced overfitting on the training set.

**What didn't work well:**
- **Regression objective:** As mentioned, trying to map classes to a regression line felt intuitive for "Low/Medium/High" but performed slightly worse on the hard class boundaries compared to the raw softmax probability of a classifier.

---

## 4. Model Comparison (LightGBM vs. XGBoost)

How did the models compare in my work?

- **Training Speed:** **LightGBM** was the clear winner here. It completed its Optuna trials significantly faster than XGBoost. This speed allowed me to run more iterations in the same amount of time, which actually resulted in a more refined model.
- **Complexity Control:** I used `gamma` for XGBoost and `reg_alpha` (L1) for LightGBM. Both were effective at controlling tree complexity, but LightGBM's leaf-wise growth required more careful depth limiting to avoid overfitting on the large dataset.
- **Overall Performance:** Both models reached a similar accuracy ceiling (~0.98), but LightGBM felt more efficient for a fast-paced Kaggle environment. 

---

## 5. Summary Takeaway
Gradient boosting is incredibly sensitive to the learning rate and the interaction between tree depth and estimator count. By aligning both models to a robust classification pipeline and using automated tuning (Optuna), I was able to squeeze out those final fractions of accuracy that the baseline models missed.
