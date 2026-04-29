Activity:
We will build on the feature engineering work from last class. There are two demo notebooks that focus on reducing complexity, understanding model behavior, and improving predictions.

You may use the provided dataset with 377 engineered features or create your own feature set. Starting from a large number of features, you should reduce the set to 20 or fewer features. Use the tools we discussed to help you in this work.
Train at least two models that differ in a meaningful way, such as model type or tuning choices, and use your reduced feature sets for these models.
Tune your models beyond their default settings. There is no fixed number of parameters you must tune, but you should make a reasonable effort to improve performance and demonstrate that your tuning choices had an effect.
Combine your models using stacking by implementing a meta learner that uses out-of-fold predictions, and you should compare the performance of your base models with the stacked model.
In a markdown cell at the end, evaluate your results by describing how performance changed as you reduced features, which features were consistently important, which features you removed and why, whether stacking improved performance, and what you learned about how your model behaves.
It is expected that this activity will take 1-2 hours total. It is not meant to be an exhaustive exploration. 