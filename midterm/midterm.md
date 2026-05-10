Predicting Customer Response to an Outreach Campaign
You will build and evaluate models to predict whether a customer accepts an offer (accepted_offer = 1 if accepted, 0 otherwise) using the provided midterm dataset.
Approach this as a notebook you would present to a potential employer. Your work should clearly show your workflow and decisions. Assume the reader understands machine learning but is not familiar with your dataset.
Your notebook should be readable and well-organized. You are not expected to write long narrative sections, but your code must be clearly annotated. Use brief Markdown explanations or comments to explain what you are doing and why. Avoid large blocks of unexplained code.
You are encouraged to document meaningful approaches that did not work, but keep this concise.

Your notebook must include the following labeled sections (please include these labels):
Exploratory Data Analysis: Include a small number of relevant summaries or plots and focus on insights that influenced your modeling.
Data Preparation: Show the preprocessing steps used for modeling.
Feature Engineering or Feature Selection: Evaluate whether feature engineering or feature selection improved or simplified your model.
Modeling and Evaluation: Build and evaluate at least two models that differ in a meaningful way. Models should be tuned beyond default values. Use an appropriate validation strategy and evaluation metric.
Ensembling: Evaluate whether combining models improves performance. The models you combine should differ in a meaningful way.
Results Summary: Summarize the models you evaluated and identify your final model. Include model name, key preprocessing or features, validation method, evaluation metric, validation score, and a brief reason for your final choice. Present this in a clear, structured format.
Final Model and Predictions: Train your final model and generate predictions for the test set. Do not use the test set for tuning.

Submit:
A link to your notebook in GitHub
A CSV file named lastname_firstname_predictions.csv with the format id,prediction where prediction is 0 or 1 and id matches the id column in midterm_test.csv.
Grading will be based on completion and quality of the required sections, clarity of your workflow, evidence-based model comparison, and the quality of your predictions. A portion of the grade will be based on predictive performance.