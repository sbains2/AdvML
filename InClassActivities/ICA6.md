We will discuss the provided demo notebooks together in class. The notebook is simplified so it runs during class time. During/after class, you should create your own notebook that builds a Naive Bayes model using the Kaggle competition irrigation data.

Create a new notebook for this assignment. 
Build a Gaussian Naive Bayes model to predict irrigation need using the data for the Kaggle playground series competition S6E4. You may use your own existing data preprocessing pipeline or adapt from the in-class adult notebook. You should train and test using only the Kaggle train data. 
Generate predicted probabilities using your model.
Create baseline predictions using the default rule (choose the class with the highest probability).
Choose one class to focus on (preferably a rare or important class) and select one evaluation metric (e.g., F1, recall, or balanced accuracy).
Apply a threshold to your chosen class by deciding what probability is “high enough” to assign that class. Generate a second set of predictions using this threshold.
Using a markdown cell, discuss which class you selected, what threshold you chose, how the metric changed, what tradeoff you observed, and how Naive Bayes compares to your existing model(s).  You should use a classification report to assist with this discussion.
It is expected that this activity will take no more than one hour total. You are not required to generate predictions for the Kaggle test data (use only the Kaggle train data) or to submit those predictions to the Kaggle competition.