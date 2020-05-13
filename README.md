This web app classifiies text message into 36 types of disasters. Please follow the steps to install and open the webapp. You can input your messages and it will tell you which type of disaster it belongs to with necessary interactive plots. Enjoy!

# 1. Installation
nltk(3.3)
scikit-learn(0.19.1)
plotly(3.3.0)
The code should run with no issues using Python versions 3.*.

# 2. Project Motivation
Create a machine learning pipeline to categorize these events so that users can send the messages to an appropriate disaster relief agency.

# 3. File Descriptions
data/disaster_messages.csv data/disaster_categories.csv : original data
data/process_data.py: to run ETL pipeline that cleans data and stores in database
data/DisasterResponse.db: database that stores cleaned data
models/train_classifier.py: to run ML pipeline that trains classifier and saves
models/classifier.pkl: a pickle file which saves model
data/: a Flask framework for presenting data

# 4. Results
The web app shows visualizations about data.
The web app can use the trained model to input text and return classification results

# 5. How to run the web app?
Before run the web app:

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
To run ML pipeline that trains classifier and saves python ```models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
Run the following command in the app's directory to run the web app.

```python run.py```
```env|grep WORK```
Then go to ```https://SPACEID-3001.SPACEDOMAIN```

Enjoy!
