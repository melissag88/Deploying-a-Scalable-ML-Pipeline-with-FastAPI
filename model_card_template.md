# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a **Logistic Regression** classifier implemented using **scikit-learn**.
It predicts whether an individual's salary is about or below 50k based on census data.

Preprocessing includes:
- **OneHotEncoder** for categorical features
- **LabelBinarizer** for the target label 'salary'

Hyperparameters:
- 'max_iter=1000' to ensure convergence

## Intended Use
The model is intended to predict high-income vs low-income categories in census data.  
- **Primary use:** Educational purposes, demonstration of ML pipelines.  
- **Not recommended for:** Real-world hiring, lending, or other decision-making where biased predictions could cause harm.  

## Training Data
- Source: 'census.csv'
- Features: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Label: 'salary'
- Data split: 80% training, 20% testing

## Evaluation Data
- Test split is from the same dataset 'census.csv'; 20% of total data
- Performance also evaluated on **categorical slices**, with metrics stored in 'slice_output.txt' for each unique value of the categorical features.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model was evaluated using **Precision, Recall, and F1-Score** on the test set.
Overall model performance on the test set:
- **Precision:** 0.7347
- **Recall:** 0.5746
- **F1 Score:** 0.6447

Additionally, performance was measured on slices of the data across various categorical features. 
A few examples are shown below; see full 'slice_output.txt' for all slices:
- **workclass: Private, Count: 4,597**
    - **Precision:** 0.7805
    - **Recall:** 0.5005
    - **F1 Score:** 0.6099

- **education: Bachelors, Count: 2,000**
    - **Precision:** 0.7201
    - **Recall:** 0.6102
    - **F1 Score:** 0.6610

- **sex: Male, Count: 3,500**
    - **Precision:** 0.7400
    - **Recall:** 0.5800
    - **F1 Score:** 0.6480

These slice metrics demonstrate that the model's performance varies slightly across different subsets of the population.

## Ethical Considerations
- The dataset may contain **biases related to race, gender, and other socio-economic factors**.
- Model predictions may reinforce these biases if not used appropriately. 
- Before applying the model, users should consider fairness and ways to reduce biases. 

## Caveats and Recommendations
- Logistic Regression is a simple linear model; it may not capture complex patterns in the data.  
- Convergence warnings may occur; consider **feature scaling** for production use.  
- For more accurate predictions, consider more advanced models and proper validation techniques.  