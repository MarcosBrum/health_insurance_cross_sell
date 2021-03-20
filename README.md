# Health_insurance_cross_sell

Prediction of most probable customers to purchase a new _vehicle insurance_.

<img src="/images/car_insurance.jpg" height="413" width="400">


## 1 Business problem

A company that traditionally provides Health Insurance intends to offer its customers a new product, a Vehicle Insurance. In order to achieve this purpose efficiently, it gathered some information about their customers and asked if they would be interested in purchasing a new vehicle insurance. The features, together with their meanings, are:

* __id__: Unique ID for the customer.
* __Gender__: Gender of the customer.
* __Age__: Age of the customer
* __Driving License__: Whether the customer has a driving license, or not.
* __Region Code__: Unique code for the location of the customer.
* __Previously Insured__: Whether the customer already has a vehicle insurance, or not.
* __Vehicle Age__: Age of the vehicle.
* __Vehicle Damage__: Whether the vehicle has suffered damage in the past, or not.
* __Annual Premium__: The amount paied by the customer annually.
* __Policy Sales Channel__: Anonymized code for the channel of outreaching the customer.
* __Vintage__: Amount of time (in days) since the customer is associated with the company.
* __Response__: Whether the customer is interested in purchasing the vehicle insurance, or not.

With this information at hand, the company needs to define a data-driven marketing strategy to sell their new product. It hired a Data Science consulting and asked them to clarify a few issues:

1. Among all the features gathered, which show more evidence of intent of the customers to purchase the car insurance?

2. If the sales team is able to make 20.000 calls, which fraction of the interested customers will be reached?

3. If the sales team is now able to make 40.000 calls, which fraction of the interested customers will be reached?

4. How many calls does the sales team need to make in order to reach 80\% of the interested customers?

In order to answer these questions, the machine learning model (more information below) must inform the probability that each client will purchase the vehicle insurance, and the database must be sorted by this information. Having the clients with higher probabilities on top, the questions above will be addressed.


## 2 Business Results

The answers to the above questions are:

1. The most important features are: "age", "policy sales channel", "previously insured", "annual premium", "vintage", "vehicle hist", "gender".
	- The feature "vehicle hist" was created in the analysis process. It mixes the informations contained in the variables "vehicle damage" and "previously insured".
2. If the sales team is able to make 20.00 calls, it will reach 68.56\% of the interested customers.
3. If the sales team is now able to make 40.000 calls, it will reach 99.2\% of the interested customers.
4. In order to reach 80\% of the interested customers, the sales team must make 24.309 calls.

__Quantitative gain__: At the mark of 80\% of interested customers reached, the sales team would have contacted about 32\% of all the clients in the database, with a lift in gain of 2.5. This means that the cost in making the calls will be diminished to 40\%.


## 3 Business Assumptions

* The dataset presents customers from Pakistan. The curency presented by the source is Rs, Pakistani rupee.
* Every client in the database is above minimum driving age.
* For all customers, the vintage is less than one year.


## 4 Solution Strategy

The strategy adopted was the following:

__Step 01. Data Description__: I searched for NAs, checked data types (and adapted some of them for analysis) and presented a statistical description.

__Step 02. Feature Engineering__: New features were created to make possible a more thorough analysis.

__Step 03. Data Filtering__: Entries containing no information or containing information which does not match the scope of the project were filtered out.

__Step 04. Exploratory Data Analysis__: I performed univariate, bivariate and multivariate data analysis, obtaining statistical properties of each of them, correlations and testing hypothesis (the most important of them are detailed in the following section).

__Step 05. Data Preparation__: The dataset had to be balanced because only 12\% of the clients responded that they were interested in purchasing the new insurance. The balancing consisted in undersampling the majority class and oversampling the minority class. This step is necessary both for feature selection and for the machine learning models. Regarding the data types, numerical data was rescaled and categorical data was encoded.

__Step 06. Feature selection__: The statistically most relevant features were selected using the Boruta package. Alternatively, I performed the feature selection using the tree classifiers Extra Trees and Balanced Random Forest, obtaining the "feature importances" from both of them. In the next steps, the machine learning models trained using the features selected by Boruta presented a better generalizability performance.

__Step 07. Machine learning modelling__: Some machine learning models were trained. The one that presented best results after cross-validation went through a further stage of hyperparameter fine tunning to optimize the model's generalizability.

__Step 08. Model-to-business__: The models performance is converted into business values.

__Step 09. Deploy Model to Production__: The model is deployed on a cloud environment to make possible that other stakeholders and services access its results.


## 5 Top 3 Data insights

1. Elderly clients are less prone to purchase the vehicle insurance.
2. Female clients are less prone to purchase the vehicle insurance.
3. Clients whose vehicle has already suffered damage are more prone to purchase the vehicle insurance.


## 6 Machine Learning Models Applied

The following machine learning models were trained:
* Balanced Random Forest Classifier;
* Balanced Bagging Classifier;
* Easy Ensemble Classifier;
* Logistic Regression Classifier;
* Nearest Neighbors Classifier;
* Random Forest Classifier;
* Random Under Sampler Classifier;
* Stochastic Gradient Descent Classifier;
* XGBoost Classifier.

All of them were cross-validated.


## 7 Machine Learning Model Performance

The models "Random Forest Classifier" and "XGBoost Classifier" presented a better generalizability performance than the other models, but due to storage issues, the "XGBoost Classifier" was chosen. The most adequate graphs that exhibit the performance of the model in this ranking problem are the cumulative gain curve, the lift curve ahd the roc curve. These three curves are displayed below.

<img src="/images/model_performance.png" height="450" width="723">

The trained (cross-validated and fine tuned) model was also applied on a dataset of potential customers who did not participate in the initial poll. Therefore, this last dataset does not contain a "response" variable. In this case, the probability that each potential client will purchase the vehicle insurance is calculated, the dataset is sorted according to the probability and the sales team receives the sorted dataset to offer the vehicle insurance for the most prone individuales registered in the set.

The model was deployed to production as an application on Heroku. This app was linked to a Google Sheet containing the dataset through a script in Google Scripts that posts requests sending the samples of the dataset, receives the predicted probability and writes this result in an additional column on the sheet. A sample of the resulting dataset with the probabilities (unsorted) is in the image below. I remark the menu _"Health Insurance Prediction"_, added by the script, and the column "score", containing the probabilities.

<img src="/images/deploy_sheet.png" height="627" width="907">

## 8 Conclusions

The sales forecast and the generated insights provide the CEO with valuable tools to decide the amount of budget that is going to be dedicated to the restoration of each store.


## 9 Lessons Learned

* The exploratory data analysis provides important insights to the business problem, many of which contradict the initial hypothesis. This information is valuable for the understanding of business and for planning future actions. This step also provides a preview of the result of the feature selection step.
* The machine learning model performance must be evaluated in the learning and generalization stages. A balance between bias and variance must be achieved based on the uniqueness of the problem.


## 10 Next steps and improvements

Some hypothesis made when filling missing data would be reviewed in a following CRISP cycle, and other ones would be tested in the exploratory data analysis step. Also, other machine learning models would be employed (in particular, gradient boost models).

Besides, the model is deployed to production in an App at Heroku. One can send a request from an external application (such as Postman, for example). The app receives a JSON file and returns the sales forecast for the following six weeks (the amount is displayed in the Brazilian currency BRL). This app is also receiving requests from another app hooked to a Telegram Bot. In this case, one must only pass the number of the store to the Bot to obtain the sales forecast. This second App is also hosted at Heroku.
