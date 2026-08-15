# 🏥 CMS Hospital Quality Analysis
This project analyzes publicly available hospital data from the Centers for Medicare & Medicaid Services (CMS) to examine hospital characteristics and quality performance across the United States. The project demonstrates an end-to-end healthcare analytics workflow. 


## 🤖 Technologies
- Python
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- SciPy
- CMS Data API
- Jupyter Notebook
- VS Code
- Git/GitHub


## 🧩 Features
What can you do with this project: 

- Automatically retrieve the latest CMS hospital dataset through the CMS API
- Clean and validate hospital quality data
- Explore hospital ratings across hospital types, ownership, states, and emergency services
- Create quality-performance features from CMS measure coutns
- Test relationships between hospital characteristics and high hospital ratings
- Train and compare multiple classification models
- Generate reusable visualizations and processed datasets


## ⏳ The Process
I started with the CMS Hospital General Information dataset, which contains information on Medicare-registered hospitals, including their location, hospital type, ownership, emergency services, overall rating, and several groups of quality measures.

Rather than downloading the dataset manually and treating it as a static file, I wanted the project to be reproducible. So I built the first part of the pipeline around the CMS Provider Data API. The pipeline retrieves the dataset's current download URL from the dataset metadata and downloads the raw CSV into the project.

The initial dataset contained:

- **5,419 hospitals**
- **38 variables**
- Hospital characteristics
- Overall hospital ratings
- Mortality measures
- Safety measures
- Readmission measures
- Patient experience measures
- Timeliness of care measures

The raw data is preserved in `data/raw/` so that the cleaning process can be reproduced from the original source.

Before transforming anything, I inspected the structure of the dataset. One of the first things I noticed was that several variables were stored as text even though they represented numeric quantities. For example, the number of measures a hospital performed better or worse on was represented as strings such as `"0"`, `"1"`, and `"Not Available"`.

There were also substantial amounts of missing data in some variables. That being said, I did not automatically treat every missing value as a data-quality problem. In this dataset, `"Not Available"` can represent a measure that was not available for a hospital rather than an error in the dataset. So I treated missingness as something to understand rather than something to blindly replace.

The cleaning pipeline standardizes the dataset and converts applicable quality measures into numeric values. I also added validation checks rather than assuming the transformation worked correctly.

For example, the CMS quality groups provide counts for:

- Total facility measures
- Measures where the facility performed better
- Measures where the facility performed no differently
- Measures where the facility performed worse

I checked whether these components reconciled with the total number of measures available for each hospital.

The integrity checks passed for:

- **MORT:** 4,096 records
- **Safety:** 3,344 records
- **READM:** 4,264 records

The cleaned dataset retained all 5,419 hospital records and contained no duplicate facility IDs.

The processed dataset is saved to: `data/processed/hospital_quality_clean.csv`

Once the data was cleaned, I wanted to understand what the hospitals in the dataset actually looked like before building a model.

The overall rating distribution showed that:

- 198 hospitals received a rating of 1
- 661 received a rating of 2
- 985 received a rating of 3
- 946 received a rating of 4
- 384 received a rating of 5
- 2,245 did not have an available overall rating

I also explored differences across hospital type, ownership, emergency services, and state. Here, I encountered what initially appeared to be an outlier in the hospital-type rating distribution. Rather than automatically removing it, I investigated the underlying records. The apparent outliers belonged to **Acute Care - Veterans Administration** hospitals and represented hospitals with a rating of 2 in a group where most available ratings were 4 or 5. Because these were legitimate hospital records rather than erroneous observations, I kept them in the dataset. That was an important reminder that an outlier is not automatically a mistake.

The raw quality measures provide counts, but I wanted variables that could be compared more directly across hospitals. 

So I created performance rates for three quality categories:

- Mortality
- Safety
- Readmission

For each category, I calculated the proportion of available measures on which a hospital performed:

- Better
- Worse

This resulted in six engineered variables:

- `mort_better_rate`
- `mort_worse_rate`
- `safety_better_rate`
- `safety_worse_rate`
- `readm_better_rate`
- `readm_worse_rate`

These features give the modeling stage a more interpretable representation of hospital quality performance.

At this point, I shifted from descriptive analysis to prediction. I defined a binary target representing whether a hospital had a high overall rating. 

The modeling dataset contained 3,174 hospitals with the information required for the analysis:

- 1,844 lower-rated hospitals
- 1,330 high-rated hospitals

I used a stratified train/test split so that the proportion of high-rated and lower-rated hospitals remained approximately consistent between the training and testing datasets.

The final split contained:

- 2,539 training records
- 635 testing records

Before adding quality measures, I wanted to answer a simpler question:

*How much can hospital characteristics alone tell us about whether a hospital has a high overall rating?*

I built a logistic regression model using:

- Hospital type
- Hospital ownership
- Emergency services
- State

This became the baseline model. Its ROC-AUC was 0.624, suggesting that hospital characteristics alone provided some predictive information, but there was considerable room for improvement.

The next question was whether the engineered quality measures actually added predictive value. I expanded the logistic regression model to include the mortality, safety, and readmission performance features. The result was a substantial improvement.

ROC-AUC increased from: 0.624 → 0.816

This suggested that quality performance contained considerably more information about hospital ratings than hospital characteristics alone.

The model also achieved:

- Accuracy: 0.729
- Precision: 0.684
- Recall: 0.658
- F1: 0.670

I didn't want to assume that logistic regression was the best approach, so I built a Random Forest model using the same modeling dataset.

The Random Forest achieved:

- Accuracy: 0.721
- Precision: 0.654
- Recall: 0.711
- F1: 0.681
- ROC-AUC: 0.792

The comparison revealed an interesting tradeoff. The logistic regression model achieved the highest ROC-AUC, while the Random Forest achieved higher recall and F1. Rather than declaring one model universally "best," I treated the result as dependent on what the model would ultimately be used for.

Next, I also to distinguish between predictive performance and simple association. I used chi-square tests to examine whether categorical hospital characteristics were associated with the high-rating outcome and calculated Cramer's V to describe the strength of those associations.

The analysis found evidence of an association for:

- Hospital type
- Hospital ownership
- State

There was insufficient evidence of an association between emergency services and high rating at the 0.05 significance level. These results provided additional context for interpreting the predictive models.

### Key Findings
- Quality measures substantially improved predictive performance compared with hospital characteristics alone.
- Logistic regression with quality measures achieved the highest ROC-AUC: 0.816
- Random Forest achieved the highest recall: 0.711
- Random Forest also achieved the highest F1 score: 0.681
- Hospital type, ownership, and state showed statistically significant associations with high ratings.
- Emergency service availability did not show sufficient evidence of association at the 0.05 significance level.
- Legitimate unusual observations were retained rather than automatically removed as outliers.


## 📚 What I Learned
During this project, a few things that stood out to me: 

- Data cleaning is part of the analysis. I learned that understanding the data comes before modeling it. Working with the CMS dataset included investigating missing values, looking into inconsistent formats, and validating calculated measures.
- Not all missing data should be fixed. A large portion of the dataset contained unavailable ratings and quality measures. Instead of filling or removing them, I learned to consider why the data was missing and how that could affect the analysis.
- Data exploration can raise better questions than answers. Looking at hospital ratings across hospital type, ownership, emergency services, and state helped identify patterns worth investigating further rather than jumping straight into modeling.
- Statistical significance needs context to make sense. The chi-square tests showed statistically significant association between some hospital characteristics and high ratings, the Cramer's V showed that statistical significance doesn't always mean the relationship is strong.
- Different models gives different things. Logistic regression gave me an interpretable way to understand how features were associated with high ratings. The random forest provided a different approach to prediction. Comparing the models helped me think beyond simply choosing whichever model had the highest accuracy.
- Each model has tradeoffs. The logistic regression model had the strongest ROC-AUC (0.816), while the random forest model achieved slightly higher recall (0.711) and F1 (0.681). There wasn't one model that was best at everything!


## 💭 Future Steps
There's plenty Of things I would like to explore with this project. Such steps include:

- Improve the modeling approach by testing more models.
- Address the class and sample-size differences across hospital types, ownership, and states to make comparison more robust.
- Look into furthering feature importance to better understand which quality measures contribute most to predicting higher ratings.
- Build a dashboard to make the findings more accessible and interactive.
- Keep on documenting! 
