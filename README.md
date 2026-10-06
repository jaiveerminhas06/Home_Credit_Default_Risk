# Home Credit Default Risk

Predicting whether a loan applicant will have trouble repaying, using the data from Kaggle's [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition. The project is a single Jupyter notebook that goes from exploratory analysis to a LightGBM model with hand-built features from applicants' credit bureau history.

**Best result:** ROC-AUC of **0.764** on a 20% hold-out split, up from 0.691 for the logistic regression baseline.

## The problem

Home Credit lends to people with little or no credit history. Each row in the main table is one loan application, and the target is binary:

- `TARGET = 0`: the loan was repaid on time
- `TARGET = 1`: the client had payment difficulties

The classes are heavily imbalanced (about 8% of applications are defaults), so models are compared on ROC-AUC rather than accuracy.

## Data

The data files are included in the repository as zipped CSVs, and the notebook reads them directly without unzipping.

| File | Rows | Used in notebook | Description |
| --- | --- | --- | --- |
| `application_train.csv.zip` | 307,511 | Yes | One row per loan application, 122 columns including `TARGET` |
| `bureau.csv.zip` | 1,716,428 | Yes | Applicants' previous credits at other institutions, linked by `SK_ID_CURR` |
| `application_test.csv.zip` | 48,744 | Not yet | Kaggle test applications (no target) |
| `bureau_balance.csv.zip` | 27,299,925 | Not yet | Monthly balance history for each bureau credit |

## Approach

### Part 1: application data only

1. **Exploration.** Checked the target distribution, column types and missing values (64 columns have missing data, some close to 70%).
2. **Cleaning.**
   - Dropped the 4 rows where `CODE_GENDER` is `XNA`.
   - `DAYS_EMPLOYED` contains the placeholder value `365243` for about 55,000 applicants. These were replaced with `NaN`, and a `DAYS_EMPLOYED_ANOM` flag was added because this group defaults less often (5.4% vs 8.7%).
3. **Encoding.** Label encoding for categorical columns with two categories and one-hot encoding for the rest, giving 241 features.
4. **Analysis.**
   - `EXT_SOURCE_1/2/3` have the strongest (negative) correlation with default.
   - Default rate falls steadily with age, from about 12% for ages 20–25 to under 8% for ages 40–45.
5. **Feature engineering.** Degree-3 polynomial features from `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` and `DAYS_BIRTH`.
6. **Models.** Median imputation and min-max scaling, then logistic regression, random forest and LightGBM on an 80/20 split.

### Part 2: adding credit bureau history

`bureau.csv` has many rows per applicant, so it is aggregated to one row per `SK_ID_CURR` before merging:

- **Previous loan count** per applicant.
- **Numeric aggregates:** `count`, `mean`, `max`, `min` and `sum` of every numeric column (60 features).
- **Categorical aggregates:** count and normalised count for each category of `CREDIT_ACTIVE`, `CREDIT_CURRENCY` and `CREDIT_TYPE` (46 features).

This takes the feature set from 241 to 348 columns. Imputation and scaling are fitted on the training split only, and LightGBM is trained with balanced class weights and early stopping on validation AUC.

## Results

All scores are ROC-AUC on the 20% hold-out split (`random_state=42`).

| Model | Features | ROC-AUC |
| --- | --- | --- |
| Logistic regression (`C=0.0001`) | Application data | 0.6906 |
| LightGBM (10,000 rounds, no early stopping) | Application data | 0.7071 |
| Random forest (100 trees) | Application data | 0.7107 |
| Random forest (100 trees) | Application data + polynomial features | 0.7167 |
| **LightGBM (early stopping, best at round 326)** | **Application data + bureau aggregates** | **0.7639** |

The bureau features made the biggest difference. Among them, `bureau_DAYS_CREDIT_mean` (how recently the applicant's previous credits were opened, on average) has a stronger correlation with default than any raw application feature other than the three `EXT_SOURCE` scores.

## Repository structure

```
.
├── home_credit_default_risk (6).ipynb   # Full analysis and modelling notebook
├── application_train.csv.zip
├── application_test.csv.zip
├── bureau.csv.zip
├── bureau_balance.csv.zip
└── README.md
```

## Running it

Requires Python 3 with Jupyter. The notebook was last run on Python 3.13 with LightGBM 4.6.

```bash
git clone https://github.com/jaiveerminhas06/Home_Credit_Default_Risk.git
cd Home_Credit_Default_Risk
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm jupyter
jupyter notebook "home_credit_default_risk (6).ipynb"
```

Run the cells from top to bottom.

## Limitations and next steps

- Scores come from a single hold-out split of the training data. Cross-validation and a Kaggle submission using `application_test.csv` would give a more reliable number.
- `bureau_balance.csv` is not used yet, and the other competition tables (previous applications, instalment payments, credit card and POS balances) are not included.
- The first LightGBM model runs all 10,000 boosting rounds with no early stopping, which likely explains why it scores below the random forest. The Part 2 model uses early stopping.
- `SK_ID_CURR` is an identifier and should be dropped from the feature set.
- LightGBM hyperparameters have not been tuned.

## Acknowledgements

Data provided by [Home Credit Group](https://www.homecredit.net/) through Kaggle.
