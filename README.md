# Housing Price Prediction

Predicting King County house prices using Linear, Polynomial, and Ridge Regression. This project includes data cleaning, feature engineering, model training, and evaluation.

## Dataset
kc_house_data.csv — contains house features and sale prices in King County, WA.

## How to Run
1. Clone the repository:
git clone <your-repo-url>
2. Install required packages:
pip install -r requirements.txt
3. Run the script:
python housing_price.py

## Results
- Test R²: ~81%
- Average prediction error: ~$94,000
- Models evaluated: Linear Regression, Polynomial Regression, Ridge Regression

## Key Techniques
- Regression models: Linear, Polynomial, Ridge
- Feature engineering: new variables such as lat_long, bath_per_bed, lot_ratio_15
- Model evaluation: Cross-
