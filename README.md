# Housing Price Prediction

Predicting King County house prices using Linear, Polynomial, and Ridge Regression. This project includes data cleaning, feature engineering, model training, and evaluation.

## Files
- `housing_price.py` — Python script with the full code  
- `kc_house_data.csv` — dataset  

## How to Run
1. Download both files to the same folder.  
2. Install required packages (if not already installed):
```bash
pip install pandas numpy matplotlib scikit-learn
```
3. Run the Python script:
```bash
python housing_price.py
```

## Results
- Test R²: ~81%  
- Average prediction error: ~$94,000  

## Key Techniques
- Linear, Polynomial, and Ridge Regression  
- Feature engineering: lat_long, bath_per_bed, lot_ratio_15  
- Model evaluation: Cross-validation, MSE, RMSE, R²
