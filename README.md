# Sunflower Oil Sales Forecasting System

🎯 **Business Impact**: This machine learning solution successfully helped a B2B supply chain company achieve accurate sales forecasting, enabling better inventory planning and significant cost savings through optimized order management.

## Project Overview

A production-ready XGBoost forecasting system that predicts daily sunflower oil sales with high accuracy. Built for real-world B2B supply chain operations using 7+ years of historical data (2017-2024).

**Key Achievement**: Delivered excellent prediction scores that directly improved business planning and inventory management.

## 💼 Business Value Delivered

- ✅ **Improved Order Planning**: Accurate demand forecasts enabled optimal inventory levels
- ✅ **Cost Reduction**: Minimized overstock and stockout situations
- ✅ **Risk Management**: Successfully incorporated external factors (COVID-19, war impacts, seasonal trends)
- ✅ **Operational Efficiency**: Automated forecasting process replaced manual estimation

## 🚀 Technical Implementation

### Core Features
- **Advanced ML Pipeline**: XGBoost with Bayesian hyperparameter optimization
- **Smart Feature Engineering**: 30+ features including lags, rolling statistics, seasonal indicators
- **External Factor Integration**: COVID periods, war impacts, pricing data, geographic factors
- **Robust Validation**: Time series split and expanding window validation methods

### Project Structure
```
├── main.py                     # Main execution pipeline
├── data_preprocessing.py       # Data cleaning and feature creation
├── feature_engineering.py     # Time series features and validation
├── xgboost_model.py           # Model training and optimization
├── visualization.py           # Business insights and trend analysis
├── periods_costs.py           # External factors and cost mapping
└── model_hyperparameters.py   # Optimized model configuration
```

## 📊 Key Features

**Data Processing**:
- Historical sales data with daily granularity
- COVID-19, war period, and seasonal pattern mapping
- Geographic classification (coastal vs mountain regions)
- Dynamic pricing and cost integration

**Machine Learning**:
- XGBoost regression with GPU acceleration
- Bayesian optimization for hyperparameter tuning
- Multiple validation strategies to prevent overfitting
- Feature importance analysis for business insights

## 🛠️ Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn

# Run the forecasting system
python main.py

# Choose validation method and optimization preferences
# System will train model and provide performance metrics
```

## 📈 Performance Metrics

The model evaluates success using:
- **MAE (Mean Absolute Error)**: Average prediction accuracy
- **R² Score**: Model fit quality
- **Business KPIs**: Inventory optimization and cost reduction

## 🎯 Real-World Impact

This system proved its value by:
1. **Replacing Manual Forecasting**: Automated accurate predictions
2. **Enabling Data-Driven Decisions**: Clear insights into demand drivers
3. **Improving Supply Chain Efficiency**: Better inventory turnover rates
4. **Managing External Shocks**: Successfully adapted to COVID-19 and geopolitical events

## 🔧 Technical Highlights

- **Production Ready**: Modular design for easy deployment and maintenance
- **Scalable Architecture**: Can adapt to other commodity forecasting needs
- **Comprehensive Analysis**: Includes visualization tools for business stakeholders
- **Optimized Performance**: GPU-accelerated training with automated hyperparameter tuning

---

**Result**: A proven ML solution that transformed supply chain planning from reactive to predictive, delivering measurable business value through accurate demand forecasting.
