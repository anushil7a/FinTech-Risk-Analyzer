# FinTech Risk Analyzer 🚀

A comprehensive web application that analyzes financial transactions, assigns risk scores, and flags anomalies using Machine Learning and rule-based systems.

## 🎯 Project Overview

**Goal**: Build a web app that analyzes financial transactions, assigns risk scores, and flags anomalies using ML + rules.

**Tech Stack**: 
- **Backend**: Flask (API)
- **Frontend**: Angular (UI) 
- **Database**: Snowflake/SQLite (DB)
- **ML**: scikit-learn, XGBoost, LightGBM
- **Deployment**: Docker

**Outcome**: Showcases full-stack, data engineering, and AI skills with a finance use-case.

## 🏗️ Architecture

```
FinTech Risk Analyzer/
├── api/                 # Flask backend API
├── frontend/           # Angular frontend
├── models/             # ML models and training scripts
├── scripts/            # Data generation and utility scripts
├── data/               # Data files and databases
├── tests/              # Test suite
└── docs/               # Documentation
```

## 🚀 Quick Start

### 1. Generate Sample Data

```bash
# Install dependencies
pip install -r requirements.txt

# Generate fake transaction data
python scripts/make_fake_data.py
```

This will create:
- `data/transactions.csv` - Full dataset (50,000 transactions)
- `data/transactions_sample.csv` - Sample dataset (1,000 transactions)

### 2. Run the Application

```bash
# Start Flask backend
cd api
python app.py

# Start Angular frontend (in another terminal)
cd frontend
ng serve
```

## 📊 Data Schema

The generated transaction data includes:

| Field | Type | Description |
|-------|------|-------------|
| `transaction_id` | String | Unique transaction identifier |
| `timestamp` | DateTime | Transaction timestamp |
| `amount` | Float | Transaction amount in USD |
| `merchant` | String | Merchant name |
| `mcc` | String | Merchant Category Code |
| `device_type` | String | Device used for transaction |
| `latitude` | Float | Transaction location (lat) |
| `longitude` | Float | Transaction location (long) |
| `merchant_risk_level` | String | Low/Medium/High risk merchant |
| `device_risk_level` | String | Low/Medium/High risk device |
| `hour` | Integer | Hour of transaction (0-23) |
| `day_of_week` | Integer | Day of week (0-6) |
| `month` | Integer | Month (1-12) |
| `is_weekend` | Boolean | Weekend transaction flag |
| `is_business_hours` | Boolean | Business hours flag |
| `anomaly_type` | String | Type of anomaly detected |
| `risk_score` | Float | Calculated risk score (0-100) |
| `is_fraud` | Boolean | Fraud flag |

## 🔍 Anomaly Detection

The system detects several types of anomalies:

1. **High Amount Anomalies** (1%): Transactions with unusually high amounts
2. **Geographic Anomalies** (0.5%): Transactions from unusual locations
3. **Time-based Anomalies** (0.3%): Transactions at unusual hours
4. **Device-Merchant Mismatch** (0.2%): High-risk devices with low-risk merchants

## 🎯 Risk Scoring

Risk scores are calculated using a combination of:
- Merchant risk level (Low=1, Medium=2, High=3)
- Device risk level (Low=1, Medium=2, High=3)
- Transaction amount (normalized)
- Time factors (non-business hours, weekends)
- Anomaly detection results

## 🛠️ Development

### Prerequisites
- Python 3.8+
- Node.js 16+
- Angular CLI

### Setup Development Environment

```bash
# Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Run tests
pytest
```

## 📈 ML Model Features

- **Feature Engineering**: Time-based features, geographic clustering, merchant patterns
- **Model Types**: Random Forest, XGBoost, LightGBM, Neural Networks
- **Evaluation Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Model Interpretability**: SHAP values, feature importance

## 🚀 Deployment

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Production

- **Backend**: Deploy to AWS/GCP with auto-scaling
- **Frontend**: Serve via CDN (CloudFront/Cloud CDN)
- **Database**: Use Snowflake for production, SQLite for development
- **ML Models**: Deploy via MLflow or custom API endpoints

## 📚 API Endpoints

### Risk Analysis
- `POST /api/analyze` - Analyze single transaction
- `POST /api/batch-analyze` - Analyze multiple transactions
- `GET /api/risk-summary` - Get risk summary statistics

### Model Management
- `POST /api/models/train` - Retrain ML models
- `GET /api/models/status` - Get model performance metrics
- `POST /api/models/update` - Update model parameters

### Data Management
- `GET /api/transactions` - Get transaction data
- `POST /api/transactions` - Add new transaction
- `GET /api/analytics` - Get analytics dashboard data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Financial transaction patterns inspired by real-world fraud detection systems
- ML techniques based on industry best practices
- Architecture patterns from modern fintech applications

---

**Built with ❤️ for showcasing full-stack, data engineering, and AI skills**
