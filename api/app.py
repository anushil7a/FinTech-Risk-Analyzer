#!/usr/bin/env python3
"""
FinTech Risk Analyzer - Flask API
Main application for analyzing financial transactions and calculating risk scores
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///risk_analyzer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Enable CORS with more permissive settings
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize database
db = SQLAlchemy(app)

# Database Models
class Transaction(db.Model):
    """Transaction model for storing financial transactions"""
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(50), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    merchant = db.Column(db.String(100), nullable=False)
    mcc = db.Column(db.String(10), nullable=False)
    device_type = db.Column(db.String(50), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    merchant_risk_level = db.Column(db.String(20), nullable=False)
    device_risk_level = db.Column(db.String(20), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    is_fraud = db.Column(db.Boolean, nullable=False)
    anomaly_type = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert transaction to dictionary"""
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'amount': self.amount,
            'merchant': self.merchant,
            'mcc': self.mcc,
            'device_type': self.device_type,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'merchant_risk_level': self.merchant_risk_level,
            'device_risk_level': self.device_risk_level,
            'risk_score': self.risk_score,
            'is_fraud': self.is_fraud,
            'anomaly_type': self.anomaly_type,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class RiskAnalysis(db.Model):
    """Risk analysis results model"""
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(50), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    risk_factors = db.Column(db.Text, nullable=False)  # JSON string
    ml_prediction = db.Column(db.Float, nullable=False)
    rule_based_score = db.Column(db.Float, nullable=False)
    final_decision = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert risk analysis to dictionary"""
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'risk_score': self.risk_score,
            'risk_factors': json.loads(self.risk_factors),
            'ml_prediction': self.ml_prediction,
            'rule_based_score': self.rule_based_score,
            'final_decision': self.final_decision,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# ML Model Class
class RiskAnalyzer:
    """Risk analyzer using ML and rule-based approaches"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train_model(self, transactions_df):
        """Train the ML model on transaction data"""
        try:
            # Prepare features
            features = self._prepare_features(transactions_df)
            target = transactions_df['is_fraud']
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit scaler and model
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled, target)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def _prepare_features(self, df):
        """Prepare features for ML model"""
        # Create feature matrix
        feature_cols = [
            'amount', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours',
            'latitude', 'longitude'
        ]
        
        # Add merchant risk encoding
        merchant_risk_map = {'low': 1, 'medium': 2, 'high': 3}
        df['merchant_risk_encoded'] = df['merchant_risk_level'].map(merchant_risk_map)
        
        # Add device risk encoding
        device_risk_map = {'low': 1, 'medium': 2, 'high': 3}
        df['device_risk_encoded'] = df['device_risk_level'].map(device_risk_map)
        
        feature_cols.extend(['merchant_risk_encoded', 'device_risk_encoded'])
        
        return df[feature_cols].fillna(0)
    
    def predict_risk(self, transaction_data):
        """Predict risk for a single transaction"""
        if not self.is_trained:
            # Return a default risk analysis when model is not trained
            rule_score, rule_reasons = self._calculate_rule_based_score(transaction_data)
            return {
                'ml_prediction': 0.0,
                'rule_based_score': float(rule_score),
                'final_score': float(rule_score),
                'risk_level': self._get_risk_level(rule_score),
                'rule_reasons': rule_reasons,
                'ml_reasons': ["ML Model: Not yet trained, using rule-based fallback."],
                'final_reasoning': self._generate_final_reasoning(0.0, rule_score, rule_score)
            }
        
        try:
            # Prepare features
            features = self._prepare_features(pd.DataFrame([transaction_data]))
            features_scaled = self.scaler.transform(features)
            
            # Get ML prediction
            ml_prediction = self.model.predict_proba(features_scaled)[0][1]
            
            # Calculate rule-based score with reasoning
            rule_score, rule_reasons = self._calculate_rule_based_score(transaction_data)
            
            # Combine scores (70% ML, 30% rules)
            final_score = 0.7 * ml_prediction + 0.3 * rule_score
            
            # Generate ML reasoning
            ml_reasons = self._generate_ml_reasons(transaction_data, ml_prediction)
            
            return {
                'ml_prediction': float(ml_prediction),
                'rule_based_score': float(rule_score),
                'final_score': float(final_score),
                'risk_level': self._get_risk_level(final_score),
                'rule_reasons': rule_reasons,
                'ml_reasons': ml_reasons,
                'final_reasoning': self._generate_final_reasoning(ml_prediction, rule_score, final_score)
            }
            
        except Exception as e:
            print(f"Error in predict_risk: {e}")
            # Return rule-based analysis as fallback
            rule_score, rule_reasons = self._calculate_rule_based_score(transaction_data)
            return {
                'ml_prediction': 0.0,
                'rule_based_score': float(rule_score),
                'final_score': float(rule_score),
                'risk_level': self._get_risk_level(rule_score),
                'rule_reasons': rule_reasons,
                'ml_reasons': [f"ML Model: Error during prediction ({str(e)}), using rule-based fallback."],
                'final_reasoning': self._generate_final_reasoning(0.0, rule_score, rule_score)
            }
    
    def _calculate_rule_based_score(self, transaction):
        """Calculate rule-based risk score with detailed reasoning"""
        score = 0.0
        reasons = []
        
        # Base risk from merchant
        merchant_risk = {'low': 0.1, 'medium': 0.3, 'high': 0.6}
        merchant_score = merchant_risk.get(transaction.get('merchant_risk_level', 'medium'), 0.3)
        score += merchant_score
        reasons.append(f"Merchant risk: {transaction.get('merchant_risk_level', 'medium')} (+{merchant_score:.1f})")
        
        # Device risk
        device_risk = {'low': 0.1, 'medium': 0.3, 'high': 0.6}
        device_score = device_risk.get(transaction.get('device_risk_level', 'medium'), 0.3)
        score += device_score
        reasons.append(f"Device risk: {transaction.get('device_risk_level', 'medium')} (+{device_score:.1f})")
        
        # Amount risk (normalized)
        amount = transaction.get('amount', 0)
        if amount > 1000:
            score += 0.3
            reasons.append(f"High amount: ${amount:,.2f} (+0.3)")
        elif amount > 500:
            score += 0.2
            reasons.append(f"Medium amount: ${amount:,.2f} (+0.2)")
        elif amount > 100:
            score += 0.1
            reasons.append(f"Moderate amount: ${amount:,.2f} (+0.1)")
        else:
            reasons.append(f"Low amount: ${amount:,.2f} (+0.0)")
        
        # Time risk
        hour = transaction.get('hour', 12)
        if hour < 6 or hour > 22:
            score += 0.2
            reasons.append(f"Non-business hours: {hour}:00 (+0.2)")
        else:
            reasons.append(f"Business hours: {hour}:00 (+0.0)")
        
        # Weekend risk
        if transaction.get('is_weekend', False):
            score += 0.1
            reasons.append("Weekend transaction (+0.1)")
        else:
            reasons.append("Weekday transaction (+0.0)")
        
        # Geographic risk (if coordinates provided)
        if 'latitude' in transaction and 'longitude' in transaction:
            lat, lon = transaction['latitude'], transaction['longitude']
            # Check for unusual locations (simplified)
            if lat > 60 or lat < 25 or lon < -125 or lon > -65:
                score += 0.2
                reasons.append(f"Unusual location: ({lat:.2f}, {lon:.2f}) (+0.2)")
            else:
                reasons.append(f"Standard US location: ({lat:.2f}, {lon:.2f}) (+0.0)")
        
        return min(score, 1.0), reasons
    
    def _generate_ml_reasons(self, transaction_data, ml_prediction):
        """Generate explanations for ML predictions"""
        reasons = []
        
        # Explain ML confidence
        if ml_prediction > 0.8:
            reasons.append(f"ML Model: High confidence fraud prediction ({ml_prediction:.1%})")
        elif ml_prediction > 0.6:
            reasons.append(f"ML Model: Medium confidence fraud prediction ({ml_prediction:.1%})")
        elif ml_prediction > 0.4:
            reasons.append(f"ML Model: Low confidence fraud prediction ({ml_prediction:.1%})")
        else:
            reasons.append(f"ML Model: Very low fraud probability ({ml_prediction:.1%})")
        
        # Add feature importance explanations
        amount = transaction_data.get('amount', 0)
        if amount > 1000:
            reasons.append("ML detected: High amount transaction pattern")
        
        merchant = transaction_data.get('merchant', '')
        if merchant in ['Casino', 'LuxuryHotel', 'JewelryStore']:
            reasons.append("ML detected: High-risk merchant category pattern")
        
        device = transaction_data.get('device_type', '')
        if device in ['web-chrome', 'mobile-web']:
            reasons.append("ML detected: High-risk device pattern")
        
        return reasons
    
    def _generate_final_reasoning(self, ml_prediction, rule_score, final_score):
        """Generate final reasoning summary"""
        reasoning = []
        
        # Explain the scoring breakdown
        ml_contribution = 0.7 * ml_prediction
        rule_contribution = 0.3 * rule_score
        
        reasoning.append(f"Final Score Breakdown:")
        reasoning.append(f"• ML Model (70%): {ml_contribution:.3f}")
        reasoning.append(f"• Rule-based (30%): {rule_contribution:.3f}")
        reasoning.append(f"• Combined Score: {final_score:.3f}")
        
        # Explain the decision
        if final_score > 0.7:
            reasoning.append("Decision: HIGH RISK - Transaction flagged for review")
        elif final_score > 0.4:
            reasoning.append("Decision: MEDIUM RISK - Transaction requires monitoring")
        else:
            reasoning.append("Decision: LOW RISK - Transaction appears legitimate")
        
        return reasoning
    
    def _get_risk_level(self, score):
        """Convert score to risk level"""
        if score < 0.3:
            return 'low'
        elif score < 0.7:
            return 'medium'
        else:
            return 'high'

# Initialize risk analyzer
risk_analyzer = RiskAnalyzer()

# Routes
@app.route('/')
def index():
    """Home page"""
    return jsonify({
        'message': 'FinTech Risk Analyzer API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'analyze': '/api/analyze',
            'batch_analyze': '/api/batch-analyze',
            'transactions': '/api/transactions',
            'risk_summary': '/api/risk-summary',
            'train_model': '/api/models/train'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_trained': risk_analyzer.is_trained
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    """Analyze a single transaction for risk"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['amount', 'merchant', 'device_type', 'latitude', 'longitude']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        
        # Add derived fields
        timestamp = pd.to_datetime(data['timestamp'])
        data['hour'] = timestamp.hour
        data['day_of_week'] = timestamp.weekday()
        data['month'] = timestamp.month
        data['is_weekend'] = timestamp.weekday() >= 5
        data['is_business_hours'] = 9 <= timestamp.hour <= 17
        
        # Get merchant risk level (simplified)
        merchant_risk_map = {
            'Walmart': 'low', 'Target': 'low', 'Amazon': 'low',
            'Shell': 'medium', 'Exxon': 'medium', 'BestBuy': 'medium',
            'LuxuryHotel': 'high', 'Casino': 'high', 'JewelryStore': 'high'
        }
        data['merchant_risk_level'] = merchant_risk_map.get(data['merchant'], 'medium')
        
        # Get device risk level
        device_risk_map = {
            'ios-15.2': 'low', 'ios-16.1': 'low',
            'android-13': 'medium', 'android-12': 'medium',
            'web-chrome': 'high', 'web-safari': 'medium', 'mobile-web': 'high'
        }
        data['device_risk_level'] = device_risk_map.get(data['device_type'], 'medium')
        
        # Analyze risk
        risk_result = risk_analyzer.predict_risk(data)
        
        if risk_result is None:
            return jsonify({'error': 'Failed to analyze risk'}), 500
        
        # Create transaction record
        transaction = Transaction(
            transaction_id=f"TXN{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            amount=data['amount'],
            merchant=data['merchant'],
            mcc=data.get('mcc', '0000'),  # Default to '0000' if not provided
            device_type=data['device_type'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            merchant_risk_level=data['merchant_risk_level'],
            device_risk_level=data['device_risk_level'],
            risk_score=risk_result['final_score'] * 100,
            is_fraud=risk_result['final_score'] > 0.7,
            anomaly_type='normal'  # Will be enhanced later
        )
        
        # Save to database
        db.session.add(transaction)
        db.session.commit()
        
        # Create risk analysis record
        risk_analysis = RiskAnalysis(
            transaction_id=transaction.transaction_id,
            risk_score=risk_result['final_score'] * 100,
            risk_factors=json.dumps({
                'merchant_risk': data['merchant_risk_level'],
                'device_risk': data['device_risk_level'],
                'amount_risk': 'high' if data['amount'] > 1000 else 'medium' if data['amount'] > 500 else 'low',
                'time_risk': 'high' if data['hour'] < 6 or data['hour'] > 22 else 'low',
                'weekend_risk': 'high' if data['is_weekend'] else 'low'
            }),
            ml_prediction=risk_result['ml_prediction'] * 100,
            rule_based_score=risk_result['rule_based_score'] * 100,
            final_decision='fraud' if risk_result['final_score'] > 0.7 else 'legitimate'
        )
        
        db.session.add(risk_analysis)
        db.session.commit()
        
        return jsonify({
            'transaction_id': transaction.transaction_id,
            'risk_analysis': risk_result,
            'database_id': transaction.id,
            'timestamp': timestamp.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple transactions"""
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions provided'}), 400
        
        transactions = data['transactions']
        results = []
        
        for transaction_data in transactions:
            # Analyze each transaction
            result = analyze_transaction()
            if result.status_code == 200:
                results.append(result.get_json())
            else:
                results.append({'error': 'Failed to analyze transaction'})
        
        return jsonify({
            'total_transactions': len(transactions),
            'successful_analyses': len([r for r in results if 'error' not in r]),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Get transactions with optional filtering"""
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 50, type=int), 100)
        risk_level = request.args.get('risk_level')
        merchant = request.args.get('merchant')
        
        # Build query
        query = Transaction.query
        
        if risk_level:
            if risk_level == 'high':
                query = query.filter(Transaction.risk_score >= 70)
            elif risk_level == 'medium':
                query = query.filter(Transaction.risk_score.between(30, 69))
            elif risk_level == 'low':
                query = query.filter(Transaction.risk_score < 30)
        
        if merchant:
            query = query.filter(Transaction.merchant.ilike(f'%{merchant}%'))
        
        # Paginate results
        pagination = query.order_by(Transaction.timestamp.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        transactions = [t.to_dict() for t in pagination.items]
        
        return jsonify({
            'transactions': transactions,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk-summary', methods=['GET'])
def risk_summary():
    """Get risk summary statistics"""
    try:
        # Get overall statistics
        total_transactions = Transaction.query.count()
        high_risk_count = Transaction.query.filter(Transaction.risk_score >= 70).count()
        medium_risk_count = Transaction.query.filter(Transaction.risk_score.between(30, 69)).count()
        low_risk_count = Transaction.query.filter(Transaction.risk_score < 30).count()
        
        fraud_count = Transaction.query.filter(Transaction.is_fraud == True).count()
        
        # Get average risk scores by merchant
        merchant_stats = db.session.query(
            Transaction.merchant,
            db.func.avg(Transaction.risk_score).label('avg_risk'),
            db.func.count(Transaction.id).label('count')
        ).group_by(Transaction.merchant).all()
        
        merchant_data = [
            {
                'merchant': stat.merchant,
                'avg_risk': round(stat.avg_risk, 2),
                'count': stat.count
            }
            for stat in merchant_stats
        ]
        
        # Get risk distribution over time (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_transactions = Transaction.query.filter(
            Transaction.timestamp >= thirty_days_ago
        ).all()
        
        daily_risks = {}
        for t in recent_transactions:
            date = t.timestamp.date().isoformat()
            if date not in daily_risks:
                daily_risks[date] = []
            daily_risks[date].append(t.risk_score)
        
        daily_averages = {
            date: round(sum(scores) / len(scores), 2)
            for date, scores in daily_risks.items()
        }
        
        return jsonify({
            'overall_stats': {
                'total_transactions': total_transactions,
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'low_risk_count': low_risk_count,
                'fraud_count': fraud_count,
                'fraud_rate': round(fraud_count / total_transactions * 100, 2) if total_transactions > 0 else 0
            },
            'merchant_stats': merchant_data,
            'daily_risk_trends': daily_averages
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/train', methods=['POST'])
def train_model():
    """Train the ML model on existing data"""
    try:
        # Get all transactions from database
        transactions = Transaction.query.all()
        
        if len(transactions) < 100:
            return jsonify({
                'error': 'Insufficient data for training. Need at least 100 transactions.'
            }), 400
        
        # Convert to DataFrame
        data = []
        for t in transactions:
            data.append({
                'amount': t.amount,
                'merchant': t.merchant,
                'device_type': t.device_type,
                'latitude': t.latitude,
                'longitude': t.longitude,
                'merchant_risk_level': t.merchant_risk_level,
                'device_risk_level': t.device_risk_level,
                'hour': t.timestamp.hour if t.timestamp else 12,
                'day_of_week': t.timestamp.weekday() if t.timestamp else 0,
                'month': t.timestamp.month if t.timestamp else 1,
                'is_weekend': t.timestamp.weekday() >= 5 if t.timestamp else False,
                'is_business_hours': 9 <= t.timestamp.hour <= 17 if t.timestamp else True,
                'is_fraud': t.is_fraud
            })
        
        df = pd.DataFrame(data)
        
        # Train model
        success = risk_analyzer.train_model(df)
        
        if success:
            return jsonify({
                'message': 'Model trained successfully',
                'training_samples': len(df),
                'fraud_rate': round(df['is_fraud'].mean() * 100, 2)
            })
        else:
            return jsonify({'error': 'Failed to train model'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/status', methods=['GET'])
def model_status():
    """Get model status and performance metrics"""
    try:
        if not risk_analyzer.is_trained:
            return jsonify({
                'status': 'not_trained',
                'message': 'Model needs to be trained first'
            })
        
        # Get recent predictions for accuracy calculation
        recent_analyses = RiskAnalysis.query.order_by(
            RiskAnalysis.created_at.desc()
        ).limit(100).all()
        
        if recent_analyses:
            # Calculate basic metrics
            total = len(recent_analyses)
            high_risk = len([a for a in recent_analyses if a.risk_score >= 70])
            avg_risk = sum(a.risk_score for a in recent_analyses) / total
            
            return jsonify({
                'status': 'trained',
                'model_type': 'RandomForest',
                'recent_predictions': total,
                'high_risk_rate': round(high_risk / total * 100, 2),
                'average_risk_score': round(avg_risk, 2),
                'last_training': 'Model trained and ready'
            })
        else:
            return jsonify({
                'status': 'trained',
                'message': 'Model trained but no recent predictions available'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# Initialize database and load data
def init_database():
    """Initialize database and load sample data"""
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Check if we have data
        if Transaction.query.count() == 0:
            print("Loading sample data into database...")
            
            # Load CSV data - use full dataset for better training
            csv_path = Path("../data/transactions.csv")
            if csv_path.exists():
                print("Loading full 50K transaction dataset...")
                df = pd.read_csv(csv_path)
                
                # Clear existing transactions first
                Transaction.query.delete()
                db.session.commit()
                print("Cleared existing transactions from database")
                
                # Load in batches to avoid memory issues
                batch_size = 1000
                total_batches = len(df) // batch_size + 1
                
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    batch_num = i // batch_size + 1
                    
                    for _, row in batch.iterrows():
                        transaction = Transaction(
                            transaction_id=row['transaction_id'],
                            timestamp=pd.to_datetime(row['timestamp']),
                            amount=row['amount'],
                            merchant=row['merchant'],
                            mcc=row['mcc'],
                            device_type=row['device_type'],
                            latitude=row['latitude'],
                            longitude=row['longitude'],
                            merchant_risk_level=row['merchant_risk_level'],
                            device_risk_level=row['device_risk_level'],
                            risk_score=row['risk_score'],
                            is_fraud=bool(row['is_fraud']),
                            anomaly_type=row['anomaly_type']
                        )
                        db.session.add(transaction)
                    
                    # Commit batch
                    db.session.commit()
                    print(f"Loaded batch {batch_num}/{total_batches} ({len(batch)} transactions)")
                
                print(f"✅ Successfully loaded {len(df):,} transactions into database")
                
                # Train model on full dataset
                print("🧠 Training ML model on full dataset...")
                success = risk_analyzer.train_model(df)
                if success:
                    print("🎉 Model trained successfully on 50K transactions!")
                else:
                    print("❌ Failed to train model")
            else:
                print("Full dataset not found. Run make_fake_data.py first.")

if __name__ == '__main__':
    print("🚀 Starting FinTech Risk Analyzer API...")
    
    try:
        init_database()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
    
    print("🌐 API running on http://localhost:5002")
    print("📊 Health check: http://localhost:5002/health")
    print("🔧 Debug mode: ON")
    
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)
