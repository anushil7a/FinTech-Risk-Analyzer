#!/usr/bin/env python3
"""
Script to load the full 50K transaction dataset into the database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db, Transaction, risk_analyzer
import pandas as pd
from pathlib import Path

def load_full_dataset():
    """Load the full 50K transaction dataset"""
    with app.app_context():
        print("🧠 Loading full 50K transaction dataset...")
        
        # Path to the full dataset
        csv_path = Path("../data/transactions.csv")
        if not csv_path.exists():
            print("❌ Full dataset not found at:", csv_path)
            return False
        
        # Read the full dataset
        print("📖 Reading CSV file...")
        df = pd.read_csv(csv_path)
        print(f"📊 Found {len(df):,} transactions in CSV")
        
        # Clear existing transactions
        print("🗑️  Clearing existing transactions from database...")
        Transaction.query.delete()
        db.session.commit()
        print("✅ Database cleared")
        
        # Load in batches to avoid memory issues
        batch_size = 1000
        total_batches = len(df) // batch_size + 1
        
        print(f"📦 Loading data in {total_batches} batches...")
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"📥 Loading batch {batch_num}/{total_batches} ({len(batch)} transactions)...")
            
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
            print(f"✅ Batch {batch_num} loaded and committed")
        
        print(f"🎉 Successfully loaded {len(df):,} transactions into database!")
        
        # Train the ML model on the full dataset
        print("🧠 Training ML model on full dataset...")
        success = risk_analyzer.train_model(df)
        
        if success:
            print("🎯 Model trained successfully on 50K transactions!")
            print(f"📈 Dataset stats:")
            print(f"   - Total transactions: {len(df):,}")
            print(f"   - Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
            print(f"   - Average risk score: {df['risk_score'].mean():.2f}")
            print(f"   - Risk score range: {df['risk_score'].min():.2f} - {df['risk_score'].max():.2f}")
        else:
            print("❌ Failed to train model")
            return False
        
        return True

if __name__ == "__main__":
    print("🚀 Starting full dataset load...")
    success = load_full_dataset()
    
    if success:
        print("\n🎉 Full dataset loaded successfully!")
        print("🌐 Your API now has 50K transactions and a trained ML model!")
    else:
        print("\n❌ Failed to load full dataset")
        sys.exit(1)
