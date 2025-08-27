#!/usr/bin/env python3
"""
Generate fake financial transaction data for FinTech Risk Analyzer
Creates realistic transaction patterns with anomalies for ML model testing
"""

import pandas as pd
import numpy as np
import random
import datetime as dt
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_fake_transactions(n=50000):
    """Generate realistic financial transaction data with anomalies"""
    
    # Base parameters
    start_date = dt.datetime(2025, 1, 1)
    
    # Merchant categories and their typical risk profiles
    merchants = {
        "Walmart": {"mcc": "5411", "risk_level": "low", "avg_amount": 45.0},
        "Target": {"mcc": "5411", "risk_level": "low", "avg_amount": 38.0},
        "Shell": {"mcc": "5541", "risk_level": "medium", "avg_amount": 65.0},
        "Exxon": {"mcc": "5541", "risk_level": "medium", "avg_amount": 58.0},
        "Amazon": {"mcc": "5999", "risk_level": "low", "avg_amount": 89.0},
        "BestBuy": {"mcc": "5732", "risk_level": "medium", "avg_amount": 245.0},
        "Starbucks": {"mcc": "5814", "risk_level": "low", "avg_amount": 12.0},
        "McDonalds": {"mcc": "5814", "risk_level": "low", "avg_amount": 18.0},
        "LuxuryHotel": {"mcc": "7011", "risk_level": "high", "avg_amount": 450.0},
        "Casino": {"mcc": "7990", "risk_level": "high", "avg_amount": 1200.0},
        "OnlineGaming": {"mcc": "7990", "risk_level": "high", "avg_amount": 85.0},
        "JewelryStore": {"mcc": "5944", "risk_level": "high", "avg_amount": 850.0},
        "Electronics": {"mcc": "5732", "risk_level": "medium", "avg_amount": 320.0},
        "GasStation": {"mcc": "5541", "risk_level": "medium", "avg_amount": 55.0},
        "GroceryStore": {"mcc": "5411", "risk_level": "low", "avg_amount": 42.0}
    }
    
    # Device types and their risk profiles
    devices = {
        "ios-15.2": {"risk_level": "low", "fraud_prob": 0.001},
        "ios-16.1": {"risk_level": "low", "fraud_prob": 0.001},
        "android-13": {"risk_level": "medium", "fraud_prob": 0.003},
        "android-12": {"risk_level": "medium", "fraud_prob": 0.003},
        "web-chrome": {"risk_level": "high", "fraud_prob": 0.008},
        "web-safari": {"risk_level": "medium", "fraud_prob": 0.004},
        "mobile-web": {"risk_level": "high", "fraud_prob": 0.010}
    }
    
    # Generate base transaction data
    print(f"Generating {n:,} transactions...")
    
    # Transaction IDs
    transaction_ids = [f"TXN{i:08d}" for i in range(n)]
    
    # Timestamps with realistic patterns (more transactions during business hours)
    timestamps = []
    for i in range(n):
        # Add some randomness to business hours
        hour = np.random.normal(14, 4)  # Peak around 2 PM
        hour = max(0, min(23, int(hour)))
        minute = np.random.randint(0, 60)
        
        # Random day within 2025
        days_offset = np.random.randint(0, 365)
        timestamp = start_date + dt.timedelta(days=days_offset, hours=hour, minutes=minute)
        timestamps.append(timestamp)
    
    # Sort timestamps chronologically
    timestamps.sort()
    
    # Generate merchant and device data
    merchant_list = list(merchants.keys())
    device_list = list(devices.keys())
    
    selected_merchants = np.random.choice(merchant_list, n)
    selected_devices = np.random.choice(device_list, n)
    
    # Generate amounts based on merchant characteristics
    amounts = []
    for merchant in selected_merchants:
        base_amount = merchants[merchant]["avg_amount"]
        # Add some variance
        amount = np.random.gamma(2.2, base_amount/2.2)
        amounts.append(round(amount, 2))
    
    # Generate geographic data (US coordinates)
    latitudes = np.random.uniform(25, 49, n)
    longitudes = np.random.uniform(-124, -67, n)
    
    # Create base DataFrame
    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "timestamp": timestamps,
        "amount": amounts,
        "merchant": selected_merchants,
        "mcc": [merchants[m]["mcc"] for m in selected_merchants],
        "device_type": selected_devices,
        "latitude": latitudes,
        "longitude": longitudes,
        "merchant_risk_level": [merchants[m]["risk_level"] for m in selected_merchants],
        "device_risk_level": [devices[d]["risk_level"] for d in selected_devices]
    })
    
    # Add derived features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
    
    # Inject anomalies for ML model training
    print("Injecting anomalies...")
    
    # 1. High amount anomalies (1% of transactions)
    high_amount_idx = np.random.choice(n, size=int(n*0.01), replace=False)
    df.loc[high_amount_idx, "amount"] *= np.random.uniform(8, 25, size=len(high_amount_idx))
    df.loc[high_amount_idx, "anomaly_type"] = "high_amount"
    
    # 2. Geographic anomalies (0.5% of transactions)
    geo_anomaly_idx = np.random.choice(n, size=int(n*0.005), replace=False)
    # Place some transactions in unusual locations
    df.loc[geo_anomaly_idx, "latitude"] = np.random.uniform(60, 70, size=len(geo_anomaly_idx))  # Alaska
    df.loc[geo_anomaly_idx, "longitude"] = np.random.uniform(-180, -170, size=len(geo_anomaly_idx))
    df.loc[geo_anomaly_idx, "anomaly_type"] = "geographic"
    
    # 3. Time-based anomalies (0.3% of transactions)
    time_anomaly_idx = np.random.choice(n, size=int(n*0.003), replace=False)
    # Transactions at unusual hours
    df.loc[time_anomaly_idx, "hour"] = np.random.choice([0, 1, 2, 3, 4, 5], size=len(time_anomaly_idx))
    df.loc[time_anomaly_idx, "anomaly_type"] = "time_based"
    
    # 4. Device-merchant mismatch anomalies (0.2% of transactions)
    device_anomaly_idx = np.random.choice(n, size=int(n*0.002), replace=False)
    # High-risk devices with low-risk merchants
    df.loc[device_anomaly_idx, "device_type"] = "web-chrome"
    df.loc[device_anomaly_idx, "merchant"] = "Starbucks"
    df.loc[device_anomaly_idx, "anomaly_type"] = "device_merchant_mismatch"
    
    # Fill NaN anomaly types with "normal"
    df["anomaly_type"] = df["anomaly_type"].fillna("normal")
    
    # Calculate risk score (simple heuristic for now)
    df["risk_score"] = 0.0
    
    # Base risk from merchant
    risk_mapping = {"low": 1, "medium": 2, "high": 3}
    df["risk_score"] += df["merchant_risk_level"].map(risk_mapping)
    
    # Add risk from device
    df["risk_score"] += df["device_risk_level"].map(risk_mapping)
    
    # Add risk from amount (higher amounts = higher risk)
    df["risk_score"] += (df["amount"] / 100).clip(0, 5)
    
    # Add risk from time (non-business hours = higher risk)
    df["risk_score"] += (1 - df["is_business_hours"]) * 2
    
    # Add risk from weekend transactions
    df["risk_score"] += df["is_weekend"] * 1.5
    
    # Normalize risk score to 0-100 scale
    df["risk_score"] = (df["risk_score"] / df["risk_score"].max() * 100).round(2)
    
    # Add fraud flag based on risk score and anomaly type
    df["is_fraud"] = ((df["risk_score"] > 70) | 
                      (df["anomaly_type"] != "normal")).astype(int)
    
    return df

def main():
    """Main function to generate and save fake data"""
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate data
    df = generate_fake_transactions(n=50000)
    
    # Save to CSV
    output_file = data_dir / "transactions.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Generated {len(df):,} transactions")
    print(f"📁 Saved to: {output_file}")
    print(f"💰 Amount range: ${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
    print(f"🎯 Risk score range: {df['risk_score'].min():.2f} - {df['risk_score'].max():.2f}")
    print(f"🚨 Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"🔍 Anomaly distribution:")
    print(df['anomaly_type'].value_counts())
    
    # Save sample for quick testing
    sample_file = data_dir / "transactions_sample.csv"
    df.sample(min(1000, len(df))).to_csv(sample_file, index=False)
    print(f"📋 Sample saved to: {sample_file}")

if __name__ == "__main__":
    main()
