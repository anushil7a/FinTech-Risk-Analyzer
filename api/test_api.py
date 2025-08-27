#!/usr/bin/env python3
"""
Simple test script for Flask API
"""

import requests
import json

def test_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing FinTech Risk Analyzer API...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test transaction analysis
    test_transaction = {
        "amount": 1500.00,
        "merchant": "LuxuryHotel",
        "device_type": "web-chrome",
        "latitude": 40.7128,
        "longitude": -74.0060
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/analyze",
            json=test_transaction,
            headers={"Content-Type": "application/json"}
        )
        print(f"✅ Transaction analysis: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Transaction ID: {result.get('transaction_id')}")
            print(f"   Risk Score: {result.get('risk_analysis', {}).get('final_score', 'N/A')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Transaction analysis failed: {e}")

if __name__ == "__main__":
    test_api()
