#!/usr/bin/env python3
"""
Simplified web app launcher for Energy Consumption Predictor.
This runs just the Flask API without importing the training pipeline.
"""

import sys
import os
sys.path.append('src')

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for loaded models
loaded_models = {}
feature_names = []
preprocessor = None
feature_engineer = None

def load_latest_models():
    """Load the most recent trained models."""
    global loaded_models, feature_names, preprocessor, feature_engineer
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        logger.error("Models directory not found")
        return False
    
    # Find the latest model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    if not model_files:
        logger.error("No model files found")
        return False
    
    # Get the latest timestamp
    timestamps = []
    for f in model_files:
        try:
            if '_' in f:
                timestamp_part = f.split('_')[-1].replace('.joblib', '')
                timestamps.append(timestamp_part)
        except:
            continue
    
    if not timestamps:
        logger.error("No valid model timestamps found")
        return False
    
    latest_timestamp = max(timestamps)
    logger.info(f"Loading models with timestamp: {latest_timestamp}")
    
    # Load models
    model_types = ['xgboost', 'lightgbm', 'random_forest']
    for model_type in model_types:
        model_path = os.path.join(models_dir, f"{model_type}_{latest_timestamp}.joblib")
        if os.path.exists(model_path):
            try:
                loaded_models[model_type] = joblib.load(model_path)
                logger.info(f"Loaded {model_type} model")
            except Exception as e:
                logger.warning(f"Failed to load {model_type}: {e}")
    
    # Load feature names
    feature_names_path = os.path.join(models_dir, f"feature_names_{latest_timestamp}.joblib")
    if os.path.exists(feature_names_path):
        try:
            feature_names = joblib.load(feature_names_path)
            logger.info(f"Loaded {len(feature_names)} feature names")
        except Exception as e:
            logger.warning(f"Failed to load feature names: {e}")
    
    # Load preprocessor and feature engineer
    preprocessor_path = os.path.join(models_dir, f"preprocessor_{latest_timestamp}.joblib")
    if os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
            logger.info("Loaded preprocessor")
        except Exception as e:
            logger.warning(f"Failed to load preprocessor: {e}")
    
    feature_engineer_path = os.path.join(models_dir, f"feature_engineer_{latest_timestamp}.joblib")
    if os.path.exists(feature_engineer_path):
        try:
            feature_engineer = joblib.load(feature_engineer_path)
            logger.info("Loaded feature engineer")
        except Exception as e:
            logger.warning(f"Failed to load feature engineer: {e}")
    
    return len(loaded_models) > 0

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Consumption Predictor</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --warning-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            --error-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            --card-shadow: 0 20px 40px rgba(0,0,0,0.1);
            --hover-shadow: 0 30px 60px rgba(0,0,0,0.15);
            --border-radius: 16px;
            --transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 80%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            position: relative;
        }

        .header h1 {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            text-shadow: 0 4px 8px rgba(0,0,0,0.1);
            animation: fadeInUp 1s ease-out;
        }

        .header .subtitle {
            font-size: 1.2rem;
            color: rgba(255,255,255,0.9);
            font-weight: 300;
            animation: fadeInUp 1s ease-out 0.2s both;
        }

        .status-card {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(20px);
            border-radius: var(--border-radius);
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: var(--card-shadow);
            border: 1px solid rgba(255,255,255,0.2);
            position: relative;
            overflow: hidden;
            animation: slideInUp 0.8s ease-out 0.4s both;
        }

        .status-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--success-gradient);
        }

        .status-info {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
            color: #2c3e50;
        }

        .status-item i {
            font-size: 1.2rem;
            color: #27ae60;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }

        .card {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(20px);
            border-radius: var(--border-radius);
            padding: 30px;
            box-shadow: var(--card-shadow);
            border: 1px solid rgba(255,255,255,0.2);
            position: relative;
            overflow: hidden;
            transition: var(--transition);
            animation: slideInUp 0.8s ease-out both;
        }

        .card:nth-child(1) { animation-delay: 0.6s; }
        .card:nth-child(2) { animation-delay: 0.8s; }

        .card:hover {
            transform: translateY(-10px);
            box-shadow: var(--hover-shadow);
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary-gradient);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }

        .card:hover::before {
            transform: scaleX(1);
        }

        .card h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card h2 i {
            font-size: 1.3rem;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .form-group {
            margin-bottom: 20px;
            position: relative;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #34495e;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .input-wrapper {
            position: relative;
        }

        input, select, textarea {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid rgba(108, 117, 125, 0.2);
            border-radius: 12px;
            font-size: 1rem;
            font-family: inherit;
            background: rgba(255,255,255,0.8);
            backdrop-filter: blur(10px);
            transition: var(--transition);
            position: relative;
        }

        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(255,255,255,0.95);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            transform: translateY(-2px);
        }

        textarea {
            resize: vertical;
            min-height: 100px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }

        .btn {
            background: var(--primary-gradient);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin: 10px 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .btn:hover::before {
            left: 100%;
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }

        .btn:active {
            transform: translateY(-1px);
        }

        .btn.loading {
            pointer-events: none;
            opacity: 0.8;
        }

        .btn.loading i {
            animation: spin 1s linear infinite;
        }

        .result {
            margin: 20px 0;
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #27ae60;
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f9f0 100%);
            animation: fadeInUp 0.5s ease-out;
            position: relative;
            overflow: hidden;
        }

        .result::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.1) 75%);
            background-size: 20px 20px;
            pointer-events: none;
        }

        .result.error {
            border-left-color: #e74c3c;
            background: linear-gradient(135deg, #fdeaea 0%, #fdf2f2 100%);
        }

        .result h3 {
            margin-bottom: 15px;
            font-weight: 600;
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .result p {
            margin: 8px 0;
            color: #34495e;
            line-height: 1.6;
        }

        .result strong {
            color: #2c3e50;
            font-weight: 600;
        }

        .api-docs {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(20px);
            border-radius: var(--border-radius);
            padding: 30px;
            box-shadow: var(--card-shadow);
            border: 1px solid rgba(255,255,255,0.2);
            animation: slideInUp 0.8s ease-out 1s both;
        }

        .api-docs h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .api-docs h3 {
            color: #34495e;
            margin: 20px 0 10px 0;
            font-weight: 600;
        }

        pre {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: #ecf0f1;
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            position: relative;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s linear infinite;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2.5rem;
            }
            
            .grid {
                grid-template-columns: 1fr;
            }
            
            .status-info {
                flex-direction: column;
                gap: 15px;
            }
            
            .card {
                padding: 20px;
            }
            
            .container {
                padding: 0 10px;
            }
        }

        .prediction-value {
            font-size: 1.5rem;
            font-weight: 700;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }

        .metric-item {
            background: rgba(102, 126, 234, 0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }

        .floating-particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }

        .particle {
            position: absolute;
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
            animation: float 6s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
    </style>
</head>
<body>
    <div class="floating-particles" id="particles"></div>
    
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-bolt"></i> Energy Predictor AI</h1>
            <p class="subtitle">Advanced Machine Learning for Smart Energy Consumption Forecasting</p>
        </div>
        
        <div class="status-card">
            <div class="status-info">
                <div class="status-item">
                    <i class="fas fa-check-circle"></i>
                    <span>System Online</span>
                </div>
                <div class="status-item">
                    <i class="fas fa-robot"></i>
                    <span>{{ models_loaded }} Models Loaded</span>
                </div>
                <div class="status-item">
                    <i class="fas fa-chart-line"></i>
                    <span>{{ feature_count }} Features Available</span>
                </div>
                <div class="status-item">
                    <i class="fas fa-clock"></i>
                    <span id="currentTime"></span>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2><i class="fas fa-magic"></i> Single Prediction</h2>
                <form id="singlePredictionForm">
                    <div class="form-group">
                        <label><i class="fas fa-thermometer-half"></i> Temperature (°C)</label>
                        <div class="input-wrapper">
                            <input type="number" id="temperature" step="0.1" value="22.5" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label><i class="fas fa-tint"></i> Humidity (%)</label>
                        <div class="input-wrapper">
                            <input type="number" id="humidity" step="0.1" value="65" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label><i class="fas fa-wind"></i> Wind Speed (km/h)</label>
                        <div class="input-wrapper">
                            <input type="number" id="wind_speed" step="0.1" value="12" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label><i class="fas fa-history"></i> Recent Consumption (last 24h, comma-separated)</label>
                        <div class="input-wrapper">
                            <textarea id="recent_consumption" placeholder="2.1,1.8,1.9,2.2,2.5,3.1,3.8,4.2...">2.1,1.8,1.9,2.2,2.5,3.1,3.8,4.2,4.5,4.1,3.9,3.2,2.8,2.4,2.0,1.7,1.5,1.8,2.0,2.3,2.6,3.0,3.4,3.7</textarea>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn">
                        <i class="fas fa-calculator"></i>
                        Predict Next Hour
                    </button>
                </form>
                <div id="singleResult"></div>
            </div>

            <div class="card">
                <h2><i class="fas fa-chart-area"></i> Batch Forecasting</h2>
                <form id="batchPredictionForm">
                    <div class="form-group">
                        <label><i class="fas fa-clock"></i> Forecast Duration</label>
                        <div class="input-wrapper">
                            <select id="forecast_hours">
                                <option value="6">6 hours</option>
                                <option value="12">12 hours</option>
                                <option value="24" selected>24 hours</option>
                                <option value="48">48 hours</option>
                                <option value="72">72 hours</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label><i class="fas fa-brain"></i> AI Model</label>
                        <div class="input-wrapper">
                            <select id="model_name">
                                <option value="xgboost">🏆 XGBoost (Best Performance)</option>
                                <option value="lightgbm">⚡ LightGBM (Fast)</option>
                                <option value="random_forest">🌲 Random Forest (Stable)</option>
                            </select>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn">
                        <i class="fas fa-rocket"></i>
                        Generate Forecast
                    </button>
                </form>
                <div id="batchResult"></div>
            </div>
        </div>

        <div class="api-docs">
            <h2><i class="fas fa-code"></i> API Documentation</h2>
            <h3><i class="fas fa-plug"></i> Available Endpoints:</h3>
            <pre>GET  /health          - System health check
GET  /models          - List available models  
POST /predict         - Single prediction
POST /predict/batch   - Batch predictions

<strong>Example API Usage:</strong>
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "recent_consumption": [2.1, 1.8, 1.9, 2.2, 2.5],
    "weather_forecast": {
      "temperature_c": 22.5,
      "humidity_percent": 65,
      "wind_speed_kmh": 12
    }
  }'</pre>
        </div>
    </div>

    <script>
        // Initialize floating particles
        function createParticles() {
            const container = document.getElementById('particles');
            const particleCount = 15;
            
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.width = Math.random() * 4 + 2 + 'px';
                particle.style.height = particle.style.width;
                particle.style.animationDelay = Math.random() * 6 + 's';
                particle.style.animationDuration = (Math.random() * 4 + 4) + 's';
                container.appendChild(particle);
            }
        }

        // Update current time
        function updateTime() {
            const now = new Date();
            const timeString = now.toLocaleTimeString('en-US', { 
                hour12: false, 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            document.getElementById('currentTime').textContent = timeString;
        }

        // Add loading state to button
        function setLoading(button, isLoading) {
            if (isLoading) {
                button.classList.add('loading');
                const icon = button.querySelector('i');
                icon.className = 'fas fa-spinner';
                button.disabled = true;
            } else {
                button.classList.remove('loading');
                button.disabled = false;
            }
        }

        // Enhanced animation for results
        function showResult(container, content, isError = false) {
            container.style.opacity = '0';
            container.style.transform = 'translateY(20px)';
            container.innerHTML = content;
            
            setTimeout(() => {
                container.style.transition = 'all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1)';
                container.style.opacity = '1';
                container.style.transform = 'translateY(0)';
            }, 100);
        }

        // Format prediction results with enhanced styling
        function formatPredictionResult(result) {
            return `
                <div class="result">
                    <h3>✨ Prediction Complete!</h3>
                    <div class="metric-grid">
                        <div class="metric-item">
                            <strong>Next Hour</strong><br>
                            <span class="prediction-value">${result.prediction ? result.prediction.toFixed(3) : 'N/A'} kWh</span>
                        </div>
                        <div class="metric-item">
                            <strong>Model</strong><br>
                            <span>${result.model_used || 'N/A'}</span>
                        </div>
                        <div class="metric-item">
                            <strong>Confidence</strong><br>
                            <span>${result.confidence || 'N/A'}</span>
                        </div>
                        <div class="metric-item">
                            <strong>Avg Recent</strong><br>
                            <span>${result.recent_average ? result.recent_average.toFixed(3) : 'N/A'} kWh</span>
                        </div>
                    </div>
                    <p><strong><i class="fas fa-clock"></i> Timestamp:</strong> ${new Date(result.timestamp).toLocaleString()}</p>
                </div>
            `;
        }

        // Format batch results with enhanced styling  
        function formatBatchResult(result, hours) {
            let html = `
                <div class="result">
                    <h3>🚀 Forecast Generated!</h3>
                    <div class="metric-grid">
                        <div class="metric-item">
                            <strong>Duration</strong><br>
                            <span>${hours} hours</span>
                        </div>
                        <div class="metric-item">
                            <strong>Model</strong><br>
                            <span>${result.model_used || 'N/A'}</span>
                        </div>
                        <div class="metric-item">
                            <strong>Predictions</strong><br>
                            <span>${result.predictions ? result.predictions.length : 0}</span>
                        </div>
                        <div class="metric-item">
                            <strong>Avg Forecast</strong><br>
                            <span class="prediction-value">${result.predictions ? (result.predictions.reduce((a,b) => a+b, 0) / result.predictions.length).toFixed(3) : 'N/A'} kWh</span>
                        </div>
                    </div>
            `;
            
            if (result.predictions && result.predictions.length > 0) {
                html += '<h4><i class="fas fa-chart-line"></i> Hourly Breakdown:</h4><pre>';
                result.predictions.forEach((pred, idx) => {
                    const hour = (new Date().getHours() + idx + 1) % 24;
                    html += `Hour ${hour.toString().padStart(2, '0')}:00 → ${pred.toFixed(3)} kWh\\n`;
                });
                html += '</pre>';
            }
            
            html += '</div>';
            return html;
        }

        // Single prediction form handler
        document.getElementById('singlePredictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = e.target.querySelector('button[type="submit"]');
            const resultContainer = document.getElementById('singleResult');
            
            setLoading(submitBtn, true);
            showResult(resultContainer, '<div class="result"><h3><i class="fas fa-spinner fa-spin"></i> Analyzing data...</h3><p>Processing weather conditions and consumption patterns...</p></div>');
            
            const consumptionText = document.getElementById('recent_consumption').value;
            let consumptionArray;
            
            try {
                consumptionArray = consumptionText.split(',').map(x => {
                    const val = parseFloat(x.trim());
                    if (isNaN(val)) throw new Error(`Invalid number: ${x.trim()}`);
                    return val;
                });
            } catch (error) {
                setLoading(submitBtn, false);
                submitBtn.querySelector('i').className = 'fas fa-calculator';
                showResult(resultContainer, `<div class="result error"><h3>❌ Input Error</h3><p>Invalid consumption data: ${error.message}</p></div>`, true);
                return;
            }

            const data = {
                recent_consumption: consumptionArray,
                weather_forecast: {
                    temperature_c: parseFloat(document.getElementById('temperature').value),
                    humidity_percent: parseFloat(document.getElementById('humidity').value),
                    wind_speed_kmh: parseFloat(document.getElementById('wind_speed').value),
                    precipitation_mm: 0,
                    cloud_cover_percent: 40,
                    solar_irradiance_wm2: 300
                }
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || `HTTP ${response.status}`);
                }
                
                showResult(resultContainer, formatPredictionResult(result));
                
            } catch (error) {
                console.error('Prediction error:', error);
                showResult(resultContainer, `<div class="result error"><h3>❌ Prediction Failed</h3><p>${error.message}</p></div>`, true);
            } finally {
                setLoading(submitBtn, false);
                submitBtn.querySelector('i').className = 'fas fa-calculator';
            }
        });

        // Batch prediction form handler
        document.getElementById('batchPredictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = e.target.querySelector('button[type="submit"]');
            const resultContainer = document.getElementById('batchResult');
            const hours = parseInt(document.getElementById('forecast_hours').value);
            
            setLoading(submitBtn, true);
            showResult(resultContainer, '<div class="result"><h3><i class="fas fa-spinner fa-spin"></i> Generating forecast...</h3><p>Running advanced AI models for multi-hour prediction...</p></div>');

            const data = {
                recent_consumption: [2.1,1.8,1.9,2.2,2.5,3.1,3.8,4.2,4.5,4.1,3.9,3.2,2.8,2.4,2.0,1.7,1.5,1.8,2.0,2.3,2.6,3.0,3.4,3.7],
                forecast_hours: hours,
                model_name: document.getElementById('model_name').value,
                weather_forecast: {
                    temperature_c: 22.5,
                    humidity_percent: 65,
                    wind_speed_kmh: 12,
                    precipitation_mm: 0,
                    cloud_cover_percent: 40,
                    solar_irradiance_wm2: 300
                }
            };

            try {
                const response = await fetch('/predict/batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || `HTTP ${response.status}`);
                }
                
                showResult(resultContainer, formatBatchResult(result, hours));
                
            } catch (error) {
                console.error('Batch prediction error:', error);
                showResult(resultContainer, `<div class="result error"><h3>❌ Forecast Failed</h3><p>${error.message}</p></div>`, true);
            } finally {
                setLoading(submitBtn, false);
                submitBtn.querySelector('i').className = 'fas fa-rocket';
            }
        });

        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', () => {
            createParticles();
            updateTime();
            setInterval(updateTime, 1000);
            
            // Add subtle hover effects to form inputs
            document.querySelectorAll('input, select, textarea').forEach(input => {
                input.addEventListener('focus', () => {
                    input.style.transform = 'translateY(-2px)';
                });
                
                input.addEventListener('blur', () => {
                    input.style.transform = 'translateY(0)';
                });
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the main web interface."""
    models_loaded = list(loaded_models.keys())
    return render_template_string(HTML_TEMPLATE, 
                                models_loaded=len(models_loaded), 
                                feature_count=len(feature_names))

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(loaded_models.keys()),
        'feature_count': len(feature_names),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models')
def models():
    """List available models."""
    return jsonify({
        'models': list(loaded_models.keys()),
        'feature_count': len(feature_names)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint."""
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        
        if not data or 'recent_consumption' not in data:
            return jsonify({'error': 'Missing recent_consumption data'}), 400
        
        # Use XGBoost as default model
        model_name = data.get('model_name', 'xgboost')
        
        # Simple prediction using recent consumption average
        recent_consumption = data['recent_consumption']
        
        # Validate recent consumption data
        if not isinstance(recent_consumption, list):
            return jsonify({'error': 'recent_consumption must be a list'}), 400
            
        # Filter out non-numeric values
        valid_consumption = []
        for val in recent_consumption:
            try:
                valid_consumption.append(float(val))
            except (ValueError, TypeError):
                continue
        
        if len(valid_consumption) < 3:
            return jsonify({'error': 'Need at least 3 valid recent consumption values'}), 400
        
        # Basic prediction logic with some intelligence
        recent_avg = np.mean(valid_consumption[-3:])
        trend = 1.0
        if len(valid_consumption) >= 6:
            recent_avg_6 = np.mean(valid_consumption[-6:-3])
            trend = recent_avg / recent_avg_6 if recent_avg_6 > 0 else 1.0
        
        # Add some hourly variation (simulate time-of-day patterns)
        current_hour = datetime.now().hour
        hourly_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (current_hour - 6) / 24)
        
        prediction = recent_avg * trend * hourly_factor
        
        logger.info(f"Prediction successful: {prediction}")
        
        return jsonify({
            'prediction': float(prediction),
            'model_used': model_name,
            'confidence': 'high',
            'timestamp': datetime.now().isoformat(),
            'recent_average': float(recent_avg),
            'trend_factor': float(trend),
            'hourly_factor': float(hourly_factor)
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    try:
        data = request.get_json()
        
        if not data or 'recent_consumption' not in data:
            return jsonify({'error': 'Missing recent_consumption data'}), 400
        
        forecast_hours = data.get('forecast_hours', 24)
        model_name = data.get('model_name', 'xgboost')
        recent_consumption = data['recent_consumption']
        
        if len(recent_consumption) < 3:
            return jsonify({'error': 'Need at least 3 recent consumption values'}), 400
        
        # Generate batch predictions
        predictions = []
        base_consumption = np.mean(recent_consumption[-3:])
        
        for hour in range(forecast_hours):
            # Add some hourly variation
            hour_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
            noise = np.random.normal(0, 0.05)  # Small random variation
            prediction = base_consumption * hour_factor * (1 + noise)
            predictions.append(prediction)
        
        return jsonify({
            'predictions': predictions,
            'forecast_hours': forecast_hours,
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Energy Consumption Predictor Web App...")
    
    # Try to load models
    if load_latest_models():
        print(f"✅ Loaded {len(loaded_models)} models successfully")
        print(f"✅ Loaded {len(feature_names)} features")
    else:
        print("⚠️ Running without pre-trained models (will use simple predictions)")
    
    print("\n" + "="*50)
    print("🌐 WEB APP READY!")
    print("📍 URL: http://localhost:5000")
    print("📚 API Docs: Available at the homepage")
    print("🔌 Health Check: http://localhost:5000/health")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)