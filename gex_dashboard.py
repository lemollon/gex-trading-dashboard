# GEX Strategy Dashboard with Databricks Integration
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import yfinance as yf
import time
from typing import Dict, List, Optional
import sqlite3
import os

# Page configuration
st.set_page_config(
    page_title="GEX Trading Strategy Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and readability
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .strategy-box {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        color: #2d3748;
    }
    
    .strategy-box h2 {
        color: #1a202c !important;
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    
    .strategy-box h3 {
        color: #2d3748 !important;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    .strategy-box p {
        color: #4a5568 !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .bullish-signal {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(72, 187, 120, 0.3);
        border: none;
        transform: translateY(0);
        transition: all 0.3s ease;
    }
    
    .bullish-signal:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(72, 187, 120, 0.4);
    }
    
    .bullish-signal h3 {
        color: white !important;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }
    
    .bullish-signal p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem;
        margin: 0.3rem 0;
    }
    
    .bullish-signal strong {
        color: #f7fafc !important;
        font-weight: 700;
    }
    
    .bearish-signal {
        background: linear-gradient(135deg, #f56565, #e53e3e);
        color: white !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(245, 101, 101, 0.3);
        border: none;
        transform: translateY(0);
        transition: all 0.3s ease;
    }
    
    .bearish-signal:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(245, 101, 101, 0.4);
    }
    
    .bearish-signal h3 {
        color: white !important;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }
    
    .bearish-signal p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem;
        margin: 0.3rem 0;
    }
    
    .bearish-signal strong {
        color: #f7fafc !important;
        font-weight: 700;
    }
    
    .neutral-signal {
        background: linear-gradient(135deg, #ed8936, #dd6b20);
        color: white !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(237, 137, 54, 0.3);
        border: none;
        transform: translateY(0);
        transition: all 0.3s ease;
    }
    
    .neutral-signal:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(237, 137, 54, 0.4);
    }
    
    .neutral-signal h3 {
        color: white !important;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }
    
    .neutral-signal p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem;
        margin: 0.3rem 0;
    }
    
    .neutral-signal strong {
        color: #f7fafc !important;
        font-weight: 700;
    }
    
    /* Streamlit specific overrides */
    .stMarkdown {
        color: white;
    }
    
    .stSelectbox label {
        color: white !important;
        font-weight: 600;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stMetric [data-testid="metric-container"] {
        background: transparent;
        border: none;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    
    .stSidebar .stMarkdown {
        color: white;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Make text visible everywhere */
    .stApp .element-container {
        color: white;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 700;
    }
    
    p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem;
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class DatabricksMockAccount:
    """Mock trading account with $100k starting balance - Databricks integrated"""
    
    def __init__(self):
        self.initial_balance = 100000
        try:
            self.databricks_config = {
                'server_hostname': st.secrets.get('databricks_hostname', 'demo-mode'),
                'http_path': st.secrets.get('databricks_http_path', 'demo-mode'),
                'access_token': st.secrets.get('databricks_token', 'demo-mode')
            }
        except:
            self.databricks_config = {
                'server_hostname': 'demo-mode',
                'http_path': 'demo-mode',
                'access_token': 'demo-mode'
            }
        self.init_databricks_tables()
    
    def init_databricks_tables(self):
        """Initialize Databricks Delta tables for portfolio tracking"""
        
        create_trades_table = """
        CREATE TABLE IF NOT EXISTS g
