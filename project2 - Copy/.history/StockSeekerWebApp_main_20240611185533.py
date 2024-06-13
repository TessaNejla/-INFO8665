#Import Required Libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas_ta as ta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
#Suppress Warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
#Load Custom CSS for Styling
# Function to load local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Provide the path to the style.css file
style_css_path = "path/to/your/style.css"  # Update the path as needed
local_css(style_css_path)
#Sidebar Setup
