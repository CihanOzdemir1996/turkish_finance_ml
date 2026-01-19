"""
Data Collection Module for Turkish Financial Data
Collects BIST stock prices and macroeconomic indicators from various sources
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


class TurkishFinancialDataCollector:
    """
    Collects Turkish financial data from multiple sources:
    1. CBRT EVDS API - Macroeconomic indicators (inflation, interest rates)
    2. Yahoo Finance - BIST stock prices
    3. Kaggle datasets (manual download)
    """
    
    def __init__(self, data_dir=None, evds_api_key=None):
        """
        Initialize collector
        
        Args:
            data_dir: Directory to save collected data
            evds_api_key: CBRT EVDS API key (get from https://evds2.tcmb.gov.tr/)
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data" / "raw"
        else:
            self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.evds_api_key = evds_api_key
        self.evds_base_url = "https://evds2.tcmb.gov.tr/service/evds"
        
    def collect_cbrt_macroeconomic_data(self, start_date="01-01-2000", end_date=None):
        """
        Collect macroeconomic data from CBRT EVDS API
        
        Key Series Codes:
        - TP.DK.YTL.A.YTL: Consumer Price Index (CPI/TÜFE) - Annual
        - TP.DK.USD.A.YTL: USD/TRY Exchange Rate
        - TP.INT.RSK.A: Policy Interest Rate (Repo Rate)
        - TP.DK.GB.A.YTL: Producer Price Index (PPI/ÜFE) - Annual
        
        Args:
            start_date: Start date in format "dd-mm-yyyy"
            end_date: End date in format "dd-mm-yyyy" (default: today)
        
        Returns:
            DataFrame with macroeconomic indicators
        """
        if self.evds_api_key is None:
            print("⚠️  EVDS API key not provided. Please get one from: https://evds2.tcmb.gov.tr/")
            print("   For now, returning sample data structure...")
            return self._create_sample_macro_data()
        
        if end_date is None:
            end_date = datetime.now().strftime("%d-%m-%Y")
        
        # Key macroeconomic series codes
        series_codes = {
            "TP.DK.YTL.A.YTL": "CPI_Annual",  # Consumer Price Index (Annual)
            "TP.DK.USD.A.YTL": "USD_TRY",     # USD/TRY Exchange Rate
            "TP.INT.RSK.A": "Policy_Rate",    # Policy Interest Rate
            "TP.DK.GB.A.YTL": "PPI_Annual",   # Producer Price Index (Annual)
        }
        
        all_data = []
        
        for series_code, name in series_codes.items():
            try:
                url = f"{self.evds_base_url}/dataseries/{series_code}"
                params = {
                    "key": self.evds_api_key,
                    "startDate": start_date,
                    "endDate": end_date,
                    "type": "json",
                    "formulas": "0"  # 0 = raw data, 1 = YoY change, etc.
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if 'items' in data:
                    df = pd.DataFrame(data['items'])
                    if 'Tarih' in df.columns:
                        df['Date'] = pd.to_datetime(df['Tarih'], format='%d-%m-%Y')
                        df = df.rename(columns={df.columns[1]: name})
                        df = df[['Date', name]].copy()
                        df[name] = pd.to_numeric(df[name], errors='coerce')
                        all_data.append(df)
                        print(f"✅ Collected {name}: {len(df)} records")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error collecting {name}: {str(e)}")
                continue
        
        if all_data:
            # Merge all series on Date
            macro_df = all_data[0]
            for df in all_data[1:]:
                macro_df = macro_df.merge(df, on='Date', how='outer')
            
            macro_df = macro_df.sort_values('Date').reset_index(drop=True)
            
            # Save to file
            output_file = self.data_dir / "cbrt_macroeconomic_data.csv"
            macro_df.to_csv(output_file, index=False)
            print(f"\n✅ Saved macroeconomic data to: {output_file}")
            print(f"   Shape: {macro_df.shape}")
            print(f"   Date range: {macro_df['Date'].min()} to {macro_df['Date'].max()}")
            
            return macro_df
        else:
            return pd.DataFrame()
    
    def collect_bist_stock_data(self, tickers=None, start_date="2000-01-01", end_date=None):
        """
        Collect BIST stock prices from Yahoo Finance
        
        Args:
            tickers: List of BIST tickers (e.g., ['AKBNK.IS', 'GARAN.IS'])
                    If None, collects BIST-100 index
            start_date: Start date in format "yyyy-mm-dd"
            end_date: End date in format "yyyy-mm-dd" (default: today)
        
        Returns:
            DataFrame with stock prices
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if tickers is None:
            # Default: Collect BIST-100 index
            tickers = ['XU100.IS']  # BIST-100 index
        
        all_stocks = []
        
        for ticker in tickers:
            try:
                print(f"📊 Collecting {ticker}...")
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    hist = hist.reset_index()
                    hist['Ticker'] = ticker
                    all_stocks.append(hist)
                    print(f"   ✅ {ticker}: {len(hist)} records ({hist['Date'].min()} to {hist['Date'].max()})")
                else:
                    print(f"   ⚠️  No data for {ticker}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"   ❌ Error collecting {ticker}: {str(e)}")
                continue
        
        if all_stocks:
            stock_df = pd.concat(all_stocks, ignore_index=True)
            
            # Save to file
            output_file = self.data_dir / "bist_stock_prices.csv"
            stock_df.to_csv(output_file, index=False)
            print(f"\n✅ Saved stock data to: {output_file}")
            print(f"   Shape: {stock_df.shape}")
            print(f"   Tickers: {stock_df['Ticker'].unique()}")
            
            return stock_df
        else:
            return pd.DataFrame()
    
    def collect_bist_100_companies(self):
        """
        Collect data for major BIST-100 companies
        Common tickers: AKBNK.IS, GARAN.IS, THYAO.IS, TUPRS.IS, etc.
        """
        # Major BIST-100 companies
        major_tickers = [
            'AKBNK.IS',  # Akbank
            'GARAN.IS',  # Garanti BBVA
            'THYAO.IS',  # Turkish Airlines
            'TUPRS.IS',  # Tüpraş
            'SAHOL.IS',  # Hacı Ömer Sabancı Holding
            'BIMAS.IS',  # BIM
            'ARCLK.IS',  # Arçelik
            'KOZAL.IS',  # Koza Altın
            'SASA.IS',   # Sasa
            'PETKM.IS',  # Petkim
        ]
        
        return self.collect_bist_stock_data(tickers=major_tickers)
    
    def _create_sample_macro_data(self):
        """Create sample macroeconomic data structure for demonstration"""
        dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='M')
        
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'Date': dates,
            'CPI_Annual': 20 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'USD_TRY': 5 + np.cumsum(np.random.randn(len(dates)) * 0.1),
            'Policy_Rate': 15 + np.random.randn(len(dates)) * 2,
            'PPI_Annual': 25 + np.cumsum(np.random.randn(len(dates)) * 0.6),
        })
        
        output_file = self.data_dir / "cbrt_macroeconomic_data_sample.csv"
        sample_data.to_csv(output_file, index=False)
        print(f"📝 Created sample data structure: {output_file}")
        print("   ⚠️  This is SAMPLE data. Get real data from CBRT EVDS API!")
        
        return sample_data


def get_evds_api_key_instructions():
    """Print instructions for getting EVDS API key"""
    print("""
    📋 How to Get CBRT EVDS API Key:
    
    1. Visit: https://evds2.tcmb.gov.tr/
    2. Click "Register" or "Login" if you have an account
    3. After logging in, go to your profile/settings
    4. Find "API Key" section and generate a new key
    5. Copy the API key and use it in the collector
    
    The API is FREE and provides official Turkish macroeconomic data!
    """)


if __name__ == "__main__":
    # Example usage
    collector = TurkishFinancialDataCollector()
    
    print("=" * 60)
    print("Turkish Financial Data Collector")
    print("=" * 60)
    
    # Show API key instructions
    get_evds_api_key_instructions()
    
    # Example: Collect BIST-100 index data (no API key needed)
    print("\n" + "=" * 60)
    print("Collecting BIST-100 Index Data...")
    print("=" * 60)
    bist_data = collector.collect_bist_stock_data()
    
    # Example: Collect macroeconomic data (requires API key)
    print("\n" + "=" * 60)
    print("To collect macroeconomic data, set your EVDS API key:")
    print("collector = TurkishFinancialDataCollector(evds_api_key='YOUR_KEY')")
    print("macro_data = collector.collect_cbrt_macroeconomic_data()")
    print("=" * 60)
