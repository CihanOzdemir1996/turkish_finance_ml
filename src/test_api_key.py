"""
Quick script to test your CBRT EVDS API key
Run this after getting your API key to verify it works
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data_collection import TurkishFinancialDataCollector

def test_api_key(api_key):
    """
    Test if the API key works by making a small data request
    
    Args:
        api_key: Your CBRT EVDS API key
    """
    print("=" * 60)
    print("Testing CBRT EVDS API Key")
    print("=" * 60)
    print(f"\nAPI Key: {api_key[:10]}...{api_key[-5:]} (hidden for security)")
    print("\nAttempting to collect test data...")
    
    try:
        # Initialize collector
        collector = TurkishFinancialDataCollector(evds_api_key=api_key)
        
        # Try to collect a small sample of data (just 2023)
        print("\n📊 Testing with CPI data from 2023...")
        test_data = collector.collect_cbrt_macroeconomic_data(
            start_date="01-01-2023",
            end_date="31-12-2023"
        )
        
        if not test_data.empty:
            print("\n" + "=" * 60)
            print("✅ SUCCESS! Your API key works!")
            print("=" * 60)
            print(f"\nCollected {len(test_data)} records")
            print(f"Columns: {test_data.columns.tolist()}")
            print(f"\nSample data:")
            print(test_data.head())
            print("\n✅ You can now use this API key in your notebooks!")
            return True
        else:
            print("\n" + "=" * 60)
            print("⚠️  WARNING: No data returned")
            print("=" * 60)
            print("\nPossible reasons:")
            print("  - API key might be invalid")
            print("  - Date range might not have data")
            print("  - Series codes might have changed")
            print("\nTry checking your API key on the EVDS website.")
            return False
            
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR: API key test failed")
        print("=" * 60)
        print(f"\nError message: {str(e)}")
        print("\nPossible issues:")
        print("  1. Invalid API key - check you copied it correctly")
        print("  2. Network connection issue")
        print("  3. EVDS service temporarily unavailable")
        print("\nTroubleshooting:")
        print("  - Verify your API key at: https://evds2.tcmb.gov.tr/")
        print("  - Make sure you're logged into your EVDS account")
        print("  - Check your internet connection")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CBRT EVDS API Key Tester")
    print("=" * 60)
    print("\nThis script tests if your CBRT EVDS API key is working.")
    print("You need to get your API key from: https://evds2.tcmb.gov.tr/")
    print("\n" + "-" * 60)
    
    # Get API key from user
    api_key = input("\nEnter your CBRT EVDS API key: ").strip()
    
    if not api_key or api_key.lower() == "your_api_key_here":
        print("\n❌ Please enter a valid API key!")
        print("   Get one from: https://evds2.tcmb.gov.tr/")
        sys.exit(1)
    
    # Test the key
    success = test_api_key(api_key)
    
    if success:
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("1. Save your API key securely (use .env file)")
        print("2. Open notebooks/01_data_collection.ipynb")
        print("3. Set EVDS_API_KEY = 'your_key_here'")
        print("4. Run the notebook to collect full dataset!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Need Help?")
        print("=" * 60)
        print("See CBRT_API_SETUP.md for detailed registration instructions")
        print("=" * 60)
