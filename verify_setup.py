"""
Quick verification script to check if everything is set up correctly
before running the data collection notebook
"""

import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has API key"""
    print("=" * 60)
    print("Checking .env file...")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print(f"   Expected location: {env_file}")
        print("\n   Create .env file with:")
        print("   EVDS_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ .env file found: {env_file}")
    
    # Check contents
    with open(env_file, 'r') as f:
        content = f.read()
        if 'EVDS_API_KEY' in content:
            # Extract key
            for line in content.split('\n'):
                if line.strip().startswith('EVDS_API_KEY'):
                    key_value = line.split('=', 1)[1].strip() if '=' in line else ''
                    if key_value and key_value != 'your_api_key_here':
                        print(f"✅ API key found: {key_value[:10]}...{key_value[-5:]}")
                        return True
                    else:
                        print("⚠️  API key in .env file is not set (still has placeholder)")
                        return False
    
    print("❌ EVDS_API_KEY not found in .env file")
    return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 60)
    print("Checking dependencies...")
    print("=" * 60)
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'requests': 'requests',
        'yfinance': 'yfinance',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - NOT INSTALLED")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_folders():
    """Check if required folders exist"""
    print("\n" + "=" * 60)
    print("Checking folder structure...")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    required_folders = [
        project_root / "data" / "raw",
        project_root / "src",
        project_root / "notebooks",
    ]
    
    all_exist = True
    for folder in required_folders:
        if folder.exists():
            print(f"✅ {folder.relative_to(project_root)}")
        else:
            print(f"❌ {folder.relative_to(project_root)} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_files():
    """Check if required files exist"""
    print("\n" + "=" * 60)
    print("Checking required files...")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    required_files = [
        project_root / "src" / "data_collection.py",
        project_root / "src" / "load_env.py",
        project_root / "notebooks" / "01_data_collection.ipynb",
    ]
    
    all_exist = True
    for file in required_files:
        if file.exists():
            print(f"✅ {file.relative_to(project_root)}")
        else:
            print(f"❌ {file.relative_to(project_root)} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_api_key_loading():
    """Test if API key can be loaded from .env"""
    print("\n" + "=" * 60)
    print("Testing API key loading...")
    print("=" * 60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from load_env import get_evds_api_key
        
        api_key = get_evds_api_key()
        if api_key:
            print(f"✅ API key loaded successfully: {api_key[:10]}...{api_key[-5:]}")
            return True
        else:
            print("❌ API key could not be loaded from .env file")
            return False
    except Exception as e:
        print(f"❌ Error loading API key: {str(e)}")
        return False


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("PRE-RUN VERIFICATION CHECKLIST")
    print("=" * 60)
    print("\nThis script verifies your setup before running data collection.\n")
    
    checks = [
        ("Environment File", check_env_file),
        ("Dependencies", check_dependencies),
        ("Folder Structure", check_folders),
        ("Required Files", check_files),
        ("API Key Loading", test_api_key_loading),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name} check: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED! You're ready to run the notebook.")
        print("\nNext steps:")
        print("1. Open notebooks/01_data_collection.ipynb")
        print("2. Run all cells in order")
        print("3. Data will be saved to data/raw/")
    else:
        print("❌ SOME CHECKS FAILED. Please fix the issues above.")
        print("\nSee PRE_RUN_CHECKLIST.md for detailed help.")
    print("=" * 60)


if __name__ == "__main__":
    main()
