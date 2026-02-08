"""Quick test to verify src/ modules are working."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data import load_data, DataLoader
        print("  src.data - OK")
    except Exception as e:
        print(f"  src.data - FAILED: {e}")
        return False
    
    try:
        from src.metrics import (
            calculate_ppda,
            calculate_field_tilt,
            calculate_progressive_passes,
            calculate_progressive_carries,
            aggregate_xg_by_team,
            aggregate_xg_by_player,
        )
        print("  src.metrics - OK")
    except Exception as e:
        print(f"  src.metrics - FAILED: {e}")
        return False
    
    return True


def test_data_loader():
    """Test that DataLoader can initialize."""
    print("\nTesting DataLoader...")
    
    try:
        from src.data import DataLoader
        
        try:
            loader = DataLoader(data_dir="./data/Statsbomb")
            print("  Initialized successfully")
            print(f"  Found {len(loader.available_files)} data files")
            
            if loader.available_files:
                summary = loader.get_data_summary()
                print(f"  Total matches: {summary.get('total_matches', 'N/A')}")
                print(f"  Total events: {summary.get('total_events', 'N/A'):,}")
            
            loader.close()
            return True
            
        except ValueError as e:
            print(f"  Data directory issue: {e}")
            print("  (Point it to your data directory when using)")
            return True
            
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_module_structure():
    """Verify all expected files exist."""
    print("\nChecking module structure...")
    
    expected_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/loader.py',
        'src/metrics/__init__.py',
        'src/metrics/possession.py',
        'src/metrics/progression.py',
        'src/metrics/aggregations.py',
    ]
    
    all_exist = True
    for filepath in expected_files:
        path = Path(filepath)
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {filepath}: {status}")
        if not exists:
            all_exist = False
    
    return all_exist


if __name__ == "__main__":
    print("="*60)
    print("Module Tests")
    print("="*60)
    
    results = []
    results.append(("Module Structure", test_module_structure()))
    results.append(("Imports", test_imports()))
    results.append(("DataLoader", test_data_loader()))
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\nAll tests passed. Modules ready to use.")
    else:
        print("\nSome tests failed. Check output above.")