"""Quick test to verify src/ modules are working."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure the root directory is in the path so we can import from src
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
            # Updated Progression Suite
            calculate_progressive_passes,
            calculate_progressive_carries,
            calculate_progressive_passes_received,
            calculate_progressive_actions,
            analyze_progression_profile,
            # Aggregations
            aggregate_xg_by_team,
            aggregate_xg_by_player,
        )
        print("  src.metrics - OK")
    except Exception as e:
        print(f"  src.metrics - FAILED: {e}")
        return False
    
    return True


def create_sample_progression_data():
    """Create minimal sample data for testing progression functions."""
    np.random.seed(42)
    n_events = 200
    
    data = {
        'match_id': [1] * n_events,
        'team': ['Team A'] * n_events,
        'player': [f'Player_{i % 11}' for i in range(n_events)],
        'type': np.random.choice(['Pass', 'Carry', 'Duel'], n_events, p=[0.6, 0.3, 0.1]),
        'play_pattern': np.random.choice(['Regular Play', 'From Counter', 'From Throw-In'], n_events, p=[0.7, 0.2, 0.1]),
        'location_x': np.random.uniform(0, 120, n_events),
        'location_y': np.random.uniform(0, 80, n_events),
        'pass_end_location_x': np.random.uniform(0, 120, n_events),
        'pass_end_location_y': np.random.uniform(0, 80, n_events),
        'carry_end_location_x': np.random.uniform(0, 120, n_events),
        'carry_end_location_y': np.random.uniform(0, 80, n_events),
        'pass_outcome': [None if np.random.random() > 0.15 else 'Incomplete' for _ in range(n_events)],
        'pass_recipient': [f'Player_{(i+1) % 11}' if np.random.random() > 0.3 else None for i in range(n_events)],
        'minute': np.random.randint(0, 91, n_events),
    }
    
    df = pd.DataFrame(data)
    return df


def test_progressive_passes():
    """Test calculate_progressive_passes with sample data."""
    print("\nTesting calculate_progressive_passes()...")
    
    try:
        from src.metrics import calculate_progressive_passes
        
        df = create_sample_progression_data()
        
        # Test with pandas DataFrame
        result = calculate_progressive_passes(df)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) > 0, "Should return results"
        assert 'progressive_passes' in result.columns, "Should have progressive_passes column"
        assert 'progressive_pass_pct' in result.columns, "Should have progressive_pass_pct column"
        
        print("  Pandas path - OK")
        print(f"    Found {len(result)} players with progressive passes")
        if len(result) > 0:
            print(f"    Max progressive passes: {result['progressive_passes'].max()}")
        
        # Test with filters
        result_filtered = calculate_progressive_passes(df, player='Player_0')
        print("  Filter by player - OK")
        
        result_match = calculate_progressive_passes(df, match_id=1)
        print("  Filter by match_id - OK")
        
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progressive_carries():
    """Test calculate_progressive_carries with sample data."""
    print("\nTesting calculate_progressive_carries()...")
    
    try:
        from src.metrics import calculate_progressive_carries
        
        df = create_sample_progression_data()
        
        # Test with pandas DataFrame
        result = calculate_progressive_carries(df)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) > 0, "Should return results"
        assert 'progressive_carries' in result.columns, "Should have progressive_carries column"
        assert 'progressive_carry_pct' in result.columns, "Should have progressive_carry_pct column"
        
        print("  Pandas path - OK")
        print(f"    Found {len(result)} players with progressive carries")
        if len(result) > 0:
            print(f"    Max progressive carries: {result['progressive_carries'].max()}")
        
        # Test with filters
        result_filtered = calculate_progressive_carries(df, player='Player_0')
        print("  Filter by player - OK")
        
        result_match = calculate_progressive_carries(df, match_id=1)
        print("  Filter by match_id - OK")
        
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progressive_passes_received():
    """Test calculate_progressive_passes_received with sample data."""
    print("\nTesting calculate_progressive_passes_received()...")
    
    try:
        from src.metrics import calculate_progressive_passes_received
        
        df = create_sample_progression_data()
        
        # Test with pandas DataFrame
        result = calculate_progressive_passes_received(df)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert 'progressive_passes_received' in result.columns, "Should have progressive_passes_received column"
        assert 'progressive_passes_received_pct' in result.columns, "Should have progressive_passes_received_pct column"
        
        print("  Pandas path - OK")
        if len(result) > 0:
            print(f"    Found {len(result)} players with progressive passes received")
            print(f"    Max progressive passes received: {result['progressive_passes_received'].max()}")
        
        # Test with filters
        result_filtered = calculate_progressive_passes_received(df, player='Player_0')
        print("  Filter by player - OK")
        
        result_match = calculate_progressive_passes_received(df, match_id=1)
        print("  Filter by match_id - OK")
        
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progressive_actions():
    """Test calculate_progressive_actions with sample data."""
    print("\nTesting calculate_progressive_actions()...")
    
    try:
        from src.metrics import calculate_progressive_actions
        
        df = create_sample_progression_data()
        
        # Test with pandas DataFrame
        result = calculate_progressive_actions(df)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert 'progressive_actions' in result.columns, "Should have progressive_actions column"
        assert 'progressive_passes' in result.columns, "Should have progressive_passes column"
        assert 'progressive_carries' in result.columns, "Should have progressive_carries column"
        assert 'progressive_passes_received' in result.columns, "Should have progressive_passes_received column"
        
        print("  Pandas path - OK")
        print(f"    Found {len(result)} players with progressive actions")
        if len(result) > 0:
            print(f"    Max progressive actions: {result['progressive_actions'].max()}")
            print(f"    Columns: {', '.join(result.columns.tolist())}")
        
        # Test with filters
        result_filtered = calculate_progressive_actions(df, player='Player_0')
        print("  Filter by player - OK")
        
        result_match = calculate_progressive_actions(df, match_id=1)
        print("  Filter by match_id - OK")
        
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze_progression_profile():
    """Test analyze_progression_profile with sample data."""
    print("\nTesting analyze_progression_profile()...")
    
    try:
        from src.metrics import analyze_progression_profile
        
        df = create_sample_progression_data()
        
        # Test with pandas DataFrame
        result = analyze_progression_profile(df, min_minutes=0)  # Set low threshold for test data
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert 'progression_type' in result.columns, "Should have progression_type column"
        assert 'total_progressive_actions_p90' in result.columns, "Should have total_progressive_actions_p90 column"
        
        print("  Pandas path - OK")
        print(f"    Found {len(result)} players in profile")
        if len(result) > 0:
            print(f"    Progression types: {result['progression_type'].unique().tolist()}")
            print(f"    Top type: {result.iloc[0]['progression_type']} ({result.iloc[0]['player']})")
        
        # Test with filters
        result_filtered = analyze_progression_profile(df, match_id=1, min_minutes=0)
        print("  Filter by match_id - OK")
        
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test that DataLoader can initialize."""
    print("\nTesting DataLoader...")
    
    try:
        from src.data import DataLoader
        
        try:
            # Note: Update path to match your local setup
            loader = DataLoader(data_dir="./data/Statsbomb")
            print("  Initialized successfully")
            
            if hasattr(loader, 'available_files') and loader.available_files:
                print(f"  Found {len(loader.available_files)} data files")
                summary = loader.get_data_summary()
                print(f"  Total matches: {summary.get('total_matches', 'N/A')}")
                print(f"  Total events: {summary.get('total_events', 'N/A'):,}")
            else:
                print("  No data files found (Check data directory)")
            
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
    
    # Updated to include the separate progression and possession files
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
        path = Path(__file__).parent / filepath
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {filepath}: {status}")
        if not exists:
            all_exist = False
    
    return all_exist


if __name__ == "__main__":
    print("="*70)
    print("Module Integrity & Import Tests")
    print("="*70)
    
    results = []
    results.append(("Module Structure", test_module_structure()))
    results.append(("Imports", test_imports()))
    results.append(("DataLoader", test_data_loader()))
    
    # Progression function tests
    results.append(("Progressive Passes", test_progressive_passes()))
    results.append(("Progressive Carries", test_progressive_carries()))
    results.append(("Progressive Passes Received", test_progressive_passes_received()))
    results.append(("Progressive Actions", test_progressive_actions()))
    results.append(("Analyze Progression Profile", test_analyze_progression_profile()))
    
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name.ljust(35)}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests passed. Your Scouting Toolkit is ready.")
    else:
        print("✗ Some tests failed. Check errors above.")
    print("="*70)