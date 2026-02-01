"""
Create train/validation/test splits.
"""
import pandas as pd
import json
from datetime import datetime


def create_temporal_splits(matches, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    # sort by date and split into id arrays
    m_sorted = matches.sort_values('match_date').reset_index(drop=True)
    n = len(m_sorted)
    t_size = int(train_ratio * n)
    v_size = int(val_ratio * n)

    train_ids = m_sorted.iloc[:t_size]['match_id'].values
    val_ids = m_sorted.iloc[t_size:t_size+v_size]['match_id'].values
    test_ids = m_sorted.iloc[t_size+v_size:]['match_id'].values

    return {'train': train_ids, 'val': val_ids, 'test': test_ids}


def split_by_match_ids(df, match_id_splits):
    return {
        'train': df[df['match_id'].isin(match_id_splits['train'])].copy(),
        'val': df[df['match_id'].isin(match_id_splits['val'])].copy(),
        'test': df[df['match_id'].isin(match_id_splits['test'])].copy()
    }


def create_splits(matches, events, lineups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits for matches, events, and lineups.
    Returns a dict with 'train','val','test' and 'metadata'.
    """
    # ensure datetime
    matches['match_date'] = pd.to_datetime(matches['match_date'])

    ids = create_temporal_splits(matches, train_ratio, val_ratio, test_ratio)

    ms = split_by_match_ids(matches, ids)
    es = split_by_match_ids(events, ids)
    ls = split_by_match_ids(lineups, ids)

    # date ranges
    t_dates = ms['train']['match_date']
    v_dates = ms['val']['match_date']
    s_dates = ms['test']['match_date']

    meta = {
        'train_match_ids': ids['train'].tolist(),
        'val_match_ids': ids['val'].tolist(),
        'test_match_ids': ids['test'].tolist(),
        'train_size': len(ids['train']),
        'val_size': len(ids['val']),
        'test_size': len(ids['test']),
        'train_date_range': (str(t_dates.min().date()), str(t_dates.max().date())),
        'val_date_range': (str(v_dates.min().date()), str(v_dates.max().date())),
        'test_date_range': (str(s_dates.min().date()), str(s_dates.max().date())),
        'created_at': str(datetime.now())
    }

    print(f"Splits created: train={meta['train_size']}, val={meta['val_size']}, test={meta['test_size']}")

    return {
        'train': {'matches': ms['train'], 'events': es['train'], 'lineups': ls['train']},
        'val': {'matches': ms['val'], 'events': es['val'], 'lineups': ls['val']},
        'test': {'matches': ms['test'], 'events': es['test'], 'lineups': ls['test']},
        'metadata': meta
    }