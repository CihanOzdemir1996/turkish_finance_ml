"""
Time Series Validation Utilities for Financial Data

This module provides proper validation strategies for time-series financial data,
preventing look-ahead bias and data leakage.

Key Features:
- TimeSeriesSplit: Chronological cross-validation
- Walk-Forward Analysis: Industry-standard validation for trading systems
- Proper scaler handling: Fit only on training data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesValidator:
    """
    Time Series Cross-Validation for Financial Data
    
    Ensures no data leakage by:
    1. Fitting scalers only on training data
    2. Respecting temporal order
    3. Providing proper train/validation/test splits
    """
    
    def __init__(self, n_splits=5, test_size=None, gap=0):
        """
        Initialize Time Series Validator
        
        Args:
            n_splits: Number of splits for cross-validation
            test_size: Size of test set (if None, uses default split)
            gap: Gap between train and test to prevent leakage
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    
    def split(self, X, y=None):
        """Generate train/test indices respecting temporal order"""
        return self.tscv.split(X, y)
    
    def validate_with_scaler(self, X, y, model, scaler_type='standard', 
                            return_scalers=False, verbose=True):
        """
        Perform time series cross-validation with proper scaler handling
        
        Args:
            X: Feature matrix (DataFrame or array)
            y: Target vector
            model: Model to validate (must have fit, predict, score methods)
            scaler_type: 'standard' or 'minmax'
            return_scalers: If True, return fitted scalers for each fold
            verbose: Print progress
            
        Returns:
            Dictionary with validation results
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'confusion_matrices': []
        }
        
        scalers = [] if return_scalers else None
        
        for fold, (train_idx, val_idx) in enumerate(self.split(X)):
            # Split data
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # CRITICAL: Fit scaler ONLY on training data
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler_type: {scaler_type}")
            
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)  # Transform, don't fit!
            
            if return_scalers:
                scalers.append(scaler)
            
            # Train model
            model.fit(X_train_scaled, y_train_fold)
            
            # Predictions
            y_val_pred = model.predict(X_val_scaled)
            
            # Probabilities (if available)
            try:
                y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
                roc_auc = roc_auc_score(y_val_fold, y_val_proba)
            except:
                roc_auc = None
            
            # Metrics
            acc = accuracy_score(y_val_fold, y_val_pred)
            prec = precision_score(y_val_fold, y_val_pred, zero_division=0)
            rec = recall_score(y_val_fold, y_val_pred, zero_division=0)
            f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)
            cm = confusion_matrix(y_val_fold, y_val_pred)
            
            cv_scores['accuracy'].append(acc)
            cv_scores['precision'].append(prec)
            cv_scores['recall'].append(rec)
            cv_scores['f1'].append(f1)
            if roc_auc is not None:
                cv_scores['roc_auc'].append(roc_auc)
            cv_scores['confusion_matrices'].append(cm)
            
            if verbose:
                print(f"Fold {fold+1}/{self.n_splits}: "
                      f"Accuracy={acc:.4f}, Precision={prec:.4f}, "
                      f"Recall={rec:.4f}, F1={f1:.4f}")
        
        # Summary statistics
        results = {
            'mean_accuracy': np.mean(cv_scores['accuracy']),
            'std_accuracy': np.std(cv_scores['accuracy']),
            'mean_precision': np.mean(cv_scores['precision']),
            'std_precision': np.std(cv_scores['precision']),
            'mean_recall': np.mean(cv_scores['recall']),
            'std_recall': np.std(cv_scores['recall']),
            'mean_f1': np.mean(cv_scores['f1']),
            'std_f1': np.std(cv_scores['f1']),
            'fold_scores': cv_scores
        }
        
        if cv_scores['roc_auc']:
            results['mean_roc_auc'] = np.mean(cv_scores['roc_auc'])
            results['std_roc_auc'] = np.std(cv_scores['roc_auc'])
        
        if return_scalers:
            results['scalers'] = scalers
        
        return results


class WalkForwardValidator:
    """
    Walk-Forward Analysis for Trading Systems
    
    Industry-standard validation that mimics real trading:
    - Train on past data
    - Test on future data
    - Slide forward in time
    """
    
    def __init__(self, train_size=1000, test_size=250, step=250):
        """
        Initialize Walk-Forward Validator
        
        Args:
            train_size: Size of training window
            test_size: Size of test window
            step: Step size for sliding window
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
    
    def split(self, data_length):
        """
        Generate train/test splits for walk-forward analysis
        
        Yields:
            (train_start, train_end, test_start, test_end) tuples
        """
        for i in range(0, data_length - self.train_size - self.test_size, self.step):
            train_start = i
            train_end = i + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size
            
            if test_end > data_length:
                break
            
            yield (train_start, train_end, test_start, test_end)
    
    def validate(self, X, y, model, scaler_type='standard', verbose=True):
        """
        Perform walk-forward validation
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to validate
            scaler_type: 'standard' or 'minmax'
            verbose: Print progress
            
        Returns:
            DataFrame with results for each walk-forward period
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        results = []
        
        for train_start, train_end, test_start, test_end in self.split(len(X)):
            # Split data
            X_train = X[train_start:train_end]
            X_test = X[test_start:test_end]
            y_train = y[train_start:train_end]
            y_test = y[test_start:test_end]
            
            # Fit scaler ONLY on training data
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler_type: {scaler_type}")
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_test_pred = model.predict(X_test_scaled)
            
            # Probabilities (if available)
            try:
                y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                roc_auc = roc_auc_score(y_test, y_test_proba)
            except:
                roc_auc = None
            
            # Metrics
            acc = accuracy_score(y_test, y_test_pred)
            prec = precision_score(y_test, y_test_pred, zero_division=0)
            rec = recall_score(y_test, y_test_pred, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            results.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': roc_auc
            })
            
            if verbose:
                print(f"Period {len(results)}: "
                      f"Train[{train_start}:{train_end}] -> Test[{test_start}:{test_end}], "
                      f"Accuracy={acc:.4f}")
        
        results_df = pd.DataFrame(results)
        
        # Summary statistics
        summary = {
            'mean_accuracy': results_df['accuracy'].mean(),
            'std_accuracy': results_df['accuracy'].std(),
            'mean_precision': results_df['precision'].mean(),
            'std_precision': results_df['precision'].std(),
            'mean_recall': results_df['recall'].mean(),
            'std_recall': results_df['recall'].std(),
            'mean_f1': results_df['f1'].mean(),
            'std_f1': results_df['f1'].std(),
        }
        
        if results_df['roc_auc'].notna().any():
            summary['mean_roc_auc'] = results_df['roc_auc'].mean()
            summary['std_roc_auc'] = results_df['roc_auc'].std()
        
        return results_df, summary


def create_proper_train_test_split(X, y, test_size=0.2, validation_size=0.1, 
                                   scaler_type='standard', return_scaler=True):
    """
    Create proper train/validation/test split with scaler handling
    
    CRITICAL: Fits scaler ONLY on training data to prevent data leakage
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for test set
        validation_size: Proportion of data for validation set (from training data)
        scaler_type: 'standard' or 'minmax'
        return_scaler: If True, return fitted scaler
        
    Returns:
        Dictionary with splits and optionally scaler
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Time series split: train -> validation -> test (chronological)
    n_samples = len(X)
    test_start = int(n_samples * (1 - test_size))
    val_start = int(test_start * (1 - validation_size / (1 - test_size)))
    
    # Split indices
    train_idx = slice(0, val_start)
    val_idx = slice(val_start, test_start)
    test_idx = slice(test_start, n_samples)
    
    # Split data
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    # CRITICAL: Fit scaler ONLY on training data
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)  # Transform, don't fit!
    X_test_scaled = scaler.transform(X_test)  # Transform, don't fit!
    
    result = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_indices': test_idx
    }
    
    if return_scaler:
        result['scaler'] = scaler
    
    return result
