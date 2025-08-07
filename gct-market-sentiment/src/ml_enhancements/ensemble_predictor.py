"""
Advanced Ensemble Predictor for TraderAI
Combines XGBoost, LightGBM, and CatBoost for improved predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import logging

logger = logging.getLogger(__name__)


class AdvancedEnsemblePredictor(BaseEstimator, RegressorMixin):
    """
    Advanced ensemble predictor combining multiple gradient boosting models
    with optimized hyperparameters for financial time series
    """
    
    def __init__(self, 
                 use_gpu: bool = False,
                 optimize_for_speed: bool = False):
        """
        Initialize ensemble predictor
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
            optimize_for_speed: Trade some accuracy for faster predictions
        """
        self.use_gpu = use_gpu
        self.optimize_for_speed = optimize_for_speed
        
        # Initialize models with financial market-optimized parameters
        self._init_models()
        
    def _init_models(self):
        """Initialize individual models with optimized parameters"""
        # XGBoost - Good for handling missing data and regularization
        xgb_params = {
            'n_estimators': 200 if self.optimize_for_speed else 300,
            'learning_rate': 0.01,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 1,
            'n_jobs': -1,
            'random_state': 42
        }
        
        if self.use_gpu:
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
            
        # LightGBM - Fastest and memory efficient
        lgb_params = {
            'n_estimators': 200 if self.optimize_for_speed else 300,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 1,
            'n_jobs': -1,
            'random_state': 42,
            'importance_type': 'gain'
        }
        
        if self.use_gpu:
            lgb_params['device'] = 'gpu'
            
        # CatBoost - Best for categorical features
        cat_params = {
            'iterations': 200 if self.optimize_for_speed else 300,
            'learning_rate': 0.01,
            'depth': 7,
            'l2_leaf_reg': 3,
            'min_data_in_leaf': 20,
            'random_strength': 0.5,
            'bagging_temperature': 0.5,
            'od_type': 'Iter',
            'od_wait': 50,
            'random_seed': 42,
            'verbose': False
        }
        
        if self.use_gpu:
            cat_params['task_type'] = 'GPU'
            
        self.models = {
            'xgb': XGBRegressor(**xgb_params),
            'lgb': LGBMRegressor(**lgb_params),
            'cat': CatBoostRegressor(**cat_params)
        }
        
        # Voting ensemble with optimized weights
        self.ensemble = VotingRegressor(
            list(self.models.items()),
            weights=[0.35, 0.35, 0.30]  # Slightly favor XGB and LGB
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            categorical_features: Optional[List[int]] = None,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Fit the ensemble model
        
        Args:
            X: Training features
            y: Training targets
            categorical_features: Indices of categorical features
            validation_data: Optional (X_val, y_val) for early stopping
        """
        logger.info("Training advanced ensemble predictor...")
        
        # Convert to DataFrame for better handling
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        # Prepare data for each model
        if categorical_features:
            # CatBoost handles categoricals natively
            self.models['cat'].set_params(cat_features=categorical_features)
            
        # Train with early stopping if validation data provided
        if validation_data is not None:
            X_val, y_val = validation_data
            
            # XGBoost early stopping
            self.models['xgb'].set_params(
                early_stopping_rounds=50,
                eval_metric='rmse'
            )
            self.models['xgb'].fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # LightGBM early stopping
            self.models['lgb'].fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # CatBoost early stopping
            self.models['cat'].fit(
                X, y,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=False
            )
        else:
            # Standard fit for ensemble
            self.ensemble.fit(X, y)
            
        # Calculate feature importances
        self._calculate_feature_importances(X.shape[1])
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the ensemble"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        return self.ensemble.predict(X)
        
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimates
        
        Returns:
            Dictionary with 'mean', 'std', and individual model predictions
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
            
        # Calculate ensemble statistics
        all_preds = np.column_stack(list(predictions.values()))
        
        return {
            'mean': np.mean(all_preds, axis=1),
            'std': np.std(all_preds, axis=1),
            'predictions': predictions
        }
        
    def _calculate_feature_importances(self, n_features: int):
        """Calculate aggregated feature importances"""
        importances = np.zeros(n_features)
        
        # Average importances across models
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_ / len(self.models)
                
        self.feature_importances_ = importances
        
    def save_model(self, filepath: str):
        """Save the ensemble model"""
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
        
    @staticmethod
    def load_model(filepath: str):
        """Load a saved ensemble model"""
        return joblib.load(filepath)
        

class TimeSeriesEnsemblePredictor(AdvancedEnsemblePredictor):
    """
    Specialized ensemble for time series with feature engineering
    """
    
    def __init__(self, lookback_window: int = 24, **kwargs):
        super().__init__(**kwargs)
        self.lookback_window = lookback_window
        
    def create_features(self, data: pd.DataFrame, 
                       price_col: str = 'close',
                       volume_col: str = 'volume') -> pd.DataFrame:
        """
        Create time series features from raw data
        
        Args:
            data: DataFrame with time series data
            price_col: Name of price column
            volume_col: Name of volume column
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data[price_col].pct_change()
        features['log_returns'] = np.log(data[price_col] / data[price_col].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = data[price_col].rolling(window).mean()
            features[f'ma_ratio_{window}'] = data[price_col] / features[f'ma_{window}']
            
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            
        # Volume features
        if volume_col in data.columns:
            features['volume_ma_5'] = data[volume_col].rolling(5).mean()
            features['volume_ratio'] = data[volume_col] / features['volume_ma_5']
            features['price_volume'] = data[price_col] * data[volume_col]
            
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data[price_col])
        bb_upper, bb_lower = self._calculate_bollinger_bands(data[price_col])
        features['bb_position'] = (data[price_col] - bb_lower) / (bb_upper - bb_lower)
        
        # Lag features
        for lag in range(1, self.lookback_window + 1):
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            
        # Drop NaN rows
        features = features.dropna()
        
        return features
        
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, 
                                  window: int = 20, 
                                  num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, lower