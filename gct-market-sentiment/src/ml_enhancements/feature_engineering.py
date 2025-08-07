"""
Advanced Feature Engineering for Financial Time Series
Implements automated feature extraction using tsfresh and custom indicators
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import ta
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FinancialFeatureEngineer:
    """
    Comprehensive feature engineering for financial time series
    combining technical indicators, statistical features, and ML-based features
    """
    
    def __init__(self,
                 use_tsfresh: bool = True,
                 use_ta: bool = True,
                 use_custom: bool = True,
                 feature_selection: bool = True,
                 n_jobs: int = -1):
        """
        Initialize feature engineer
        
        Args:
            use_tsfresh: Whether to use tsfresh for automated features
            use_ta: Whether to use ta library technical indicators
            use_custom: Whether to use custom financial features
            feature_selection: Whether to perform feature selection
            n_jobs: Number of parallel jobs
        """
        self.use_tsfresh = use_tsfresh
        self.use_ta = use_ta
        self.use_custom = use_custom
        self.feature_selection = feature_selection
        self.n_jobs = n_jobs
        
        self.scaler = None
        self.selected_features = None
        self.feature_importance = None
        
    def create_features(self, 
                       df: pd.DataFrame,
                       target_col: Optional[str] = None,
                       price_col: str = 'close',
                       volume_col: str = 'volume',
                       high_col: str = 'high',
                       low_col: str = 'low',
                       open_col: str = 'open') -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            target_col: Target column for feature selection
            price_col: Close price column name
            volume_col: Volume column name
            high_col: High price column name
            low_col: Low price column name
            open_col: Open price column name
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        features_list = []
        
        # 1. Basic price and volume features
        if self.use_custom:
            custom_features = self._create_custom_features(
                df, price_col, volume_col, high_col, low_col, open_col
            )
            features_list.append(custom_features)
            logger.info(f"Created {len(custom_features.columns)} custom features")
            
        # 2. Technical Analysis features
        if self.use_ta:
            ta_features = self._create_ta_features(
                df, high_col, low_col, price_col, volume_col
            )
            features_list.append(ta_features)
            logger.info(f"Created {len(ta_features.columns)} technical indicators")
            
        # 3. TSFresh automated features
        if self.use_tsfresh and len(df) > 50:  # Need sufficient data
            tsfresh_features = self._create_tsfresh_features(
                df, price_col, volume_col
            )
            if tsfresh_features is not None:
                features_list.append(tsfresh_features)
                logger.info(f"Created {len(tsfresh_features.columns)} tsfresh features")
                
        # Combine all features
        if features_list:
            all_features = pd.concat(features_list, axis=1)
            
            # Remove duplicate columns
            all_features = all_features.loc[:, ~all_features.columns.duplicated()]
            
            # Handle infinities and NaNs
            all_features = all_features.replace([np.inf, -np.inf], np.nan)
            all_features = all_features.fillna(method='ffill').fillna(0)
            
            # Feature selection if target provided
            if self.feature_selection and target_col is not None and target_col in df.columns:
                all_features = self._select_features(
                    all_features, df[target_col]
                )
                
            logger.info(f"Total features created: {len(all_features.columns)}")
            return all_features
        else:
            logger.warning("No features were created")
            return pd.DataFrame(index=df.index)
            
    def _create_custom_features(self, df, price_col, volume_col, 
                               high_col, low_col, open_col) -> pd.DataFrame:
        """Create custom financial features"""
        features = pd.DataFrame(index=df.index)
        
        # Returns at different frequencies
        for period in [1, 5, 10, 20]:
            features[f'returns_{period}'] = df[price_col].pct_change(period)
            features[f'log_returns_{period}'] = np.log(df[price_col] / df[price_col].shift(period))
            
        # Price ratios and spreads
        features['high_low_ratio'] = df[high_col] / df[low_col]
        features['close_open_ratio'] = df[price_col] / df[open_col]
        features['high_low_spread'] = (df[high_col] - df[low_col]) / df[price_col]
        features['close_open_spread'] = (df[price_col] - df[open_col]) / df[open_col]
        
        # Volume features
        if volume_col in df.columns:
            features['volume_price'] = df[volume_col] * df[price_col]
            features['volume_change'] = df[volume_col].pct_change()
            
            # VWAP
            typical_price = (df[high_col] + df[low_col] + df[price_col]) / 3
            features['vwap'] = (typical_price * df[volume_col]).rolling(20).sum() / df[volume_col].rolling(20).sum()
            features['price_vwap_ratio'] = df[price_col] / features['vwap']
            
            # Volume patterns
            for period in [5, 10, 20]:
                features[f'volume_ma_{period}'] = df[volume_col].rolling(period).mean()
                features[f'volume_ratio_{period}'] = df[volume_col] / features[f'volume_ma_{period}']
                
        # Volatility features
        for period in [5, 10, 20, 50]:
            features[f'volatility_{period}'] = features['returns_1'].rolling(period).std()
            features[f'volatility_ratio_{period}'] = (
                features[f'volatility_{period}'] / features['volatility_20']
            )
            
        # Price position features
        for period in [10, 20, 50]:
            rolling_high = df[high_col].rolling(period).max()
            rolling_low = df[low_col].rolling(period).min()
            features[f'price_position_{period}'] = (
                (df[price_col] - rolling_low) / (rolling_high - rolling_low)
            )
            
        # Microstructure features
        features['tick_direction'] = np.sign(df[price_col].diff())
        features['tick_direction_run'] = (
            features['tick_direction'].groupby(
                (features['tick_direction'] != features['tick_direction'].shift()).cumsum()
            ).cumcount()
        )
        
        # Time-based features (if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['day_of_month'] = df.index.day
            features['is_month_start'] = df.index.is_month_start.astype(int)
            features['is_month_end'] = df.index.is_month_end.astype(int)
            
        return features
        
    def _create_ta_features(self, df, high, low, close, volume) -> pd.DataFrame:
        """Create technical analysis features using ta library"""
        features = pd.DataFrame(index=df.index)
        
        # Trend Indicators
        features['sma_10'] = ta.trend.sma_indicator(df[close], window=10)
        features['sma_20'] = ta.trend.sma_indicator(df[close], window=20)
        features['sma_50'] = ta.trend.sma_indicator(df[close], window=50)
        features['ema_10'] = ta.trend.ema_indicator(df[close], window=10)
        features['ema_20'] = ta.trend.ema_indicator(df[close], window=20)
        
        # MACD
        macd = ta.trend.MACD(df[close])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df[close])
        features['bb_high'] = bb.bollinger_hband()
        features['bb_low'] = bb.bollinger_lband()
        features['bb_mid'] = bb.bollinger_mavg()
        features['bb_width'] = bb.bollinger_wband()
        features['bb_position'] = (df[close] - features['bb_low']) / (features['bb_high'] - features['bb_low'])
        
        # RSI
        features['rsi_14'] = ta.momentum.RSIIndicator(df[close], window=14).rsi()
        features['rsi_7'] = ta.momentum.RSIIndicator(df[close], window=7).rsi()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df[high], df[low], df[close])
        features['stoch_k'] = stoch.stoch()
        features['stoch_d'] = stoch.stoch_signal()
        
        # ATR
        features['atr_14'] = ta.volatility.AverageTrueRange(df[high], df[low], df[close]).average_true_range()
        
        # ADX
        adx = ta.trend.ADXIndicator(df[high], df[low], df[close])
        features['adx'] = adx.adx()
        features['adx_pos'] = adx.adx_pos()
        features['adx_neg'] = adx.adx_neg()
        
        # Volume indicators
        if volume in df.columns:
            features['obv'] = ta.volume.OnBalanceVolumeIndicator(df[close], df[volume]).on_balance_volume()
            features['vpt'] = ta.volume.VolumePriceTrendIndicator(df[close], df[volume]).volume_price_trend()
            
            # Chaikin Money Flow
            cmf = ta.volume.ChaikinMoneyFlowIndicator(df[high], df[low], df[close], df[volume])
            features['cmf'] = cmf.chaikin_money_flow()
            
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df[high], df[low])
        features['ichimoku_a'] = ichimoku.ichimoku_a()
        features['ichimoku_b'] = ichimoku.ichimoku_b()
        features['ichimoku_base'] = ichimoku.ichimoku_base_line()
        features['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        return features
        
    def _create_tsfresh_features(self, df, price_col, volume_col, 
                                max_features: int = 100) -> Optional[pd.DataFrame]:
        """Create automated features using tsfresh"""
        try:
            # Prepare data for tsfresh
            tsfresh_df = pd.DataFrame({
                'id': 1,  # Single time series
                'time': range(len(df)),
                'price': df[price_col].values,
                'returns': df[price_col].pct_change().fillna(0).values
            })
            
            if volume_col in df.columns:
                tsfresh_df['volume'] = df[volume_col].values
                
            # Extract features
            extracted_features = extract_features(
                tsfresh_df,
                column_id='id',
                column_sort='time',
                n_jobs=self.n_jobs,
                disable_progressbar=True
            )
            
            # Impute missing values
            impute(extracted_features)
            
            # Limit number of features
            if len(extracted_features.columns) > max_features:
                # Select most important features based on variance
                variances = extracted_features.var()
                top_features = variances.nlargest(max_features).index
                extracted_features = extracted_features[top_features]
                
            # Align with original index
            features = pd.DataFrame(
                extracted_features.values[0].reshape(1, -1),
                columns=extracted_features.columns,
                index=[df.index[-1]]
            )
            
            # Expand to match original index (forward fill)
            features = features.reindex(df.index, method='bfill')
            
            return features
            
        except Exception as e:
            logger.warning(f"TSFresh feature extraction failed: {e}")
            return None
            
    def _select_features(self, features: pd.DataFrame, 
                        target: pd.Series,
                        max_features: int = 50) -> pd.DataFrame:
        """Select most important features"""
        # Align features and target
        common_index = features.index.intersection(target.index)
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Remove any remaining NaN
        mask = ~(features_aligned.isna().any(axis=1) | target_aligned.isna())
        features_clean = features_aligned[mask]
        target_clean = target_aligned[mask]
        
        if len(features_clean) < 10:
            logger.warning("Not enough data for feature selection")
            return features
            
        try:
            # Calculate mutual information
            mi_scores = mutual_info_regression(features_clean, target_clean)
            mi_scores = pd.Series(mi_scores, index=features_clean.columns)
            
            # Select top features
            top_features = mi_scores.nlargest(max_features).index.tolist()
            self.selected_features = top_features
            self.feature_importance = mi_scores
            
            logger.info(f"Selected {len(top_features)} features based on mutual information")
            return features[top_features]
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return features
            
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using saved feature pipeline"""
        if self.selected_features is not None:
            # Apply same feature selection
            available_features = [f for f in self.selected_features if f in df.columns]
            return df[available_features]
        return df
        
    def fit_scaler(self, features: pd.DataFrame, 
                  scaler_type: str = 'robust') -> pd.DataFrame:
        """Fit and transform feature scaler"""
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        scaled_features = pd.DataFrame(
            self.scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        
        return scaled_features
        
    def create_rolling_features(self, features: pd.DataFrame,
                               windows: List[int] = [5, 10, 20],
                               operations: List[str] = ['mean', 'std', 'skew']) -> pd.DataFrame:
        """Create rolling window features"""
        rolling_features = []
        
        for col in features.columns:
            for window in windows:
                for op in operations:
                    if op == 'mean':
                        feat = features[col].rolling(window).mean()
                    elif op == 'std':
                        feat = features[col].rolling(window).std()
                    elif op == 'skew':
                        feat = features[col].rolling(window).skew()
                    elif op == 'kurt':
                        feat = features[col].rolling(window).kurt()
                        
                    rolling_features.append(
                        feat.rename(f'{col}_rolling_{op}_{window}')
                    )
                    
        return pd.concat([features] + rolling_features, axis=1)