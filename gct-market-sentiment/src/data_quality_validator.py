"""
Data Quality Validation and Anomaly Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Validation rule definition"""
    field: str
    rule_type: str  # 'range', 'type', 'pattern', 'custom'
    params: Dict[str, Any]
    severity: str  # 'error', 'warning', 'info'
    message: str


@dataclass
class ValidationResult:
    """Result of validation check"""
    passed: bool
    field: str
    rule: str
    severity: str
    message: str
    value: Any


class DataQualityValidator:
    """Comprehensive data validation and quality checks"""
    
    def __init__(self):
        self.validation_rules = self._init_validation_rules()
        self.anomaly_detector = AnomalyDetector()
        self.quality_metrics = QualityMetrics()
        
    def _init_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules"""
        return [
            # GCT variable ranges
            ValidationRule('psi', 'range', {'min': 0.0, 'max': 1.0}, 
                         'error', 'psi must be between 0 and 1'),
            ValidationRule('rho', 'range', {'min': 0.0, 'max': 1.0}, 
                         'error', 'rho must be between 0 and 1'),
            ValidationRule('q_raw', 'range', {'min': 0.0, 'max': float('inf')}, 
                         'error', 'q_raw must be non-negative'),
            ValidationRule('f', 'range', {'min': 0.0, 'max': 1.0}, 
                         'error', 'f must be between 0 and 1'),
            
            # Coherence checks
            ValidationRule('coherence', 'range', {'min': 0.0, 'max': 10.0}, 
                         'warning', 'coherence unusually high'),
            
            # Type checks
            ValidationRule('timestamp', 'type', {'expected': (str, datetime)}, 
                         'error', 'timestamp must be string or datetime'),
            ValidationRule('ticker', 'pattern', {'pattern': r'^[A-Z]{1,5}$'}, 
                         'warning', 'ticker should be 1-5 uppercase letters'),
            
            # Derivative checks
            ValidationRule('dc_dt', 'range', {'min': -1.0, 'max': 1.0}, 
                         'warning', 'derivative unusually large'),
            
            # Article fields
            ValidationRule('title', 'type', {'expected': str}, 
                         'error', 'title must be string'),
            ValidationRule('body', 'type', {'expected': str}, 
                         'error', 'body must be string'),
            ValidationRule('source', 'type', {'expected': str}, 
                         'error', 'source must be string'),
        ]
    
    def validate_data(self, data: Dict) -> Tuple[Dict, List[ValidationResult]]:
        """Validate data and return cleaned data with issues"""
        results = []
        cleaned = data.copy()
        
        # Apply validation rules
        for rule in self.validation_rules:
            if rule.field in data:
                result = self._apply_rule(rule, data[rule.field])
                results.append(result)
                
                if not result.passed:
                    # Apply cleaning based on severity
                    if rule.severity == 'error':
                        cleaned[rule.field] = self._get_default_value(rule.field)
                    elif rule.severity == 'warning':
                        cleaned[rule.field] = self._sanitize_value(
                            rule.field, data[rule.field], rule
                        )
        
        # Check for anomalies
        anomalies = self.anomaly_detector.detect(cleaned)
        for anomaly in anomalies:
            results.append(ValidationResult(
                passed=False,
                field=anomaly['field'],
                rule='anomaly',
                severity='warning',
                message=anomaly['message'],
                value=anomaly['value']
            ))
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(results)
        cleaned['_quality_score'] = quality_score
        
        return cleaned, results
    
    def _apply_rule(self, rule: ValidationRule, value: Any) -> ValidationResult:
        """Apply a single validation rule"""
        passed = True
        
        if rule.rule_type == 'range':
            min_val = rule.params.get('min', float('-inf'))
            max_val = rule.params.get('max', float('inf'))
            
            try:
                num_val = float(value)
                passed = min_val <= num_val <= max_val
            except (TypeError, ValueError):
                passed = False
                
        elif rule.rule_type == 'type':
            expected_types = rule.params['expected']
            passed = isinstance(value, expected_types)
            
        elif rule.rule_type == 'pattern':
            import re
            pattern = rule.params['pattern']
            passed = bool(re.match(pattern, str(value)))
            
        elif rule.rule_type == 'custom':
            func = rule.params['function']
            passed = func(value)
        
        return ValidationResult(
            passed=passed,
            field=rule.field,
            rule=rule.rule_type,
            severity=rule.severity,
            message=rule.message if not passed else '',
            value=value
        )
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for field"""
        defaults = {
            'psi': 0.5,
            'rho': 0.5,
            'q_raw': 0.0,
            'f': 0.5,
            'coherence': 0.0,
            'dc_dt': 0.0,
            'd2c_dt2': 0.0,
            'timestamp': datetime.now(),
            'ticker': 'UNKNOWN',
            'title': '',
            'body': '',
            'source': 'unknown'
        }
        return defaults.get(field, None)
    
    def _sanitize_value(self, field: str, value: Any, rule: ValidationRule) -> Any:
        """Sanitize value based on rule"""
        if rule.rule_type == 'range':
            min_val = rule.params.get('min', float('-inf'))
            max_val = rule.params.get('max', float('inf'))
            
            try:
                num_val = float(value)
                return max(min_val, min(num_val, max_val))  # Clamp to range
            except:
                return self._get_default_value(field)
                
        elif rule.rule_type == 'pattern' and field == 'ticker':
            # Clean ticker symbol
            cleaned = ''.join(c for c in str(value).upper() if c.isalpha())
            return cleaned[:5] if cleaned else 'UNKNOWN'
            
        return value
    
    def _calculate_quality_score(self, results: List[ValidationResult]) -> float:
        """Calculate overall data quality score"""
        if not results:
            return 1.0
        
        # Weight by severity
        severity_weights = {
            'error': 1.0,
            'warning': 0.5,
            'info': 0.1
        }
        
        total_weight = 0
        failed_weight = 0
        
        for result in results:
            weight = severity_weights.get(result.severity, 0.1)
            total_weight += weight
            if not result.passed:
                failed_weight += weight
        
        if total_weight == 0:
            return 1.0
            
        return max(0.0, 1.0 - (failed_weight / total_weight))
    
    def validate_batch(self, data_list: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Validate batch of data and return summary"""
        cleaned_list = []
        all_results = []
        
        for data in data_list:
            cleaned, results = self.validate_data(data)
            cleaned_list.append(cleaned)
            all_results.extend(results)
        
        # Generate summary
        summary = self._generate_validation_summary(all_results, len(data_list))
        
        return cleaned_list, summary
    
    def _generate_validation_summary(self, results: List[ValidationResult], 
                                   total_records: int) -> Dict:
        """Generate validation summary statistics"""
        failed_by_field = {}
        failed_by_severity = {'error': 0, 'warning': 0, 'info': 0}
        
        for result in results:
            if not result.passed:
                failed_by_field[result.field] = failed_by_field.get(result.field, 0) + 1
                failed_by_severity[result.severity] += 1
        
        total_issues = sum(failed_by_severity.values())
        
        return {
            'total_records': total_records,
            'total_issues': total_issues,
            'issues_by_severity': failed_by_severity,
            'issues_by_field': failed_by_field,
            'quality_rate': (total_records - total_issues) / max(total_records, 1) * 100,
            'timestamp': datetime.now().isoformat()
        }


class AnomalyDetector:
    """Detect anomalies in data using statistical methods"""
    
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        self.historical_stats = {}
        
    def detect(self, data: Dict) -> List[Dict]:
        """Detect anomalies in data"""
        anomalies = []
        
        # Check numerical fields for statistical anomalies
        numerical_fields = ['coherence', 'dc_dt', 'd2c_dt2', 'psi', 'rho', 'q_raw', 'f']
        
        for field in numerical_fields:
            if field in data:
                value = data[field]
                if self._is_anomalous(field, value):
                    anomalies.append({
                        'field': field,
                        'value': value,
                        'type': 'statistical',
                        'message': f'{field} value {value} is anomalous'
                    })
        
        # Check for logical inconsistencies
        logical_anomalies = self._check_logical_consistency(data)
        anomalies.extend(logical_anomalies)
        
        return anomalies
    
    def _is_anomalous(self, field: str, value: float) -> bool:
        """Check if value is statistically anomalous"""
        if field not in self.historical_stats:
            # Initialize with first value
            self.historical_stats[field] = {
                'values': [value],
                'mean': value,
                'std': 0
            }
            return False
        
        stats = self.historical_stats[field]
        
        # Calculate z-score
        if stats['std'] > 0:
            z_score = abs((value - stats['mean']) / stats['std'])
            is_anomaly = z_score > self.z_threshold
        else:
            is_anomaly = False
        
        # Update statistics (online algorithm)
        self._update_stats(field, value)
        
        return is_anomaly
    
    def _update_stats(self, field: str, value: float):
        """Update running statistics"""
        stats = self.historical_stats[field]
        stats['values'].append(value)
        
        # Keep only recent values (sliding window)
        if len(stats['values']) > 1000:
            stats['values'] = stats['values'][-1000:]
        
        # Recalculate statistics
        stats['mean'] = np.mean(stats['values'])
        stats['std'] = np.std(stats['values'])
    
    def _check_logical_consistency(self, data: Dict) -> List[Dict]:
        """Check for logical inconsistencies"""
        anomalies = []
        
        # Check if components sum to coherence (approximately)
        if all(k in data for k in ['coherence', 'psi', 'rho', 'q_opt', 'f']):
            expected_coherence = (
                data['psi'] + 
                data['rho'] * data['psi'] + 
                data['q_opt'] + 
                data['f'] * data['psi']
            )
            
            if abs(data['coherence'] - expected_coherence) > 0.1:
                anomalies.append({
                    'field': 'coherence',
                    'value': data['coherence'],
                    'type': 'logical',
                    'message': f'Coherence {data["coherence"]} does not match expected {expected_coherence}'
                })
        
        # Check sentiment consistency with derivative
        if 'sentiment' in data and 'dc_dt' in data:
            if data['sentiment'] == 'bullish' and data['dc_dt'] < -0.1:
                anomalies.append({
                    'field': 'sentiment',
                    'value': data['sentiment'],
                    'type': 'logical',
                    'message': 'Bullish sentiment with negative derivative'
                })
            elif data['sentiment'] == 'bearish' and data['dc_dt'] > 0.1:
                anomalies.append({
                    'field': 'sentiment',
                    'value': data['sentiment'],
                    'type': 'logical',
                    'message': 'Bearish sentiment with positive derivative'
                })
        
        return anomalies


class QualityMetrics:
    """Track and report data quality metrics"""
    
    def __init__(self):
        self.metrics_history = []
        
    def record_quality_metrics(self, summary: Dict):
        """Record quality metrics for tracking"""
        self.metrics_history.append(summary)
        
        # Keep only recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_quality_trends(self) -> Dict:
        """Get quality trends over time"""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        return {
            'avg_quality_rate': df['quality_rate'].mean(),
            'quality_trend': self._calculate_trend(df['quality_rate']),
            'total_issues_trend': self._calculate_trend(df['total_issues']),
            'issues_by_severity_avg': df['issues_by_severity'].apply(pd.Series).mean().to_dict(),
            'most_common_issues': self._get_most_common_issues(df)
        }
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return 'stable'
        
        # Simple linear regression
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series, 1)
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'degrading'
        else:
            return 'stable'
    
    def _get_most_common_issues(self, df: pd.DataFrame) -> List[str]:
        """Get most common issue fields"""
        all_issues = {}
        
        for issues_dict in df['issues_by_field']:
            for field, count in issues_dict.items():
                all_issues[field] = all_issues.get(field, 0) + count
        
        # Sort by frequency
        sorted_issues = sorted(all_issues.items(), key=lambda x: x[1], reverse=True)
        
        return [field for field, _ in sorted_issues[:5]]


def generate_quality_report(db_path: str) -> Dict:
    """Generate comprehensive data quality report"""
    validator = DataQualityValidator()
    
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        # Check for missing data
        missing_data = pd.read_sql_query("""
            SELECT 
                COUNT(CASE WHEN psi IS NULL THEN 1 END) as missing_psi,
                COUNT(CASE WHEN rho IS NULL THEN 1 END) as missing_rho,
                COUNT(CASE WHEN q_raw IS NULL THEN 1 END) as missing_q,
                COUNT(CASE WHEN f IS NULL THEN 1 END) as missing_f,
                COUNT(*) as total_records
            FROM GCTScores
        """, conn)
        
        # Check for outliers
        outliers = pd.read_sql_query("""
            SELECT 
                COUNT(CASE WHEN coherence > 10 THEN 1 END) as extreme_coherence,
                COUNT(CASE WHEN ABS(dc_dt) > 1 THEN 1 END) as extreme_derivatives,
                AVG(coherence) as avg_coherence,
                STDEV(coherence) as std_coherence
            FROM GCTScores
            WHERE created_at > datetime('now', '-24 hours')
        """, conn)
        
        # Data freshness
        freshness = pd.read_sql_query("""
            SELECT 
                COUNT(CASE WHEN created_at > datetime('now', '-1 hour') THEN 1 END) as last_hour,
                COUNT(CASE WHEN created_at > datetime('now', '-24 hours') THEN 1 END) as last_day,
                COUNT(*) as total,
                MAX(created_at) as latest_record
            FROM GCTScores
        """, conn)
    
    return {
        'missing_data': missing_data.to_dict('records')[0],
        'outliers': outliers.to_dict('records')[0],
        'freshness': freshness.to_dict('records')[0],
        'recommendations': _generate_recommendations(missing_data, outliers, freshness),
        'timestamp': datetime.now().isoformat()
    }


def _generate_recommendations(missing_data, outliers, freshness) -> List[str]:
    """Generate data quality recommendations"""
    recommendations = []
    
    # Check missing data
    missing_pct = missing_data.iloc[0]['missing_psi'] / missing_data.iloc[0]['total_records'] * 100
    if missing_pct > 5:
        recommendations.append(f"High missing data rate ({missing_pct:.1f}%) - investigate data collection")
    
    # Check outliers
    if outliers.iloc[0]['extreme_coherence'] > 0:
        recommendations.append("Extreme coherence values detected - review calculation logic")
    
    # Check data freshness
    if freshness.iloc[0]['last_hour'] == 0:
        recommendations.append("No recent data in last hour - check data pipeline")
    
    return recommendations