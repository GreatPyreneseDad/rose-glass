"""
Creative Flow Database Schema and Management
Store and retrieve creative session data
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
from pathlib import Path

from .creative_flow_engine import CreativeState, CreativeMetrics, BiometricData


class CreativeFlowDatabase:
    """Database for storing creative flow sessions and analytics"""
    
    def __init__(self, db_path: str = "creative_flow.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize_schema()
        
    def _initialize_schema(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Creative sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS creative_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                project_name TEXT,
                session_type TEXT,
                total_coherence REAL,
                peak_coherence REAL,
                flow_duration_seconds INTEGER DEFAULT 0,
                breakthroughs INTEGER DEFAULT 0,
                completed BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Creative states timeline
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS creative_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                state TEXT NOT NULL,
                coherence REAL NOT NULL,
                psi REAL,
                rho REAL,
                q_raw REAL,
                f REAL,
                dc_dt REAL,
                d2c_dt2 REAL,
                FOREIGN KEY (session_id) REFERENCES creative_sessions(id)
            )
        """)
        
        # Creative metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS creative_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                novelty_score REAL,
                fluency_rate REAL,
                flexibility_index REAL,
                elaboration_depth REAL,
                convergence_ratio REAL,
                flow_intensity REAL,
                breakthrough_probability REAL,
                FOREIGN KEY (session_id) REFERENCES creative_sessions(id)
            )
        """)
        
        # Biometric data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS biometric_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                hrv REAL,
                eeg_alpha REAL,
                eeg_theta REAL,
                eeg_gamma REAL,
                gsr REAL,
                eye_movement_entropy REAL,
                posture_stability REAL,
                FOREIGN KEY (session_id) REFERENCES creative_sessions(id)
            )
        """)
        
        # Creative outputs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS creative_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                output_type TEXT,
                content TEXT,
                quality_score REAL,
                tags TEXT,  -- JSON array
                FOREIGN KEY (session_id) REFERENCES creative_sessions(id)
            )
        """)
        
        # Breakthrough events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS breakthrough_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                coherence_before REAL,
                coherence_after REAL,
                state_before TEXT,
                state_after TEXT,
                description TEXT,
                impact_score REAL,
                FOREIGN KEY (session_id) REFERENCES creative_sessions(id)
            )
        """)
        
        # Environment settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS environment_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                lighting_temperature INTEGER,
                lighting_brightness REAL,
                sound_type TEXT,
                sound_volume REAL,
                interruptions_blocked BOOLEAN,
                workspace_config TEXT,  -- JSON
                FOREIGN KEY (session_id) REFERENCES creative_sessions(id)
            )
        """)
        
        # Team collaborations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_collaborations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                team_member_id TEXT NOT NULL,
                team_member_name TEXT,
                role TEXT,
                coherence_avg REAL,
                interaction_score REAL,
                contribution_score REAL,
                FOREIGN KEY (session_id) REFERENCES creative_sessions(id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON creative_sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_states_session ON creative_states(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session ON creative_metrics(session_id)")
        
        self.conn.commit()
    
    def create_session(self, user_id: str, project_name: str, 
                      session_type: str = "individual") -> int:
        """Create a new creative session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO creative_sessions 
            (user_id, start_time, project_name, session_type)
            VALUES (?, ?, ?, ?)
        """, (user_id, datetime.now(), project_name, session_type))
        self.conn.commit()
        return cursor.lastrowid
    
    def end_session(self, session_id: int):
        """End a creative session and calculate summary statistics"""
        cursor = self.conn.cursor()
        
        # Get session statistics
        cursor.execute("""
            SELECT 
                AVG(coherence) as avg_coherence,
                MAX(coherence) as peak_coherence,
                COUNT(DISTINCT state) as state_changes
            FROM creative_states
            WHERE session_id = ?
        """, (session_id,))
        
        stats = cursor.fetchone()
        
        # Calculate flow duration
        cursor.execute("""
            SELECT timestamp, state
            FROM creative_states
            WHERE session_id = ? AND state = 'flow'
            ORDER BY timestamp
        """, (session_id,))
        
        flow_states = cursor.fetchall()
        flow_duration = self._calculate_state_duration(flow_states)
        
        # Count breakthroughs
        cursor.execute("""
            SELECT COUNT(*) as breakthrough_count
            FROM breakthrough_events
            WHERE session_id = ?
        """, (session_id,))
        
        breakthroughs = cursor.fetchone()['breakthrough_count']
        
        # Update session
        cursor.execute("""
            UPDATE creative_sessions
            SET end_time = ?,
                total_coherence = ?,
                peak_coherence = ?,
                flow_duration_seconds = ?,
                breakthroughs = ?,
                completed = TRUE
            WHERE id = ?
        """, (
            datetime.now(),
            stats['avg_coherence'] or 0,
            stats['peak_coherence'] or 0,
            flow_duration,
            breakthroughs,
            session_id
        ))
        
        self.conn.commit()
    
    def record_creative_state(self, session_id: int, state: CreativeState,
                            gct_result: Dict, variables: Dict):
        """Record a creative state measurement"""
        cursor = self.conn.cursor()
        
        components = gct_result.get('components', {})
        
        cursor.execute("""
            INSERT INTO creative_states
            (session_id, timestamp, state, coherence, psi, rho, q_raw, f, dc_dt, d2c_dt2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now(),
            state.value,
            gct_result.get('coherence', 0),
            components.get('psi', 0),
            components.get('rho', 0),
            components.get('q_raw', 0),
            components.get('f', 0),
            gct_result.get('dc_dt', 0),
            gct_result.get('d2c_dt2', 0)
        ))
        
        self.conn.commit()
    
    def record_creative_metrics(self, session_id: int, metrics: CreativeMetrics):
        """Record creative metrics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO creative_metrics
            (session_id, timestamp, novelty_score, fluency_rate, flexibility_index,
             elaboration_depth, convergence_ratio, flow_intensity, breakthrough_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now(),
            metrics.novelty_score,
            metrics.fluency_rate,
            metrics.flexibility_index,
            metrics.elaboration_depth,
            metrics.convergence_ratio,
            metrics.flow_intensity,
            metrics.breakthrough_probability
        ))
        
        self.conn.commit()
    
    def record_biometric_data(self, session_id: int, biometrics: BiometricData):
        """Record biometric measurements"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO biometric_data
            (session_id, timestamp, hrv, eeg_alpha, eeg_theta, eeg_gamma,
             gsr, eye_movement_entropy, posture_stability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now(),
            biometrics.hrv,
            biometrics.eeg_alpha,
            biometrics.eeg_theta,
            biometrics.eeg_gamma,
            biometrics.gsr,
            biometrics.eye_movement_entropy,
            biometrics.posture_stability
        ))
        
        self.conn.commit()
    
    def record_breakthrough(self, session_id: int, coherence_before: float,
                          coherence_after: float, state_before: str,
                          state_after: str, description: str = "",
                          impact_score: float = 0.5):
        """Record a creative breakthrough event"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO breakthrough_events
            (session_id, timestamp, coherence_before, coherence_after,
             state_before, state_after, description, impact_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now(),
            coherence_before,
            coherence_after,
            state_before,
            state_after,
            description,
            impact_score
        ))
        
        self.conn.commit()
    
    def get_session_history(self, user_id: str, 
                          days_back: int = 30) -> pd.DataFrame:
        """Get session history for a user"""
        query = """
            SELECT *
            FROM creative_sessions
            WHERE user_id = ?
            AND start_time >= datetime('now', '-{} days')
            ORDER BY start_time DESC
        """.format(days_back)
        
        return pd.read_sql_query(query, self.conn, params=(user_id,))
    
    def get_session_timeline(self, session_id: int) -> pd.DataFrame:
        """Get complete timeline for a session"""
        query = """
            SELECT 
                cs.timestamp,
                cs.state,
                cs.coherence,
                cs.psi,
                cs.rho,
                cs.q_raw,
                cs.f,
                cm.novelty_score,
                cm.flow_intensity,
                cm.breakthrough_probability
            FROM creative_states cs
            LEFT JOIN creative_metrics cm
                ON cs.session_id = cm.session_id
                AND cs.timestamp = cm.timestamp
            WHERE cs.session_id = ?
            ORDER BY cs.timestamp
        """
        
        return pd.read_sql_query(query, self.conn, params=(session_id,))
    
    def get_flow_statistics(self, user_id: str) -> Dict:
        """Get flow state statistics for a user"""
        cursor = self.conn.cursor()
        
        # Total flow time
        cursor.execute("""
            SELECT SUM(flow_duration_seconds) as total_flow_seconds
            FROM creative_sessions
            WHERE user_id = ? AND completed = TRUE
        """, (user_id,))
        
        total_flow = cursor.fetchone()['total_flow_seconds'] or 0
        
        # Average flow per session
        cursor.execute("""
            SELECT AVG(flow_duration_seconds) as avg_flow_seconds
            FROM creative_sessions
            WHERE user_id = ? AND completed = TRUE
        """, (user_id,))
        
        avg_flow = cursor.fetchone()['avg_flow_seconds'] or 0
        
        # Best flow session
        cursor.execute("""
            SELECT project_name, flow_duration_seconds, peak_coherence
            FROM creative_sessions
            WHERE user_id = ? AND completed = TRUE
            ORDER BY flow_duration_seconds DESC
            LIMIT 1
        """, (user_id,))
        
        best_session = cursor.fetchone()
        
        return {
            'total_flow_hours': total_flow / 3600,
            'avg_flow_minutes': avg_flow / 60,
            'best_session': dict(best_session) if best_session else None
        }
    
    def get_breakthrough_history(self, user_id: str) -> pd.DataFrame:
        """Get breakthrough events for a user"""
        query = """
            SELECT 
                be.*,
                cs.project_name,
                cs.session_type
            FROM breakthrough_events be
            JOIN creative_sessions cs ON be.session_id = cs.id
            WHERE cs.user_id = ?
            ORDER BY be.timestamp DESC
        """
        
        return pd.read_sql_query(query, self.conn, params=(user_id,))
    
    def get_creative_patterns(self, user_id: str) -> Dict:
        """Analyze creative patterns for a user"""
        cursor = self.conn.cursor()
        
        # Most common states
        cursor.execute("""
            SELECT 
                state,
                COUNT(*) as count,
                AVG(coherence) as avg_coherence
            FROM creative_states cs
            JOIN creative_sessions sess ON cs.session_id = sess.id
            WHERE sess.user_id = ?
            GROUP BY state
            ORDER BY count DESC
        """, (user_id,))
        
        state_distribution = {
            row['state']: {
                'count': row['count'],
                'avg_coherence': row['avg_coherence']
            }
            for row in cursor.fetchall()
        }
        
        # Peak creative times
        cursor.execute("""
            SELECT 
                strftime('%H', timestamp) as hour,
                AVG(coherence) as avg_coherence,
                COUNT(*) as measurements
            FROM creative_states cs
            JOIN creative_sessions sess ON cs.session_id = sess.id
            WHERE sess.user_id = ? AND state = 'flow'
            GROUP BY hour
            ORDER BY avg_coherence DESC
        """, (user_id,))
        
        peak_hours = [
            {'hour': int(row['hour']), 'coherence': row['avg_coherence']}
            for row in cursor.fetchall()
        ]
        
        return {
            'state_distribution': state_distribution,
            'peak_creative_hours': peak_hours[:3]  # Top 3 hours
        }
    
    def _calculate_state_duration(self, state_records: List) -> int:
        """Calculate total duration in a specific state"""
        if not state_records:
            return 0
        
        total_seconds = 0
        for i in range(len(state_records) - 1):
            start = datetime.fromisoformat(state_records[i]['timestamp'])
            end = datetime.fromisoformat(state_records[i + 1]['timestamp'])
            duration = (end - start).total_seconds()
            
            # Cap individual durations at 5 minutes to handle gaps
            total_seconds += min(duration, 300)
        
        return int(total_seconds)
    
    def close(self):
        """Close database connection"""
        self.conn.close()