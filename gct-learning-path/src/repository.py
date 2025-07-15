"""
Course Repository Management
Handles storage and retrieval of learning modules and their content
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class Module:
    """Learning module representation"""
    module_id: str
    title: str
    description: str
    topic_tags: List[str]
    difficulty: float  # 0-1 scale
    duration_minutes: int
    prerequisites: List[str]
    learning_objectives: List[str]
    content_type: str  # 'video', 'text', 'interactive', 'quiz'
    content_url: Optional[str] = None
    cognitive_load: float = 0.5  # 0-1 scale
    engagement_score: float = 0.7  # Historical average
    completion_rate: float = 0.8  # Historical average
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class CourseRepository:
    """Manages course content storage and retrieval"""
    
    def __init__(self, storage_config: Dict[str, Any]):
        """
        Initialize repository with storage configuration
        
        Args:
            storage_config: {
                'type': 'sqlite' | 'json' | 'api',
                'path': str,
                'api_key': Optional[str]
            }
        """
        self.config = storage_config
        self.storage_type = storage_config.get('type', 'sqlite')
        
        if self.storage_type == 'sqlite':
            self.db_path = storage_config.get('path', 'learning_modules.db')
            self._init_database()
        elif self.storage_type == 'json':
            self.json_path = Path(storage_config.get('path', 'modules.json'))
            self._init_json_storage()
        elif self.storage_type == 'api':
            self.api_endpoint = storage_config.get('endpoint')
            self.api_key = storage_config.get('api_key')
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS modules (
                module_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                topic_tags TEXT,  -- JSON array
                difficulty REAL,
                duration_minutes INTEGER,
                prerequisites TEXT,  -- JSON array
                learning_objectives TEXT,  -- JSON array
                content_type TEXT,
                content_url TEXT,
                cognitive_load REAL,
                engagement_score REAL,
                completion_rate REAL,
                metadata TEXT,  -- JSON object
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS module_relationships (
                from_module TEXT,
                to_module TEXT,
                relationship_type TEXT,  -- 'prerequisite', 'builds_on', 'related'
                strength REAL,
                FOREIGN KEY (from_module) REFERENCES modules(module_id),
                FOREIGN KEY (to_module) REFERENCES modules(module_id),
                PRIMARY KEY (from_module, to_module, relationship_type)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _init_json_storage(self):
        """Initialize JSON file storage"""
        if not self.json_path.exists():
            default_data = {
                'modules': [],
                'relationships': [],
                'metadata': {
                    'version': '1.0',
                    'created_at': datetime.now().isoformat()
                }
            }
            with open(self.json_path, 'w') as f:
                json.dump(default_data, f, indent=2)
    
    def list_modules(self, 
                    topic_filter: Optional[List[str]] = None,
                    difficulty_range: Optional[Tuple[float, float]] = None) -> List[Dict]:
        """
        Return metadata for all available modules
        
        Args:
            topic_filter: List of topic tags to filter by
            difficulty_range: (min, max) difficulty values
            
        Returns:
            List of module dictionaries
        """
        if self.storage_type == 'sqlite':
            return self._list_modules_sqlite(topic_filter, difficulty_range)
        elif self.storage_type == 'json':
            return self._list_modules_json(topic_filter, difficulty_range)
        else:
            return self._list_modules_api(topic_filter, difficulty_range)
    
    def _list_modules_sqlite(self, topic_filter, difficulty_range) -> List[Dict]:
        """List modules from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM modules WHERE 1=1"
        params = []
        
        if difficulty_range:
            query += " AND difficulty >= ? AND difficulty <= ?"
            params.extend(difficulty_range)
        
        cursor.execute(query, params)
        modules = []
        
        for row in cursor.fetchall():
            module_dict = dict(row)
            # Parse JSON fields
            module_dict['topic_tags'] = json.loads(module_dict['topic_tags'] or '[]')
            module_dict['prerequisites'] = json.loads(module_dict['prerequisites'] or '[]')
            module_dict['learning_objectives'] = json.loads(module_dict['learning_objectives'] or '[]')
            module_dict['metadata'] = json.loads(module_dict['metadata'] or '{}')
            
            # Filter by topics if specified
            if topic_filter:
                if any(tag in module_dict['topic_tags'] for tag in topic_filter):
                    modules.append(module_dict)
            else:
                modules.append(module_dict)
        
        conn.close()
        return modules
    
    def _list_modules_json(self, topic_filter, difficulty_range) -> List[Dict]:
        """List modules from JSON file"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        modules = data.get('modules', [])
        
        # Apply filters
        if topic_filter:
            modules = [m for m in modules 
                      if any(tag in m.get('topic_tags', []) for tag in topic_filter)]
        
        if difficulty_range:
            modules = [m for m in modules 
                      if difficulty_range[0] <= m.get('difficulty', 0) <= difficulty_range[1]]
        
        return modules
    
    def _list_modules_api(self, topic_filter, difficulty_range) -> List[Dict]:
        """List modules from API endpoint"""
        # Placeholder for API integration
        import requests
        
        params = {}
        if topic_filter:
            params['topics'] = ','.join(topic_filter)
        if difficulty_range:
            params['min_difficulty'] = difficulty_range[0]
            params['max_difficulty'] = difficulty_range[1]
        
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        
        try:
            response = requests.get(f"{self.api_endpoint}/modules", 
                                  params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API error: {e}")
            return []
    
    def fetch_module_content(self, module_id: str) -> Dict:
        """
        Retrieve video URLs, text, and assessments for a module
        
        Args:
            module_id: Unique identifier for the module
            
        Returns:
            Dictionary with full module content including:
            - metadata
            - content (video URLs, text, etc.)
            - assessments (quiz questions, exercises)
            - resources (supplementary materials)
        """
        if self.storage_type == 'sqlite':
            return self._fetch_module_sqlite(module_id)
        elif self.storage_type == 'json':
            return self._fetch_module_json(module_id)
        else:
            return self._fetch_module_api(module_id)
    
    def _fetch_module_sqlite(self, module_id: str) -> Dict:
        """Fetch module from SQLite"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM modules WHERE module_id = ?", (module_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise ValueError(f"Module {module_id} not found")
        
        module_dict = dict(row)
        # Parse JSON fields
        module_dict['topic_tags'] = json.loads(module_dict['topic_tags'] or '[]')
        module_dict['prerequisites'] = json.loads(module_dict['prerequisites'] or '[]')
        module_dict['learning_objectives'] = json.loads(module_dict['learning_objectives'] or '[]')
        module_dict['metadata'] = json.loads(module_dict['metadata'] or '{}')
        
        # Fetch relationships
        cursor.execute("""
            SELECT to_module, relationship_type, strength
            FROM module_relationships
            WHERE from_module = ?
        """, (module_id,))
        
        module_dict['relationships'] = [
            {'to_module': r[0], 'type': r[1], 'strength': r[2]}
            for r in cursor.fetchall()
        ]
        
        conn.close()
        return module_dict
    
    def _fetch_module_json(self, module_id: str) -> Dict:
        """Fetch module from JSON"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        modules = data.get('modules', [])
        for module in modules:
            if module.get('module_id') == module_id:
                return module
        
        raise ValueError(f"Module {module_id} not found")
    
    def _fetch_module_api(self, module_id: str) -> Dict:
        """Fetch module from API"""
        import requests
        
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        
        try:
            response = requests.get(f"{self.api_endpoint}/modules/{module_id}", 
                                  headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError(f"Failed to fetch module {module_id}: {e}")
    
    def add_module(self, module: Module) -> bool:
        """Add a new module to the repository"""
        if self.storage_type == 'sqlite':
            return self._add_module_sqlite(module)
        elif self.storage_type == 'json':
            return self._add_module_json(module)
        else:
            return self._add_module_api(module)
    
    def _add_module_sqlite(self, module: Module) -> bool:
        """Add module to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO modules (
                    module_id, title, description, topic_tags, difficulty,
                    duration_minutes, prerequisites, learning_objectives,
                    content_type, content_url, cognitive_load,
                    engagement_score, completion_rate, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                module.module_id,
                module.title,
                module.description,
                json.dumps(module.topic_tags),
                module.difficulty,
                module.duration_minutes,
                json.dumps(module.prerequisites),
                json.dumps(module.learning_objectives),
                module.content_type,
                module.content_url,
                module.cognitive_load,
                module.engagement_score,
                module.completion_rate,
                json.dumps(module.metadata or {})
            ))
            
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def _add_module_json(self, module: Module) -> bool:
        """Add module to JSON storage"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Check if module already exists
        existing_ids = [m['module_id'] for m in data['modules']]
        if module.module_id in existing_ids:
            return False
        
        data['modules'].append(module.to_dict())
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    def _add_module_api(self, module: Module) -> bool:
        """Add module via API"""
        import requests
        
        headers = {
            'Authorization': f'Bearer {self.api_key}' if self.api_key else {},
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                f"{self.api_endpoint}/modules",
                headers=headers,
                json=module.to_dict()
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to add module: {e}")
            return False
    
    def get_module_graph(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get graph representation of module relationships
        
        Returns:
            Dict mapping module_id to list of (related_module_id, strength) tuples
        """
        if self.storage_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT from_module, to_module, strength
                FROM module_relationships
            """)
            
            graph = {}
            for from_mod, to_mod, strength in cursor.fetchall():
                if from_mod not in graph:
                    graph[from_mod] = []
                graph[from_mod].append((to_mod, strength))
            
            conn.close()
            return graph
        
        # Simplified for other storage types
        return {}
    
    def update_engagement_metrics(self, module_id: str, 
                                engagement_data: Dict[str, float]) -> bool:
        """
        Update module engagement metrics based on learner feedback
        
        Args:
            module_id: Module to update
            engagement_data: {
                'completion_rate': float,
                'engagement_score': float,
                'cognitive_load': float
            }
        """
        if self.storage_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update with exponential moving average
            alpha = 0.1  # Learning rate
            
            cursor.execute("""
                UPDATE modules
                SET completion_rate = completion_rate * ? + ? * ?,
                    engagement_score = engagement_score * ? + ? * ?,
                    cognitive_load = cognitive_load * ? + ? * ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE module_id = ?
            """, (
                1-alpha, engagement_data.get('completion_rate', 0), alpha,
                1-alpha, engagement_data.get('engagement_score', 0), alpha,
                1-alpha, engagement_data.get('cognitive_load', 0), alpha,
                module_id
            ))
            
            conn.commit()
            conn.close()
            return True
        
        return False