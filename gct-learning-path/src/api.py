"""
FastAPI endpoints for Learning Path Generator
RESTful API for path generation and management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd
import json
import logging
from pathlib import Path

# Import our modules
from repository import CourseRepository, Module
from metadata_extractor import MetadataExtractor
from gct_engine import LearningGCTEngine, GCTWeights
from path_optimizer import LearningPathOptimizer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GCT Learning Path API",
    description="Adaptive learning path generation using Grounded Coherence Theory",
    version="0.1.0"
)

# Global instances (in production, use dependency injection)
repo = None
extractor = None
gct_engine = None
optimizer_cache = {}


# Pydantic models for API
class PathGenerationRequest(BaseModel):
    learner_id: str = Field(..., description="Unique learner identifier")
    start_module: str = Field(..., description="Starting module ID")
    target_module: str = Field(..., description="Target module ID")
    max_steps: int = Field(10, description="Maximum number of modules in path")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Additional constraints")
    
    class Config:
        json_schema_extra = {
            "example": {
                "learner_id": "u123",
                "start_module": "intro_python",
                "target_module": "advanced_ml",
                "max_steps": 10,
                "constraints": {
                    "max_daily_minutes": 60,
                    "avoid_topics": ["statistics"]
                }
            }
        }


class LearnerProfile(BaseModel):
    learner_id: str
    skill_level: float = Field(0.5, ge=0, le=1)
    learning_style: str = Field("balanced", description="visual|reading|kinesthetic|balanced")
    goals: List[str] = []
    interests: List[str] = []
    constraints: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "learner_id": "u123",
                "skill_level": 0.6,
                "learning_style": "visual",
                "goals": ["Learn machine learning", "Build AI projects"],
                "interests": ["computer_vision", "nlp"],
                "constraints": {"max_daily_minutes": 90}
            }
        }


class ModuleFeedback(BaseModel):
    learner_id: str
    module_id: str
    completion_time: int = Field(..., description="Minutes to complete")
    difficulty_rating: float = Field(..., ge=1, le=5)
    engagement_rating: float = Field(..., ge=1, le=5)
    quiz_score: float = Field(..., ge=0, le=1)
    would_recommend: bool = True
    notes: Optional[str] = None


class PathResponse(BaseModel):
    path: List[str]
    coherence_score: float
    estimated_duration: int
    difficulty_progression: List[float]
    explanation: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global repo, extractor, gct_engine
    
    # Initialize repository
    repo_config = {
        'type': 'sqlite',
        'path': 'data/learning_modules.db'
    }
    repo = CourseRepository(repo_config)
    
    # Initialize metadata extractor
    extractor = MetadataExtractor()
    
    # Initialize GCT engine
    gct_engine = LearningGCTEngine()
    
    logger.info("API components initialized successfully")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "GCT Learning Path API",
        "version": "0.1.0",
        "endpoints": {
            "generate_path": "/generate-path",
            "learner_profile": "/learner/{learner_id}",
            "modules": "/modules",
            "feedback": "/feedback"
        }
    }


@app.post("/generate-path", response_model=PathResponse)
async def generate_learning_path(request: PathGenerationRequest):
    """
    Generate an optimal learning path between two modules
    """
    try:
        # Load learner profile
        learner_profile = await get_learner_profile_data(request.learner_id)
        
        # Get or create optimizer for this learner
        if request.learner_id not in optimizer_cache:
            optimizer_cache[request.learner_id] = LearningPathOptimizer(
                gct_engine, learner_profile
            )
        optimizer = optimizer_cache[request.learner_id]
        
        # Load module metadata
        modules = repo.list_modules()
        if not modules:
            raise HTTPException(status_code=404, detail="No modules found in repository")
        
        # Extract features
        metadata_df = extractor.extract_topic_vectors(modules)
        difficulty_df = extractor.assign_difficulty_scores(modules)
        
        # Merge dataframes
        full_metadata = pd.merge(metadata_df, difficulty_df, on='module_id')
        
        # Calculate coherence scores
        score_matrix = gct_engine.score_transitions(full_metadata)
        
        # Create transition graph
        module_graph = gct_engine.create_transition_graph(score_matrix)
        
        # Generate path
        learning_path = optimizer.build_path(
            request.start_module,
            request.target_module,
            module_graph,
            full_metadata,
            request.max_steps
        )
        
        # Generate explanation
        explanation = optimizer.explain_path_choice(learning_path, full_metadata)
        
        return PathResponse(
            path=learning_path.modules,
            coherence_score=learning_path.total_coherence,
            estimated_duration=learning_path.estimated_duration,
            difficulty_progression=learning_path.difficulty_curve,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Error generating path: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modules")
async def list_modules(
    topic: Optional[str] = None,
    min_difficulty: Optional[float] = None,
    max_difficulty: Optional[float] = None
):
    """List available learning modules with optional filters"""
    try:
        topic_filter = [topic] if topic else None
        difficulty_range = None
        
        if min_difficulty is not None or max_difficulty is not None:
            difficulty_range = (
                min_difficulty or 0,
                max_difficulty or 1
            )
        
        modules = repo.list_modules(topic_filter, difficulty_range)
        
        return {
            "count": len(modules),
            "modules": modules
        }
        
    except Exception as e:
        logger.error(f"Error listing modules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modules/{module_id}")
async def get_module_details(module_id: str):
    """Get detailed information about a specific module"""
    try:
        module_data = repo.fetch_module_content(module_id)
        
        # Get coherence scores for possible next modules
        next_modules = gct_engine.recommend_next_modules(
            module_id,
            None,  # Would need score matrix
            n_recommendations=5
        )
        
        module_data['recommended_next'] = next_modules
        
        return module_data
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching module: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/modules")
async def create_module(module: Module):
    """Add a new learning module"""
    try:
        success = repo.add_module(module)
        
        if success:
            return {"message": f"Module {module.module_id} created successfully"}
        else:
            raise HTTPException(
                status_code=409,
                detail=f"Module {module.module_id} already exists"
            )
            
    except Exception as e:
        logger.error(f"Error creating module: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/learner/{learner_id}")
async def get_learner_profile(learner_id: str):
    """Get learner profile"""
    try:
        profile = await get_learner_profile_data(learner_id)
        return profile
        
    except Exception as e:
        logger.error(f"Error fetching learner profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/learner/{learner_id}")
async def update_learner_profile(learner_id: str, profile: LearnerProfile):
    """Update learner profile"""
    try:
        # Save profile (in production, use proper database)
        profile_path = Path(f"data/learners/{learner_id}.json")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(profile_path, 'w') as f:
            json.dump(profile.dict(), f, indent=2)
        
        # Clear optimizer cache to use new profile
        if learner_id in optimizer_cache:
            del optimizer_cache[learner_id]
        
        return {"message": f"Profile updated for learner {learner_id}"}
        
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(feedback: ModuleFeedback, background_tasks: BackgroundTasks):
    """Submit learning feedback for a completed module"""
    try:
        # Get optimizer for learner
        if feedback.learner_id in optimizer_cache:
            optimizer = optimizer_cache[feedback.learner_id]
            
            # Update optimizer with feedback
            background_tasks.add_task(
                optimizer.update_on_feedback,
                feedback.module_id,
                feedback.dict()
            )
        
        # Update module engagement metrics
        engagement_data = {
            'completion_rate': 1.0 if feedback.quiz_score >= 0.7 else 0.5,
            'engagement_score': feedback.engagement_rating / 5,
            'cognitive_load': feedback.difficulty_rating / 5
        }
        
        background_tasks.add_task(
            repo.update_engagement_metrics,
            feedback.module_id,
            engagement_data
        )
        
        return {"message": "Feedback received successfully"}
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/learner/{learner_id}")
async def get_learner_analytics(learner_id: str):
    """Get learning analytics for a specific learner"""
    try:
        # Load learner history
        profile = await get_learner_profile_data(learner_id)
        
        analytics = {
            "learner_id": learner_id,
            "total_modules_completed": len(profile.get('completed_modules', [])),
            "average_quiz_score": calculate_average_score(profile),
            "preferred_topics": extract_preferred_topics(profile),
            "learning_velocity": calculate_learning_velocity(profile),
            "strengths": identify_strengths(profile),
            "areas_for_improvement": identify_weaknesses(profile)
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/{learner_id}")
async def get_recommendations(
    learner_id: str,
    n_recommendations: int = 5
):
    """Get personalized module recommendations"""
    try:
        profile = await get_learner_profile_data(learner_id)
        
        # Get last completed module
        completed = profile.get('completed_modules', [])
        if not completed:
            # Recommend beginner modules
            modules = repo.list_modules(difficulty_range=(0, 0.3))
            return {
                "recommendations": modules[:n_recommendations],
                "reason": "Beginner-friendly modules to start your journey"
            }
        
        last_module = completed[-1]
        
        # Use GCT engine for recommendations
        # (Would need full implementation with score matrix)
        recommendations = []
        
        return {
            "recommendations": recommendations,
            "based_on": last_module
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
async def get_learner_profile_data(learner_id: str) -> Dict:
    """Load learner profile from storage"""
    profile_path = Path(f"data/learners/{learner_id}.json")
    
    if profile_path.exists():
        with open(profile_path, 'r') as f:
            return json.load(f)
    else:
        # Return default profile
        return {
            "learner_id": learner_id,
            "skill_level": 0.5,
            "learning_style": "balanced",
            "goals": [],
            "interests": [],
            "constraints": {},
            "completed_modules": [],
            "performance_history": []
        }


def calculate_average_score(profile: Dict) -> float:
    """Calculate average quiz score from history"""
    history = profile.get('performance_history', [])
    if not history:
        return 0.0
    
    scores = [entry.get('score', 0) for entry in history]
    return sum(scores) / len(scores)


def calculate_learning_velocity(profile: Dict) -> Dict:
    """Calculate learning speed metrics"""
    history = profile.get('performance_history', [])
    if len(history) < 2:
        return {"status": "insufficient_data"}
    
    # Simple velocity: modules per week
    first_date = pd.Timestamp(history[0].get('timestamp', datetime.now()))
    last_date = pd.Timestamp(history[-1].get('timestamp', datetime.now()))
    
    weeks = (last_date - first_date).days / 7
    velocity = len(history) / max(weeks, 1)
    
    return {
        "modules_per_week": round(velocity, 2),
        "trend": "increasing" if velocity > 2 else "steady"
    }


def extract_preferred_topics(profile: Dict) -> List[str]:
    """Extract topics with highest engagement"""
    # Simplified implementation
    completed = profile.get('completed_modules', [])
    # Would need to look up module topics
    return ["python", "data_science"]  # Placeholder


def identify_strengths(profile: Dict) -> List[str]:
    """Identify learner strengths from performance"""
    strengths = []
    avg_score = calculate_average_score(profile)
    
    if avg_score > 0.8:
        strengths.append("Excellent quiz performance")
    
    # Add more strength detection logic
    
    return strengths


def identify_weaknesses(profile: Dict) -> List[str]:
    """Identify areas for improvement"""
    weaknesses = []
    
    # Analyze performance patterns
    # This is simplified - would need more sophisticated analysis
    
    return weaknesses


# Health check endpoint
@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "repository": repo is not None,
            "extractor": extractor is not None,
            "gct_engine": gct_engine is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)