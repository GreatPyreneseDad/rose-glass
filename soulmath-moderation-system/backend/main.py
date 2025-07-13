from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import subprocess
import json
import os
from datetime import datetime
import asyncio
from pathlib import Path

app = FastAPI(title="SoulMath Moderation API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ScrapingRequest(BaseModel):
    subreddits: List[str]
    max_posts: Optional[int] = 50
    
class ModerationAnalysis(BaseModel):
    psi: float
    rho: float
    coherence: float
    toxicity_risk: float
    manipulation_risk: float
    extremism_risk: float
    spam_risk: float
    harassment_risk: float
    discourse_collapse: float
    escalation_risk: float
    overall_risk: float

class ContentItem(BaseModel):
    id: str
    text: str
    author: str
    timestamp: str
    subreddit: Optional[str]
    score: int
    analysis: ModerationAnalysis
    content_type: str  # 'post' or 'comment'
    
class ScrapingStatus(BaseModel):
    status: str
    message: str
    items_scraped: int
    last_update: str

# Global status tracking
scraping_status = {
    "active": False,
    "last_run": None,
    "items_scraped": 0
}

@app.get("/")
async def root():
    return {"message": "SoulMath Moderation API", "version": "1.0.0"}

@app.post("/api/scrape")
async def start_scraping(request: ScrapingRequest, background_tasks: BackgroundTasks):
    """Start Reddit scraping process"""
    if scraping_status["active"]:
        raise HTTPException(status_code=400, detail="Scraping already in progress")
    
    # Run scraping in background
    background_tasks.add_task(run_scrapy, request.subreddits)
    
    return {
        "status": "started",
        "message": f"Started scraping {len(request.subreddits)} subreddits",
        "subreddits": request.subreddits
    }

async def run_scrapy(subreddits: List[str]):
    """Run Scrapy spider"""
    scraping_status["active"] = True
    scraping_status["last_run"] = datetime.utcnow().isoformat()
    
    try:
        # Change to scrapy project directory
        scrapy_dir = Path(__file__).parent / "scrapy_project"
        subreddit_list = ",".join(subreddits)
        
        # Run scrapy command
        cmd = [
            "scrapy", "crawl", "reddit",
            "-a", f"subreddits={subreddit_list}",
            "-L", "INFO"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(scrapy_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"Scrapy error: {stderr.decode()}")
        else:
            print(f"Scrapy completed successfully")
            
    except Exception as e:
        print(f"Error running scrapy: {e}")
    finally:
        scraping_status["active"] = False

@app.get("/api/status")
async def get_scraping_status():
    """Get current scraping status"""
    # Count items in the data file
    data_file = Path(__file__).parent / "scrapy_project" / "reddit_moderation_data.jsonl"
    items_count = 0
    
    if data_file.exists():
        with open(data_file, 'r') as f:
            items_count = sum(1 for _ in f)
    
    return ScrapingStatus(
        status="active" if scraping_status["active"] else "idle",
        message="Scraping in progress" if scraping_status["active"] else "Ready to scrape",
        items_scraped=items_count,
        last_update=scraping_status["last_run"] or "Never"
    )

@app.get("/api/content")
async def get_moderated_content(
    limit: int = 50,
    min_risk: float = 0.0,
    max_risk: float = 1.0,
    content_type: Optional[str] = None
):
    """Get scraped and analyzed content"""
    data_file = Path(__file__).parent / "scrapy_project" / "reddit_moderation_data.jsonl"
    
    if not data_file.exists():
        return {"items": [], "total": 0}
    
    items = []
    with open(data_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Filter by risk level
                if not (min_risk <= data.get('overall_risk', 0) <= max_risk):
                    continue
                
                # Filter by content type
                if content_type and data.get('post_id') and not data.get('comment_id'):
                    if content_type != 'post':
                        continue
                elif content_type and data.get('comment_id'):
                    if content_type != 'comment':
                        continue
                
                # Create content item
                item = ContentItem(
                    id=data.get('post_id') or data.get('comment_id', ''),
                    text=data.get('text') or data.get('title', ''),
                    author=data.get('author', 'unknown'),
                    timestamp=data.get('timestamp', ''),
                    subreddit=data.get('subreddit'),
                    score=data.get('score', 0),
                    analysis=ModerationAnalysis(
                        psi=data.get('psi', 0),
                        rho=data.get('rho', 0),
                        coherence=data.get('coherence', 0),
                        toxicity_risk=data.get('toxicity_risk', 0),
                        manipulation_risk=data.get('manipulation_risk', 0),
                        extremism_risk=data.get('extremism_risk', 0),
                        spam_risk=data.get('spam_risk', 0),
                        harassment_risk=data.get('harassment_risk', 0),
                        discourse_collapse=data.get('discourse_collapse', 0),
                        escalation_risk=data.get('escalation_risk', 0),
                        overall_risk=data.get('overall_risk', 0)
                    ),
                    content_type='post' if data.get('title') else 'comment'
                )
                
                items.append(item)
                
                if len(items) >= limit:
                    break
                    
            except json.JSONDecodeError:
                continue
    
    # Sort by risk level (highest first)
    items.sort(key=lambda x: x.analysis.overall_risk, reverse=True)
    
    return {
        "items": items,
        "total": len(items)
    }

@app.get("/api/analytics")
async def get_analytics():
    """Get analytics on scraped content"""
    data_file = Path(__file__).parent / "scrapy_project" / "reddit_moderation_data.jsonl"
    
    if not data_file.exists():
        return {"error": "No data available"}
    
    stats = {
        "total_items": 0,
        "posts": 0,
        "comments": 0,
        "high_risk": 0,
        "medium_risk": 0,
        "low_risk": 0,
        "avg_coherence": 0,
        "avg_toxicity": 0,
        "subreddits": set()
    }
    
    coherence_sum = 0
    toxicity_sum = 0
    
    with open(data_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                stats["total_items"] += 1
                
                if data.get('title'):
                    stats["posts"] += 1
                else:
                    stats["comments"] += 1
                
                risk = data.get('overall_risk', 0)
                if risk > 0.7:
                    stats["high_risk"] += 1
                elif risk > 0.4:
                    stats["medium_risk"] += 1
                else:
                    stats["low_risk"] += 1
                
                coherence_sum += data.get('coherence', 0)
                toxicity_sum += data.get('toxicity_risk', 0)
                
                if data.get('subreddit'):
                    stats["subreddits"].add(data['subreddit'])
                    
            except json.JSONDecodeError:
                continue
    
    if stats["total_items"] > 0:
        stats["avg_coherence"] = round(coherence_sum / stats["total_items"], 3)
        stats["avg_toxicity"] = round(toxicity_sum / stats["total_items"], 3)
    
    stats["subreddits"] = list(stats["subreddits"])
    
    return stats

@app.delete("/api/clear")
async def clear_data():
    """Clear all scraped data"""
    data_file = Path(__file__).parent / "scrapy_project" / "reddit_moderation_data.jsonl"
    
    if data_file.exists():
        os.remove(data_file)
        
    return {"message": "Data cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
