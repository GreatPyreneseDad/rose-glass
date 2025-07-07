import scrapy
from datetime import datetime

class RedditPostItem(scrapy.Item):
    # Post metadata
    post_id = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()
    author = scrapy.Field()
    subreddit = scrapy.Field()
    url = scrapy.Field()
    timestamp = scrapy.Field()
    score = scrapy.Field()
    num_comments = scrapy.Field()
    
    # SoulMath analysis fields
    psi = scrapy.Field()
    rho = scrapy.Field()
    coherence = scrapy.Field()
    toxicity_risk = scrapy.Field()
    manipulation_risk = scrapy.Field()
    extremism_risk = scrapy.Field()
    spam_risk = scrapy.Field()
    harassment_risk = scrapy.Field()
    discourse_collapse = scrapy.Field()
    escalation_risk = scrapy.Field()
    overall_risk = scrapy.Field()
    
    # Metadata
    scraped_at = scrapy.Field()
    
class RedditCommentItem(scrapy.Item):
    # Comment metadata
    comment_id = scrapy.Field()
    post_id = scrapy.Field()
    parent_id = scrapy.Field()
    text = scrapy.Field()
    author = scrapy.Field()
    timestamp = scrapy.Field()
    score = scrapy.Field()
    depth = scrapy.Field()
    
    # SoulMath analysis fields
    psi = scrapy.Field()
    rho = scrapy.Field()
    coherence = scrapy.Field()
    toxicity_risk = scrapy.Field()
    manipulation_risk = scrapy.Field()
    extremism_risk = scrapy.Field()
    spam_risk = scrapy.Field()
    harassment_risk = scrapy.Field()
    discourse_collapse = scrapy.Field()
    escalation_risk = scrapy.Field()
    overall_risk = scrapy.Field()
    
    # Metadata
    scraped_at = scrapy.Field()