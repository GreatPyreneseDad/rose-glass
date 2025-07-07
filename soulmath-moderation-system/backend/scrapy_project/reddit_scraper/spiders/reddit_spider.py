import scrapy
import json
from urllib.parse import urljoin
from reddit_scraper.items import RedditPostItem, RedditCommentItem
from datetime import datetime

class RedditSpider(scrapy.Spider):
    name = 'reddit'
    allowed_domains = ['reddit.com', 'old.reddit.com']
    
    custom_settings = {
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS': 4,
    }
    
    def __init__(self, subreddits=None, *args, **kwargs):
        super(RedditSpider, self).__init__(*args, **kwargs)
        
        # Default subreddits for moderation analysis
        default_subreddits = [
            'politics', 'worldnews', 'news', 'science', 'technology',
            'AskReddit', 'unpopularopinion', 'changemyview'
        ]
        
        if subreddits:
            self.subreddits = subreddits.split(',')
        else:
            self.subreddits = default_subreddits
        
        # Generate start URLs using old.reddit.com for easier parsing
        self.start_urls = [
            f'https://old.reddit.com/r/{sub}/new/.json?limit=25' 
            for sub in self.subreddits
        ]
    
    def parse(self, response):
        """Parse subreddit listing page"""
        try:
            data = json.loads(response.text)
            posts = data['data']['children']
            
            for post in posts:
                post_data = post['data']
                
                # Skip stickied posts
                if post_data.get('stickied'):
                    continue
                
                # Create post item
                post_item = RedditPostItem()
                post_item['post_id'] = post_data['id']
                post_item['title'] = post_data['title']
                post_item['text'] = post_data.get('selftext', '')
                post_item['author'] = post_data['author']
                post_item['subreddit'] = post_data['subreddit']
                post_item['url'] = f"https://reddit.com{post_data['permalink']}"
                post_item['timestamp'] = datetime.fromtimestamp(post_data['created_utc']).isoformat()
                post_item['score'] = post_data['score']
                post_item['num_comments'] = post_data['num_comments']
                
                yield post_item
                
                # Fetch comments if there are any
                if post_data['num_comments'] > 0:
                    comment_url = f"https://old.reddit.com{post_data['permalink']}.json?limit=100"
                    yield scrapy.Request(
                        url=comment_url,
                        callback=self.parse_comments,
                        meta={'post_id': post_data['id']}
                    )
            
            # Follow pagination
            after = data['data'].get('after')
            if after:
                next_page = f"{response.url.split('?')[0]}?limit=25&after={after}"
                yield scrapy.Request(url=next_page, callback=self.parse)
                
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON from {response.url}")
    
    def parse_comments(self, response):
        """Parse comments from a post"""
        try:
            data = json.loads(response.text)
            
            # data[1] contains the comments
            if len(data) > 1 and data[1]['data']['children']:
                post_id = response.meta['post_id']
                yield from self._extract_comments(
                    data[1]['data']['children'], 
                    post_id, 
                    parent_id=post_id,
                    depth=0
                )
                
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse comments JSON from {response.url}")
    
    def _extract_comments(self, comments, post_id, parent_id, depth):
        """Recursively extract comments and replies"""
        for comment in comments:
            if comment['kind'] != 't1':  # Skip non-comment items
                continue
                
            comment_data = comment['data']
            
            # Skip deleted/removed comments
            if comment_data.get('body') in ['[deleted]', '[removed]', None]:
                continue
            
            # Create comment item
            comment_item = RedditCommentItem()
            comment_item['comment_id'] = comment_data['id']
            comment_item['post_id'] = post_id
            comment_item['parent_id'] = parent_id
            comment_item['text'] = comment_data['body']
            comment_item['author'] = comment_data.get('author', '[deleted]')
            comment_item['timestamp'] = datetime.fromtimestamp(comment_data['created_utc']).isoformat()
            comment_item['score'] = comment_data['score']
            comment_item['depth'] = depth
            
            yield comment_item
            
            # Process replies
            if comment_data.get('replies') and isinstance(comment_data['replies'], dict):
                replies = comment_data['replies']['data']['children']
                yield from self._extract_comments(
                    replies, 
                    post_id, 
                    parent_id=comment_data['id'],
                    depth=depth + 1
                )