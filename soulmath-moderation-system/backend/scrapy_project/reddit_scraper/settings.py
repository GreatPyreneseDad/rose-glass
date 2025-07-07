# Scrapy settings for reddit_scraper project

BOT_NAME = 'reddit_scraper'

SPIDER_MODULES = ['reddit_scraper.spiders']
NEWSPIDER_MODULE = 'reddit_scraper.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 8

# Configure a delay for requests for the same website (default: 0)
DOWNLOAD_DELAY = 1
RANDOMIZE_DOWNLOAD_DELAY = True

# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 4

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# User agent
USER_AGENT = 'SoulMathModeration/1.0 (+https://github.com/GreatPyreneseDad/GCT)'

# Configure pipelines
ITEM_PIPELINES = {
    'reddit_scraper.pipelines.SoulMathAnalysisPipeline': 300,
    'reddit_scraper.pipelines.JsonWriterPipeline': 400,
}

# Configure extensions
EXTENSIONS = {
    'scrapy.extensions.telnet.TelnetConsole': None,
}

# Configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# Request headers
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en',
}

# Export settings
FEED_EXPORT_ENCODING = 'utf-8'

# Reddit-specific settings
REDDIT_DOMAINS = ['reddit.com', 'old.reddit.com']
MAX_COMMENTS_PER_POST = 100
MAX_POSTS_PER_SUBREDDIT = 50

# Retry settings
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# AutoThrottle extension settings
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 4.0
AUTOTHROTTLE_DEBUG = False

# Memory usage settings
MEMUSAGE_ENABLED = True
MEMUSAGE_LIMIT_MB = 512
MEMUSAGE_WARNING_MB = 256

# Logging
LOG_LEVEL = 'INFO'