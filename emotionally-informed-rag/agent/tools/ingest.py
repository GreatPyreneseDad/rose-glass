#!/usr/bin/env python3
"""
Document ingestion with emotional signature tagging for Qdrant

Ingests documents into Qdrant vector database with:
- Vector embeddings
- Emotional signatures (Ψ, ρ, q, f, τ)
- Metadata

Usage:
    python ingest.py /path/to/documents/
    python ingest.py /path/to/file.txt --title "Custom Title"
"""

from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import sys
import argparse
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("Qdrant not available. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Sentence transformers not available. Install with: pip install sentence-transformers")
    TRANSFORMERS_AVAILABLE = False

try:
    from core.rose_glass_v2 import RoseGlassV2
    ROSE_GLASS_AVAILABLE = True
except ImportError:
    logger.warning("Rose Glass not available - emotional signatures will be basic")
    ROSE_GLASS_AVAILABLE = False


class DocumentIngester:
    """Ingest documents into Qdrant with emotional signatures"""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "knowledge_base",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name

        # Initialize Qdrant
        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant client not available")

        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Initialize encoder
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Sentence transformers not available")

        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

        # Initialize Rose Glass
        if ROSE_GLASS_AVAILABLE:
            self.rose_glass = RoseGlassV2()
            logger.info("✓ Rose Glass initialized")
        else:
            self.rose_glass = None
            logger.warning("⚠ Running without Rose Glass")

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            self.qdrant.get_collection(self.collection_name)
            logger.info(f"✓ Collection '{self.collection_name}' exists")
        except:
            logger.info(f"Creating collection '{self.collection_name}'...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=Distance.COSINE
                )
            )
            logger.info("✓ Collection created")

    def _analyze_emotional_signature(self, text: str) -> Dict:
        """Analyze text emotional signature"""
        if self.rose_glass:
            signature = self.rose_glass.analyze(text)
            return {
                'psi': signature.get('psi', 0.5),
                'rho': signature.get('rho', 0.5),
                'q': signature.get('q', 0.5),
                'f': signature.get('f', 0.5)
            }
        else:
            # Fallback: basic heuristics
            text_lower = text.lower()
            return {
                'psi': 0.6,
                'rho': 0.3 if 'understand' in text_lower else 0.5,
                'q': 0.7 if any(w in text_lower for w in ['urgent', 'worried', 'help']) else 0.3,
                'f': 0.4
            }

    def ingest_text(
        self,
        text: str,
        title: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Ingest single document with emotional signature

        Args:
            text: Document content
            title: Document title
            metadata: Additional metadata (optional)

        Returns:
            Document ID
        """
        # Generate embedding
        embedding = self.encoder.encode(text).tolist()

        # Analyze emotional signature
        signature = self._analyze_emotional_signature(text)

        # Create document ID
        doc_id = hashlib.md5((title + text).encode()).hexdigest()

        # Prepare payload
        payload = {
            "content": text,
            "title": title,
            **signature,
            **(metadata or {})
        }

        # Store in Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload
            )]
        )

        logger.info(f"✓ Ingested: {title} (q={signature['q']:.2f}, ρ={signature['rho']:.2f})")
        return doc_id

    def ingest_file(self, file_path: Path, title: Optional[str] = None) -> str:
        """Ingest single file"""
        if title is None:
            title = file_path.stem

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        return self.ingest_text(
            text=content,
            title=title,
            metadata={"source": str(file_path), "filename": file_path.name}
        )

    def ingest_directory(
        self,
        path: Path,
        extensions: List[str] = ['.txt', '.md', '.pdf'],
        recursive: bool = True
    ) -> int:
        """
        Ingest all matching files from directory

        Args:
            path: Directory path
            extensions: File extensions to include
            recursive: Recursively scan subdirectories

        Returns:
            Number of documents ingested
        """
        count = 0
        glob_pattern = '**/*' if recursive else '*'

        for file_path in path.glob(glob_pattern):
            if file_path.is_file() and file_path.suffix in extensions:
                try:
                    self.ingest_file(file_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {e}")

        return count

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "total_documents": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Ingest documents into Qdrant with emotional signatures'
    )
    parser.add_argument('path', help='File or directory to ingest')
    parser.add_argument('--title', help='Custom title for single file')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--collection', default='knowledge_base', help='Collection name')
    parser.add_argument('--extensions', nargs='+', default=['.txt', '.md'],
                       help='File extensions to ingest')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Disable recursive directory scan')

    args = parser.parse_args()

    # Initialize ingester
    try:
        ingester = DocumentIngester(
            qdrant_host=args.host,
            qdrant_port=args.port,
            collection_name=args.collection
        )
    except Exception as e:
        logger.error(f"Failed to initialize ingester: {e}")
        return 1

    # Ingest
    path = Path(args.path)

    if path.is_file():
        doc_id = ingester.ingest_file(path, title=args.title)
        print(f"✓ Ingested: {path}")
        print(f"  Document ID: {doc_id}")
    elif path.is_dir():
        print(f"Ingesting from: {path}")
        count = ingester.ingest_directory(
            path,
            extensions=args.extensions,
            recursive=not args.no_recursive
        )
        print(f"\n✓ Ingested {count} documents")
    else:
        print(f"Error: {path} does not exist")
        return 1

    # Show stats
    stats = ingester.get_stats()
    print("\nCollection Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
