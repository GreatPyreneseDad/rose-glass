#!/usr/bin/env python3
"""
Enhanced Document Ingestion with Metadata Classification
=========================================================

Adds intelligent document type detection and case classification:
- doc_type: evidence, motion, deposition, transcript, framework
- case: maslowsky, fardell, kowitz, stokes, general
- speaker: For depositions/transcripts
"""

from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import sys
import argparse
import logging
import numpy as np
import re
from collections import Counter

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
    logger.error("Qdrant not available. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False
    sys.exit(1)

try:
    from core.rose_glass_v2 import RoseGlassV2
    ROSE_GLASS_AVAILABLE = True
except ImportError:
    logger.warning("Rose Glass not available - emotional signatures will be basic")
    ROSE_GLASS_AVAILABLE = False


class EnhancedDocumentIngester:
    """Ingest documents with intelligent type and case classification"""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "knowledge_base",
        embedding_dim: int = 384
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # Initialize Qdrant
        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant client not available")

        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info(f"✓ Connected to Qdrant at {qdrant_host}:{qdrant_port}")

        # Initialize Rose Glass
        if ROSE_GLASS_AVAILABLE:
            self.rose_glass = RoseGlassV2()
            logger.info("✓ Rose Glass initialized")
        else:
            self.rose_glass = None
            logger.warning("⚠ Running without Rose Glass")

        # Build vocabulary
        self.vocabulary = self._build_legal_vocabulary()

        # Ensure collection exists
        self._ensure_collection()

    def _build_legal_vocabulary(self) -> List[str]:
        """Build vocabulary of common legal and emotional terms"""
        return [
            # Legal terms
            "motion", "court", "case", "defendant", "plaintiff", "evidence", "testimony",
            "hearing", "trial", "judge", "attorney", "counsel", "jury", "verdict",
            "objection", "ruling", "order", "filing", "petition", "complaint",
            "discovery", "deposition", "subpoena", "witness", "document", "exhibit",
            "contract", "agreement", "breach", "damages", "liability", "negligence",
            "custody", "divorce", "family", "child", "support", "visitation",
            "restraining", "protective", "harassment", "abuse", "domestic",
            # Case-specific
            "maslowsky", "macgregor", "fardell", "stokes", "kowitz",
            # Emotional/trauma-informed terms
            "trauma", "emotional", "distress", "anxiety", "fear", "worried", "concerned",
            "urgent", "crisis", "help", "support", "safety", "protection", "harm",
        ]

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Create simple TF-IDF-style embedding (fallback if no transformers)"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_counts = Counter(words)

        embedding = np.zeros(self.embedding_dim)

        for idx, vocab_term in enumerate(self.vocabulary[:self.embedding_dim]):
            if vocab_term in word_counts:
                tf = word_counts[vocab_term] / len(words) if words else 0
                embedding[idx] = tf

        non_vocab_words = [w for w in words if w not in self.vocabulary]
        for word in non_vocab_words[:50]:
            hash_val = hash(word)
            for i in range(3):
                idx = (hash_val + i * 17) % self.embedding_dim
                embedding[idx] += 0.1

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

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
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info("✓ Collection created")

    def _classify_doc_type(self, title: str, content: str) -> str:
        """
        Intelligently classify document type

        Types:
        - motion: Legal motions, briefs
        - evidence: Exhibits, evidence files
        - deposition: Depositions, sworn statements
        - transcript: Court transcripts, hearing records
        - framework: GCT/philosophical papers
        - discovery: Discovery requests/responses
        - order: Court orders, rulings
        """
        title_lower = title.lower()
        content_lower = content.lower()

        # Motion detection
        if any(word in title_lower for word in ['motion', 'brief', 'petition', 'complaint', 'objection']):
            return 'motion'

        # Evidence detection
        if any(word in title_lower for word in ['exhibit', 'evidence', 'attachment']):
            return 'evidence'

        # Deposition/Transcript detection
        if any(word in title_lower for word in ['deposition', 'testimony', 'sworn']):
            return 'deposition'
        if any(word in title_lower for word in ['transcript', 'hearing', 'proceeding']):
            return 'transcript'

        # Framework/theory papers
        if any(word in title_lower for word in ['grounded coherence', 'gct', 'framework', 'theory', 'philosophy', 'codex']):
            return 'framework'

        # Discovery
        if any(word in title_lower for word in ['discovery', 'interrogator', 'request for production']):
            return 'discovery'

        # Orders
        if any(word in title_lower for word in ['order', 'ruling', 'decision']):
            return 'order'

        # Default: Check content
        if 'grounded coherence' in content_lower or 'rose glass' in content_lower:
            return 'framework'

        return 'document'

    def _classify_case(self, title: str, content: str, file_path: Path) -> str:
        """
        Classify which case the document belongs to

        Cases:
        - maslowsky: Maslowsky v MacGregor
        - fardell: Fardell family law case
        - kowitz: Kowitz case
        - stokes: Stokes-related
        - general: Not case-specific
        """
        title_lower = title.lower()
        content_lower = content.lower()
        path_lower = str(file_path).lower()

        # Check path first (most reliable)
        if 'maslowsky' in path_lower:
            return 'maslowsky'
        if 'fardell' in path_lower:
            return 'fardell'
        if 'kowitz' in path_lower:
            return 'kowitz'
        if 'stokes' in path_lower:
            return 'stokes'

        # Check title
        if 'maslowsky' in title_lower or 'macgregor' in title_lower:
            return 'maslowsky'
        if 'fardell' in title_lower:
            return 'fardell'
        if 'kowitz' in title_lower:
            return 'kowitz'
        if 'stokes' in title_lower:
            return 'stokes'

        # Check content (first 500 chars)
        content_sample = content_lower[:500]
        if 'maslowsky' in content_sample:
            return 'maslowsky'
        if 'fardell' in content_sample:
            return 'fardell'

        return 'general'

    def _extract_speaker(self, title: str, content: str) -> Optional[str]:
        """Extract speaker name for depositions/transcripts"""
        title_lower = title.lower()

        # Look for deposition patterns
        depo_pattern = r'deposition.*?(of|by)\s+([a-z\s]+?)(?:$|\.|,)'
        match = re.search(depo_pattern, title_lower)
        if match:
            return match.group(2).strip().title()

        # Look for testimony patterns
        testimony_pattern = r'testimony.*?(of|by|from)\s+([a-z\s]+?)(?:$|\.|,)'
        match = re.search(testimony_pattern, title_lower)
        if match:
            return match.group(2).strip().title()

        return None

    def _analyze_emotional_signature(self, text: str) -> Dict:
        """Analyze text emotional signature"""
        if self.rose_glass:
            try:
                # Try RoseGlassV2 translate method
                result = self.rose_glass.translate(
                    text=text,
                    source_lens="modern_digital",
                    target_lens="trauma_informed"
                )
                return {
                    'psi': result.get('psi', 0.5),
                    'rho': result.get('rho', 0.5),
                    'q': result.get('q', 0.5),
                    'f': result.get('f', 0.5)
                }
            except Exception as e:
                logger.debug(f"Rose Glass analysis failed: {e}, using fallback")

        # Fallback: basic heuristics
        text_lower = text.lower()

        # Emotional activation (q)
        crisis_words = ['urgent', 'worried', 'help', 'crisis', 'emergency', 'scared', 'terrified', 'desperate']
        q_score = min(1.0, sum(1 for word in crisis_words if word in text_lower) * 0.2 + 0.3)

        # Wisdom depth (rho)
        wisdom_words = ['understand', 'consider', 'reflect', 'principle', 'philosophy', 'wisdom', 'insight']
        rho_score = min(1.0, sum(1 for word in wisdom_words if word in text_lower) * 0.15 + 0.3)

        # Social belonging (f)
        social_words = ['we', 'together', 'community', 'family', 'relationship', 'support']
        f_score = min(1.0, sum(1 for word in social_words if word in text_lower) * 0.15 + 0.3)

        # Internal consistency (psi) - default moderate
        psi_score = 0.6

        return {
            'psi': psi_score,
            'rho': rho_score,
            'q': q_score,
            'f': f_score
        }

    def ingest_text(
        self,
        text: str,
        title: str,
        file_path: Path,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Ingest single document with enhanced metadata

        Args:
            text: Document content
            title: Document title
            file_path: Original file path
            metadata: Additional metadata (optional)

        Returns:
            Document ID
        """
        # Generate embedding
        embedding = self._simple_embedding(text)

        # Analyze emotional signature
        signature = self._analyze_emotional_signature(text)

        # Classify document
        doc_type = self._classify_doc_type(title, text)
        case = self._classify_case(title, text, file_path)
        speaker = self._extract_speaker(title, text) if doc_type in ['deposition', 'transcript'] else None

        # Create document ID
        doc_id = hashlib.md5((title + text[:100]).encode()).hexdigest()

        # Prepare enhanced payload
        payload = {
            "content": text,
            "title": title,
            "doc_type": doc_type,
            "case": case,
            "source": str(file_path),
            "filename": file_path.name,
            **signature,
            **(metadata or {})
        }

        # Add speaker if available
        if speaker:
            payload["speaker"] = speaker

        # Store in Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=doc_id,
                vector=embedding.tolist(),
                payload=payload
            )]
        )

        logger.info(f"✓ {title} | type={doc_type} case={case} q={signature['q']:.2f}")
        return doc_id

    def ingest_file(self, file_path: Path, title: Optional[str] = None) -> str:
        """Ingest single file"""
        if title is None:
            title = file_path.stem

        # Read file content based on extension
        if file_path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
            except ImportError:
                logger.warning(f"PyPDF2 not available - skipping PDF: {file_path}")
                return None
            except Exception as e:
                logger.error(f"Error reading PDF {file_path}: {e}")
                return None
        else:
            # Text files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        if not content.strip():
            logger.warning(f"Empty content in {file_path} - skipping")
            return None

        return self.ingest_text(
            text=content,
            title=title,
            file_path=file_path
        )

    def ingest_directory(
        self,
        path: Path,
        extensions: List[str] = ['.txt', '.md', '.pdf'],
        recursive: bool = True,
        exclude_patterns: List[str] = None
    ) -> int:
        """
        Ingest all matching files from directory

        Returns:
            Number of documents ingested
        """
        count = 0
        glob_pattern = '**/*' if recursive else '*'
        exclude_patterns = exclude_patterns or []

        for file_path in path.glob(glob_pattern):
            # Check exclusions
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue

            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    doc_id = self.ingest_file(file_path)
                    if doc_id:
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
        description='Enhanced document ingestion with intelligent classification'
    )
    parser.add_argument('path', help='File or directory to ingest')
    parser.add_argument('--title', help='Custom title for single file')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--collection', default='knowledge_base', help='Collection name')
    parser.add_argument('--extensions', nargs='+', default=['.txt', '.md', '.pdf'],
                       help='File extensions to ingest')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Disable recursive directory scan')
    parser.add_argument('--exclude', nargs='+', default=[],
                       help='Patterns to exclude')

    args = parser.parse_args()

    # Initialize ingester
    try:
        ingester = EnhancedDocumentIngester(
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
        if doc_id:
            print(f"✓ Ingested: {path}")
            print(f"  Document ID: {doc_id}")
        else:
            print(f"✗ Failed to ingest: {path}")
            return 1
    elif path.is_dir():
        print(f"Ingesting from: {path}")
        print(f"Extensions: {args.extensions}")
        if args.exclude:
            print(f"Excluding: {args.exclude}")
        count = ingester.ingest_directory(
            path,
            extensions=args.extensions,
            recursive=not args.no_recursive,
            exclude_patterns=args.exclude
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
