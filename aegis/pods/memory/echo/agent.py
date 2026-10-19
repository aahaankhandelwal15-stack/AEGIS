"""Echo Agent - Long-Term Memory Backbone
Provides durable, queryable, versioned long-term memory with fast working-memory recall.
Implements FR-MEM-001 (durability) and FR-MEM-002 (versioned KG point-in-time queries).
"""

import json
import pickle
import hashlib
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer
import threading
import queue
import logging
from dataclasses import dataclass, asdict

from aegis.core.models import MemoryType, StimulusEnvelope, PrivacyClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryEntity:
    """Represents an entity in the knowledge graph."""
    entity_id: str
    entity_type: str
    attributes: Dict[str, Any]
    created_at: float
    updated_at: float
    version: int = 1

@dataclass
class MemoryRelation:
    """Represents a relationship between entities in the knowledge graph."""
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float
    created_at: float
    updated_at: float
    version: int = 1

class VectorStore:
    """Handles semantic memory storage and retrieval using FAISS."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.id_to_text: Dict[int, str] = {}
        self.text_to_id: Dict[str, int] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.next_id = 0
        self._lock = threading.RLock()

    def add_text(self, text: str) -> int:
        """Add text to vector store and return its ID."""
        with self._lock:
            if text in self.text_to_id:
                return self.text_to_id[text]

            # Get or compute embedding
            if text in self.embeddings_cache:
                embedding = self.embeddings_cache[text]
            else:
                embedding = self.model.encode([text], normalize_embeddings=True)[0]
                self.embeddings_cache[text] = embedding

            # Add to FAISS index
            self.index.add(np.array([embedding], dtype=np.float32))
            text_id = self.next_id
            self.id_to_text[text_id] = text
            self.text_to_id[text] = text_id
            self.next_id += 1

            return text_id

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar texts and return (text, score) tuples."""
        with self._lock:
            if self.index.ntotal == 0:
                return []

            query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
            scores, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                k
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx in self.id_to_text:
                    results.append((self.id_to_text[idx], float(score)))

            return results

    def get_vector_count(self) -> int:
        """Return number of vectors in store."""
        return self.index.ntotal

    def save(self, path: Path):
        """Save vector store to disk."""
        with self._lock:
            data = {
                'id_to_text': self.id_to_text,
                'text_to_id': self.text_to_id,
                'next_id': self.next_id,
                'embeddings_cache': self.embeddings_cache
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            # Save FAISS index separately
            faiss.write_index(self.index, str(path.with_suffix('.faiss')))

    def load(self, path: Path):
        """Load vector store from disk."""
        with self._lock:
            index_path = path.with_suffix('.faiss')
            if path.exists() and index_path.exists():
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.id_to_text = data['id_to_text']
                self.text_to_id = data['text_to_id']
                self.next_id = data['next_id']
                self.embeddings_cache = data['embeddings_cache']
                self.index = faiss.read_index(str(index_path))
            else:
                logger.warning(f"No vector store found at {path}")

class KnowledgeGraph:
    """Handles versioned knowledge graph storage and point-in-time queries."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.events: List[Dict] = []  # Event log for versioning
        self.snapshots: Dict[float, Dict] = {}  # timestamp -> graph snapshot
        self._lock = threading.RLock()
        self._snapshot_interval = 300  # 5 minutes
        self._last_snapshot = time.time()

    def add_entity(self, entity: MemoryEntity) -> str:
        """Add entity to knowledge graph and log event."""
        with self._lock:
            event = {
                'timestamp': time.time(),
                'type': 'add_entity',
                'entity_id': entity.entity_id,
                'data': asdict(entity)
            }
            self.events.append(event)

            # Add to current graph
            self.graph.add_node(
                entity.entity_id,
                entity_type=entity.entity_type,
                attributes=entity.attributes,
                created_at=entity.created_at,
                updated_at=entity.updated_at,
                version=entity.version
            )

            # Check if we need a snapshot
            if time.time() - self._last_snapshot > self._snapshot_interval:
                self._take_snapshot()

            return entity.entity_id

    def add_relation(self, relation: MemoryRelation) -> str:
        """Add relation to knowledge graph and log event."""
        with self._lock:
            event = {
                'timestamp': time.time(),
                'type': 'add_relation',
                'relation_id': relation.relation_id,
                'data': asdict(relation)
            }
            self.events.append(event)

            # Add to current graph
            self.graph.add_edge(
                relation.subject_id,
                relation.object_id,
                key=relation.relation_id,
                predicate=relation.predicate,
                confidence=relation.confidence,
                created_at=relation.created_at,
                updated_at=relation.updated_at,
                version=relation.version
            )

            # Check if we need a snapshot
            if time.time() - self._last_snapshot > self._snapshot_interval:
                self._take_snapshot()

            return relation.relation_id

    def _take_snapshot(self):
        """Take a snapshot of the current graph state."""
        snapshot = {
            'timestamp': time.time(),
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': dict(self.graph.edges(data=True)),
            'event_count': len(self.events)
        }
        self.snapshots[snapshot['timestamp']] = snapshot
        self._last_snapshot = snapshot['timestamp']
        logger.debug(f"Took knowledge graph snapshot at {snapshot['timestamp']}")

    def get_state_at_time(self, timestamp: float) -> nx.MultiDiGraph:
        """Reconstruct knowledge graph state at given timestamp."""
        with self._lock:
            # Find the latest snapshot before or at timestamp
            valid_snapshots = {ts: snap for ts, snap in self.snapshots.items() if ts <= timestamp}
            if not valid_snapshots:
                # Start from empty graph and replay all events up to timestamp
                graph = nx.MultiDiGraph()
                start_event_idx = 0
            else:
                latest_snapshot_ts = max(valid_snapshots.keys())
                snapshot = valid_snapshots[latest_snapshot_ts]
                graph = nx.MultiDiGraph()
                graph.add_nodes_from(snapshot['nodes'])
                graph.add_edges_from(snapshot['edges'])
                start_event_idx = snapshot['event_count']

            # Replay events from start_event_idx up to timestamp
            for event in self.events[start_event_idx:]:
                if event['timestamp'] > timestamp:
                    break

                if event['type'] == 'add_entity':
                    data = event['data']
                    entity = MemoryEntity(**data)
                    graph.add_node(
                        entity.entity_id,
                        entity_type=entity.entity_type,
                        attributes=entity.attributes,
                        created_at=entity.created_at,
                        updated_at=entity.updated_at,
                        version=entity.version
                    )
                elif event['type'] == 'add_relation':
                    data = event['data']
                    relation = MemoryRelation(**data)
                    graph.add_edge(
                        relation.subject_id,
                        relation.object_id,
                        key=relation.relation_id,
                        predicate=relation.predicate,
                        confidence=relation.confidence,
                        created_at=relation.created_at,
                        updated_at=relation.updated_at,
                        version=relation.version
                    )

            return graph

    def query_entities(self,
                      entity_type: Optional[str] = None,
                      attribute_filters: Optional[Dict[str, Any]] = None) -> List[MemoryEntity]:
        """Query entities in current graph with filters."""
        with self._lock:
            entities = []
            for node_id, node_data in self.graph.nodes(data=True):
                if entity_type and node_data.get('entity_type') != entity_type:
                    continue

                if attribute_filters:
                    match = True
                    for key, value in attribute_filters.items():
                        if node_data.get('attributes', {}).get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                entities.append(MemoryEntity(
                    entity_id=node_id,
                    entity_type=node_data.get('entity_type', 'unknown'),
                    attributes=node_data.get('attributes', {}),
                    created_at=node_data.get('created_at', time.time()),
                    updated_at=node_data.get('updated_at', time.time()),
                    version=node_data.get('version', 1)
                ))

            return entities

    def query_relations(self,
                       subject_id: Optional[str] = None,
                       predicate: Optional[str] = None,
                       object_id: Optional[str] = None,
                       min_confidence: float = 0.0) -> List[MemoryRelation]:
        """Query relations in current graph with filters."""
        with self._lock:
            relations = []
            for u, v, key, edge_data in self.graph.edges(keys=True, data=True):
                if subject_id and u != subject_id:
                    continue
                if object_id and v != object_id:
                    continue
                if predicate and edge_data.get('predicate') != predicate:
                    continue
                if edge_data.get('confidence', 0.0) < min_confidence:
                    continue

                relations.append(MemoryRelation(
                    relation_id=key,
                    subject_id=u,
                    predicate=edge_data.get('predicate', 'unknown'),
                    object_id=v,
                    confidence=edge_data.get('confidence', 0.0),
                    created_at=edge_data.get('created_at', time.time()),
                    updated_at=edge_data.get('updated_at', time.time()),
                    version=edge_data.get('version', 1)
                ))

            return relations

    def save(self, path: Path):
        """Save knowledge graph to disk."""
        with self._lock:
            data = {
                'events': self.events,
                'snapshots': self.snapshots,
                '_last_snapshot': self._last_snapshot
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    def load(self, path: Path):
        """Load knowledge graph from disk."""
        with self._lock:
            if path.exists():
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.events = data.get('events', [])
                self.snapshots = data.get('snapshots', {})
                self._last_snapshot = data.get('_last_snapshot', time.time())

                # Rebuild current graph from events
                self.graph = nx.MultiDiGraph()
                for event in self.events:
                    if event['type'] == 'add_entity':
                        data = event['data']
                        entity = MemoryEntity(**data)
                        self.graph.add_node(
                            entity.entity_id,
                            entity_type=entity.entity_type,
                            attributes=entity.attributes,
                            created_at=entity.created_at,
                            updated_at=entity.updated_at,
                            version=entity.version
                        )
                    elif event['type'] == 'add_relation':
                        data = event['data']
                        relation = MemoryRelation(**data)
                        self.graph.add_edge(
                            relation.subject_id,
                            relation.object_id,
                            key=relation.relation_id,
                            predicate=relation.predicate,
                            confidence=relation.confidence,
                            created_at=relation.created_at,
                            updated_at=relation.updated_at,
                            version=relation.version
                        )
            else:
                logger.warning(f"No knowledge graph found at {path}")

class EpisodicMemory:
    """Handles episodic memory storage and consolidation."""

    def __init__(self, max_episodic: int = 10000):
        self.episodic: List[StimulusEnvelope] = []
        self.max_episodic = max_episodic
        self._lock = threading.RLock()

    def add_episode(self, episode: StimulusEnvelope):
        """Add an episodic memory."""
        with self._lock:
            self.episodic.append(episode)
            # Keep only the most recent episodes
            if len(self.episodic) > self.max_episodic:
                self.episodic = self.episodic[-self.max_episodic:]

    def get_recent_episodes(self, count: int = 100) -> List[StimulusEnvelope]:
        """Get most recent episodic memories."""
        with self._lock:
            return self.episodic[-count:] if self.episodic else []

    def get_episodes_by_time_range(self,
                                 start_time: float,
                                 end_time: float) -> List[StimulusEnvelope]:
        """Get episodes within a time range."""
        with self._lock:
            return [
                ep for ep in self.episodic
                if start_time <= ep.timestamp <= end_time
            ]

    def consolidate_to_semantic(self,
                              vector_store: VectorStore,
                              knowledge_graph: KnowledgeGraph) -> int:
        """
        Consolidate episodic memories into semantic memory.
        Returns number of episodes consolidated.
        """
        consolidated_count = 0
        with self._lock:
            # Simple consolidation: extract entities and relations from episodic content
            # In a real implementation, this would use NLP techniques
            episodes_to_remove = []

            for i, episode in enumerate(self.episodic):
                # For demonstration, we'll treat each episode as a semantic fact
                # In practice, you'd extract entities, relations, etc.
                content_str = str(episode.content)
                if len(content_str.strip()) > 10:  # Only consolidate meaningful content
                    # Add to vector store as semantic memory
                    vector_store.add_text(f"Episode {episode.stimulus_id}: {content_str[:200]}")

                    # Create a semantic entity from the episode
                    entity_id = f"semantic_{episode.stimulus_id}"
                    entity = MemoryEntity(
                        entity_id=entity_id,
                        entity_type="semantic_episode",
                        attributes={
                            "source_episode": episode.stimulus_id,
                            "content_summary": content_str[:500],
                            "timestamp": episode.timestamp
                        },
                        created_at=episode.timestamp,
                        updated_at=episode.timestamp
                    )
                    knowledge_graph.add_entity(entity)

                    episodes_to_remove.append(i)
                    consolidated_count += 1

            # Remove consolidated episodes (in reverse order to maintain indices)
            for i in reversed(episodes_to_remove):
                del self.episodic[i]

        return consolidated_count

class Echo:
    """
    Echo Agent - Long-Term Memory Backbone
    Main interface for interacting with the memory system.
    """

    def __init__(self,
                 storage_path: Union[str, Path] = "./echo_storage",
                 vector_dimension: int = 384,
                 max_episodic: int = 10000,
                 consolidation_interval: int = 3600):  # 1 hour
        """
        Initialize Echo agent.

        Args:
            storage_path: Path to store persistent data
            vector_dimension: Dimension for vector embeddings
            max_episodic: Maximum episodic memories to keep
            consolidation_interval: Seconds between consolidation runs
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Initialize components
        self.vector_store = VectorStore(dimension=vector_dimension)
        self.knowledge_graph = KnowledgeGraph()
        self.episodic_memory = EpisodicMemory(max_episodic=max_episodic)

        # Consolidation settings
        self.consolidation_interval = consolidation_interval
        self._last_consolidation = time.time()
        self._consolidation_queue = queue.Queue()
        self._consolidation_thread = None
        self._stop_consolidation = False

        # Load persisted data
        self._load_persisted_data()

        # Start background consolidation thread
        self._start_consolidation_thread()

        logger.info(f"Echo agent initialized with storage at {self.storage_path}")

    def _load_persisted_data(self):
        """Load persisted data from disk."""
        try:
            # Load vector store
            vector_store_path = self.storage_path / "vector_store.pkl"
            self.vector_store.load(vector_store_path)

            # Load knowledge graph
            kg_path = self.storage_path / "knowledge_graph.pkl"
            self.knowledge_graph.load(kg_path)

            # Load episodic memory (simple approach: store as pickle)
            episodic_path = self.storage_path / "episodic_memory.pkl"
            if episodic_path.exists():
                with open(episodic_path, 'rb') as f:
                    self.episodic_memory.episodic = pickle.load(f)

            logger.info("Loaded persisted Echo data")
        except Exception as e:
            logger.warning(f"Could not load persisted data: {e}. Starting fresh.")

    def _save_persisted_data(self):
        """Save persisted data to disk."""
        try:
            # Save vector store
            vector_store_path = self.storage_path / "vector_store.pkl"
            self.vector_store.save(vector_store_path)

            # Save knowledge graph
            kg_path = self.storage_path / "knowledge_graph.pkl"
            self.knowledge_graph.save(kg_path)

            # Save episodic memory
            episodic_path = self.storage_path / "episodic_memory.pkl"
            with open(episodic_path, 'wb') as f:
                pickle.dump(self.episodic_memory.episodic, f)

            logger.debug("Saved persisted Echo data")
        except Exception as e:
            logger.error(f"Failed to save persisted data: {e}")

    def _start_consolidation_thread(self):
        """Start background thread for memory consolidation."""
        self._consolidation_thread = threading.Thread(
            target=self._consolidation_worker,
            daemon=True
        )
        self._consolidation_thread.start()
        logger.debug("Started consolidation thread")

    def _consolidation_worker(self):
        """Worker thread for periodic memory consolidation."""
        while not self._stop_consolidation:
            try:
                # Wait for consolidation interval or signal
                time.sleep(self.consolidation_interval)

                # Check if it's time to consolidate
                if time.time() - self._last_consolidation >= self.consolidation_interval:
                    self._perform_consolidation()

            except Exception as e:
                logger.error(f"Error in consolidation worker: {e}")

    def _perform_consolidation(self):
        """Perform memory consolidation from episodic to semantic."""
        logger.info("Starting memory consolidation...")
        start_time = time.time()

        try:
            consolidated = self.episodic_memory.consolidate_to_semantic(
                self.vector_store,
                self.knowledge_graph
            )

            # Save persisted data after consolidation
            self._save_persisted_data()

            self._last_consolidation = time.time()
            logger.info(
                f"Consolidation completed. Consolidated {consolidated} episodes in "
                f"{time.time() - start_time:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

    def store_stimulus(self,
                      content: Any,
                      source: str = "unknown",
                      metadata: Optional[Dict[str, Any]] = None,
                      memory_type: MemoryType = MemoryType.EPISODIC,
                      modality: str = "unknown",
                      confidence: float = 1.0) -> str:
        """
        Store a stimulus envelope in memory.

        Args:
            content: The content to store
            source: Source of the stimulus
            metadata: Additional metadata
            memory_type: Type of memory (episodic, semantic, procedural)
            modality: Broad class of stimulus (e.g., 'calendar', 'browser', 'voice')
            confidence: Confidence/probability value in [0.0, 1.0] range

        Returns:
            stimulus_id: Unique ID for the stored stimulus
        """
        stimulus_id = str(uuid.uuid4())
        timestamp = time.time()

        # Extract modality and confidence from metadata if present, otherwise use parameters
        metadata_dict = metadata or {}
        final_modality = metadata_dict.get("modality", modality)
        final_confidence = metadata_dict.get("confidence", confidence)

        # Ensure confidence is within valid range
        final_confidence = max(0.0, min(1.0, final_confidence))

        envelope = StimulusEnvelope(
            stimulus_id=stimulus_id,
            content=content,
            timestamp=timestamp,
            source=source,
            modality=final_modality,
            confidence=final_confidence,
            privacy_classification=PrivacyClassification.PUBLIC,  # Default to PUBLIC
            metadata=metadata_dict,
            memory_type=memory_type
        )

        # Store in appropriate memory system
        if memory_type == MemoryType.EPISODIC:
            self.episodic_memory.add_episode(envelope)
        elif memory_type == MemoryType.SEMANTIC:
            # Store as semantic memory
            content_str = str(content)
            if content_str.strip():
                self.vector_store.add_text(content_str)

                # Also create a semantic entity
                entity_id = f"semantic_{stimulus_id}"
                entity = MemoryEntity(
                    entity_id=entity_id,
                    entity_type="semantic_stimulus",
                    attributes={
                        "content": content_str[:1000],  # Truncate for storage
                        "source": source,
                        "timestamp": timestamp
                    },
                    created_at=timestamp,
                    updated_at=timestamp
                )
                self.knowledge_graph.add_entity(entity)
        # Procedural memory would be handled differently (skills, procedures)

        logger.debug(f"Stored stimulus {stimulus_id} in {memory_type.value} memory")
        return stimulus_id

    def recall_semantic(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Recall semantically similar memories.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (memory_text, similarity_score) tuples
        """
        return self.vector_store.search(query, k)

    def recall_episodic(self,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       count: Optional[int] = None) -> List[StimulusEnvelope]:
        """
        Recall episodic memories.

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            count: Number of most recent episodes to return (ignored if time range given)

        Returns:
            List of StimulusEnvelope objects
        """
        if start_time is not None and end_time is not None:
            return self.episodic_memory.get_episodes_by_time_range(start_time, end_time)
        elif count is not None:
            return self.episodic_memory.get_recent_episodes(count)
        else:
            return self.episodic_memory.get_recent_episodes(100)  # Default

    def query_knowledge_graph(
        self,
        timestamp: Optional[float] = None,
        **kwargs) -> Dict[str, Any]:

        if timestamp is not None:
            graph = self.knowledge_graph.get_state_at_time(timestamp)
            logger.warning(
                "Point-in-time KG query not fully implemented; using current state"
            )

        entity_kwargs = {}
        relation_kwargs = {}

        # Entity filters
        if "entity_type" in kwargs:
            entity_kwargs["entity_type"] = kwargs["entity_type"]

        if "attribute_filters" in kwargs:
            entity_kwargs["attribute_filters"] = kwargs["attribute_filters"]

        # Relation filters
        if "subject_id" in kwargs:
            relation_kwargs["subject_id"] = kwargs["subject_id"]

        if "predicate" in kwargs:
            relation_kwargs["predicate"] = kwargs["predicate"]

        if "object_id" in kwargs:
            relation_kwargs["object_id"] = kwargs["object_id"]

        if "min_confidence" in kwargs:
            relation_kwargs["min_confidence"] = kwargs["min_confidence"]

        entities = self.knowledge_graph.query_entities(**entity_kwargs)
        relations = self.knowledge_graph.query_relations(**relation_kwargs)

        return {
            "entities": [asdict(e) for e in entities],
            "relations": [asdict(r) for r in relations]
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory systems."""
        return {
            'vector_store_count': self.vector_store.get_vector_count(),
            'episodic_count': len(self.episodic_memory.episodic),
            'knowledge_graph_nodes': self.knowledge_graph.graph.number_of_nodes(),
            'knowledge_graph_edges': self.knowledge_graph.graph.number_of_edges(),
            'knowledge_graph_events': len(self.knowledge_graph.events),
            'last_consolidation': self._last_consolidation,
            'storage_path': str(self.storage_path)
        }

    def shutdown(self):
        """Shutdown the Echo agent and save persisted data."""
        logger.info("Shutting down Echo agent... (called from %s)", threading.current_thread().name)
        import traceback
        logger.info("Shutdown traceback: %s", traceback.format_stack())
        self._stop_consolidation = True
        if self._consolidation_thread:
            self._consolidation_thread.join(timeout=5)
        self._save_persisted_data()
        logger.info("Echo agent shut down")

# Example usage and testing
if __name__ == "__main__":
    # Initialize Echo
    echo = Echo(storage_path="./test_echo_storage")

    # Store some test stimuli
    echo.store_stimulus(
        content="The user prefers morning meetings and dislikes late afternoon slots.",
        source="user_interview",
        metadata={"topic": "scheduling_preferences", "confidence": 0.9}
    )

    echo.store_stimulus(
        content="Project deadline is set for October 15, 2026.",
        source="project_manager",
        metadata={"project": "Phoenix", "priority": "high"}
    )

    # Store semantic memory directly
    echo.store_stimulus(
        content="Users typically check email first thing in the morning.",
        source="research_study",
        memory_type=MemoryType.SEMANTIC,
        metadata={"study": "email_habits_2025", "sample_size": 1000}
    )

    # Test semantic recall
    print("\nSemantic recall for 'morning routine':")
    results = echo.recall_semantic("morning routine", k=3)
    for text, score in results:
        print(f"  - {text[:80]}... (score: {score:.3f})")

    # Test episodic recall
    print("\nRecent episodic memories:")
    recent = echo.recall_episodic(count=5)
    for ep in recent:
        print(f"  - [{ep.source}] {str(ep.content)[:60]}...")

    # Test knowledge graph query
    print("\nKnowledge graph entities:")
    kg_result = echo.query_knowledge_graph(entity_type="semantic_stimulus")
    for entity in kg_result['entities']:
        print(f"  - {entity['entity_id']}: {entity['attributes']}")

    # Print stats
    print("\nMemory stats:")
    stats = echo.get_memory_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    # Shutdown
    echo.shutdown()

    # Clean up test storage
    import shutil
    shutil.rmtree("./test_echo_storage", ignore_errors=True)
    print("\nTest completed and cleaned up.")

