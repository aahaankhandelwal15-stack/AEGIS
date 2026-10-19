"""IRIS - Perception Layer Ingestion Gateway for AEGIS.

IRIS (Intelligent Recognition and Ingestion System) serves as the perception
layer ingestion gateway that receives external events from various sources,
normalizes them to StimulusEnvelope format, validates events, applies privacy
classification, handles deduplication, manages rate limiting and backpressure,
and routes events to appropriate consumers via the existing AEGIS architecture.
"""

import hashlib
import time
import uuid
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List, Callable, Tuple
from threading import Lock, RLock
from dataclasses import dataclass, field
from enum import Enum
import logging

from aegis.core.models import (
    StimulusEnvelope,
    PrivacyClassification,
    MemoryType,
    AgentEnvelope,
    AgentResult
)
from aegis.core.protocol import KAIRO_AGENT_ID, MEMORY_POD_ID, IRIS_AGENT_ID, PERCEPTION_POD_ID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingResult(Enum):
    """Result of processing an incoming event."""
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    RATE_LIMITED = "rate_limited"
    INVALID = "invalid"
    ERROR = "error"


@dataclass
class SourceAdapter:
    """Base class for source adapters that ingest events into IRIS."""
    name: str
    enabled: bool = True
    last_poll: float = field(default_factory=time.time)
    poll_interval: float = 5.0  # seconds

    def poll(self) -> List[Dict[str, Any]]:
        """Poll for new events from the source.

        Returns:
            List of raw event dictionaries to be processed by IRIS.
        """
        raise NotImplementedError("Source adapters must implement poll() method")

    def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw event into a format suitable for StimulusEnvelope creation.

        Args:
            raw_event: Raw event data from the source

        Returns:
            Normalized event dictionary with keys matching StimulusEnvelope fields
        """
        raise NotImplementedError("Source adapters must implement normalize() method")


@dataclass
class DeduplicationCache:
    """Tracks recently seen events for deduplication."""
    max_size: int = 1000
    ttl_seconds: float = 300.0  # 5 minutes
    _cache: Dict[str, float] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def is_duplicate(self, event_id: str) -> bool:
        """Check if an event has been seen recently.

        Args:
            event_id: Unique identifier for the event

        Returns:
            True if event is a duplicate, False otherwise
        """
        with self._lock:
            # Clean expired entries
            now = time.time()
            expired_keys = [
                key for key, timestamp in self._cache.items()
                if now - timestamp > self.ttl_seconds
            ]
            for key in expired_keys:
                del self._cache[key]

            # Check if current event is duplicate
            if event_id in self._cache:
                return True

            # Add to cache
            self._cache[event_id] = now

            # Enforce size limit
            if len(self._cache) > self.max_size:
                # Remove oldest entries
                sorted_items = sorted(self._cache.items(), key=lambda x: x[1])
                for key, _ in sorted_items[:len(self._cache) - self.max_size]:
                    del self._cache[key]

            return False


@dataclass
class RateLimiter:
    """Simple token bucket rate limiter."""
    max_events_per_second: float = 10.0
    burst_capacity: float = 20.0
    _tokens: float = field(default_factory=lambda: 20.0)
    _last_update: float = field(default_factory=time.time)
    _lock: Lock = field(default_factory=Lock)

    def allow_event(self) -> bool:
        """Check if an event is allowed under rate limiting.

        Returns:
            True if event is allowed, False if rate limited
        """
        with self._lock:
            now = time.time()
            # Add tokens based on time passed
            time_passed = now - self._last_update
            self._tokens = min(
                self.burst_capacity,
                self._tokens + time_passed * self.max_events_per_second
            )
            self._last_update = now

            # Check if we have enough tokens
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class IrisAgent:
    """IRIS agent: Perception layer ingestion gateway."""

    def __init__(
        self,
        source_adapters: Optional[List[SourceAdapter]] = None,
        deduplication_enabled: bool = True,
        rate_limit_enabled: bool = True,
        max_events_per_second: float = 10.0,
        burst_capacity: float = 20.0,
        deduplication_ttl: float = 300.0,
        deduplication_max_size: int = 1000
    ):
        """Initialize IRIS agent.

        Args:
            source_adapters: List of source adapters to ingest events from
            deduplication_enabled: Whether to enable deduplication
            rate_limit_enabled: Whether to enable rate limiting
            max_events_per_second: Maximum events per second allowed
            burst_capacity: Burst capacity for rate limiter
            deduplication_ttl: Time to live for deduplication cache (seconds)
            deduplication_max_size: Maximum size of deduplication cache
        """
        self.source_adapters = source_adapters or []
        self.deduplication_enabled = deduplication_enabled
        self.rate_limit_enabled = rate_limit_enabled

        # Initialize components
        self.deduplication_cache = DeduplicationCache(
            max_size=deduplication_max_size,
            ttl_seconds=deduplication_ttl
        ) if deduplication_enabled else None

        self.rate_limiter = RateLimiter(
            max_events_per_second=max_events_per_second,
            burst_capacity=burst_capacity
        ) if rate_limit_enabled else None

        # Event routing - in a full implementation, this would connect to
        # other agents via AgentEnvelope, but for now we'll simulate
        self.event_handlers: Dict[str, Callable[[StimulusEnvelope], None]] = {}

        # Statistics
        self.stats = {
            'events_received': 0,
            'events_accepted': 0,
            'events_duplicate': 0,
            'events_rate_limited': 0,
            'events_invalid': 0,
            'events_error': 0,
            'last_reset': time.time()
        }
        self._stats_lock = Lock()

        logger.info("IRIS agent initialized")

    def add_source_adapter(self, adapter: SourceAdapter) -> None:
        """Add a source adapter to IRIS.

        Args:
            adapter: Source adapter to add
        """
        self.source_adapters.append(adapter)
        logger.info(f"Added source adapter: {adapter.name}")

    def remove_source_adapter(self, adapter_name: str) -> bool:
        """Remove a source adapter by name.

        Args:
            adapter_name: Name of adapter to remove

        Returns:
            True if adapter was removed, False if not found
        """
        for i, adapter in enumerate(self.source_adapters):
            if adapter.name == adapter_name:
                del self.source_adapters[i]
                logger.info(f"Removed source adapter: {adapter_name}")
                return True
        return False

    def register_event_handler(self, event_type: str, handler: Callable[[StimulusEnvelope], None]) -> None:
        """Register a handler for processed events.

        Args:
            event_type: Type of event to handle (for filtering)
            handler: Function to call when event is processed
        """
        self.event_handlers[event_type] = handler
        logger.info(f"Registered event handler for type: {event_type}")

    def _generate_event_id(self, raw_event: Dict[str, Any]) -> str:
        """Generate a unique identifier for deduplication.

        Args:
            raw_event: Raw event data

        Returns:
            String identifier for the event
        """
        # Create a deterministic ID based on event content
        event_str = str(sorted(raw_event.items()))
        return hashlib.md5(event_str.encode()).hexdigest()

    def _apply_privacy_classification(
        self,
        raw_event: Dict[str, Any],
        default_classification: PrivacyClassification = PrivacyClassification.PUBLIC
    ) -> PrivacyClassification:
        """Determine appropriate privacy classification for an event.

        Args:
            raw_event: Raw event data
            default_classification: Default classification to use

        Returns:
            PrivacyClassification for the event
        """
        # Check if privacy classification is explicitly provided
        if 'privacy_classification' in raw_event:
            try:
                return PrivacyClassification(raw_event['privacy_classification'])
            except ValueError:
                logger.warning(
                    f"Invalid privacy classification '{raw_event['privacy_classification']}', "
                    f"using default: {default_classification}"
                )

        # Check for privacy hints in metadata
        metadata = raw_event.get('metadata', {})
        if 'privacy_level' in metadata:
            try:
                return PrivacyClassification(metadata['privacy_level'])
            except ValueError:
                logger.warning(
                    f"Invalid privacy level in metadata '{metadata['privacy_level']}', "
                    f"using default: {default_classification}"
                )

        # Heuristic-based classification (simplified)
        source = raw_event.get('source', '').lower()
        content_str = str(raw_event.get('content', '')).lower()

        # Personal/sensitive indicators
        personal_indicators = ['email', 'phone', 'address', 'name', 'ssn', 'birthday']
        sensitive_indicators = ['password', 'token', 'key', 'secret', 'credit_card']

        if any(indicator in source or indicator in content_str
               for indicator in sensitive_indicators):
            return PrivacyClassification.RESTRICTED
        elif any(indicator in source or indicator in content_str
                 for indicator in personal_indicators):
            return PrivacyClassification.PERSONAL

        return default_classification

    def _create_stimulus_envelope(self, normalized_event: Dict[str, Any]) -> StimulusEnvelope:
        """Create a StimulusEnvelope from a normalized event.

        Args:
            normalized_event: Normalized event data

        Returns:
            StimulusEnvelope object
        """
        # Extract or generate required fields
        stimulus_id = normalized_event.get('stimulus_id', str(uuid.uuid4()))
        timestamp = normalized_event.get('timestamp', time.time())
        source = normalized_event.get('source', 'unknown')
        modality = normalized_event.get('modality', 'unknown')

        # Validate and clamp confidence
        confidence = normalized_event.get('confidence', 1.0)
        confidence = max(0.0, min(1.0, confidence))

        # Determine privacy classification
        privacy_classification = self._apply_privacy_classification(
            normalized_event.get('raw_event', {}),
            PrivacyClassification.PUBLIC
        )

        # Extract content
        content = normalized_event.get('content', normalized_event.get('raw_event', {}))

        # Extract optional fields
        session_id = normalized_event.get('session_id')
        correlation_id = normalized_event.get('correlation_id')
        metadata = normalized_event.get('metadata', {})

        # Add processing metadata
        metadata.update({
            'processed_by': 'iris',
            'processed_at': time.time(),
            'ingestion_source': normalized_event.get('source_adapter', 'unknown')
        })

        # Determine memory type (default to episodic for perception events)
        memory_type_str = normalized_event.get('memory_type', 'episodic')
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            logger.warning(
                f"Invalid memory type '{memory_type_str}', defaulting to episodic"
            )
            memory_type = MemoryType.EPISODIC

        envelope = StimulusEnvelope(
            stimulus_id=stimulus_id,
            content=content,
            timestamp=timestamp,
            source=source,
            modality=modality,
            confidence=confidence,
            privacy_classification=privacy_classification,
            session_id=session_id,
            correlation_id=correlation_id,
            metadata=metadata
        )

        return envelope

    def _process_event(self, raw_event: Dict[str, Any], source_adapter: SourceAdapter) -> ProcessingResult:
        """Process a single raw event through the IRIS pipeline.

        Args:
            raw_event: Raw event data from source
            source_adapter: Source adapter that provided the event

        Returns:
            ProcessingResult indicating outcome
        """
        try:
            # Update statistics
            with self._stats_lock:
                self.stats['events_received'] += 1

            # Check rate limiting
            if self.rate_limit_enabled and self.rate_limiter:
                if not self.rate_limiter.allow_event():
                    with self._stats_lock:
                        self.stats['events_rate_limited'] += 1
                    logger.debug(f"Event rate limited from {source_adapter.name}")
                    return ProcessingResult.RATE_LIMITED

            # Normalize the event
            try:
                normalized = source_adapter.normalize(raw_event)
                # Keep reference to raw event for privacy classification
                normalized['raw_event'] = raw_event
                normalized['source_adapter'] = source_adapter.name
            except Exception as e:
                logger.error(f"Failed to normalize event from {source_adapter.name}: {e}")
                with self._stats_lock:
                    self.stats['events_invalid'] += 1
                return ProcessingResult.INVALID

            # Check for duplicates
            if self.deduplication_enabled and self.deduplication_cache:
                event_id = self._generate_event_id(raw_event)
                if self.deduplication_cache.is_duplicate(event_id):
                    with self._stats_lock:
                        self.stats['events_duplicate'] += 1
                    logger.debug(f"Duplicate event filtered from {source_adapter.name}")
                    return ProcessingResult.DUPLICATE

            # Create StimulusEnvelope
            try:
                envelope = self._create_stimulus_envelope(normalized)
            except Exception as e:
                logger.error(f"Failed to create StimulusEnvelope: {e}")
                with self._stats_lock:
                    self.stats['events_invalid'] += 1
                return ProcessingResult.INVALID

            # Route event to handlers
            try:
                self._route_event(envelope)
                with self._stats_lock:
                    self.stats['events_accepted'] += 1
                logger.debug(f"Event accepted and routed from {source_adapter.name}")
                return ProcessingResult.ACCEPTED
            except Exception as e:
                logger.error(f"Failed to route event: {e}")
                with self._stats_lock:
                    self.stats['events_error'] += 1
                return ProcessingResult.ERROR

        except Exception as e:
            logger.error(f"Unexpected error processing event from {source_adapter.name}: {e}")
            with self._stats_lock:
                self.stats['events_error'] += 1
            return ProcessingResult.ERROR

    def _route_event(self, envelope: StimulusEnvelope) -> None:
        """Route a processed event to registered handlers.

        In a full implementation, this would send AgentEnvelopes to other agents.
        For now, we'll call registered handler functions.

        Args:
            envelope: Processed StimulusEnvelope to route
        """
        # Call all registered handlers
        for event_type, handler in self.event_handlers.items():
            try:
                handler(envelope)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")

        # If no specific handlers, log the event
        if not self.event_handlers:
            logger.info(
                f"IRIS processed event: {envelope.stimulus_id[:8]} "
                f"from {envelope.source} ({envelope.modality}) "
                f"with confidence {envelope.confidence:.2f}"
            )

    def poll_sources(self) -> Dict[str, int]:
        """Poll all source adapters for new events and process them.

        Returns:
            Dictionary mapping source adapter names to number of events processed
        """
        results = defaultdict(int)
        now = time.time()

        for adapter in self.source_adapters:
            if not adapter.enabled:
                continue

            # Check if it's time to poll this adapter
            if now - adapter.last_poll < adapter.poll_interval:
                continue

            try:
                # Poll for new events
                raw_events = adapter.poll()
                adapter.last_poll = now

                # Process each event
                for raw_event in raw_events:
                    result = self._process_event(raw_event, adapter)
                    results[adapter.name] += 1

                    # Log significant results
                    if result != ProcessingResult.ACCEPTED:
                        logger.debug(
                            f"Event from {adapter.name} resulted in {result.value}"
                        )

            except Exception as e:
                logger.error(f"Error polling source adapter {adapter.name}: {e}")

        return dict(results)

    def get_stats(self) -> Dict[str, Any]:
        """Get IRIS processing statistics.

        Returns:
            Dictionary of statistics
        """
        with self._stats_lock:
            stats_copy = self.stats.copy()
            stats_copy['uptime_seconds'] = time.time() - stats_copy['last_reset']
            return stats_copy

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        with self._stats_lock:
            self.stats = {
                'events_received': 0,
                'events_accepted': 0,
                'events_duplicate': 0,
                'events_rate_limited': 0,
                'events_invalid': 0,
                'events_error': 0,
                'last_reset': time.time()
            }
        logger.info("IRIS statistics reset")


# Dummy source adapter for testing and demonstration
class DummySourceAdapter(SourceAdapter):
    """A dummy source adapter that generates test events."""

    def __init__(
        self,
        name: str = "dummy",
        event_interval: float = 2.0,
        max_events: Optional[int] = None
    ):
        """Initialize dummy source adapter.

        Args:
            name: Name of the adapter
            event_interval: Seconds between generated events
            max_events: Maximum number of events to generate (None for unlimited)
        """
        super().__init__(name=name)
        # Set poll_interval to match event_interval for responsive testing
        self.poll_interval = event_interval
        self.event_interval = event_interval
        self.max_events = max_events
        self._events_generated = 0
        self._last_event_time = 0.0
        self._event_templates = [
            {
                'source': 'dummy_sensor_1',
                'modality': 'temperature',
                'content': {
                    'temperature_celsius': 22.5,
                    'humidity_percent': 65,
                    'location': 'living_room'
                },
                'confidence': 0.95,
                'metadata': {
                    'unit': 'celsius',
                    'sensor_id': 'temp_001'
                }
            },
            {
                'source': 'user_input',
                'modality': 'text',
                'content': 'Hello, how can I help you today?',
                'confidence': 0.9,
                'metadata': {
                    'input_method': 'keyboard',
                    'language': 'en'
                }
            },
            {
                'source': 'calendar_api',
                'modality': 'calendar',
                'content': {
                    'event': 'Team Meeting',
                    'start_time': '2026-08-18T10:00:00Z',
                    'end_time': '2026-08-18T11:00:00Z',
                    'attendees': ['alice@example.com', 'bob@example.com']
                },
                'confidence': 0.99,
                'metadata': {
                    'calendar_id': 'primary',
                    'event_id': 'meeting_123'
                },
                'session_id': 'work_day_2026_08_18',
                'correlation_id': 'calendar_sync_001'
            }
        ]

    def poll(self) -> List[Dict[str, Any]]:
        """Poll for new dummy events.

        Returns:
            List of generated event dictionaries
        """
        now = time.time()

        # Check if we should generate a new event
        if now - self._last_event_time < self.event_interval:
            return []

        # Check if we've reached the maximum
        if self.max_events is not None and self._events_generated >= self.max_events:
            return []

        # Generate event
        template = self._event_templates[
            self._events_generated % len(self._event_templates)
        ]

        # Add unique identifiers
        event = template.copy()
        # Handle content copying safely for different types
        if hasattr(event['content'], 'copy') and callable(getattr(event['content'], 'copy')):
            event['content'] = event['content'].copy()
        # For other types (strings, numbers, etc.), we keep the reference as it's just test data
        event['generated_at'] = now
        event['sequence_id'] = self._events_generated

        self._events_generated += 1
        self._last_event_time = now

        logger.debug(f"Generated dummy event #{self._events_generated}")
        return [event]

    def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a dummy event.

        For the dummy adapter, normalization is mostly a pass-through
        with some field mapping.

        Args:
            raw_event: Raw event data from the dummy source

        Returns:
            Normalized event dictionary
        """
        # Extract relevant fields
        normalized = {
            'source': raw_event.get('source', 'dummy'),
            'modality': raw_event.get('modality', 'unknown'),
            'content': raw_event.get('content', {}),
            'confidence': raw_event.get('confidence', 1.0),
            'metadata': raw_event.get('metadata', {}),
        }

        # Copy optional fields if present
        if 'session_id' in raw_event:
            normalized['session_id'] = raw_event['session_id']
        if 'correlation_id' in raw_event:
            normalized['correlation_id'] = raw_event['correlation_id']
        if 'timestamp' in raw_event:
            normalized['timestamp'] = raw_event['timestamp']

        return normalized


# Example usage and testing
if __name__ == "__main__":
    # Create IRIS agent
    iris = IrisAgent(
        deduplication_enabled=True,
        rate_limit_enabled=True,
        max_events_per_second=5.0,  # Allow 5 events per second
        burst_capacity=10.0
    )

    # Add dummy source adapter
    dummy_source = DummySourceAdapter(name="test_dummy", event_interval=1.0)
    iris.add_source_adapter(dummy_source)

    # Define a simple event handler
    def event_handler(envelope: StimulusEnvelope) -> None:
        """Handle processed events."""
        print(f"[HANDLER] Received event: {envelope.stimulus_id[:8]} "
              f"from {envelope.source} "
              f"modality={envelope.modality} "
              f"confidence={envelope.confidence:.2f} "
              f"privacy={envelope.privacy_classification.value}")

    # Register the handler
    iris.register_event_handler("*", event_handler)

    print("IRIS agent started. Processing events from dummy source...")
    print("Press Ctrl+C to stop.")

    try:
        # Main processing loop
        while True:
            # Poll sources for new events
            results = iris.poll_sources()

            # Print statistics every 10 seconds
            if int(time.time()) % 10 == 0:
                stats = iris.get_stats()
                if stats['events_received'] > 0:
                    print(f"\n[STATS] Received: {stats['events_received']}, "
                          f"Accepted: {stats['events_accepted']}, "
                          f"Duplicate: {stats['events_duplicate']}, "
                          f"Rate Limited: {stats['events_rate_limited']}, "
                          f"Invalid: {stats['events_invalid']}, "
                          f"Error: {stats['events_error']}")

            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nShutting down IRIS agent...")
        final_stats = iris.get_stats()
        print(f"Final stats: {final_stats}")
        print("IRIS agent stopped.")