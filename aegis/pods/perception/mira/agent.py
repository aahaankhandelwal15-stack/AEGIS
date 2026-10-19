"""MIRA - Market and Financial Perception Agent for AEGIS Perception Pod.

MIRA (Market Intelligence and Recognition Agent) specializes in ingesting
market, financial, and macroeconomic data, normalizing heterogeneous market
information into consistent AEGIS representation, preserving provenance,
detecting significant market movements using configurable thresholds, and
emitting structured signals for downstream systems.
"""

import time
import uuid
import hashlib
from collections import defaultdict
from typing import Dict, Any, Optional, List, Callable, Tuple
from threading import Lock, RLock
from dataclasses import dataclass, field
from enum import Enum
import logging

from aegis.core.models import (
    StimulusEnvelope,
    PrivacyClassification,
    AgentResult
)
from aegis.core.protocol import PERCEPTION_POD_ID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingResult(Enum):
    """Result of processing a market event."""
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    RATE_LIMITED = "rate_limited"
    INVALID = "invalid"
    ERROR = "error"


@dataclass
class MarketSourceAdapter:
    """Base class for market data source adapters that ingest events into MIRA."""
    name: str
    enabled: bool = True
    last_poll: float = field(default_factory=time.time)
    poll_interval: float = 5.0  # seconds

    def poll(self) -> List[Dict[str, Any]]:
        """Poll for new market events from the source.

        Returns:
            List of raw market event dictionaries to be processed by MIRA.
        """
        raise NotImplementedError("Market source adapters must implement poll() method")

    def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw market event into a format suitable for StimulusEnvelope creation.

        Args:
            raw_event: Raw market event data from the source

        Returns:
            Normalized event dictionary with keys matching StimulusEnvelope fields
            plus any market-specific fields needed for processing
        """
        raise NotImplementedError("Market source adapters must implement normalize() method")


@dataclass
class DeduplicationCache:
    """Tracks recently seen market events for deduplication."""
    max_size: int = 1000
    ttl_seconds: float = 300.0  # 5 minutes
    _cache: Dict[str, float] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def is_duplicate(self, event_id: str) -> bool:
        """Check if a market event has been seen recently.

        Args:
            event_id: Unique identifier for the market event

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


class MiraAgent:
    """MIRA agent: Market and financial perception agent."""

    def __init__(
        self,
        market_source_adapters: Optional[List[MarketSourceAdapter]] = None,
        deduplication_enabled: bool = True,
        rate_limit_enabled: bool = True,
        max_events_per_second: float = 10.0,
        burst_capacity: float = 20.0,
        deduplication_ttl: float = 300.0,
        deduplication_max_size: int = 1000,
        anomaly_threshold: float = 0.05  # 5% default threshold for price movements
    ):
        """Initialize MIRA agent.

        Args:
            market_source_adapters: List of market source adapters to ingest events from
            deduplication_enabled: Whether to enable deduplication
            rate_limit_enabled: Whether to enable rate limiting
            max_events_per_second: Maximum events per second allowed
            burst_capacity: Burst capacity for rate limiter
            deduplication_ttl: Time to live for deduplication cache (seconds)
            deduplication_max_size: Maximum size of deduplication cache
            anomaly_threshold: Threshold for detecting significant market movements (as decimal, e.g., 0.05 = 5%)
        """
        self.market_source_adapters = market_source_adapters or []
        self.deduplication_enabled = deduplication_enabled
        self.rate_limit_enabled = rate_limit_enabled
        self.anomaly_threshold = anomaly_threshold

        # Initialize components
        self.deduplication_cache = DeduplicationCache(
            max_size=deduplication_max_size,
            ttl_seconds=deduplication_ttl
        ) if deduplication_enabled else None

        self.rate_limiter = RateLimiter(
            max_events_per_second=max_events_per_second,
            burst_capacity=burst_capacity
        ) if rate_limit_enabled else None

        # Event routing - handlers for processed market signals
        self.signal_handlers: Dict[str, Callable[[StimulusEnvelope], None]] = {}

        # Store previous prices for anomaly detection (symbol -> last_price)
        self._previous_prices: Dict[str, float] = {}
        self._prices_lock = Lock()

        # Statistics
        self.stats = {
            'events_received': 0,
            'signals_accepted': 0,
            'signals_duplicate': 0,
            'signals_rate_limited': 0,
            'signals_invalid': 0,
            'signals_error': 0,
            'anomalies_detected': 0,
            'normal_updates': 0,
            'last_reset': time.time()
        }
        self._stats_lock = Lock()

        logger.info(f"MIRA agent initialized with anomaly threshold: {anomaly_threshold}")

    def set_anomaly_threshold(self, threshold: float) -> None:
        """Set the anomaly detection threshold.

        Args:
            threshold: Threshold for detecting significant market movements (as decimal)
        """
        self.anomaly_threshold = threshold
        logger.info(f"MIRA anomaly threshold updated to: {threshold}")

    def add_market_source_adapter(self, adapter: MarketSourceAdapter) -> None:
        """Add a market source adapter to MIRA.

        Args:
            adapter: Market source adapter to add
        """
        self.market_source_adapters.append(adapter)
        logger.info(f"Added market source adapter: {adapter.name}")

    def remove_market_source_adapter(self, adapter_name: str) -> bool:
        """Remove a market source adapter by name.

        Args:
            adapter_name: Name of adapter to remove

        Returns:
            True if adapter was removed, False if not found
        """
        for i, adapter in enumerate(self.market_source_adapters):
            if adapter.name == adapter_name:
                del self.market_source_adapters[i]
                logger.info(f"Removed market source adapter: {adapter_name}")
                return True
        return False

    def register_signal_handler(self, signal_type: str, handler: Callable[[StimulusEnvelope], None]) -> None:
        """Register a handler for processed market signals.

        Args:
            signal_type: Type of signal to handle (for filtering)
            handler: Function to call when signal is processed
        """
        self.signal_handlers[signal_type] = handler
        logger.info(f"Registered signal handler for type: {signal_type}")

    def _generate_event_id(self, raw_event: Dict[str, Any]) -> str:
        """Generate a unique identifier for deduplication.

        Args:
            raw_event: Raw market event data

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
        """Determine appropriate privacy classification for a market event.

        Args:
            raw_event: Raw market event data
            default_classification: Default classification to use

        Returns:
            PrivacyClassification for the event
        """
        # Market data is typically PUBLIC, but check for explicit classification
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

        return default_classification

    def _create_stimulus_envelope(
        self,
        normalized_event: Dict[str, Any],
        is_anomaly: bool = False
    ) -> StimulusEnvelope:
        """Create a StimulusEnvelope from a normalized market event.

        Args:
            normalized_event: Normalized market event data
            is_anomaly: Whether this event represents a significant market movement

        Returns:
            StimulusEnvelope object
        """
        # Extract or generate required fields
        stimulus_id = normalized_event.get('stimulus_id', str(uuid.uuid4()))
        timestamp = normalized_event.get('timestamp', time.time())
        source = normalized_event.get('source', 'unknown')
        modality = normalized_event.get('modality', 'market.data')

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

        # Add market-specific metadata
        metadata = normalized_event.get('metadata', {})
        metadata.update({
            'processed_by': 'mira',
            'processed_at': time.time(),
            'ingestion_source': normalized_event.get('source_adapter', 'unknown'),
            'is_anomaly': is_anomaly,
            'anomaly_threshold': self.anomaly_threshold
        })

        # Extract optional fields
        session_id = normalized_event.get('session_id')
        correlation_id = normalized_event.get('correlation_id')

        # Determine memory type (default to episodic for perception events)
        memory_type_str = normalized_event.get('memory_type', 'episodic')
        # Note: We don't actually use MemoryType in StimulusEnvelope v1, but keep for compatibility

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

    def _detect_anomaly(self, symbol: str, current_price: float) -> bool:
        """Detect if a price movement represents a significant anomaly.

        Args:
            symbol: Market symbol/instrument identifier
            current_price: Current price value

        Returns:
            True if price movement exceeds anomaly threshold, False otherwise
        """
        with self._prices_lock:
            last_price = self._previous_prices.get(symbol)

            # If we don't have a previous price, this is the first observation
            if last_price is None:
                self._previous_prices[symbol] = current_price
                return False

            # Calculate price change percentage
            if last_price == 0:
                # Avoid division by zero
                price_change = 0.0 if current_price == 0 else float('inf')
            else:
                price_change = abs((current_price - last_price) / last_price)

            # Update the stored price
            self._previous_prices[symbol] = current_price

            # Check if change exceeds threshold
            is_anomaly = price_change >= self.anomaly_threshold

            if is_anomaly:
                logger.debug(
                    f"Anomaly detected for {symbol}: "
                    f"price change {price_change:.4f} >= threshold {self.anomaly_threshold}"
                )

            return is_anomaly

    def _process_market_event(
        self,
        raw_event: Dict[str, Any],
        source_adapter: MarketSourceAdapter
    ) -> Tuple[ProcessingResult, Optional[StimulusEnvelope]]:
        """Process a single raw market event through the MIRA pipeline.

        Args:
            raw_event: Raw market event data from source
            source_adapter: Market source adapter that provided the event

        Returns:
            Tuple of (ProcessingResult, StimulusEnvelope if accepted)
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
                    logger.debug(f"Market event rate limited from {source_adapter.name}")
                    return ProcessingResult.RATE_LIMITED, None

            # Normalize the event
            try:
                normalized = source_adapter.normalize(raw_event)
                # Keep reference to raw event for privacy classification
                normalized['raw_event'] = raw_event
                normalized['source_adapter'] = source_adapter.name
            except Exception as e:
                logger.error(f"Failed to normalize market event from {source_adapter.name}: {e}")
                with self._stats_lock:
                    self.stats['events_invalid'] += 1
                return ProcessingResult.INVALID, None

            # Check for duplicates
            if self.deduplication_enabled and self.deduplication_cache:
                event_id = self._generate_event_id(raw_event)
                if self.deduplication_cache.is_duplicate(event_id):
                    with self._stats_lock:
                        self.stats['events_duplicate'] += 1
                    logger.debug(f"Duplicate market event filtered from {source_adapter.name}")
                    return ProcessingResult.DUPLICATE, None

            # Extract symbol and price for anomaly detection (if available)
            symbol = normalized.get('symbol') or normalized.get('ticker') or normalized.get('instrument')
            price_fields = ['price', 'last_price', 'close', 'open', 'value']
            price = None
            for field in price_fields:
                if field in normalized:
                    try:
                        price = float(normalized[field])
                        break
                    except (ValueError, TypeError):
                        continue

            # Detect anomaly if we have symbol and price data
            is_anomaly = False
            if symbol and price is not None:
                is_anomaly = self._detect_anomaly(str(symbol), price)
                if is_anomaly:
                    with self._stats_lock:
                        self.stats['anomalies_detected'] += 1
                else:
                    with self._stats_lock:
                        self.stats['normal_updates'] += 1

            # Create StimulusEnvelope
            try:
                envelope = self._create_stimulus_envelope(normalized, is_anomaly=is_anomaly)
            except Exception as e:
                logger.error(f"Failed to create StimulusEnvelope: {e}")
                with self._stats_lock:
                    self.stats['events_invalid'] += 1
                return ProcessingResult.INVALID, None

            # Route signal to handlers
            try:
                self._route_signal(envelope)
                with self._stats_lock:
                    self.stats['signals_accepted'] += 1
                logger.debug(f"Market signal accepted and routed from {source_adapter.name}")
                return ProcessingResult.ACCEPTED, envelope
            except Exception as e:
                logger.error(f"Failed to route market signal: {e}")
                with self._stats_lock:
                    self.stats['signals_error'] += 1
                return ProcessingResult.ERROR, None

        except Exception as e:
            logger.error(f"Unexpected error processing market event from {source_adapter.name}: {e}")
            with self._stats_lock:
                self.stats['signals_error'] += 1
            return ProcessingResult.ERROR, None

    def _route_signal(self, envelope: StimulusEnvelope) -> None:
        """Route a processed market signal to registered handlers.

        Args:
            envelope: Processed StimulusEnvelope to route
        """
        # Call all registered handlers
        for signal_type, handler in self.signal_handlers.items():
            try:
                handler(envelope)
            except Exception as e:
                logger.error(f"Error in signal handler for {signal_type}: {e}")

        # If no specific handlers, log the signal
        if not self.signal_handlers:
            anomaly_status = "ANOMALY" if envelope.metadata.get('is_anomaly', False) else "UPDATE"
            logger.info(
                f"MIRA processed {anomaly_status} signal: {envelope.stimulus_id[:8]} "
                f"from {envelope.source} ({envelope.modality}) "
                f"with confidence {envelope.confidence:.2f}"
            )

    def poll_sources(self) -> Dict[str, int]:
        """Poll all market source adapters for new events and process them.

        Returns:
            Dictionary mapping source adapter names to number of signals processed
        """
        results = defaultdict(int)
        now = time.time()

        for adapter in self.market_source_adapters:
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
                    result, envelope = self._process_market_event(raw_event, adapter)
                    results[adapter.name] += 1

                    # Log significant results
                    if result != ProcessingResult.ACCEPTED:
                        logger.debug(
                            f"Market event from {adapter.name} resulted in {result.value}"
                        )

            except Exception as e:
                logger.error(f"Error polling market source adapter {adapter.name}: {e}")

        return dict(results)

    def get_stats(self) -> Dict[str, Any]:
        """Get MIRA processing statistics.

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
                'signals_accepted': 0,
                'signals_duplicate': 0,
                'signals_rate_limited': 0,
                'signals_invalid': 0,
                'signals_error': 0,
                'anomalies_detected': 0,
                'normal_updates': 0,
                'last_reset': time.time()
            }
            # Also reset price history when resetting stats
            with self._prices_lock:
                self._previous_prices.clear()
        logger.info("MIRA statistics and price history reset")


# Example market source adapter for testing
class DemoMarketSourceAdapter(MarketSourceAdapter):
    """A demo market source adapter that generates test market data."""

    def __init__(
        self,
        name: str = "demo_market",
        event_interval: float = 2.0,
        max_events: Optional[int] = None
    ):
        """Initialize demo market source adapter.

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
        self._base_price = 100.0  # Starting price for demo symbol

        # Define different market event templates
        self._event_templates = [
            {
                'source': 'nasdaq_feed',
                'modality': 'equity.price',
                'symbol': 'AAPL',
                'price': 150.0,
                'volume': 1000000,
                'timestamp': time.time(),
                'confidence': 0.95,
                'metadata': {
                    'exchange': 'NASDAQ',
                    'currency': 'USD'
                }
            },
            {
                'source': 'forex_feed',
                'modality': 'forex.pair',
                'symbol': 'EUR/USD',
                'price': 1.0850,
                'volume': 500000,
                'timestamp': time.time(),
                'confidence': 0.9,
                'metadata': {
                    'exchange': 'FOREX',
                    'base': 'EUR',
                    'quote': 'USD'
                }
            },
            {
                'source': 'crypto_feed',
                'modality': 'crypto.price',
                'symbol': 'BTC',
                'price': 42000.0,
                'volume': 25000,
                'timestamp': time.time(),
                'confidence': 0.85,
                'metadata': {
                    'exchange': 'CRYPTO',
                    'currency': 'USD'
                }
            }
        ]

    def poll(self) -> List[Dict[str, Any]]:
        """Poll for new demo market events.

        Returns:
            List of raw market event dictionaries
        """
        # Check if we've reached max events
        if self.max_events is not None and self._events_generated >= self.max_events:
            return []

        # Generate events based on interval
        now = time.time()
        if now - self._last_event_time < self.event_interval:
            return []

        self._last_event_time = now
        self._events_generated += 1

        # Select a template and vary the price slightly for demo
        import random
        template = self._event_templates[self._events_generated % len(self._event_templates)].copy()

        # Add some random walk to prices for demo purposes
        if 'price' in template:
            # Random walk: +/- 2% max change
            change_factor = 1.0 + (random.random() - 0.5) * 0.04  # -0.02 to +0.02
            template['price'] = template['price'] * change_factor

        # Update timestamp
        template['timestamp'] = now

        return [template]

    def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw market event into a format suitable for StimulusEnvelope creation.

        For the demo adapter, the raw event is already in a suitable format.
        We simply return it as-is, ensuring it has the required fields.

        Args:
            raw_event: Raw market event data from the source

        Returns:
            Normalized event dictionary with keys matching StimulusEnvelope fields
        """
        # Ensure required fields are present; if missing, provide defaults
        normalized = raw_event.copy()
        # Ensure we have a source
        if 'source' not in normalized:
            normalized['source'] = self.name
        # Ensure we have a modality
        if 'modality' not in normalized:
            normalized['modality'] = 'market.data'
        # Ensure we have content (if not present, use the whole event)
        if 'content' not in normalized:
            normalized['content'] = normalized
        # Ensure we have confidence
        if 'confidence' not in normalized:
            normalized['confidence'] = 0.9
        # Ensure we have timestamp
        if 'timestamp' not in normalized:
            normalized['timestamp'] = time.time()
        # Ensure we have metadata
        if 'metadata' not in normalized:
            normalized['metadata'] = {}
        return normalized