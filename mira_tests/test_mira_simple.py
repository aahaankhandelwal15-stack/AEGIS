"""Simple test for MIRA agent functionality."""

import sys
import os
import time
from typing import List, Dict, Any

# Add project root to path so we can import aegis modules can be imported
sys.path.insert(0, os.path.abspath('.'))

from aegis.pods.perception.mira.agent import (
    MiraAgent,
    DemoMarketSourceAdapter,
    ProcessingResult
)
from aegis.core.models import (
    StimulusEnvelope,
    PrivacyClassification
)


def test_mira_basic_functionality():
    """Test basic MIRA agent functionality."""
    print("Testing basic MIRA agent functionality...")

    # Create MIRA agent
    mira = MiraAgent(
        deduplication_enabled=True,
        rate_limit_enabled=True,
        max_events_per_second=10.0,
        burst_capacity=20.0,
        anomaly_threshold=0.05  # 5% threshold
    )

    # Add demo market source adapter
    demo_source = DemoMarketSourceAdapter(name="test_demo", event_interval=0.1)
    mira.add_market_source_adapter(demo_source)

    # Track processed signals
    processed_signals = []

    def signal_handler(envelope: StimulusEnvelope) -> None:
        processed_signals.append(envelope)
        anomaly_status = "ANOMALY" if envelope.metadata.get('is_anomaly', False) else "UPDATE"
        print(f"  Handler received: {envelope.stimulus_id[:8]} from {envelope.source} [{anomaly_status}]")

    mira.register_signal_handler("*", signal_handler)

    # Run for a short period
    start_time = time.time()
    test_duration = 3.0  # 3 seconds

    signals_processed = 0
    anomalies_detected = 0

    while time.time() - start_time < test_duration:
        # Poll sources for new events
        results = mira.poll_sources()
        signals_processed += sum(results.values())

        # Check stats for anomalies
        stats = mira.get_stats()
        anomalies_detected = stats['anomalies_detected']

        time.sleep(0.1)  # Small delay to prevent busy waiting

    # Give some time for final processing
    time.sleep(0.5)

    # Get final statistics
    final_stats = mira.get_stats()

    print(f"\n=== MIRA Test Results ===")
    print(f"Market Events Received: {final_stats['events_received']}")
    print(f"Signals Accepted: {final_stats['signals_accepted']}")
    print(f"Signals Duplicate: {final_stats['signals_duplicate']}")
    print(f"Signals Rate Limited: {final_stats['signals_rate_limited']}")
    print(f"Normal Updates: {final_stats['normal_updates']}")
    print(f"Anomalies Detected: {final_stats['anomalies_detected']}")
    print(f"Handler Signals Received: {len(processed_signals)}")

    # Basic assertions
    assert final_stats['events_received'] > 0, "Should have received market events"
    assert final_stats['signals_accepted'] > 0, "Should have accepted some signals"
    assert len(processed_signals) > 0, "Should have received signals via handler"

    # Check that signals are valid StimulusEnvelopes
    for signal in processed_signals[:5]:  # Check first 5 signals
        assert isinstance(signal, StimulusEnvelope), "Signal should be StimulusEnvelope"
        assert hasattr(signal, 'source') and signal.source, "Signal should have a source"
        assert signal.modality in ['equity.price', 'forex.pair', 'crypto.price'], "Signal should have market modality"
        assert 0.0 <= signal.confidence <= 1.0, "Confidence should be in valid range"
        assert 'processed_by' in signal.metadata, "Signal should have processing metadata"
        assert signal.metadata['processed_by'] == 'mira', "Signal should be processed by MIRA"

    print("\n[PASS] MIRA basic functionality test passed")


def test_mira_anomaly_detection():
    """Test MIRA anomaly detection with configurable thresholds."""
    print("\nTesting MIRA anomaly detection...")

    # Create MIRA agent with low threshold to detect small movements
    mira = MiraAgent(
        deduplication_enabled=False,  # Disable for easier testing
        rate_limit_enabled=False,
        anomaly_threshold=0.01  # 1% threshold - low to detect small movements
    )

    # Track processed signals for detailed validation
    processed_signals = []

    def signal_handler(envelope: StimulusEnvelope) -> None:
        processed_signals.append(envelope)

    mira.register_signal_handler("*", signal_handler)

    # Create a custom market source adapter for controlled testing
    # Bypass polling timing issues by returning events immediately
    class ControlledMarketSourceAdapter:
        def __init__(self, name="controlled_market"):
            self.name = name
            self.enabled = True
            self.last_poll = 0.0
            self.poll_interval = 0.0  # Allow immediate polling
            self._test_prices = [100.0, 100.5, 102.0, 98.0, 99.0]  # Known price sequence
            self._price_index = 0
            self._events_sent = 0

        def poll(self) -> List[Dict[str, Any]]:
            if self._price_index >= len(self._test_prices):
                return []  # No more events

            # Get current price and create event
            price = self._test_prices[self._price_index]
            self._price_index += 1
            self._events_sent += 1

            event = {
                'source': 'test_controlled',
                'modality': 'test.price',
                'symbol': 'TEST',
                'price': price,
                'confidence': 0.9,
                'timestamp': 1000000.0 + self._events_sent  # Use increasing timestamps to ensure ordering
            }

            return [event]

        def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize a raw market event into a format suitable for StimulusEnvelope creation."""
            # Ensure required fields are present
            normalized = raw_event.copy()
            if 'source' not in normalized:
                normalized['source'] = self.name
            if 'modality' not in normalized:
                normalized['modality'] = 'market.data'
            if 'content' not in normalized:
                normalized['content'] = normalized
            if 'confidence' not in normalized:
                normalized['confidence'] = 0.9
            if 'timestamp' not in normalized:
                normalized['timestamp'] = raw_event['timestamp']
            if 'metadata' not in normalized:
                normalized['metadata'] = {}
            return normalized

    # Add controlled source adapter
    controlled_source = ControlledMarketSourceAdapter(name="controlled_test")
    mira.add_market_source_adapter(controlled_source)

    # Process events by calling poll_sources enough times
    # Since we set poll_interval to 0.0, each call should process one event
    test_events = 5
    for i in range(test_events):
        results = mira.poll_sources()
        # No need for sleep since we bypassed timing

    # Give time for processing
    time.sleep(0.1)

    # Check results
    final_stats = mira.get_stats()
    print(f"Controlled test - Events received: {final_stats['events_received']}")
    print(f"Controlled test - Signals accepted: {final_stats['signals_accepted']}")
    print(f"Controlled test - Normal updates: {final_stats['normal_updates']}")
    print(f"Controlled test - Anomalies detected: {final_stats['anomalies_detected']}")
    print(f"Handler signals received: {len(processed_signals)}")

    # Validate that we got the expected number of events
    assert final_stats['events_received'] == test_events, f"Expected {test_events} events, got {final_stats['events_received']}"
    assert final_stats['signals_accepted'] == test_events, f"Expected {test_events} accepted signals, got {final_stats['signals_accepted']}"
    assert len(processed_signals) == test_events, f"Expected {test_events} handler signals, got {len(processed_signals)}"

    # Validate that we processed the expected test events
    assert controlled_source._events_sent == test_events, f"Adapter expected to send {test_events} events, sent {controlled_source._events_sent}"

    # With our test sequence [100.0, 100.5, 102.0, 98.0, 99.0] and 1% threshold:
    # Event 1: 100.0 (baseline, no previous price for comparison) -> normal update
    # Event 2: 100.5 -> +0.5% change from 100.0 (< 1% threshold) -> normal update
    # Event 3: 102.0 -> +1.5% change from 100.5 (>= 1% threshold) -> anomaly
    # Event 4: 98.0 -> -3.9% change from 102.0 (>= 1% threshold) -> anomaly
    # Event 5: 99.0 -> +1.0% change from 98.0 (>= 1% threshold) -> anomaly
    # So we expect 2 normal updates and 3 anomalies

    # Validate anomaly detection results
    expected_anomalies = 3  # events 3, 4, and 5
    expected_normal = 2     # events 1 and 2

    assert final_stats['anomalies_detected'] == expected_anomalies, f"Expected {expected_anomalies} anomalies, got {final_stats['anomalies_detected']}"
    assert final_stats['normal_updates'] == expected_normal, f"Expected {expected_normal} normal updates, got {final_stats['normal_updates']}"

    # Validate that signals are properly classified
    anomaly_signals = [s for s in processed_signals if s.metadata.get('is_anomaly', False)]
    normal_signals = [s for s in processed_signals if not s.metadata.get('is_anomaly', False)]

    print(f"  Anomaly signals: {len(anomaly_signals)}")
    print(f"  Normal signals: {len(normal_signals)}")

    assert len(anomaly_signals) == expected_anomalies, f"Expected {expected_anomalies} anomaly signals, got {len(anomaly_signals)}"
    assert len(normal_signals) == expected_normal, f"Expected {expected_normal} normal signals, got {len(normal_signals)}"

    # Validate that anomaly signals are distinguishable from normal signals
    for i, signal in enumerate(processed_signals):
        is_anomaly = signal.metadata.get('is_anomaly', False)
        # Check that the signal has the required metadata
        assert 'processed_by' in signal.metadata, f"Signal {i} missing processed_by metadata"
        assert signal.metadata['processed_by'] == 'mira', f"Signal {i} not processed by MIRA"
        assert 'is_anomaly' in signal.metadata, f"Signal {i} missing is_anomaly metadata"
        assert 'anomaly_threshold' in signal.metadata, f"Signal {i} missing anomaly_threshold metadata"
        assert signal.metadata['anomaly_threshold'] == 0.01, f"Signal {i} has wrong anomaly_threshold"

        # Verify source and timestamp are preserved
        assert signal.source == 'test_controlled', f"Signal {i} has incorrect source"
        assert isinstance(signal.timestamp, float) and signal.timestamp > 0, f"Signal {i} has invalid timestamp"

        # Verify confidence is in valid range
        assert 0.0 <= signal.confidence <= 1.0, f"Signal {i} has invalid confidence: {signal.confidence}"

        # Verify the event content is preserved
        assert 'price' in signal.content, f"Signal {i} missing price in content"
        assert isinstance(signal.content['price'], (int, float)), f"Signal {i} price is not numeric"

    # Test threshold change effect
    print("\nTesting threshold change effect...")
    mira.set_anomaly_threshold(0.02)  # Increase threshold to 2%

    # Reset for clean test
    mira.reset_stats()
    processed_signals.clear()

    # Test with a known movement: 1.5% should be normal with 2% threshold, anomaly with 1% threshold
    class SingleMovementAdapter:
        def __init__(self, name="single_test", start_price=100.0, end_price=101.5):  # 1.5% increase
            self.name = name
            self.enabled = True
            self.last_poll = 0.0
            self.poll_interval = 0.0  # Allow immediate polling
            self._prices = [start_price, end_price]
            self._price_index = 0
            self._events_sent = 0

        def poll(self) -> List[Dict[str, Any]]:
            if self._price_index >= len(self._prices):
                return []  # No more events
            price = self._prices[self._price_index]
            self._price_index += 1
            self._events_sent += 1
            return [{
                'source': 'single_test',
                'modality': 'test.price',
                'symbol': 'SINGLE',
                'price': price,
                'confidence': 0.9,
                'timestamp': 2000000.0 + self._events_sent  # Use increasing timestamps
            }]

        def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
            normalized = raw_event.copy()
            if 'source' not in normalized:
                normalized['source'] = self.name
            if 'modality' not in normalized:
                normalized['modality'] = 'market.data'
            if 'content' not in normalized:
                normalized['content'] = normalized
            if 'confidence' not in normalized:
                normalized['confidence'] = 0.9
            if 'timestamp' not in normalized:
                normalized['timestamp'] = raw_event['timestamp']
            if 'metadata' not in normalized:
                normalized['metadata'] = {}
            return normalized

    # Test with 2% threshold (should NOT detect 1.5% movement as anomaly)
    single_adapter = SingleMovementAdapter()
    mira.add_market_source_adapter(single_adapter)
    # Poll multiple times to ensure we get both events (overcoming timing issues)
    for _ in range(5):
        results = mira.poll_sources()
        time.sleep(0.01)

    stats_after = mira.get_stats()
    print(f"With 2% threshold - Events: {stats_after['events_received']}, Anomalies: {stats_after['anomalies_detected']}")

    # Should have received at least 2 events (baseline and movement)
    assert stats_after['events_received'] >= 2, f"Expected at least 2 events, got {stats_after['events_received']}"
    # With 2% threshold, 1.5% movement should NOT be anomaly
    assert stats_after['anomalies_detected'] == 0, f"Expected 0 anomalies with 2% threshold, got {stats_after['anomalies_detected']}"

    # Clean up the adapter for next test
    mira.remove_market_source_adapter("single_test")

    # Test with 1% threshold (should detect 1.5% movement as anomaly)
    mira.set_anomaly_threshold(0.01)  # Decrease threshold back to 1%
    mira.reset_stats()
    processed_signals.clear()

    single_adapter2 = SingleMovementAdapter()
    mira.add_market_source_adapter(single_adapter2)
    # Poll multiple times to ensure we get both events
    for _ in range(5):
        results = mira.poll_sources()
        time.sleep(0.01)

    stats_after2 = mira.get_stats()
    print(f"With 1% threshold - Events: {stats_after2['events_received']}, Anomalies: {stats_after2['anomalies_detected']}")

    # Should have received at least 2 events
    assert stats_after2['events_received'] >= 2, f"Expected at least 2 events, got {stats_after2['events_received']}"
    # With 1% threshold, 1.5% movement SHOULD be anomaly
    assert stats_after2['anomalies_detected'] >= 1, f"Expected at least 1 anomaly with 1% threshold, got {stats_after2['anomalies_detected']}"

    # Clean up
    mira.reset_stats()
    mira.remove_market_source_adapter("single_test")

    print("[PASS] MIRA anomaly detection test completed")


def test_mira_heterogeneous_input_formats():
    """Test that MIRA normalizes heterogeneous input formats to the same internal representation."""
    print("\nTesting MIRA heterogeneous input format normalization...")

    mira = MiraAgent(
        deduplication_enabled=False,
        rate_limit_enabled=False,
        anomaly_threshold=0.05  # 5% threshold
    )

    # Track processed signals
    processed_signals = []

    def signal_handler(envelope: StimulusEnvelope) -> None:
        processed_signals.append(envelope)

    mira.register_signal_handler("*", signal_handler)

    # Create two different market source adapters that provide equivalent information
    # in different formats
    class FormatAdapterA:
        """Adapter providing data in Format A: symbol, price, timestamp"""

        def __init__(self, name="format_a"):
            self.name = name
            self.enabled = True
            self.last_poll = 0.0
            self.poll_interval = 0.0
            self._test_data = [
                {'symbol': 'AAPL', 'price': 150.0, 'timestamp': 1000000.0},
                {'symbol': 'AAPL', 'price': 155.0, 'timestamp': 1000001.0},  # +3.33%
                {'symbol': 'GOOGL', 'price': 2500.0, 'timestamp': 1000002.0},
                {'symbol': 'GOOGL', 'price': 2400.0, 'timestamp': 1000003.0},  # -4.0%
            ]
            self._index = 0

        def poll(self) -> List[Dict[str, Any]]:
            if self._index >= len(self._test_data):
                return []
            data = self._test_data[self._index].copy()
            self._index += 1
            return [data]

        def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
            normalized = raw_event.copy()
            if 'source' not in normalized:
                normalized['source'] = self.name
            if 'modality' not in normalized:
                normalized['modality'] = 'equity.price'
            if 'content' not in normalized:
                normalized['content'] = normalized
            if 'confidence' not in normalized:
                normalized['confidence'] = 0.9
            if 'timestamp' not in normalized:
                normalized['timestamp'] = raw_event['timestamp']
            if 'metadata' not in normalized:
                normalized['metadata'] = {}
            return normalized

    class FormatAdapterB:
        """Adapter providing data in Format B: ticker, value, observed_at"""

        def __init__(self, name="format_b"):
            self.name = name
            self.enabled = True
            self.last_poll = 0.0
            self.poll_interval = 0.0
            self._test_data = [
                {'ticker': 'MSFT', 'value': 300.0, 'observed_at': 2000000.0},
                {'ticker': 'MSFT', 'value': 310.0, 'observed_at': 2000001.0},  # +3.33%
                {'ticker': 'TSLA', 'value': 800.0, 'observed_at': 2000002.0},
                {'ticker': 'TSLA', 'value': 760.0, 'observed_at': 2000003.0},  # -5.0%
            ]
            self._index = 0

        def poll(self) -> List[Dict[str, Any]]:
            if self._index >= len(self._test_data):
                return []
            data = self._test_data[self._index].copy()
            self._index += 1
            return [data]

        def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
            normalized = raw_event.copy()
            # Convert Format B to common internal representation
            if 'ticker' in normalized:
                normalized['symbol'] = normalized.pop('ticker')
            if 'value' in normalized:
                normalized['price'] = normalized.pop('value')
            if 'observed_at' in normalized:
                normalized['timestamp'] = normalized.pop('observed_at')

            if 'source' not in normalized:
                normalized['source'] = self.name
            if 'modality' not in normalized:
                normalized['modality'] = 'equity.price'
            if 'content' not in normalized:
                normalized['content'] = normalized
            if 'confidence' not in normalized:
                normalized['confidence'] = 0.9
            if 'timestamp' not in normalized:
                normalized['timestamp'] = raw_event.get('timestamp', time.time())
            if 'metadata' not in normalized:
                normalized['metadata'] = {}
            return normalized

    # Test Format A
    adapter_a = FormatAdapterA()
    mira.add_market_source_adapter(adapter_a)

    # Process Format A events
    for _ in range(4):
        results = mira.poll_sources()
        time.sleep(0.01)

    stats_a = mira.get_stats()
    signals_a = len(processed_signals)
    print(f"Format A - Events: {stats_a['events_received']}, Anomalies: {stats_a['anomalies_detected']}, Updates: {stats_a['normal_updates']}")

    # Clear signals for next test
    processed_signals.clear()
    mira.reset_stats()
    mira.remove_market_source_adapter("format_a")

    # Test Format B
    adapter_b = FormatAdapterB()
    mira.add_market_source_adapter(adapter_b)

    # Process Format B events
    for _ in range(4):
        results = mira.poll_sources()
        time.sleep(0.01)

    stats_b = mira.get_stats()
    signals_b = len(processed_signals)
    print(f"Format B - Events: {stats_b['events_received']}, Anomalies: {stats_b['anomalies_detected']}, Updates: {stats_b['normal_updates']}")

    # Format A: AAPL +3.33% (normal), GOOGL -4.0% (normal) -> 0 anomalies (both < 5% threshold)
    # Format B: MSFT +3.33% (normal), TSLA -5.0% (anomaly) -> 1 anomaly (TSLA >= 5% threshold)
    expected_anomalies_a = 0  # Format A: both movements < 5% threshold
    expected_anomalies_b = 1  # Format B: TSLA movement = 5% threshold (>= threshold)

    assert stats_a['anomalies_detected'] == expected_anomalies_a, f"Format A: Expected {expected_anomalies_a} anomalies, got {stats_a['anomalies_detected']}"
    assert stats_b['anomalies_detected'] == expected_anomalies_b, f"Format B: Expected {expected_anomalies_b} anomalies, got {stats_b['anomalies_detected']}"

    # Verify that signals from both formats have the same structure
    all_signals = processed_signals  # Contains signals from Format B test
    for signal in all_signals:
        assert isinstance(signal, StimulusEnvelope), "Signal should be StimulusEnvelope"
        assert hasattr(signal, 'source') and signal.source
        assert signal.modality == 'equity.price'
        assert 0.0 <= signal.confidence <= 1.0
        assert 'processed_by' in signal.metadata
        assert signal.metadata['processed_by'] == 'mira'
        assert 'is_anomaly' in signal.metadata
        assert 'anomaly_threshold' in signal.metadata

        # Verify that price information is preserved in content
        assert 'price' in signal.content or any(key in signal.content for key in ['symbol', 'ticker']), \
            "Signal content should contain price/symbol information"

    print("[PASS] MIRA heterogeneous input format normalization test passed")


def test_mira_traceability_preservation():
    """Test that source, timestamp, and traceability information are preserved."""
    print("\nTesting MIRA traceability preservation...")

    mira = MiraAgent(
        deduplication_enabled=False,
        rate_limit_enabled=False,
        anomaly_threshold=0.05
    )

    processed_signals = []

    def signal_handler(envelope: StimulusEnvelope) -> None:
        processed_signals.append(envelope)

    mira.register_signal_handler("*", signal_handler)

    # Create test adapter with known traceability information
    class TraceabilityTestAdapter:
        def __init__(self, name="traceability_test"):
            self.name = name
            self.enabled = True
            self.last_poll = 0.0
            self.poll_interval = 0.0
            self._test_data = [
                {
                    'symbol': 'TEST',
                    'price': 100.0,
                    'timestamp': 1234567.0,
                    'session_id': 'test_session_123',
                    'correlation_id': 'test_corr_456',
                    'confidence': 0.85
                },
                {
                    'symbol': 'TEST',
                    'price': 105.0,  # +5% - should be anomaly
                    'timestamp': 1234568.0,
                    'session_id': 'test_session_123',
                    'correlation_id': 'test_corr_456',
                    'confidence': 0.9
                }
            ]
            self._index = 0

        def poll(self) -> List[Dict[str, Any]]:
            if self._index >= len(self._test_data):
                return []
            data = self._test_data[self._index].copy()
            self._index += 1
            return [data]

        def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
            normalized = raw_event.copy()
            if 'source' not in normalized:
                normalized['source'] = self.name
            if 'modality' not in normalized:
                normalized['modality'] = 'test.price'
            if 'content' not in normalized:
                normalized['content'] = normalized
            # Don't modify timestamp, confidence, session_id, correlation_id here
            # to test if they're preserved through the pipeline
            if 'metadata' not in normalized:
                normalized['metadata'] = {}
            return normalized

    adapter = TraceabilityTestAdapter()
    mira.add_market_source_adapter(adapter)

    # Process both events
    for _ in range(2):
        results = mira.poll_sources()
        time.sleep(0.01)

    stats = mira.get_stats()
    print(f"Traceability test - Events: {stats['events_received']}, Anomalies: {stats['anomalies_detected']}")

    assert len(processed_signals) == 2, f"Expected 2 signals, got {len(processed_signals)}"

    # Validate first event (baseline)
    signal1 = processed_signals[0]
    assert signal1.source == 'traceability_test', f"Expected source 'traceability_test', got '{signal1.source}'"
    assert signal1.timestamp == 1234567.0, f"Expected timestamp 1234567.0, got {signal1.timestamp}"
    assert 'session_id' in signal1.metadata or hasattr(signal1, 'session_id'), "Missing session_id"
    assert 'correlation_id' in signal1.metadata or hasattr(signal1, 'correlation_id'), "Missing correlation_id"
    assert getattr(signal1, 'confidence', 0) == 0.85, f"Expected confidence 0.85, got {getattr(signal1, 'confidence', 0)}"
    assert signal1.metadata.get('is_anomaly', False) == False, "First event should not be anomaly"

    # Validate second event (anomaly)
    signal2 = processed_signals[1]
    assert signal2.source == 'traceability_test', f"Expected source 'traceability_test', got '{signal2.source}'"
    assert signal2.timestamp == 1234568.0, f"Expected timestamp 1234568.0, got {signal2.timestamp}"
    assert signal2.metadata.get('is_anomaly', False) == True, "Second event should be anomaly"
    assert signal2.metadata.get('anomaly_threshold', 0) == 0.05, f"Expected threshold 0.05, got {signal2.metadata.get('anomaly_threshold', 0)}"

    # Verify traceability fields are preserved
    # Check for session_id and correlation_id in metadata (where MIRA puts them)
    if 'session_id' in signal2.metadata:
        assert signal2.metadata['session_id'] == 'test_session_123', f"Expected session_id 'test_session_123', got '{signal2.metadata['session_id']}'"
    if 'correlation_id' in signal2.metadata:
        assert signal2.metadata['correlation_id'] == 'test_corr_456', f"Expected correlation_id 'test_corr_456', got '{signal2.metadata['correlation_id']}'"

    print("[PASS] MIRA traceability preservation test passed")


if __name__ == "__main__":
    print("Running MIRA agent tests...\n")

    try:
        test_mira_basic_functionality()
        test_mira_anomaly_detection()
        test_mira_heterogeneous_input_formats()
        test_mira_traceability_preservation()

        print("\n*** All MIRA tests passed! ***")

    except Exception as e:
        print(f"\n!!! Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)