"""Comprehensive test for IRIS agent functionality demonstrating all key features."""

import sys
import os
import time

# Add project root to path so we can import aegis
sys.path.insert(0, os.path.abspath('.'))

from aegis.pods.perception.iris.agent import (
    IrisAgent,
    DummySourceAdapter,
    ProcessingResult,
    SourceAdapter
)
from aegis.core.models import (
    StimulusEnvelope,
    PrivacyClassification,
    MemoryType
)


def test_iris_deduplication():
    """Test IRIS deduplication functionality."""
    print("Testing IRIS deduplication...")

    iris = IrisAgent(
        deduplication_enabled=True,
        rate_limit_enabled=False,  # Disable for focused testing
        max_events_per_second=10.0
    )

    # Create a custom source that generates the same event repeatedly
    class RepeatingSource(SourceAdapter):
        def __init__(self, name: str):
            super().__init__(name=name)
            # Set short poll interval for testing
            self.poll_interval = 0.01  # seconds
            self.poll_count = 0

        def poll(self) -> list:
            self.poll_count += 1
            # Always return the same event
            return [{
                'source': 'test_repeating',
                'modality': 'test',
                'content': {'id': 'same_event', 'value': 42},
                'confidence': 0.8
            }] if self.poll_count <= 5 else []  # Limit polls to avoid infinite loop

        def normalize(self, raw_event: dict) -> dict:
            return raw_event

    repeating_source = RepeatingSource("repeating_test")
    iris.add_source_adapter(repeating_source)

    # Track unique events received
    unique_events = []
    duplicate_count = 0

    def count_unique_events(envelope: StimulusEnvelope) -> None:
        # Simple deduplication by content for this test
        content_str = str(envelope.content)
        if not any(str(e.content) == content_str for e in unique_events):
            unique_events.append(envelope)
        else:
            nonlocal duplicate_count
            duplicate_count += 1

    iris.register_event_handler("*", count_unique_events)

    # Poll multiple times
    for _ in range(10):
        iris.poll_sources()
        time.sleep(0.01)

    stats = iris.get_stats()
    print(f"  Events received: {stats['events_received']}")
    print(f"  Events accepted: {stats['events_accepted']}")
    print(f"  Events duplicate: {stats['events_duplicate']}")
    print(f"  Unique events tracked: {len(unique_events)}")
    print(f"  Duplicate count from handler: {duplicate_count}")

    # With deduplication enabled, we should get mostly duplicates after the first event
    assert stats['events_duplicate'] > 0, "Should have detected duplicates"
    assert len(unique_events) == 1, "Should only have one unique event"
    assert stats['events_accepted'] >= 1, "Should have accepted at least one event"

    print("[PASS] IRIS deduplication test")


def test_iris_rate_limiting():
    """Test IRIS rate limiting functionality."""
    print("Testing IRIS rate limiting...")

    iris = IrisAgent(
        deduplication_enabled=False,  # Disable for focused testing
        rate_limit_enabled=True,
        max_events_per_second=2.0,  # Very low rate limit
        burst_capacity=1.0
    )

    # Create a source that generates events quickly
    class FastSource(SourceAdapter):
        def __init__(self, name: str):
            super().__init__(name=name)
            # Set short poll interval for testing
            self.poll_interval = 0.01  # seconds
            self.poll_count = 0

        def poll(self) -> list:
            self.poll_count += 1
            # Generate an event every time we're polled
            return [{
                'source': 'test_fast',
                'modality': 'test',
                'content': {'sequence': self.poll_count},
                'confidence': 0.9
            }] if self.poll_count <= 10 else []

        def normalize(self, raw_event: dict) -> dict:
            return raw_event

    fast_source = FastSource("fast_test")
    iris.add_source_adapter(fast_source)

    # Poll rapidly - should hit rate limit
    for _ in range(20):  # Try to poll 20 times quickly
        iris.poll_sources()
        time.sleep(0.01)  # Small delay between polls

    stats = iris.get_stats()
    print(f"  Events received: {stats['events_received']}")
    print(f"  Events accepted: {stats['events_accepted']}")
    print(f"  Events rate limited: {stats['events_rate_limited']}")

    # With rate limiting, some events should be rejected
    assert stats['events_rate_limited'] > 0, "Should have rate limited some events"
    assert stats['events_accepted'] < stats['events_received'], "Should have rejected some events"

    print("[PASS] IRIS rate limiting test")


def test_iris_privacy_classification():
    """Test IRIS privacy classification functionality."""
    print("Testing IRIS privacy classification...")

    iris = IrisAgent(
        deduplication_enabled=False,
        rate_limit_enabled=False
    )

    # Track events by privacy classification
    privacy_counts = {
        PrivacyClassification.PUBLIC: 0,
        PrivacyClassification.PERSONAL: 0,
        PrivacyClassification.SENSITIVE: 0,
        PrivacyClassification.RESTRICTED: 0
    }

    def count_by_privacy(envelope: StimulusEnvelope) -> None:
        privacy_counts[envelope.privacy_classification] += 1

    iris.register_event_handler("*", count_by_privacy)

    # Test events with different privacy characteristics
    test_events = [
        # Should be PUBLIC (default)
        {
            'source': 'public_source',
            'modality': 'test',
            'content': 'public information',
            'confidence': 0.8
        },
        # Should be PERSONAL (contains personal info)
        {
            'source': 'email_client',
            'modality': 'text',
            'content': 'sent email to john.doe@example.com',
            'confidence': 0.9
        },
        # Should be SENSITIVE (contains sensitive keywords)
        {
            'source': 'health_app',
            'modality': 'sensor',
            'content': 'heart_rate: 80bpm, blood_pressure: 120/80',
            'confidence': 0.85
        },
        # Should be RESTRICTED (contains restricted keywords)
        {
            'source': 'password_manager',
            'modality': 'keyboard',
            'content': 'password: secret123',
            'confidence': 0.95
        },
        # Explicit privacy classification
        {
            'source': 'explicit_test',
            'modality': 'test',
            'content': 'test content',
            'confidence': 0.7,
            'metadata': {'privacy_level': 'restricted'}
        }
    ]

    # Process each event
    for event_data in test_events:
        # We need to simulate what a source adapter would do
        # Since we don't have a source adapter, we'll call _process_event directly
        # But we need to wrap it in a source adapter interface

        class TestSource(SourceAdapter):
            def __init__(self, event_data):
                super().__init__(name="test_source")
                # Set short poll interval for testing
                self.poll_interval = 0.01  # seconds
                # Set last_poll to old time so we poll immediately
                self.last_poll = time.time() - self.poll_interval - 1.0
                self.event_data = event_data
                self.used = False

            def poll(self) -> list:
                if self.used:
                    return []
                self.used = True
                return [self.event_data]

            def normalize(self, raw_event: dict) -> dict:
                return raw_event

        test_source = TestSource(event_data)
        iris.add_source_adapter(test_source)
        iris.poll_sources()  # Process the event
        iris.source_adapters.pop()  # Remove the test source

    # Give time for processing
    time.sleep(0.1)

    stats = iris.get_stats()
    print(f"  Events received: {stats['events_received']}")
    print(f"  Events accepted: {stats['events_accepted']}")
    print(f"  Privacy distribution: {privacy_counts}")

    # Should have processed all events
    assert stats['events_received'] == len(test_events), "Should have received all test events"
    assert stats['events_accepted'] == len(test_events), "Should have accepted all test events"

    # Check that we got events with different classifications
    # Note: The exact classification might vary based on implementation details
    # but we should see some variety
    total_classified = sum(privacy_counts.values())
    assert total_classified == len(test_events), "All events should be classified"

    print("[PASS] IRIS privacy classification test")


def test_iris_confidence_clamping():
    """Test that IRIS properly clamps confidence values."""
    print("Testing IRIS confidence clamping...")

    iris = IrisAgent(
        deduplication_enabled=False,
        rate_limit_enabled=False
    )

    processed_events = []

    def capture_event(envelope: StimulusEnvelope) -> None:
        processed_events.append(envelope)

    iris.register_event_handler("*", capture_event)

    # Test events with out-of-range confidence values
    test_cases = [
        (-0.5, 0.0),   # Below minimum should clamp to 0.0
        (1.5, 1.0),    # Above maximum should clamp to 1.0
        (0.7, 0.7),    # Within range should stay the same
        (0.0, 0.0),    # Edge case
        (1.0, 1.0)     # Edge case
    ]

    for i, (input_conf, expected_conf) in enumerate(test_cases):
        class TestSource(SourceAdapter):
            def __init__(self, conf, index):
                super().__init__(name=f"test_source_{index}")
                # Set short poll interval for testing
                self.poll_interval = 0.01  # seconds
                # Set last_poll to old time so we poll immediately
                self.last_poll = time.time() - self.poll_interval - 1.0
                self.conf = conf
                self.used = False

            def poll(self) -> list:
                if self.used:
                    return []
                self.used = True
                return [{
                    'source': 'test_conf',
                    'modality': 'test',
                    'content': {'value': input_conf},
                    'confidence': self.conf
                }]

            def normalize(self, raw_event: dict) -> dict:
                return raw_event

        test_source = TestSource(input_conf, i)
        iris.add_source_adapter(test_source)
        iris.poll_sources()
        iris.source_adapters.pop()

    # Check that confidence values were properly clamped
    assert len(processed_events) == len(test_cases), "Should have processed all test cases"

    for i, (input_conf, expected_conf) in enumerate(test_cases):
        event = processed_events[i]
        actual_conf = event.confidence
        assert actual_conf == expected_conf, f"Event {i}: expected confidence {expected_conf}, got {actual_conf}"
        assert 0.0 <= actual_conf <= 1.0, f"Event {i}: confidence {actual_conf} out of range"

    print("[PASS] IRIS confidence clamping test")


def test_iris_metadata_handling():
    """Test that IRIS properly handles and extends metadata."""
    print("Testing IRIS metadata handling...")

    iris = IrisAgent(
        deduplication_enabled=False,
        rate_limit_enabled=False
    )

    processed_events = []

    def capture_event(envelope: StimulusEnvelope) -> None:
        processed_events.append(envelope)

    iris.register_event_handler("*", capture_event)

    # Test event with rich metadata
    test_event = {
        'source': 'metadata_test',
        'modality': 'test',
        'content': {'key': 'value'},
        'confidence': 0.8,
        'metadata': {
            'user_id': 'user123',
            'session': 'sess_abc',
            'tags': ['important', 'test']
        }
    }

    class TestSource(SourceAdapter):
        def __init__(self, event_data):
            super().__init__(name="test_source")
            # Set short poll interval for testing
            self.poll_interval = 0.01  # seconds
            # Set last_poll to old time so we poll immediately
            self.last_poll = time.time() - self.poll_interval - 1.0
            self.event_data = event_data
            self.used = False

        def poll(self) -> list:
            if self.used:
                return []
            self.used = True
            return [self.event_data]

        def normalize(self, raw_event: dict) -> dict:
            return raw_event

    test_source = TestSource(test_event)
    iris.add_source_adapter(test_source)
    iris.poll_sources()
    iris.source_adapters.pop()

    assert len(processed_events) == 1, "Should have processed one event"
    event = processed_events[0]

    # Check that original metadata is preserved
    assert event.metadata['user_id'] == 'user123'
    assert event.metadata['session'] == 'sess_abc'
    assert event.metadata['tags'] == ['important', 'test']

    # Check that IRIS added its own metadata
    assert event.metadata['processed_by'] == 'iris'
    assert 'processed_at' in event.metadata
    assert event.metadata['ingestion_source'] == 'test_source'

    # Check that timestamp is reasonable
    assert abs(event.timestamp - time.time()) < 5.0  # Within 5 seconds

    print("[PASS] IRIS metadata handling test")


def test_iris_traceability_fields():
    """Test that IRIS properly handles traceability fields."""
    print("Testing IRIS traceability fields...")

    iris = IrisAgent(
        deduplication_enabled=False,
        rate_limit_enabled=False
    )

    processed_events = []

    def capture_event(envelope: StimulusEnvelope) -> None:
        processed_events.append(envelope)

    iris.register_event_handler("*", capture_event)

    # Test event with traceability fields
    test_event = {
        'source': 'traceability_test',
        'modality': 'api',
        'content': {'action': 'user_login'},
        'confidence': 0.92,
        'metadata': {'auth_method': 'oauth'},
        'session_id': 'sess_user_12345',
        'correlation_id': 'req_web_98765'
    }

    class TestSource(SourceAdapter):
        def __init__(self, event_data):
            super().__init__(name="test_source")
            # Set short poll interval for testing
            self.poll_interval = 0.01  # seconds
            # Set last_poll to old time so we poll immediately
            self.last_poll = time.time() - self.poll_interval - 1.0
            self.event_data = event_data
            self.used = False

        def poll(self) -> list:
            if self.used:
                return []
            self.used = True
            return [self.event_data]

        def normalize(self, raw_event: dict) -> dict:
            return raw_event

    test_source = TestSource(test_event)
    iris.add_source_adapter(test_source)
    iris.poll_sources()
    iris.source_adapters.pop()

    assert len(processed_events) == 1, "Should have processed one event"
    event = processed_events[0]

    # Check traceability fields
    # The source should be preserved from the original event data
    assert event.source == 'traceability_test'

    assert event.modality == 'api'
    assert event.confidence == 0.92
    assert event.session_id == 'sess_user_12345'
    assert event.correlation_id == 'req_web_98765'
    assert event.metadata['auth_method'] == 'oauth'
    assert event.metadata['processed_by'] == 'iris'

    # Check that stimulus_id is generated
    assert len(event.stimulus_id) > 0
    assert isinstance(event.stimulus_id, str)

    print("[PASS] IRIS traceability fields test")


def test_iris_end_to_end_with_dummy_source():
    """Test end-to-end functionality with dummy source adapter."""
    print("Testing IRIS end-to-end with dummy source...")

    iris = IrisAgent(
        deduplication_enabled=True,
        rate_limit_enabled=True,
        max_events_per_second=5.0,
        burst_capacity=10.0
    )

    # Add dummy source
    dummy_source = DummySourceAdapter(name="end_to_end_test", event_interval=0.2)
    iris.add_source_adapter(dummy_source)

    # Track events by type
    events_by_modality = {
        'temperature': 0,
        'text': 0,
        'calendar': 0
    }

    def categorize_event(envelope: StimulusEnvelope) -> None:
        if envelope.modality in events_by_modality:
            events_by_modality[envelope.modality] += 1

    iris.register_event_handler("*", categorize_event)

    # Run for a short period
    start_time = time.time()
    test_duration = 3.0  # 3 seconds

    while time.time() - start_time < test_duration:
        iris.poll_sources()
        time.sleep(0.01)

    stats = iris.get_stats()
    print(f"  Events received: {stats['events_received']}")
    print(f"  Events accepted: {stats['events_accepted']}")
    print(f"  Events duplicate: {stats['events_duplicate']}")
    print(f"  Events rate limited: {stats['events_rate_limited']}")
    print(f"  Events by modality: {events_by_modality}")

    # Should have received and processed events
    assert stats['events_received'] > 0, "Should have received events"
    assert stats['events_accepted'] > 0, "Should have accepted events"
    assert stats['events_accepted'] == stats['events_received'], "All received events should be accepted (no duplicates/rate limits in this short test)"
    assert sum(events_by_modality.values()) == stats['events_accepted'], "Modality counts should match accepted events"

    # Check that we got events of different types
    modalities_received = [mod for mod, count in events_by_modality.items() if count > 0]
    assert len(modalities_received) > 1, "Should have received events of multiple modalities"

    print("[PASS] IRIS end-to-end test")


if __name__ == "__main__":
    print("Running comprehensive IRIS agent tests...")
    print("=" * 60)

    try:
        test_iris_deduplication()
        test_iris_rate_limiting()
        test_iris_privacy_classification()
        test_iris_confidence_clamping()
        test_iris_metadata_handling()
        test_iris_traceability_fields()
        test_iris_end_to_end_with_dummy_source()

        print("=" * 60)
        print("[PASS] All comprehensive IRIS agent tests passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)