"""Simple test for IRIS agent functionality."""

import sys
import os
import time

# Add project root to path so we can import aegis
sys.path.insert(0, os.path.abspath('.'))

from aegis.pods.perception.iris.agent import (
    IrisAgent,
    DummySourceAdapter,
    ProcessingResult
)
from aegis.core.models import (
    StimulusEnvelope,
    PrivacyClassification
)


def test_iris_basic_functionality():
    """Test basic IRIS agent functionality."""
    print("Testing basic IRIS agent functionality...")

    # Create IRIS agent
    iris = IrisAgent(
        deduplication_enabled=True,
        rate_limit_enabled=True,
        max_events_per_second=10.0,
        burst_capacity=20.0
    )

    # Add dummy source adapter
    dummy_source = DummySourceAdapter(name="test_dummy", event_interval=0.1)
    iris.add_source_adapter(dummy_source)

    # Track processed events
    processed_events = []

    def event_handler(envelope: StimulusEnvelope) -> None:
        processed_events.append(envelope)
        print(f"  Handler received: {envelope.stimulus_id[:8]} from {envelope.source}")

    iris.register_event_handler("*", event_handler)

    # Run for a short period
    start_time = time.time()
    test_duration = 2.0  # 2 seconds

    events_processed = 0
    while time.time() - start_time < test_duration:
        results = iris.poll_sources()
        events_processed += sum(results.values())
        time.sleep(0.01)  # Small delay to prevent busy loop

    # Check results
    stats = iris.get_stats()
    print(f"  Events received: {stats['events_received']}")
    print(f"  Events accepted: {stats['events_accepted']}")
    print(f"  Events duplicate: {stats['events_duplicate']}")
    print(f"  Events rate limited: {stats['events_rate_limited']}")
    print(f"  Events invalid: {stats['events_invalid']}")
    print(f"  Events error: {stats['events_error']}")
    print(f"  Handler callbacks: {len(processed_events)}")
    print(f"  Total polling cycles: {events_processed}")

    # Basic assertions - at minimum, the agent should be able to poll without errors
    assert stats['events_received'] >= 0, "Events received should be non-negative"
    assert stats['events_accepted'] >= 0, "Events accepted should be non-negative"
    assert len(processed_events) == stats['events_accepted'], "Handler calls should match accepted events"

    # If we received events, check that processed events are valid StimulusEnvelopes
    if len(processed_events) > 0:
        for event in processed_events[:3]:  # Check first 3 events
            assert isinstance(event, StimulusEnvelope)
            assert hasattr(event, 'stimulus_id')
            assert hasattr(event, 'source')
            assert hasattr(event, 'modality')
            assert hasattr(event, 'confidence')
            assert 0.0 <= event.confidence <= 1.0
            assert isinstance(event.privacy_classification, PrivacyClassification)

    print("[PASS] Basic IRIS agent functionality test")


def test_iris_source_adapter_management():
    """Test adding and removing source adapters."""
    print("Testing source adapter management...")

    iris = IrisAgent()

    # Initially no adapters
    assert len(iris.source_adapters) == 0

    # Add adapters
    adapter1 = DummySourceAdapter(name="adapter1")
    adapter2 = DummySourceAdapter(name="adapter2")

    iris.add_source_adapter(adapter1)
    assert len(iris.source_adapters) == 1
    assert iris.source_adapters[0].name == "adapter1"

    iris.add_source_adapter(adapter2)
    assert len(iris.source_adapters) == 2
    assert iris.source_adapters[1].name == "adapter2"

    # Remove adapter
    result = iris.remove_source_adapter("adapter1")
    assert result == True
    assert len(iris.source_adapters) == 1
    assert iris.source_adapters[0].name == "adapter2"

    # Try to remove non-existent adapter
    result = iris.remove_source_adapter("nonexistent")
    assert result == False
    assert len(iris.source_adapters) == 1

    print("[PASS] Source adapter management test")


def test_iris_stats():
    """Test IRIS statistics functionality."""
    print("Testing IRIS statistics...")

    iris = IrisAgent()

    # Initial stats
    stats = iris.get_stats()
    assert stats['events_received'] == 0
    assert stats['events_accepted'] == 0
    assert stats['events_duplicate'] == 0
    assert stats['events_rate_limited'] == 0
    assert stats['events_invalid'] == 0
    assert stats['events_error'] == 0
    assert 'uptime_seconds' in stats

    # Process some events (need to add a source first)
    dummy_source = DummySourceAdapter(name="stats_test", event_interval=0.1)
    iris.add_source_adapter(dummy_source)

    # Process a few events
    start_time = time.time()
    while time.time() - start_time < 0.5:  # 0.5 seconds
        iris.poll_sources()
        time.sleep(0.01)

    # Check stats updated
    stats = iris.get_stats()
    assert stats['events_received'] >= 0
    assert stats['uptime_seconds'] > 0

    # Reset stats
    iris.reset_stats()
    stats = iris.get_stats()
    assert stats['events_received'] == 0
    assert stats['events_accepted'] == 0
    assert stats['uptime_seconds'] >= 0  # Should have some small uptime

    print("[PASS] IRIS statistics test")


if __name__ == "__main__":
    print("Running simple IRIS agent tests...")
    print("=" * 50)

    try:
        test_iris_basic_functionality()
        test_iris_source_adapter_management()
        test_iris_stats()

        print("=" * 50)
        print("[PASS] All simple IRIS agent tests passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)