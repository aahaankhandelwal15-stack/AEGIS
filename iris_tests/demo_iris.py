#!/usr/bin/env python3
"""
Demonstration of IRIS agent functionality.
"""

import sys
import os
import time

# Add project root to path so we can import aegis
sys.path.insert(0, os.path.abspath('.'))

from aegis.pods.perception.iris.agent import (
    IrisAgent,
    DummySourceAdapter
)
from aegis.core.models import StimulusEnvelope


def event_handler(envelope: StimulusEnvelope) -> None:
    """Handle processed events from IRIS."""
    print(f"[EVENT] Received: {envelope.stimulus_id[:8]} "
          f"from {envelope.source} "
          f"modality={envelope.modality} "
          f"confidence={envelope.confidence:.2f} "
          f"privacy={envelope.privacy_classification.value}")


def main():
    """Demonstrate IRIS agent functionality."""
    print("IRIS (Intelligent Recognition and Ingestion System) Demo")
    print("=" * 60)

    # Create IRIS agent with moderate settings
    iris = IrisAgent(
        deduplication_enabled=True,
        rate_limit_enabled=True,
        max_events_per_second=5.0,  # Allow 5 events per second
        burst_capacity=10.0
    )

    print(f"IRIS Agent ID: {iris.agent_id if hasattr(iris, 'agent_id') else 'iris'}")
    print(f"Perception Pod ID: {iris.pod_id if hasattr(iris, 'pod_id') else 'perception_pod'}")

    # Add dummy source adapter that generates test events
    dummy_source = DummySourceAdapter(name="demo_source", event_interval=0.5)
    iris.add_source_adapter(dummy_source)
    print(f"Added source adapter: {dummy_source.name}")

    # Register event handler
    iris.register_event_handler("*", event_handler)
    print("Registered event handler")

    print("\nStarting event processing...")
    print("(Will run for 10 seconds or until interrupted)")
    print("-" * 60)

    try:
        start_time = time.time()
        demo_duration = 10.0  # Run for 10 seconds

        while time.time() - start_time < demo_duration:
            # Poll sources for new events
            results = iris.poll_sources()

            # Display statistics every 2 seconds
            elapsed = int(time.time() - start_time)
            if elapsed % 2 == 0 and elapsed > 0:
                stats = iris.get_stats()
                if stats['events_received'] > 0:
                    print(f"\n[STATS] Elapsed: {elapsed}s | "
                          f"Received: {stats['events_received']}, "
                          f"Accepted: {stats['events_accepted']}, "
                          f"Duplicate: {stats['events_duplicate']}, "
                          f"Rate Limited: {stats['events_rate_limited']}")

            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")

    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS:")
    final_stats = iris.get_stats()
    for key, value in final_stats.items():
        if key != 'last_reset':  # Skip internal timestamp
            print(f"  {key}: {value}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()