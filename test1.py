from aegis.pods.perception.lumen.agent import LumenAgent
from aegis.core.models import StimulusEnvelope, PrivacyClassification
import time

lumen = LumenAgent()

# Create a test stimulus
stimulus = StimulusEnvelope(
    stimulus_id="custom_test_001",
    content="Alice is working on the Odin project. The deadline is January 15, 2027.",
    timestamp=time.time(),
    source="user_input",
    modality="text",
    confidence=0.9,
    privacy_classification=PrivacyClassification.PERSONAL
)

# Process the stimulus
result = lumen.process_stimulus(stimulus)
print(f"Success: {result.ok}")
if result.ok:
    print(f"Entities: {len(result.data['entities'])}")
    print(f"Facts: {len(result.data['facts'])}")
    print(f"Relations: {len(result.data['relations'])}")