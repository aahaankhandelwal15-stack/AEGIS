"""Lumen - Knowledge Inference Engine for AEGIS Perception Pod.

Lumen transforms normalized stimuli into structured semantic information
by extracting entities, facts, and relationships while preserving provenance,
confidence, and privacy classifications.
"""

import re
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from aegis.core.models import (
    StimulusEnvelope,
    MemoryType,
    PrivacyClassification,
    AgentResult
)
from aegis.core.protocol import MEMORY_POD_ID, ECHO_AGENT_ID
from aegis.pods.memory.echo.agent import MemoryEntity, MemoryRelation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SemanticEntity:
    """Represents a semantic entity extracted by Lumen."""
    entity_id: str
    entity_type: str
    mention_text: str
    normalized_form: str
    attributes: Dict[str, Any]
    provenance: List[str]  # stimulus_ids that produced this entity
    confidence: float
    privacy_classification: PrivacyClassification
    created_at: float
    updated_at: float
    version: int = 1


@dataclass
class SemanticFact:
    """Represents a semantic fact extracted by Lumen."""
    fact_id: str
    subject: str
    predicate: str
    object: Any
    provenance: List[str]  # stimulus_ids that produced this fact
    confidence: float
    privacy_classification: PrivacyClassification
    created_at: float
    updated_at: float
    version: int = 1


@dataclass
class SemanticRelation:
    """Represents a semantic relation extracted by Lumen."""
    relation_id: str
    subject_id: str  # References entity_id
    predicate: str
    object_id: str   # References entity_id
    provenance: List[str]  # stimulus_ids that produced this relation
    confidence: float
    privacy_classification: PrivacyClassification
    created_at: float
    updated_at: float
    version: int = 1


class EntityType(Enum):
    """Common entity types for V1 extraction."""
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    DATE = "date"
    LOCATION = "location"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"


class LumenAgent:
    """
    Lumen Agent - Knowledge Inference Engine

    Processes StimulusEnvelopes to extract semantic information
    (entities, facts, relations) and forwards them to Echo for storage.
    """

    def __init__(
        self,
        echo_agent: Optional[Any] = None,  # Will be set via dependency injection
        entity_extractor: Optional[Callable[[str], List[SemanticEntity]]] = None,
        fact_extractor: Optional[Callable[[str], List[SemanticFact]]] = None,
        relation_extractor: Optional[Callable[[List[SemanticEntity], str], List[SemanticRelation]]] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize Lumen agent.

        Args:
            echo_agent: Reference to Echo agent for storing semantic data
            entity_extractor: Function to extract entities from text
            fact_extractor: Function to extract facts from text
            relation_extractor: Function to extract relations between entities
            confidence_threshold: Minimum confidence for extraction results
        """
        self.echo_agent = echo_agent
        self.confidence_threshold = confidence_threshold

        # Use default extractors if not provided
        self.entity_extractor = entity_extractor or self._default_entity_extractor
        self.fact_extractor = fact_extractor or self._default_fact_extractor
        self.relation_extractor = relation_extractor or self._default_relation_extractor

        # Track processed stimuli for contradiction detection
        self.processed_stimuli: Dict[str, StimulusEnvelope] = {}
        self.known_entities: Dict[str, SemanticEntity] = {}
        self.known_facts: Dict[str, SemanticFact] = {}
        self.known_relations: Dict[str, SemanticRelation] = {}

        logger.info("Lumen agent initialized")

    def set_echo_agent(self, echo_agent: Any) -> None:
        """Set the Echo agent reference for storing semantic data."""
        self.echo_agent = echo_agent
        logger.debug("Echo agent reference set in Lumen")

    def process_stimulus(self, envelope: StimulusEnvelope) -> AgentResult:
        """
        Process a StimulusEnvelope and extract semantic information.

        Args:
            envelope: The StimulusEnvelope to process

        Returns:
            AgentResult indicating success/failure and containing extracted data
        """
        try:
            logger.debug(f"Processing stimulus {envelope.stimulus_id} from {envelope.source}")

            # Store stimulus for potential contradiction detection
            self.processed_stimuli[envelope.stimulus_id] = envelope

            # Convert content to string for processing
            content_str = str(envelope.content)

            # Skip empty or very short content
            if not content_str or len(content_str.strip()) < 3:
                logger.warning(f"Skipping stimulus {envelope.stimulus_id}: insufficient content")
                return AgentResult(
                    ok=True,
                    agent_id="lumen",
                    pod_id="perception",
                    correlation_id=envelope.correlation_id,
                    data={"entities": [], "facts": [], "relations": []}
                )

            # Extract semantic information
            entities = self.entity_extractor(content_str)
            facts = self.fact_extractor(content_str)

            # Extract relations between entities
            relations = self.relation_extractor(entities, content_str)

            # Filter by confidence threshold
            entities = [e for e in entities if e.confidence >= self.confidence_threshold]
            facts = [f for f in facts if f.confidence >= self.confidence_threshold]
            relations = [r for r in relations if r.confidence >= self.confidence_threshold]

            # Preserve provenance and metadata
            self._preserve_provenance(entities, facts, relations, envelope.stimulus_id)
            self._preserve_confidence(entities, facts, relations, envelope.confidence)
            self._preserve_privacy(entities, facts, relations, envelope.privacy_classification)

            # Detect contradictions (V1: simple conflicting facts)
            contradictions = self._detect_contradictions(entities, facts, relations, envelope)

            # Store in Echo if available
            storage_success = False
            storage_error = None
            if self.echo_agent:
                try:
                    storage_success = self._store_semantic_data(entities, facts, relations)
                except Exception as e:
                    storage_error = str(e)
                    logger.error(f"Failed to store semantic data in Echo: {e}")

            # Prepare result data
            result_data = {
                "stimulus_id": envelope.stimulus_id,
                "entities": [asdict(e) for e in entities],
                "facts": [asdict(f) for f in facts],
                "relations": [asdict(r) for r in relations],
                "contradictions_detected": len(contradictions) > 0,
                "contradictions": [asdict(c) for c in contradictions] if contradictions else [],
                "stored_in_memory": storage_success,
                "storage_error": storage_error
            }

            logger.info(
                f"Processed stimulus {envelope.stimulus_id}: "
                f"{len(entities)} entities, {len(facts)} facts, {len(relations)} relations"
            )

            return AgentResult(
                ok=True,
                agent_id="lumen",
                pod_id="perception",
                correlation_id=envelope.correlation_id,
                data=result_data
            )

        except Exception as e:
            logger.error(f"Error processing stimulus {envelope.stimulus_id}: {e}")
            return AgentResult(
                ok=False,
                agent_id="lumen",
                pod_id="perception",
                correlation_id=envelope.correlation_id,
                error=f"Failed to process stimulus: {str(e)}"
            )

    def _default_entity_extractor(self, text: str) -> List[SemanticEntity]:
        """
        Default entity extractor using rule-based patterns.
        This can be replaced with an LLM-based extractor in future versions.
        """
        entities = []

        # Pattern for person names (simple detection: capitalized words)
        person_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.finditer(person_pattern, text):
            name = match.group(0)
            # Filter out common false positives
            if name not in ['The', 'This', 'That', 'Is', 'Are', 'Was', 'Were']:
                entities.append(SemanticEntity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=EntityType.PERSON.value,
                    mention_text=name,
                    normalized_form=name.lower(),
                    attributes={
                        "extraction_method": "rule_based_person",
                        "position": match.span()
                    },
                    provenance=[],  # Will be filled by _preserve_provenance
                    confidence=0.8,
                    privacy_classification=PrivacyClassification.PERSONAL,  # Default for person names
                    created_at=time.time(),
                    updated_at=time.time()
                ))

        # Pattern for organizations
        # Multi-word organization: first word capitalized, subsequent words any case (common for org names)
        multi_word_org_pattern = r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b'
        # Multi-word organization: all caps first word, subsequent words any case (e.g., "AEGIS project")
        multi_word_org_mixed_pattern = r'\b([A-Z][A-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b'
        # Multi-word organization: all caps acronyms/words
        multi_word_org_allcaps_pattern = r'\b([A-Z][A-Z]*(?:\s+[A-Z][A-Z]*)*)\b'
        # Single-word organization: all uppercase, length >= 2, not a common word
        single_word_org_pattern = r'\b([A-Z]{2,})\b'
        COMMON_UPPERCASE_WORDS = {'IS', 'OF', 'IN', 'IT', 'AS', 'AT', 'ON', 'OR', 'AND', 'THE', 'THIS', 'THAT', 'IS', 'ARE', 'WAS', 'WERE'}

        # Process multi-word organizations (first word capitalized)
        for match in re.finditer(multi_word_org_pattern, text):
            org_name = match.group(0)
            # Filter out common false positives and single letters
            if org_name not in ['The', 'This', 'That', 'Is', 'Are', 'Was', 'Were', 'A', 'I'] and len(org_name) > 1:
                # Additional filter: avoid matching common sentence starters that aren't orgs
                if not org_name.startswith(('The ', 'This ', 'That ', 'Is ', 'Are ', 'Was ', 'Were ')) or len(org_name.split()) > 1:
                    entities.append(SemanticEntity(
                        entity_id=str(uuid.uuid4()),
                        entity_type=EntityType.ORGANIZATION.value,
                        mention_text=org_name,
                        normalized_form=org_name.lower(),
                        attributes={
                            "extraction_method": "rule_based_organization_multiword",
                            "position": match.span()
                        },
                        provenance=[],
                        confidence=0.75,
                        privacy_classification=PrivacyClassification.PUBLIC,
                        created_at=time.time(),
                        updated_at=time.time()
                    ))

        # Process multi-word organizations (all caps first word, mixed case subsequent)
        for match in re.finditer(multi_word_org_mixed_pattern, text):
            org_name = match.group(0)
            # Filter out common false positives
            if org_name not in COMMON_UPPERCASE_WORDS and len(org_name) > 1:
                entities.append(SemanticEntity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=EntityType.ORGANIZATION.value,
                    mention_text=org_name,
                    normalized_form=org_name.lower(),
                    attributes={
                        "extraction_method": "rule_based_organization_mixed",
                        "position": match.span()
                    },
                    provenance=[],
                    confidence=0.75,
                    privacy_classification=PrivacyClassification.PUBLIC,
                    created_at=time.time(),
                    updated_at=time.time()
                ))

        # Process multi-word organizations (all caps)
        for match in re.finditer(multi_word_org_allcaps_pattern, text):
            org_name = match.group(0)
            # Filter out common false positives
            if org_name not in COMMON_UPPERCASE_WORDS and len(org_name) > 1:
                entities.append(SemanticEntity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=EntityType.ORGANIZATION.value,
                    mention_text=org_name,
                    normalized_form=org_name.lower(),
                    attributes={
                        "extraction_method": "rule_based_organization_multiword_allcaps",
                        "position": match.span()
                    },
                    provenance=[],
                    confidence=0.75,
                    privacy_classification=PrivacyClassification.PUBLIC,
                    created_at=time.time(),
                    updated_at=time.time()
                ))

        # Process single-word organizations
        for match in re.finditer(single_word_org_pattern, text):
            org_name = match.group(0)
            if org_name not in COMMON_UPPERCASE_WORDS:
                entities.append(SemanticEntity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=EntityType.ORGANIZATION.value,
                    mention_text=org_name,
                    normalized_form=org_name.lower(),
                    attributes={
                        "extraction_method": "rule_based_organization_singleword",
                        "position": match.span()
                    },
                    provenance=[],
                    confidence=0.75,
                    privacy_classification=PrivacyClassification.PUBLIC,
                    created_at=time.time(),
                    updated_at=time.time()
                ))

        # Pattern for dates (simple MM/DD/YYYY or Month DD, YYYY)
        date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b([A-Z][a-z]+ \d{1,2},? \d{4})\b'
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                date_str = match.group(0)
                entities.append(SemanticEntity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=EntityType.DATE.value,
                    mention_text=date_str,
                    normalized_form=date_str,
                    attributes={
                        "extraction_method": "rule_based_date",
                        "position": match.span(),
                        "raw_value": date_str
                    },
                    provenance=[],
                    confidence=0.9,
                    privacy_classification=PrivacyClassification.PUBLIC,
                    created_at=time.time(),
                    updated_at=time.time()
                ))

        return entities

    def _default_fact_extractor(self, text: str) -> List[SemanticFact]:
        """
        Default fact extractor using simple subject-predicate-object patterns.
        This can be enhanced with dependency parsing or LLM in future versions.
        """
        facts = []

        # Simple pattern for "X works on Y" or "X is working on Y"
        work_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?working\s+on\s+(.+?)(?:\.|$)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+works\s+on\s+(.+?)(?:\.|$)'
        ]

        for pattern in work_patterns:
            for match in re.finditer(pattern, text):
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                # Clean up leading articles (the, a, an) from object
                obj = re.sub(r'^(?:the|a|an)\s+', '', obj, flags=re.IGNORECASE)
                # Clean up trailing punctuation
                obj = re.sub(r'[.!?]+$', '', obj)

                facts.append(SemanticFact(
                    fact_id=str(uuid.uuid4()),
                    subject=subject,
                    predicate="works_on",
                    object=obj,
                    provenance=[],
                    confidence=0.85,
                    privacy_classification=PrivacyClassification.PERSONAL,  # Work info is personal
                    created_at=time.time(),
                    updated_at=time.time()
                ))

        # Simple pattern for deadlines: "X deadline is Y" or "X deadline: Y"
        deadline_patterns = [
            r'([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]*)*(?:\s+[a-z]+)*)\s+deadline\s+is\s+([^\.]+)',
            r'([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]*)*(?:\s+[a-z]+)*)\s+deadline\s*[:]\s*([^\.]+)'
        ]

        for pattern in deadline_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                # Clean up leading articles (the, a, an) from subject and object
                subject = re.sub(r'^(?:the|a|an)\s+', '', subject, flags=re.IGNORECASE)
                obj = re.sub(r'^(?:the|a|an)\s+', '', obj, flags=re.IGNORECASE)
                # Clean up trailing punctuation
                obj = re.sub(r'[.!?]+$', '', obj)

                facts.append(SemanticFact(
                    fact_id=str(uuid.uuid4()),
                    subject=subject,
                    predicate="deadline",
                    object=obj,
                    provenance=[],
                    confidence=0.9,
                    privacy_classification=PrivacyClassification.PUBLIC,  # Project deadlines are typically public
                    created_at=time.time(),
                    updated_at=time.time()
                ))

        # Simple pattern for preferences: "X prefers Y" or "X likes Y"
        preference_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+prefers\s+([^\.]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+likes\s+([^\.]+)'
        ]

        for pattern in preference_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                # Clean up leading articles (the, a, an) from object
                obj = re.sub(r'^(?:the|a|an)\s+', '', obj, flags=re.IGNORECASE)
                # Clean up trailing punctuation
                obj = re.sub(r'[.!?]+$', '', obj)

                facts.append(SemanticFact(
                    fact_id=str(uuid.uuid4()),
                    subject=subject,
                    predicate="prefers",
                    object=obj,
                    provenance=[],
                    confidence=0.8,
                    privacy_classification=PrivacyClassification.PERSONAL,  # Preferences are personal
                    created_at=time.time(),
                    updated_at=time.time()
                ))

        return facts

    def _default_relation_extractor(self, entities: List[SemanticEntity], text: str) -> List[SemanticRelation]:
        """
        Default relation extractor that finds relationships between entities based on proximity and patterns.
        """
        relations = []

        if len(entities) < 2:
            return relations

        # Create a mapping of mention text to entity for easy lookup
        entity_by_mention = {e.mention_text: e for e in entities}

        # Look for common relation patterns between entities
        # Patterns that capture subject and object, with object stopping at sentence boundary
        relation_patterns = [
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?working\s+on\s+(.+?)(?:\.|$)', "works_on"),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+works\s+on\s+(.+?)(?:\.|$)', "works_on"),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+part\s+of\s+(.+?)(?:\.|$)', "part_of"),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+part\s+of\s+(.+?)(?:\.|$)', "part_of"),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+uses\s+(.+?)(?:\.|$)', "uses"),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+developed\s+by\s+(.+?)(?:\.|$)', "developed_by"),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+located\s+in\s+(.+?)(?:\.|$)', "located_in")
        ]

        for pattern_str, predicate in relation_patterns:
            for match in re.finditer(pattern_str, text):
                subject_text = match.group(1).strip()
                object_text = match.group(2).strip()
                # Clean up leading articles (the, a, an) from subject and object
                subject_text = re.sub(r'^(?:the|a|an)\s+', '', subject_text, flags=re.IGNORECASE)
                object_text = re.sub(r'^(?:the|a|an)\s+', '', object_text, flags=re.IGNORECASE)
                # Clean up trailing punctuation (should already be removed by pattern, but keep for safety)
                subject_text = re.sub(r'[.!?]+$', '', subject_text)
                object_text = re.sub(r'[.!?]+$', '', object_text)

                # Find matching entities by mention_text (substring match, case-insensitive)
                subject_entity = None
                object_entity = None
                for e in entities:
                    if e.mention_text.lower() in subject_text.lower() or subject_text.lower() in e.mention_text.lower():
                        subject_entity = e
                        break
                for e in entities:
                    if e.mention_text.lower() in object_text.lower() or object_text.lower() in e.mention_text.lower():
                        object_entity = e
                        break

                if subject_entity and object_entity and subject_entity.entity_id != object_entity.entity_id:
                    relations.append(SemanticRelation(
                        relation_id=str(uuid.uuid4()),
                        subject_id=subject_entity.entity_id,
                        predicate=predicate,
                        object_id=object_entity.entity_id,
                        provenance=[],
                        confidence=0.8,
                        privacy_classification=self._merge_privacy(
                            subject_entity.privacy_classification,
                            object_entity.privacy_classification
                        ),
                        created_at=time.time(),
                        updated_at=time.time()
                    ))

        return relations

    def _preserve_provenance(self, entities: List[SemanticEntity],
                           facts: List[SemanticFact],
                           relations: List[SemanticRelation],
                           stimulus_id: str) -> None:
        """Preserve provenance by adding stimulus_id to provenance lists."""
        for entity in entities:
            entity.provenance.append(stimulus_id)

        for fact in facts:
            fact.provenance.append(stimulus_id)

        for relation in relations:
            relation.provenance.append(stimulus_id)

    def _preserve_confidence(self, entities: List[SemanticEntity],
                           facts: List[SemanticFact],
                           relations: List[SemanticRelation],
                           stimulus_confidence: float) -> None:
        """Adjust confidence based on stimulus confidence."""
        for entity in entities:
            entity.confidence = min(entity.confidence, stimulus_confidence)

        for fact in facts:
            fact.confidence = min(fact.confidence, stimulus_confidence)

        for relation in relations:
            relation.confidence = min(relation.confidence, stimulus_confidence)

    def _preserve_privacy(self, entities: List[SemanticEntity],
                        facts: List[SemanticFact],
                        relations: List[SemanticRelation],
                        stimulus_privacy: PrivacyClassification) -> None:
        """Ensure privacy classification is not downgraded."""
        privacy_order = {
            PrivacyClassification.PUBLIC: 0,
            PrivacyClassification.PERSONAL: 1,
            PrivacyClassification.SENSITIVE: 2,
            PrivacyClassification.RESTRICTED: 3
        }

        min_privacy_level = privacy_order[stimulus_privacy]

        for entity in entities:
            entity_level = privacy_order[entity.privacy_classification]
            if entity_level < min_privacy_level:
                entity.privacy_classification = stimulus_privacy

        for fact in facts:
            fact_level = privacy_order[fact.privacy_classification]
            if fact_level < min_privacy_level:
                fact.privacy_classification = stimulus_privacy

        for relation in relations:
            relation_level = privacy_order[relation.privacy_classification]
            if relation_level < min_privacy_level:
                relation.privacy_classification = stimulus_privacy

    def _merge_privacy(self, privacy1: PrivacyClassification,
                      privacy2: PrivacyClassification) -> PrivacyClassification:
        """Merge two privacy classifications, taking the more restrictive."""
        privacy_order = {
            PrivacyClassification.PUBLIC: 0,
            PrivacyClassification.PERSONAL: 1,
            PrivacyClassification.SENSITIVE: 2,
            PrivacyClassification.RESTRICTED: 3
        }

        level1 = privacy_order[privacy1]
        level2 = privacy_order[privacy2]
        max_level = max(level1, level2)

        # Find the privacy classification with this level
        for privacy, level in privacy_order.items():
            if level == max_level:
                return privacy
        return PrivacyClassification.RESTRICTED  # fallback

    def _detect_contradictions(self, entities: List[SemanticEntity],
                             facts: List[SemanticFact],
                             relations: List[SemanticRelation],
                             stimulus: StimulusEnvelope) -> List[Dict[str, Any]]:
        """
        Detect contradictions with previously stored knowledge.
        V1 implementation: detect conflicting facts with same subject and predicate.
        """
        contradictions = []

        # Check for conflicting facts
        for new_fact in facts:
            # Look for existing facts with same subject and predicate
            for fact_id, existing_fact in self.known_facts.items():
                if (existing_fact.subject.lower() == new_fact.subject.lower() and
                    existing_fact.predicate.lower() == new_fact.predicate.lower()):

                    # Check if objects conflict (simplified string comparison)
                    if str(existing_fact.object).lower() != str(new_fact.object).lower():
                        contradictions.append({
                            "type": "fact_conflict",
                            "subject": new_fact.subject,
                            "predicate": new_fact.predicate,
                            "existing_object": str(existing_fact.object),
                            "new_object": str(new_fact.object),
                            "existing_stimulus_ids": existing_fact.provenance,
                            "new_stimulus_ids": new_fact.provenance,
                            "confidence": min(existing_fact.confidence, new_fact.confidence)
                        })

        return contradictions

    def _store_semantic_data(self, entities: List[SemanticEntity],
                           facts: List[SemanticFact],
                           relations: List[SemanticRelation]) -> bool:
        """
        Store semantic data in Echo agent.
        Returns True if successful, False otherwise.
        """
        if not self.echo_agent:
            logger.warning("No Echo agent available for storage")
            return False

        try:
            # Store entities
            for entity in entities:
                memory_entity = MemoryEntity(
                    entity_id=entity.entity_id,
                    entity_type=entity.entity_type,
                    attributes={
                        **entity.attributes,
                        "mention_text": entity.mention_text,
                        "normalized_form": entity.normalized_form,
                        "provenance": entity.provenance,
                        "extraction_confidence": entity.confidence
                    },
                    created_at=entity.created_at,
                    updated_at=entity.updated_at,
                    version=entity.version
                )
                # Handle both direct Echo agent methods and knowledge_graph attribute
                if hasattr(self.echo_agent, 'knowledge_graph'):
                    self.echo_agent.knowledge_graph.add_entity(memory_entity)
                else:
                    self.echo_agent.add_entity(memory_entity)

                # Track for contradiction detection
                self.known_entities[entity.entity_id] = entity

            # Store facts as entities (since Echo doesn't have a direct fact store)
            # We'll create a special entity type for facts
            for fact in facts:
                fact_entity = MemoryEntity(
                    entity_id=fact.fact_id,
                    entity_type="semantic_fact",
                    attributes={
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": str(fact.object),
                        "provenance": fact.provenance,
                        "extraction_confidence": fact.confidence
                    },
                    created_at=fact.created_at,
                    updated_at=fact.updated_at,
                    version=fact.version
                )
                # Handle both direct Echo agent methods and knowledge_graph attribute
                if hasattr(self.echo_agent, 'knowledge_graph'):
                    self.echo_agent.knowledge_graph.add_entity(fact_entity)
                else:
                    self.echo_agent.add_entity(fact_entity)
                self.known_facts[fact.fact_id] = fact

            # Store relations
            for relation in relations:
                memory_relation = MemoryRelation(
                    relation_id=relation.relation_id,
                    subject_id=relation.subject_id,
                    predicate=relation.predicate,
                    object_id=relation.object_id,
                    confidence=relation.confidence,
                    created_at=relation.created_at,
                    updated_at=relation.updated_at,
                    version=relation.version
                )
                # Handle both direct Echo agent methods and knowledge_graph attribute
                if hasattr(self.echo_agent, 'knowledge_graph'):
                    self.echo_agent.knowledge_graph.add_relation(memory_relation)
                else:
                    self.echo_agent.add_relation(memory_relation)
                self.known_relations[relation.relation_id] = relation

            logger.debug(f"Stored {len(entities)} entities, {len(facts)} facts, {len(relations)} relations in Echo")
            return True

        except Exception as e:
            logger.error(f"Failed to store semantic data in Echo: {e}")
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about Lumen's processing."""
        return {
            "stimuli_processed": len(self.processed_stimuli),
            "entities_extracted": len(self.known_entities),
            "facts_extracted": len(self.known_facts),
            "relations_extracted": len(self.known_relations),
            "echo_agent_available": self.echo_agent is not None
        }


# Factory function for easy instantiation
def create_lumen_agent(echo_agent: Optional[Any] = None) -> LumenAgent:
    """
    Create a Lumen agent with optional Echo agent dependency.

    Args:
        echo_agent: Optional Echo agent for storing semantic data

    Returns:
        Configured LumenAgent instance
    """
    return LumenAgent(echo_agent=echo_agent)


if __name__ == "__main__":
    # Simple test when run directly
    print("Testing Lumen agent...")

    # Create a mock Echo agent for testing
    class MockEchoAgent:
        def __init__(self):
            self.entities = []
            self.relations = []

        def add_entity(self, entity):
            self.entities.append(entity)
            return entity.entity_id

        def add_relation(self, relation):
            self.relations.append(relation)
            return relation.relation_id

    # Create Lumen agent
    mock_echo = MockEchoAgent()
    lumen = LumenAgent(echo_agent=mock_echo)

    # Create test stimulus
    from aegis.core.models import StimulusEnvelope, PrivacyClassification

    test_stimulus = StimulusEnvelope(
        stimulus_id="test_001",
        content="Rahul is working on the AEGIS project. The deadline is October 15, 2026.",
        timestamp=time.time(),
        source="user_input",
        modality="text",
        confidence=0.95,
        privacy_classification=PrivacyClassification.PERSONAL
    )

    # Process the stimulus
    result = lumen.process_stimulus(test_stimulus)

    print(f"Processing result: {result.ok}")
    if result.ok:
        print(f"Entities extracted: {len(result.data['entities'])}")
        print(f"Facts extracted: {len(result.data['facts'])}")
        print(f"Relations extracted: {len(result.data['relations'])}")
        print(f"Stored in Echo: {len(mock_echo.entities)} entities, {len(mock_echo.relations)} relations")
    else:
        print(f"Error: {result.error}")

    print("Lumen agent test completed.")