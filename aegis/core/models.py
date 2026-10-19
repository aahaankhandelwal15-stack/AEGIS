"""Core data models for AEGIS."""

from enum import Enum
import time
import uuid
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Type of memory."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EXECUTION = "execution"
    GOAL = "goal"
    TASK = "task"
    # Rune-specific memory types
    PROMPT = "prompt"
    FEW_SHOT_EXAMPLE = "few_shot_example"
    REGRESSION_TEST = "regression_test"
    PROMPT_EVALUATION = "prompt_evaluation"
    OUTCOME_SIGNAL = "outcome_signal"


class MemoryScope(str, Enum):
    """Scope of memory."""

    PERSONAL = "personal"
    POD_SHARED = "pod_shared"


class PrivacyClassification(str, Enum):
    """Privacy classification for stimulus envelopes and derived entities."""

    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class StimulusEnvelope(BaseModel):
    """Normalized message format produced by the Perception Layer for every ingested event."""

    stimulus_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the stimulus"
    )
    source: str = Field(..., description="Concrete origin of the stimulus (e.g., 'google_calendar', 'chrome_extension')")
    timestamp: float = Field(
        default_factory=time.time,
        description="Timestamp when the stimulus was generated/event occurred"
    )
    modality: str = Field(..., description="Broad class of stimulus (e.g., 'calendar', 'browser', 'voice')")
    confidence: float = Field(
        ...,
        description="Confidence/probability value in [0.0, 1.0] range",
        ge=0.0,
        le=1.0
    )
    privacy_classification: PrivacyClassification = Field(
        default=PrivacyClassification.PUBLIC,
        description="Privacy classification set at ingestion"
    )
    # For V1, we keep the content in the envelope. In future versions, this could be
    # replaced with a reference to an external object store for large payloads.
    content: Any = Field(..., description="The stimulus content/payload")
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for grouping related stimuli"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for tracing across agents"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class AgentEnvelope(BaseModel):
    """Envelope for agent communication."""

    agent_id: str = Field(..., description="ID of the target agent")
    pod_id: str = Field(..., description="ID of the target pod")
    intent: str = Field(..., description="Intent to be handled by the agent")
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Payload for the intent"
    )
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for tracing"
    )


class AgentResult(BaseModel):
    """Result from an agent."""

    ok: bool = Field(..., description="Whether the operation was successful")
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for tracing"
    )
    agent_id: str = Field(..., description="ID of the agent that produced the result")
    pod_id: str = Field(..., description="ID of the pod that produced the result")
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="Result data if successful"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if unsuccessful"
    )


class ClassificationResult(BaseModel):
    """Result of a classification operation."""

    # Add fields as needed; for now, we keep it minimal since it's used in the import
    # but not obviously used in the kairo agent. We'll define it as empty for now.
    pass


class MemoryQuery(BaseModel):
    """Query for searching memory."""

    query: str = Field(..., description="Query text")
    limit: int = Field(
        default=5, description="Maximum number of results to return"
    )
    memory_types: Optional[List[MemoryType]] = Field(
        default=None, description="Filter by memory types"
    )
    session_id: Optional[str] = Field(
        default=None, description="Filter by session ID"
    )
    entity_id: Optional[str] = Field(
        default=None, description="Filter by entity ID"
    )
    requesting_agent_id: str = Field(
        default="unknown", description="ID of the requesting agent"
    )


class MemoryRecord(BaseModel):
    """A memory record."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique identifier"
    )
    project_id: str = Field(..., description="Project ID")
    content: Any = Field(..., description="Content of the memory")
    memory_type: MemoryType = Field(..., description="Type of memory")
    scope: MemoryScope = Field(..., description="Scope of the memory")
    source_agent_id: str = Field(
        ..., description="ID of the agent that created the memory"
    )
    session_id: str = Field(..., description="Session ID")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: float = Field(default_factory=time.time, description="Timestamp when this memory record was created")
    updated_at: float = Field(default_factory=time.time, description="Timestamp when this memory record was last updated")


# For forward reference in MemoryQuery
MemoryQuery.model_rebuild()


# Execution history models
class Goal(BaseModel):
    """A goal to be achieved."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., description="Description of the goal")
    source_agent_id: str = Field(..., description="ID of the agent that proposed the goal")
    created_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """A task that is part of a goal."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = Field(..., description="ID of the goal this task belongs to")
    description: str = Field(..., description="Description of the task")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    assigned_agent_id: Optional[str] = Field(
        default=None, description="ID of the agent assigned to this task"
    )
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = Field(
        default=None, description="Timestamp when the task started"
    )
    completed_at: Optional[float] = Field(
        default=None, description="Timestamp when the task was completed"
    )
    failed_at: Optional[float] = Field(
        default=None, description="Timestamp when the task failed"
    )
    failure_reason: Optional[str] = Field(
        default=None, description="Reason for failure, if any"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="List of task IDs that this task depends on"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionLog(BaseModel):
    """A log entry for an execution event."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    agent_id: str = Field(..., description="ID of the agent that generated the log")
    action: str = Field(..., description="Action performed")
    inputs: Dict[str, Any] = Field(
        default_factory=dict, description="Inputs to the action"
    )
    outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Outputs from the action"
    )
    status: str = Field(
        default="success", description="Status of the action (success, failure, etc.)"
    )
    duration: Optional[float] = Field(
        default=None, description="Duration of the action in seconds"
    )
    parent_task_id: Optional[str] = Field(
        default=None, description="ID of the parent task, if any"
    )
    parent_event_id: Optional[str] = Field(
        default=None, description="ID of the parent event, if any"
    )
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for tracing"
    )
    sequence_number: Optional[int] = Field(
        default=None, description="Sequence number within an execution trace"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the action failed"
    )
    state_transition: Optional[str] = Field(
        default=None, description="State transition information (e.g., 'idle->running')"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Rune-specific models
class ModelConfiguration(BaseModel):
    """Configuration for a model used with a prompt."""
    model_id: str = Field(..., description="Identifier for the model (e.g., 'claude-opus-4-8')")
    provider: str = Field(..., description="Provider of the model (e.g., 'anthropic')")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")


class PromptTemplate(BaseModel):
    """A specific version of a prompt template."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this version")
    prompt_id: str = Field(..., description="Identifier for the prompt (e.g., 'planner')")
    name: str = Field(..., description="Human-readable name of the prompt")
    version: int = Field(..., description="Version number (starting at 1)")
    template: str = Field(..., description="The prompt template with placeholders")
    variables: List[str] = Field(
        default_factory=list, description="List of variable names used in the template (e.g., ['user_query', 'context'])"
    )
    task_use_case: str = Field(..., description="Description of the task or use-case for this prompt")
    model_configuration: ModelConfiguration = Field(..., description="Model configuration associated with this version")
    few_shot_examples: List["FewShotExample"] = Field(
        default_factory=list, description="Few-shot examples associated with this version"
    )
    status: str = Field(
        default="draft", description="Status of this version (draft, candidate, production, archived)"
    )
    created_at: float = Field(default_factory=time.time, description="Timestamp when this version was created")
    updated_at: float = Field(default_factory=time.time, description="Timestamp when this version was last updated")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (e.g., author, source)"
    )


class FewShotExample(BaseModel):
    """A single example for few-shot learning."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the example")
    prompt_id: str = Field(..., description="Identifier for the prompt this example belongs to")
    version: Optional[int] = Field(
        default=None, description="Version of the prompt this example is associated with (if any)"
    )
    input: str = Field(..., description="Input part of the example")
    expected_output: str = Field(..., description="Expected output for the example")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the example")
    quality_score: Optional[float] = Field(
        default=None, description="Quality score of the example (e.g., from human evaluation)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: float = Field(default_factory=time.time, description="Timestamp when the example was created")
    updated_at: float = Field(default_factory=time.time, description="Timestamp when the example was last updated")


class RegressionTest(BaseModel):
    """A single test case in a regression suite."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the test")
    prompt_id: str = Field(..., description="Identifier for the prompt this test is associated with")
    version: Optional[int] = Field(
        default=None, description="Version of the prompt this test is associated with (if any)"
    )
    input: str = Field(..., description="Input for the test case")
    expected_output: str = Field(..., description="Expected output for the test case")
    evaluation_criteria: Optional[str] = Field(
        default=None, description="Criteria for evaluating the output (e.g., 'exact match', 'contains keyword')"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: float = Field(default_factory=time.time, description="Timestamp when the test was created")
    updated_at: float = Field(default_factory=time.time, description="Timestamp when the test was last updated")


class PromptEvaluationResult(BaseModel):
    """Result of evaluating a prompt version against a regression suite."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the evaluation")
    prompt_id: str = Field(..., description="Identifier for the prompt being evaluated")
    version: int = Field(..., description="Version of the prompt being evaluated")
    total_tests: int = Field(..., description="Total number of tests in the regression suite")
    passed_tests: int = Field(..., description="Number of tests that passed")
    failed_tests: int = Field(..., description="Number of tests that failed")
    score: float = Field(..., description="Overall score (e.g., percentage of tests passed)")
    regression_detected: bool = Field(
        default=False, description="Whether a regression was detected compared to the production version"
    )
    evaluation_timestamp: float = Field(default_factory=time.time, description="Timestamp when the evaluation was run")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the evaluation"
    )


class OutcomeSignal(BaseModel):
    """A signal from the Reflection Layer about prompt performance."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the signal")
    prompt_id: str = Field(..., description="Identifier for the prompt")
    version: int = Field(..., description="Version of the prompt")
    task_id: Optional[str] = Field(
        default=None, description="Identifier for the task (if available)"
    )
    execution_id: Optional[str] = Field(
        default=None, description="Identifier for the execution (if available)"
    )
    outcome: str = Field(..., description="Outcome of the execution (e.g., 'success', 'failure')")
    score: Optional[float] = Field(
        default=None, description="Performance score (e.g., quality metric)"
    )
    latency: Optional[float] = Field(
        default=None, description="Latency of the execution in seconds"
    )
    timestamp: float = Field(default_factory=time.time, description="Timestamp when the signal was generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Exports
__all__ = [
    "MemoryType",
    "MemoryScope",
    "PrivacyClassification",
    "StimulusEnvelope",
    "AgentEnvelope",
    "AgentResult",
    "ClassificationResult",
    "MemoryQuery",
    "MemoryRecord",
    "Goal",
    "TaskStatus",
    "Task",
    "ExecutionLog",
    # Rune models
    "ModelConfiguration",
    "PromptTemplate",
    "FewShotExample",
    "RegressionTest",
    "PromptEvaluationResult",
    "OutcomeSignal",
]