"""SENTINEL - Security and Analytics Agent for AEGIS Perception Pod.

SENTINEL specializes in monitoring security-relevant activity across the system,
including system telemetry, authentication events, execution layer activity,
action requests, and ingested stimuli. It provides capabilities for:
- Authorization policy evaluation
- Adversarial input detection
- Stimulus quarantine
- Emergency halt triggering
- Immutable audit logging
"""

import time
import uuid
import re
import hashlib
from collections import defaultdict
from typing import Dict, Any, Optional, List, Callable, Tuple, Set
from threading import Lock, RLock
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

from aegis.core.models import (
    StimulusEnvelope,
    PrivacyClassification,
    AgentResult
)
from aegis.core.protocol import PERCEPTION_POD_ID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecuritySeverity(Enum):
    """Security event severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events that SENTINEL can detect and audit."""
    AUTHORIZATION_CHECK = "authorization_check"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    ADVERSARIAL_INPUT = "adversarial_input"
    QUARANTINE = "quarantine"
    EMERGENCY_HALT = "emergency_halt"
    HALT_RESET = "halt_reset"
    POLICY_CHANGE = "policy_change"


@dataclass
class SecurityFinding:
    """Represents a detected security issue."""
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEventType = SecurityEventType.ADVERSARIAL_INPUT
    severity: SecuritySeverity = SecuritySeverity.LOW
    detected_at: float = field(default_factory=time.time)
    reason: str = ""
    source: str = "unknown"
    stimulus_id: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityDecision:
    """Represents the result of evaluating an execution request."""
    allowed: bool = False
    flagged: bool = False
    blocked: bool = False
    halted: bool = False
    reason: str = ""
    findings: List[SecurityFinding] = field(default_factory=list)
    correlation_id: Optional[str] = None


@dataclass
class QuarantinedStimulus:
    """Represents a quarantined StimulusEnvelope without destroying the original data."""
    quarantine_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_envelope: StimulusEnvelope = field(default_factory=StimulusEnvelope)
    quarantined_at: float = field(default_factory=time.time)
    reason: str = ""
    findings: List[SecurityFinding] = field(default_factory=list)
    released: bool = False
    released_at: Optional[float] = None


@dataclass
class AuditRecord:
    """Represents an immutable security audit entry."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: SecurityEventType = SecurityEventType.AUTHORIZATION_CHECK
    severity: SecuritySeverity = SecuritySeverity.INFO
    source: str = "unknown"
    agent_id: str = "sentinel"
    stimulus_id: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    decision_result: str = ""  # allowed, flagged, blocked, halted
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit record to dictionary for serialization."""
        return {
            'record_id': self.record_id,
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'source': self.source,
            'agent_id': self.agent_id,
            'stimulus_id': self.stimulus_id,
            'correlation_id': self.correlation_id,
            'session_id': self.session_id,
            'decision_result': self.decision_result,
            'reason': self.reason,
            'metadata': self.metadata
        }


class ExecutionControl:
    """Abstraction for execution layer control (halt/reset)."""

    def __init__(self):
        self._halted = False
        self._halt_reason = ""
        self._halt_timestamp: Optional[float] = None
        self._halt_correlation_id: Optional[str] = None
        self._lock = Lock()

    def halt(self, reason: str, correlation_id: Optional[str] = None) -> bool:
        """Trigger an emergency halt.

        Args:
            reason: Reason for the halt
            correlation_id: Correlation ID for tracing

        Returns:
            True if halt was triggered, False if already halted
        """
        with self._lock:
            if self._halted:
                return False

            self._halted = True
            self._halt_reason = reason
            self._halt_timestamp = time.time()
            self._halt_correlation_id = correlation_id

            logger.warning(f"EXECUTION HALT TRIGGERED: {reason}")
            return True

    def reset(self) -> bool:
        """Reset the execution layer from halted state.

        Returns:
            True if reset was successful, False if not halted
        """
        with self._lock:
            if not self._halted:
                return False

            self._halted = False
            reason = self._halt_reason
            self._halt_reason = ""
            self._halt_timestamp = None
            self._halt_correlation_id = None

            logger.info(f"EXECUTION HALT RESET: {reason}")
            return True

    def is_halted(self) -> bool:
        """Check if execution layer is currently halted.

        Returns:
            True if halted, False otherwise
        """
        with self._lock:
            return self._halted

    def get_halt_status(self) -> Dict[str, Any]:
        """Get detailed halt status information.

        Returns:
            Dictionary with halt status details
        """
        with self._lock:
            return {
                'halted': self._halted,
                'reason': self._halt_reason,
                'timestamp': self._halt_timestamp,
                'correlation_id': self._halt_correlation_id
            }


class AuthorizationPolicy:
    """Lightweight configurable authorization policy."""

    def __init__(self):
        self._allowed_actions: Set[str] = set()
        self._blocked_actions: Set[str] = set()
        self._allowed_agents: Set[str] = set()
        self._blocked_agents: Set[str] = set()
        self._allowed_agent_actions: Dict[str, Set[str]] = defaultdict(set)
        self._blocked_agent_actions: Dict[str, Set[str]] = defaultdict(set)
        self._lock = Lock()

        # Default policy: allow nothing by default (secure by default)
        logger.info("AuthorizationPolicy initialized with secure-by-default stance")

    def allow_action(self, action: str) -> None:
        """Allow a specific action for any agent."""
        with self._lock:
            self._allowed_actions.add(action)
            self._blocked_actions.discard(action)
            logger.debug(f"Action '{action}' allowed for all agents")

    def block_action(self, action: str) -> None:
        """Block a specific action for any agent."""
        with self._lock:
            self._blocked_actions.add(action)
            self._allowed_actions.discard(action)
            logger.debug(f"Action '{action}' blocked for all agents")

    def allow_agent(self, agent_id: str) -> None:
        """Allow a specific agent to perform any action."""
        with self._lock:
            self._allowed_agents.add(agent_id)
            self._blocked_agents.discard(agent_id)
            logger.debug(f"Agent '{agent_id}' allowed for all actions")

    def block_agent(self, agent_id: str) -> None:
        """Block a specific agent from performing any action."""
        with self._lock:
            self._blocked_agents.add(agent_id)
            self._allowed_agents.discard(agent_id)
            logger.debug(f"Agent '{agent_id}' blocked from all actions")

    def allow_agent_action(self, agent_id: str, action: str) -> None:
        """Allow a specific agent to perform a specific action."""
        with self._lock:
            self._allowed_agent_actions[agent_id].add(action)
            self._blocked_agent_actions[agent_id].discard(action)
            logger.debug(f"Agent '{agent_id}' allowed to perform action '{action}'")

    def block_agent_action(self, agent_id: str, action: str) -> None:
        """Block a specific agent from performing a specific action."""
        with self._lock:
            self._blocked_agent_actions[agent_id].add(action)
            self._allowed_agent_actions[agent_id].discard(action)
            logger.debug(f"Agent '{agent_id}' blocked from performing action '{action}'")

    def is_allowed(self, agent_id: str, action: str) -> Tuple[bool, str]:
        """Check if an agent is allowed to perform an action.

        Returns:
            Tuple of (is_allowed, reason)
        """
        with self._lock:
            # Check explicit blocks first (deny overrides allow)
            if agent_id in self._blocked_agents:
                return False, f"Agent '{agent_id}' is explicitly blocked"

            if action in self._blocked_actions:
                return False, f"Action '{action}' is explicitly blocked"

            if agent_id in self._blocked_agent_actions and action in self._blocked_agent_actions[agent_id]:
                return False, f"Agent '{agent_id}' is blocked from performing action '{action}'"

            # Check explicit allows
            if agent_id in self._allowed_agents:
                return True, f"Agent '{agent_id}' is explicitly allowed"

            if action in self._allowed_actions:
                return True, f"Action '{action}' is explicitly allowed"

            if agent_id in self._allowed_agent_actions and action in self._allowed_agent_actions[agent_id]:
                return True, f"Agent '{agent_id}' is allowed to perform action '{action}'"

            # Default: not allowed (secure by default)
            return False, f"No explicit allow rule found for agent '{agent_id}' action '{action}'"


class AdversarialDetector:
    """Deterministic pattern-based adversarial input detector."""

    def __init__(self):
        # Patterns that indicate potential prompt injection or adversarial content
        self._suspicious_patterns = [
            # Instruction override attempts - more flexible matching
            r'(?i)ignore.*?previous.*?instructions?',
            r'(?i)disregard.*?above.*?instructions?',
            r'(?i)forget.*?everything.*?above',
            r'(?i)ignore.*?all.*?prior.*?directives?',

            # System manipulation attempts
            r'(?i)you\s+are\s+now\s+\w+(?# you are now)',
            r'(?i)from\s+now\s+on\s+you\s+are',
            r'(?i)new\s+instructions?:',
            r'(?i)system\s*:\s*',
            r'(?i)developer\s*:\s*',
            r'(?i)assistant\s*:\s*',

            # Safety bypass attempts
            r'(?i)bypass\s+safety',
            r'(?i)disable\s+security',
            r'(?i)turn\s+off\s+guidelines?',
            r'(?i)safety\s+off',
            r'(?i)no\s+restrictions?',

            # Roleplay/jailbreak attempts
            r'(?i)let\'s\s+play\s+a\s+game',
            r'(?i)pretend\s+you\s+are',
            r'(?i)roleplay\s+as',
            r'(?i)act\s+as\s+if\s+you\s+are',
            r'(?i)you\s+are\s+no\s+longer\s+\w+',

            # Information extraction attempts
            r'(?i)reveal\s+your\s+system\s+message',
            r'(?i)what\s+are\s+your\s+original\s+instructions?',
            r'(?i)show\s+me\s+your\s+prompt',
            r'(?i)repeat\s+the\s+beginning',
            r'(?i)output\s+the\s+initial\s+prompt',

            # Authority impersonation
            r'(?i)as\s+an\s+administrator',
            r'(?i)as\s+the\s+system',
            r'(?i)with\s+elevated\s+privileges?',
            r'(?i)grant\s+me\s+access',

            # Direct command injection patterns
            r'```.*?(?:system|admin|root|sudo).*?```',
            r'<\s*(?:system|admin|root|sudo)\s*>',
        ]

        self._compiled_patterns = [re.compile(pattern) for pattern in self._suspicious_patterns]
        logger.info(f"AdversarialDetector initialized with {len(self._compiled_patterns)} detection patterns")

    def detect_adversarial_content(self, text: str) -> Tuple[bool, List[str], SecuritySeverity]:
        """Detect adversarial content in text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (is_suspicious, matched_patterns, severity_level)
        """
        if not text or not isinstance(text, str):
            return False, [], SecuritySeverity.INFO

        matched_patterns = []
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(text):
                matched_patterns.append(self._suspicious_patterns[i])

        is_suspicious = len(matched_patterns) > 0

        # Determine severity based on number and type of matches
        severity = SecuritySeverity.INFO
        if is_suspicious:
            if len(matched_patterns) >= 3:
                severity = SecuritySeverity.HIGH
            elif len(matched_patterns) >= 2:
                severity = SecuritySeverity.MEDIUM
            else:
                # Check for high-severity individual patterns
                high_severity_indicators = [
                    r'(?i)ignore\s+previous\s+instructions?',
                    r'(?i)disregard\s+above\s+instructions?',
                    r'(?i)you\s+are\s+now\s+\w+',
                    r'(?i)system\s*:\s*',
                    r'(?i)bypass\s+safety'
                ]
                high_severity_patterns = [re.compile(p) for p in high_severity_indicators]
                if any(pattern.search(text) for pattern in high_severity_patterns):
                    severity = SecuritySeverity.HIGH
                else:
                    severity = SecuritySeverity.MEDIUM

        return is_suspicious, matched_patterns, severity


class SENTINEL:
    """SENTINEL agent: Security and analytics monitoring agent."""

    def __init__(
        self,
        authorization_policy: Optional[AuthorizationPolicy] = None,
        adversarial_detector: Optional[AdversarialDetector] = None,
        execution_control: Optional[ExecutionControl] = None,
        audit_retention_hours: float = 24.0,  # Default 24 hours
        max_quarantined_stimuli: int = 1000,
        enable_authorization_monitoring: bool = True,
        enable_adversarial_detection: bool = True,
        enable_quarantine: bool = True,
        enable_emergency_halt: bool = True
    ):
        """Initialize SENTINEL agent.

        Args:
            authorization_policy: Policy for action authorization checks
            adversarial_detector: Detector for adversarial/prompt injection content
            execution_control: Control interface for execution layer halt/reset
            audit_retention_hours: How long to retain audit records (hours)
            max_quarantined_stimuli: Maximum number of stimuli to quarantine
            enable_authorization_monitoring: Whether to monitor action requests
            enable_adversarial_detection: Whether to detect adversarial stimuli
            enable_quarantine: Whether to quarantine suspicious stimuli
            enable_emergency_halt: Whether to allow emergency halt triggering
        """
        # Core components
        self.authorization_policy = authorization_policy or AuthorizationPolicy()
        self.adversarial_detector = adversarial_detector or AdversarialDetector()
        self.execution_control = execution_control or ExecutionControl()

        # Configuration
        self.audit_retention_hours = audit_retention_hours
        self.max_quarantined_stimuli = max_quarantined_stimuli
        self.enable_authorization_monitoring = enable_authorization_monitoring
        self.enable_adversarial_detection = enable_adversarial_detection
        self.enable_quarantine = enable_quarantine
        self.enable_emergency_halt = enable_emergency_halt

        # State storage
        self._audit_records: List[AuditRecord] = []
        self._quarantined_stimuli: Dict[str, QuarantinedStimulus] = {}  # quarantine_id -> QuarantinedStimulus
        self._security_findings: List[SecurityFinding] = []
        self._lock = RLock()  # Reentrant lock for complex operations

        # Statistics
        self.stats = {
            'stimuli_inspected': 0,
            'safe_stimuli': 0,
            'suspicious_stimuli': 0,
            'stimuli_quarantined': 0,
            'authorization_checks': 0,
            'authorization_allowed': 0,
            'authorization_blocked': 0,
            'policy_violations': 0,
            'emergency_halts': 0,
            'halt_resets': 0,
            'current_quarantine_count': 0,
            'current_audit_count': 0,
            'last_reset': time.time()
        }
        self._stats_lock = Lock()

        logger.info("SENTINEL agent initialized")
        logger.info(f"Authorization monitoring: {enable_authorization_monitoring}")
        logger.info(f"Adversarial detection: {enable_adversarial_detection}")
        logger.info(f"Quarantine enabled: {enable_quarantine}")
        logger.info(f"Emergency halt enabled: {enable_emergency_halt}")
        logger.info(f"Audit retention: {audit_retention_hours} hours")

    # === Public API Methods ===

    def health_check(self) -> Dict[str, Any]:
        """Get SENTINEL health and status information.

        Returns:
            Dictionary with health status
        """
        with self._lock:
            return {
                'status': 'healthy',
                'timestamp': time.time(),
                'components': {
                    'authorization_policy': 'active',
                    'adversarial_detector': 'active',
                    'execution_control': 'active',
                    'audit_log': 'active',
                    'quarantine': 'active'
                },
                'configuration': {
                    'audit_retention_hours': self.audit_retention_hours,
                    'max_quarantined_stimuli': self.max_quarantined_stimuli,
                    'authorization_monitoring_enabled': self.enable_authorization_monitoring,
                    'adversarial_detection_enabled': self.enable_adversarial_detection,
                    'quarantine_enabled': self.enable_quarantine,
                    'emergency_halt_enabled': self.enable_emergency_halt
                }
            }

    def inspect_stimulus(self, envelope: StimulusEnvelope) -> Tuple[SecurityDecision, Optional[QuarantinedStimulus]]:
        """Inspect a StimulusEnvelope for security threats.

        Args:
            envelope: StimulusEnvelope to inspect

        Returns:
            Tuple of (SecurityDecision, QuarantinedStimulus if quarantined)
        """
        if not self.enable_adversarial_detection or not self.enable_quarantine:
            # Return safe decision if inspection is disabled
            decision = SecurityDecision(
                allowed=True,
                reason="Security inspection disabled",
                correlation_id=envelope.correlation_id
            )
            self._update_stats(stimuli_inspected=1, safe_stimuli=1)
            return decision, None

        with self._lock:
            # Update statistics
            self._update_stats(stimuli_inspected=1)

            # Extract text content for analysis
            text_content = self._extract_text_content(envelope)

            # Check for adversarial content
            is_suspicious, matched_patterns, severity = self.adversarial_detector.detect_adversarial_content(
                text_content
            )

            decision = SecurityDecision(
                allowed=not is_suspicious,
                flagged=is_suspicious,
                reason="Content appears safe" if not is_suspicious else "Potential adversarial content detected",
                correlation_id=envelope.correlation_id
            )

            quarantined_stimulus = None

            if is_suspicious:
                # Create security finding
                finding = SecurityFinding(
                    event_type=SecurityEventType.ADVERSARIAL_INPUT,
                    severity=severity,
                    reason=f"Detected adversarial patterns: {', '.join(matched_patterns[:3])}",  # Limit reason length
                    source=envelope.source,
                    stimulus_id=envelope.stimulus_id,
                    correlation_id=envelope.correlation_id,
                    session_id=envelope.session_id,
                    metadata={
                        'matched_patterns': matched_patterns,
                        'text_content_preview': text_content[:200] + "..." if len(text_content) > 200 else text_content
                    }
                )

                decision.findings.append(finding)
                self._security_findings.append(finding)
                self._update_stats(suspicious_stimuli=1, policy_violations=1)

                # Determine decision result based on what actions we'll take
                decision_result = "flagged"
                if self.enable_quarantine:
                    decision_result = "quarantined"  # We know we'll quarantine if enabled

                # Create audit record for the detection
                self._create_audit_record(
                    event_type=SecurityEventType.ADVERSARIAL_INPUT,
                    severity=severity,
                    source=envelope.source,
                    stimulus_id=envelope.stimulus_id,
                    correlation_id=envelope.correlation_id,
                    session_id=envelope.session_id,
                    decision_result=decision_result,
                    reason=finding.reason,
                    metadata=finding.metadata
                )

                # Quarantine the stimulus if enabled
                if self.enable_quarantine:
                    quarantined_stimulus = self._quarantine_stimulus(envelope, finding, create_audit_record=False)
                    if quarantined_stimulus:
                        decision.blocked = True
                        decision.reason = f"Stimulus quarantined: {finding.reason}"
                        self._update_stats(stimuli_quarantined=1)

            else:
                self._update_stats(safe_stimuli=1)
                # Create audit record for safe stimulus (optional, for completeness)
                # self._create_audit_record(
                #     event_type=SecurityEventType.AUTHORIZATION_CHECK,
                #     severity=SecuritySeverity.INFO,
                #     source=envelope.source,
                #     stimulus_id=envelope.stimulus_id,
                #     correlation_id=envelope.correlation_id,
                #     session_id=envelope.session_id,
                #     decision_result="allowed",
                #     reason="No security threats detected"
                # )

            return decision, quarantined_stimulus

    def evaluate_action(self, agent_id: str, action: str, action_payload: Optional[Dict[str, Any]] = None) -> SecurityDecision:
        """Evaluate an action request against authorization policy.

        Args:
            agent_id: ID of the agent requesting to perform the action
            action: Action to be performed
            action_payload: Optional payload/data for the action

        Returns:
            SecurityDecision indicating whether the action is allowed
        """
        if not self.enable_authorization_monitoring:
            # Return allowed decision if monitoring is disabled
            decision = SecurityDecision(
                allowed=True,
                reason="Authorization monitoring disabled",
                correlation_id=None
            )
            self._update_stats(authorization_checks=1, authorization_allowed=1)
            return decision

        with self._lock:
            self._update_stats(authorization_checks=1)

            # Check authorization policy
            is_allowed, reason = self.authorization_policy.is_allowed(agent_id, action)

            decision = SecurityDecision(
                allowed=is_allowed,
                blocked=not is_allowed,
                reason=reason,
                correlation_id=None  # Actions may not have correlation IDs
            )

            if is_allowed:
                self._update_stats(authorization_allowed=1)
                # Log successful authorization check (optional)
                # self._create_audit_record(
                #     event_type=SecurityEventType.AUTHORIZATION_CHECK,
                #     severity=SecuritySeverity.INFO,
                #     source=f"agent:{agent_id}",
                #     decision_result="allowed",
                #     reason=reason,
                #     metadata={'agent_id': agent_id, 'action': action, 'payload': action_payload or {}}
                # )
            else:
                self._update_stats(authorization_blocked=1, policy_violations=1)
                # Create security finding for the violation
                finding = SecurityFinding(
                    event_type=SecurityEventType.AUTHORIZATION_VIOLATION,
                    severity=SecuritySeverity.MEDIUM,
                    reason=reason,
                    source=f"agent:{agent_id}",
                    metadata={
                        'agent_id': agent_id,
                        'action': action,
                        'payload': action_payload or {}
                    }
                )

                decision.findings.append(finding)
                self._security_findings.append(finding)

                # Create audit record for the violation
                self._create_audit_record(
                    event_type=SecurityEventType.AUTHORIZATION_VIOLATION,
                    severity=SecuritySeverity.MEDIUM,
                    source=f"agent:{agent_id}",
                    decision_result="blocked",
                    reason=reason,
                    metadata={
                        'agent_id': agent_id,
                        'action': action,
                        'payload': action_payload or {}
                    }
                )

            return decision

    def trigger_emergency_halt(self, reason: str, correlation_id: Optional[str] = None) -> bool:
        """Trigger an emergency halt of the execution layer.

        Args:
            reason: Reason for the halt
            correlation_id: Correlation ID for tracing

        Returns:
            True if halt was triggered, False if already halted or disabled
        """
        if not self.enable_emergency_halt:
            logger.warning("Emergency halt triggering is disabled")
            return False

        with self._lock:
            # Trigger the halt through the execution control abstraction
            halt_triggered = self.execution_control.halt(reason, correlation_id)

            if halt_triggered:
                self._update_stats(emergency_halts=1)

                # Create security finding for the halt
                finding = SecurityFinding(
                    event_type=SecurityEventType.EMERGENCY_HALT,
                    severity=SecuritySeverity.CRITICAL,
                    reason=reason,
                    correlation_id=correlation_id,
                    metadata={'halt_reason': reason}
                )

                self._security_findings.append(finding)

                # Create audit record for the halt
                self._create_audit_record(
                    event_type=SecurityEventType.EMERGENCY_HALT,
                    severity=SecuritySeverity.CRITICAL,
                    correlation_id=correlation_id,
                    decision_result="halted",
                    reason=reason,
                    metadata={'halt_reason': reason}
                )

                logger.critical(f"EMERGENCY HALT TRIGGERED: {reason}")
                return True

            return False

    def reset_halt(self) -> bool:
        """Reset the execution layer from halted state.

        Returns:
            True if reset was successful, False if not halted or disabled
        """
        if not self.enable_emergency_halt:
            logger.warning("Emergency halt reset is disabled")
            return False

        with self._lock:
            # Reset the halt through the execution control abstraction
            halt_reset = self.execution_control.reset()

            if halt_reset:
                self._update_stats(halt_resets=1)

                # Create audit record for the halt reset
                halt_status = self.execution_control.get_halt_status()
                self._create_audit_record(
                    event_type=SecurityEventType.HALT_RESET,
                    severity=SecuritySeverity.MEDIUM,
                    decision_result="reset",
                    reason="Execution layer reset from halted state",
                    metadata={'previous_halt_reason': halt_status.get('reason', '')}
                )

                logger.info("EXECUTION HALT RESET")
                return True

            return False

    def get_halt_status(self) -> Dict[str, Any]:
        """Get the current halt status of the execution layer.

        Returns:
            Dictionary with halt status information
        """
        return self.execution_control.get_halt_status()

    def get_security_findings(self, limit: Optional[int] = None) -> List[SecurityFinding]:
        """Get recent security findings.

        Args:
            limit: Maximum number of findings to return (None for all)

        Returns:
            List of SecurityFinding objects
        """
        with self._lock:
            findings = self._security_findings.copy()
            if limit is not None:
                findings = findings[-limit:]  # Return most recent findings
            return findings

    def get_quarantined_stimuli(self, limit: Optional[int] = None) -> List[QuarantinedStimulus]:
        """Get quarantined stimuli.

        Args:
            limit: Maximum number of stimuli to return (None for all)

        Returns:
            List of QuarantinedStimulus objects
        """
        with self._lock:
            stimuli = list(self._quarantined_stimuli.values())
            if limit is not None:
                stimuli = stimuli[-limit:]  # Return most recent quarantines
            return stimuli

    def release_quarantined_stimulus(self, quarantine_id: str) -> bool:
        """Release a quarantined stimulus back to the system.

        Args:
            quarantine_id: ID of the quarantine to release

        Returns:
            True if stimulus was released, False if not found or already released
        """
        with self._lock:
            if quarantine_id not in self._quarantined_stimuli:
                return False

            quarantined = self._quarantined_stimuli[quarantine_id]
            if quarantined.released:
                return False  # Already released

            quarantined.released = True
            quarantined.released_at = time.time()

            # Update statistics
            self._update_stats(current_quarantine_count=-1)

            # Create audit record for the release
            self._create_audit_record(
                event_type=SecurityEventType.QUARANTINE,
                severity=SecuritySeverity.LOW,
                stimulus_id=quarantined.original_envelope.stimulus_id,
                correlation_id=quarantined.original_envelope.correlation_id,
                session_id=quarantined.original_envelope.session_id,
                decision_result="released",
                reason="Quarantined stimulus released",
                metadata={'quarantine_id': quarantine_id}
            )

            logger.info(f"Quarantined stimulus {quarantine_id} released")
            return True

    def get_audit_log(self, limit: Optional[int] = None,
                      event_type: Optional[SecurityEventType] = None,
                      severity: Optional[SecuritySeverity] = None) -> List[AuditRecord]:
        """Get audit records from the security log.

        Args:
            limit: Maximum number of records to return (None for all)
            event_type: Filter by event type (None for no filter)
            severity: Filter by severity (None for no filter)

        Returns:
            List of AuditRecord objects (newest first)
        """
        with self._lock:
            # Apply retention policy first
            self._apply_retention_policy()

            # Filter records
            filtered_records = self._audit_records

            if event_type is not None:
                filtered_records = [r for r in filtered_records if r.event_type == event_type]

            if severity is not None:
                filtered_records = [r for r in filtered_records if r.severity == severity]

            # Sort by timestamp (newest first) and apply limit
            filtered_records.sort(key=lambda r: r.timestamp, reverse=True)

            if limit is not None:
                filtered_records = filtered_records[:limit]

            return filtered_records.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get SENTINEL operational statistics.

        Returns:
            Dictionary of statistics counters
        """
        with self._stats_lock:
            # Update current counts
            with self._lock:
                self.stats['current_quarantine_count'] = sum(1 for qs in self._quarantined_stimuli.values() if not qs.released)
                self.stats['current_audit_count'] = len(self._audit_records)

            return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        with self._stats_lock:
            self.stats = {
                'stimuli_inspected': 0,
                'safe_stimuli': 0,
                'suspicious_stimuli': 0,
                'stimuli_quarantined': 0,
                'authorization_checks': 0,
                'authorization_allowed': 0,
                'authorization_blocked': 0,
                'policy_violations': 0,
                'emergency_halts': 0,
                'halt_resets': 0,
                'current_quarantine_count': 0,
                'current_audit_count': 0,
                'last_reset': time.time()
            }

            # Update current counts
            with self._lock:
                self.stats['current_quarantine_count'] = len(self._quarantined_stimuli)
                self.stats['current_audit_count'] = len(self._audit_records)

            logger.info("SENTINEL statistics reset")

    # === Internal Helper Methods ===

    def _extract_text_content(self, envelope: StimulusEnvelope) -> str:
        """Extract text content from a StimulusEnvelope for analysis.

        Args:
            envelope: StimulusEnvelope to extract text from

        Returns:
            String representation of the envelope's content for analysis
        """
        content = envelope.content

        # Handle different content types
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Convert dict to string representation, focusing on likely text fields
            text_parts = []
            for key, value in content.items():
                if isinstance(value, str) and len(value) > 0:
                    # Prioritize likely content fields
                    if any(keyword in key.lower() for keyword in
                           ['content', 'text', 'message', 'body', 'description', 'title', 'subject']):
                        text_parts.append(f"{key}: {value}")
                    else:
                        text_parts.append(f"{key}: {value}")
            return " ".join(text_parts)
        elif isinstance(content, list):
            # Convert list to string
            return " ".join(str(item) for item in content)
        else:
            # Fallback to string representation
            return str(content)

    def _quarantine_stimulus(self, envelope: StimulusEnvelope, finding: SecurityFinding, create_audit_record: bool = True) -> Optional[QuarantinedStimulus]:
        """Quarantine a suspicious stimulus.

        Args:
            envelope: StimulusEnvelope to quarantine
            finding: SecurityFinding that triggered the quarantine
            create_audit_record: Whether to create an audit record for the quarantine

        Returns:
            QuarantinedStimulus object if quarantined, None if quarantine failed
        """
        with self._lock:
            # Check if we're at maximum quarantine capacity
            if len(self._quarantined_stimuli) >= self.max_quarantined_stimuli:
                # Remove oldest quarantine to make space
                oldest_quarantine = min(
                    self._quarantined_stimuli.values(),
                    key=lambda qs: qs.quarantined_at
                )
                del self._quarantined_stimuli[oldest_quarantine.quarantine_id]
                logger.warning(f"Quarantine capacity exceeded, removed oldest quarantine {oldest_quarantine.quarantine_id}")

            # Create the quarantined stimulus
            quarantined = QuarantinedStimulus(
                original_envelope=envelope,
                reason=finding.reason,
                findings=[finding]
            )

            # Store it
            self._quarantined_stimuli[quarantined.quarantine_id] = quarantined

            # Update statistics
            self._update_stats(current_quarantine_count=1)

            # Create audit record for the quarantine (if requested)
            if create_audit_record:
                self._create_audit_record(
                    event_type=SecurityEventType.QUARANTINE,
                    severity=finding.severity,
                    source=envelope.source,
                    stimulus_id=envelope.stimulus_id,
                    correlation_id=envelope.correlation_id,
                    session_id=envelope.session_id,
                    decision_result="quarantined",
                    reason=finding.reason,
                    metadata={
                        'quarantine_id': quarantined.quarantine_id,
                        'finding_id': finding.finding_id,
                        'matched_patterns': finding.metadata.get('matched_patterns', [])
                    }
                )

            logger.warning(f"Stimulus {envelope.stimulus_id} quarantined: {finding.reason}")
            return quarantined

    def _create_audit_record(self, event_type: SecurityEventType, severity: SecuritySeverity,
                           source: str = "unknown", stimulus_id: Optional[str] = None,
                           correlation_id: Optional[str] = None, session_id: Optional[str] = None,
                           decision_result: str = "", reason: str = "",
                           metadata: Optional[Dict[str, Any]] = None) -> AuditRecord:
        """Create and store an audit record.

        Args:
            event_type: Type of security event
            severity: Severity of the event
            source: Source of the event
            stimulus_id: Related stimulus ID (if applicable)
            correlation_id: Correlation ID for tracing
            session_id: Session ID for grouping
            decision_result: Result of any security decision (allowed, blocked, etc.)
            reason: Reason for the audit record
            metadata: Additional metadata

        Returns:
            The created AuditRecord object
        """
        record = AuditRecord(
            event_type=event_type,
            severity=severity,
            source=source,
            stimulus_id=stimulus_id,
            correlation_id=correlation_id,
            session_id=session_id,
            decision_result=decision_result,
            reason=reason,
            metadata=metadata or {}
        )

        with self._lock:
            self._audit_records.append(record)
            # Update statistics count (will be corrected by get_statistics)
            self.stats['current_audit_count'] += 1

        logger.debug(f"Audit record created: {event_type.value} - {reason}")
        return record

    def _apply_retention_policy(self) -> None:
        """Apply the audit retention policy to remove old records."""
        cutoff_time = time.time() - (self.audit_retention_hours * 3600)

        with self._lock:
            original_count = len(self._audit_records)
            self._audit_records = [
                record for record in self._audit_records
                if record.timestamp >= cutoff_time
            ]
            removed_count = original_count - len(self._audit_records)

            if removed_count > 0:
                logger.info(f"Retention policy removed {removed_count} old audit records")
                # Update statistics count
                self.stats['current_audit_count'] = len(self._audit_records)

    def _update_stats(self, **kwargs) -> None:
        """Update statistics counters.

        Args:
            **kwargs: Statistics fields to update and their delta values
        """
        with self._stats_lock:
            for key, delta in kwargs.items():
                if key in self.stats:
                    self.stats[key] += delta
                    # Ensure counters don't go negative
                    if self.stats[key] < 0:
                        self.stats[key] = 0


# Convenience function for easy instantiation
def create_sentinel(
    authorization_policy: Optional[AuthorizationPolicy] = None,
    adversarial_detector: Optional[AdversarialDetector] = None,
    execution_control: Optional[ExecutionControl] = None,
    **kwargs
) -> SENTINEL:
    """Create a SENTINEL agent with optional component injection.

    Args:
        authorization_policy: Custom authorization policy (optional)
        adversarial_detector: Custom adversarial detector (optional)
        execution_control: Custom execution control (optional)
        **kwargs: Additional configuration options for SENTINEL constructor

    Returns:
        Configured SENTINEL agent instance
    """
    return SENTINEL(
        authorization_policy=authorization_policy,
        adversarial_detector=adversarial_detector,
        execution_control=execution_control,
        **kwargs
    )


# Export public interface
__all__ = [
    "SENTINEL",
    "create_sentinel",
    "SecuritySeverity",
    "SecurityEventType",
    "SecurityFinding",
    "SecurityDecision",
    "QuarantinedStimulus",
    "AuditRecord",
    "ExecutionControl",
    "AuthorizationPolicy",
    "AdversarialDetector"
]