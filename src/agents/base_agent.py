"""
Base Agent — shared infrastructure for all agents.

Provides:
  - Reasoning trace (step-by-step log of decisions)
  - Structured result envelope
  - Timing
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningStep:
    step: str
    description: str
    result: Any
    confidence: Optional[float] = None


@dataclass
class AgentResult:
    agent_name: str
    run_id: str
    status: str                          # "success" | "error" | "skipped"
    output: Any
    reasoning_trace: List[ReasoningStep] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def add_step(self, step: str, description: str, result: Any, confidence: Optional[float] = None):
        self.reasoning_trace.append(ReasoningStep(step, description, result, confidence))

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent_name,
            "run_id": self.run_id,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "output": self.output,
            "reasoning_trace": [
                {
                    "step": s.step,
                    "description": s.description,
                    "result": s.result,
                    "confidence": s.confidence,
                }
                for s in self.reasoning_trace
            ],
            "error": self.error,
        }


class BaseAgent:
    """
    Base class for all pipeline agents.

    Subclasses must implement `run()`.
    """

    def __init__(self, name: str):
        self.name = name

    def _new_result(self, run_id: Optional[str] = None) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            run_id=run_id or str(uuid.uuid4()),
            status="success",
            output=None,
        )

    def run(self, *args, **kwargs) -> AgentResult:
        raise NotImplementedError
