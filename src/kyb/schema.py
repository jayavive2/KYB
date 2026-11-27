"""Data shapes shared across the pipeline.

`CategoryOutput` is the contract with the model: it is what every candidate model
must return for each risk category, and what the harness validates.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Company(BaseModel):
    """The entity being screened. Everything except `name` helps tell it apart from namesakes."""

    name: str = Field(description="Name used in searches, matched exactly")
    aliases: list[str] = Field(default_factory=list)
    country: str | None = None
    website: str | None = None
    industry: str | None = None
    notes: str | None = None

    def identity_block(self) -> str:
        lines = [f"- Name: {self.name}"]
        if self.aliases:
            lines.append(f"- Also known as: {', '.join(self.aliases)}")
        for label, value in (
            ("Country", self.country),
            ("Website", self.website),
            ("Industry", self.industry),
            ("Notes", self.notes),
        ):
            if value:
                lines.append(f"- {label}: {value}")
        return "\n".join(lines)


def _lower(value: object) -> object:
    return value.strip().lower() if isinstance(value, str) else value


class Finding(BaseModel):
    summary: str
    event_date: str | None = Field(default=None, description="YYYY, YYYY-MM or YYYY-MM-DD")
    severity: Literal["high", "medium", "low"]
    status: Literal["alleged", "confirmed", "resolved", "unknown"] = "unknown"
    source_ids: list[str]

    _norm = field_validator("severity", "status", mode="before")(_lower)


class RejectedSource(BaseModel):
    source_id: str
    reason: Literal["different_entity", "not_adverse", "outside_window", "not_relevant"]

    _norm = field_validator("reason", mode="before")(_lower)


class CategoryOutput(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    rejected_sources: list[RejectedSource] = Field(default_factory=list)


CATEGORY_OUTPUT_EXAMPLE = """{
  "findings": [
    {
      "summary": "One or two sentences: what happened, who acted, amounts if stated.",
      "event_date": "2024-10",
      "severity": "high",
      "status": "confirmed",
      "source_ids": ["S3", "S7"]
    }
  ],
  "rejected_sources": [
    {"source_id": "S5", "reason": "different_entity"}
  ]
}"""
