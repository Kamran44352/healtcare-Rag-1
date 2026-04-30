from __future__ import annotations
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class CodedConcept(BaseModel):
    label: str
    code: str | None = None
    system: Literal["SNOMED-CT", "ICD-10", "ATC", "MeSH", "LOINC", "free-text"] = "free-text"

    @field_validator("system", mode="before")
    @classmethod
    def _normalize_system(cls, v: str) -> str:
        if v == "SNOMED CT":
            return "SNOMED-CT"
        return v


class PopulationScope(BaseModel):
    age_min_years: float | None = None
    age_max_years: float | None = None
    pregnancy: Literal["applicable", "excluded", "specific", "unspecified"] = "unspecified"
    comorbidities_focus: list[str] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    title: str
    document_type: Literal[
        "clinical_guideline", "protocol", "drug_monograph",
        "research_paper", "sop", "patient_leaflet", "review", "other"
    ] = "other"
    issuing_body: str | None = None
    issuing_body_authority: Literal[
        "national", "international", "regional", "local", "academic", "unknown"
    ] = "unknown"
    reference_id: str | None = None
    publication_date: date | None = None
    last_updated: date | None = None
    version: str | None = None
    geographic_scope: list[str] = Field(default_factory=list)
    language: str = "en"
    specialties: list[str] = Field(default_factory=list)
    conditions: list[CodedConcept] = Field(default_factory=list)
    drugs: list[CodedConcept] = Field(default_factory=list)
    procedures: list[CodedConcept] = Field(default_factory=list)
    populations: PopulationScope = Field(default_factory=PopulationScope)
    care_setting: list[
        Literal["primary", "secondary", "tertiary", "community", "emergency", "inpatient", "outpatient"]
    ] = Field(default_factory=list)
    evidence_grading_system: str | None = None
    has_dosing_tables: bool = False
    has_red_flags: bool = False
    has_referral_criteria: bool = False
    summary: str = Field(default="", max_length=280)
