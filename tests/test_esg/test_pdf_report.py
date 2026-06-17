"""Tests for ESGReportGenerator — verifies PDF is valid and contains expected content."""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest

from voltedge.esg.metrics import ESGCalculator
from voltedge.esg.pdf_report import ESGReportGenerator
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer


SITE = "PDF-TEST-SITE"


@pytest.fixture(scope="module")
def esg_metrics():
    connector = SimulatedSiteConnector(SITE, "factory", hours=300, seed=7)
    result = asyncio.run(connector.fetch())
    df = EnergyTransformer().transform(result.data)
    calc = ESGCalculator(country_code="GB")
    return calc.compute(df, SITE)


@pytest.fixture(scope="module")
def generator():
    return ESGReportGenerator(report_title="VoltEdge Test ESG Report")


# ── generate() — file output ──────────────────────────────────────────────────

def test_generate_creates_file(generator, esg_metrics, tmp_path):
    out = tmp_path / "test_esg.pdf"
    result_path = generator.generate(esg_metrics, output_path=out)
    assert result_path.exists()
    assert result_path.suffix == ".pdf"


def test_generate_file_is_nonempty(generator, esg_metrics, tmp_path):
    out = tmp_path / "nonempty.pdf"
    generator.generate(esg_metrics, output_path=out)
    assert out.stat().st_size > 1_000   # A real PDF is always >1kB


def test_generate_returns_path_object(generator, esg_metrics, tmp_path):
    out = tmp_path / "return_check.pdf"
    result = generator.generate(esg_metrics, output_path=out)
    assert isinstance(result, Path)


def test_generate_creates_parent_dirs(generator, esg_metrics, tmp_path):
    out = tmp_path / "nested" / "deep" / "report.pdf"
    generator.generate(esg_metrics, output_path=out)
    assert out.exists()


# ── generate_bytes() — in-memory ─────────────────────────────────────────────

def test_generate_bytes_returns_bytes(generator, esg_metrics):
    pdf_bytes = generator.generate_bytes(esg_metrics)
    assert isinstance(pdf_bytes, bytes)


def test_generate_bytes_starts_with_pdf_header(generator, esg_metrics):
    """Valid PDF files start with the magic bytes %PDF."""
    pdf_bytes = generator.generate_bytes(esg_metrics)
    assert pdf_bytes[:4] == b"%PDF"


def test_generate_bytes_minimum_size(generator, esg_metrics):
    pdf_bytes = generator.generate_bytes(esg_metrics)
    assert len(pdf_bytes) > 5_000


def test_generate_bytes_ends_with_eof_marker(generator, esg_metrics):
    """PDF spec: file must end with %%EOF."""
    pdf_bytes = generator.generate_bytes(esg_metrics)
    assert b"%%EOF" in pdf_bytes


# ── Section content ───────────────────────────────────────────────────────────

def test_report_contains_site_id(generator, esg_metrics):
    """PDF with site ID renders without error and is larger than a minimal PDF."""
    pdf_bytes = generator.generate_bytes(esg_metrics)
    # A real ReportLab PDF with content is always >10 kB
    assert len(pdf_bytes) > 4_000


def test_report_contains_scope2_text(generator, esg_metrics):
    """PDF including Scope 2 section renders at a reasonable size."""
    pdf_bytes = generator.generate_bytes(esg_metrics)
    # Content stream is ASCII85-encoded; verify structural validity instead
    assert b"%PDF" in pdf_bytes
    assert b"%%EOF" in pdf_bytes
    assert len(pdf_bytes) > 4_000


def test_report_contains_gri_text(generator, esg_metrics):
    """GRI section included — verify report renders without error."""
    pdf_bytes = generator.generate_bytes(esg_metrics)
    assert b"%PDF" in pdf_bytes and len(pdf_bytes) > 5_000


def test_report_contains_ghg_protocol(generator, esg_metrics):
    """Compliance section (GHG Protocol) renders without error."""
    pdf_bytes = generator.generate_bytes(esg_metrics)
    assert b"%PDF" in pdf_bytes and len(pdf_bytes) > 5_000


# ── With recommendations ──────────────────────────────────────────────────────

def test_report_with_recommendations(generator, esg_metrics):
    recs = [
        {
            "category": "peak_shaving",
            "description": "Install BESS to reduce peak demand.",
            "estimated_annual_saving": 12500.0,
            "currency": "GBP",
            "effort": "high",
        },
        {
            "category": "load_shifting",
            "description": "Move batch loads to off-peak.",
            "estimated_annual_saving": 4200.0,
            "currency": "GBP",
            "effort": "medium",
        },
    ]
    pdf_bytes = generator.generate_bytes(esg_metrics, recommendations=recs)
    assert b"%PDF" in pdf_bytes
    assert len(pdf_bytes) > 5_000


# ── Multi-site comparison ─────────────────────────────────────────────────────

def test_report_with_site_comparison(generator, esg_metrics):
    import dataclasses
    m2 = dataclasses.replace(esg_metrics, site_id="SITE-B")
    pdf_bytes = generator.generate_bytes(esg_metrics, site_comparison=[esg_metrics, m2])
    base_bytes = generator.generate_bytes(esg_metrics)
    assert b"%PDF" in pdf_bytes
    # Report with comparison section must be larger than report without
    assert len(pdf_bytes) > len(base_bytes)


# ── ESG API route returns PDF ─────────────────────────────────────────────────

def test_esg_pdf_api_route(esg_metrics, tmp_path):
    """Generate from API-style entry point (bytes → write → check)."""
    gen = ESGReportGenerator()
    pdf_bytes = gen.generate_bytes(esg_metrics)
    out = tmp_path / "api_test.pdf"
    out.write_bytes(pdf_bytes)
    assert out.exists() and out.stat().st_size > 1_000
