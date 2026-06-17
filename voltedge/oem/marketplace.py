"""
VoltEdge — AWS Marketplace Metering Integration

Handles the SaaS metering dimension reporting required by
AWS Marketplace to enable pay-as-you-go billing.

AWS Marketplace SaaS contracts meter on dimensions:
  - active_sites      (number of monitored sites)
  - api_calls         (thousands of API calls)
  - data_gb           (GB of energy data processed)
  - ml_predictions    (thousands of ML model inferences)

Flow:
  1. Customer subscribes via AWS Marketplace
  2. AWS sends SNS notification with CustomerIdentifier
  3. VoltEdge calls resolve_customer() to validate + get product code
  4. Hourly: batch_meter_usage() reports usage for all active tenants
  5. AWS Marketplace invoices the customer based on reported usage

AWS SDK note:
  - Uses boto3.client("meteringmarketplace")
  - All metering calls are idempotent via UsageRecordTransactionId
  - Dry-run mode for local testing (no real AWS calls)

References:
  https://docs.aws.amazon.com/marketplacemetering/latest/APIReference/
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# ── Marketplace constants ─────────────────────────────────────────────────────

PRODUCT_CODE = "voltedge-energy-platform"   # Set in AWS Marketplace console

DIMENSIONS = {
    "active_sites":   "Number of actively monitored sites (per hour)",
    "api_calls":      "API calls in thousands (per hour)",
    "data_gb":        "Energy data processed in GB (per hour)",
    "ml_predictions": "ML model inferences in thousands (per hour)",
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    """Single metering record for one tenant + dimension."""
    customer_identifier: str
    dimension:           str
    quantity:            int
    timestamp:           datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transaction_id:      str = field(default="")

    def __post_init__(self):
        if not self.transaction_id:
            # Deterministic ID for idempotency
            raw = f"{self.customer_identifier}-{self.dimension}-{self.timestamp.strftime('%Y%m%d%H')}"
            self.transaction_id = hashlib.sha256(raw.encode()).hexdigest()[:32]

    @property
    def is_valid(self) -> bool:
        return (
            bool(self.customer_identifier)
            and self.dimension in DIMENSIONS
            and self.quantity >= 0
        )


@dataclass
class MeteringResult:
    """Result of a batch_meter_usage call."""
    success:          bool
    records_submitted: int
    records_failed:    int
    failed_records:    list[dict] = field(default_factory=list)
    response_raw:      dict       = field(default_factory=dict)
    dry_run:           bool       = False

    @property
    def success_rate(self) -> float:
        total = self.records_submitted + self.records_failed
        return 0.0 if total == 0 else self.records_submitted / total


# ── Metering service ──────────────────────────────────────────────────────────

class MarketplaceMeteringService:
    """
    Reports SaaS usage to AWS Marketplace Metering API.

    Args:
        product_code: AWS Marketplace product code.
        region:       AWS region (default: us-east-1, required by Marketplace).
        dry_run:      Log meter calls without making real AWS API requests.
        aws_access_key_id / aws_secret_access_key: Optional credential override.
    """

    MARKETPLACE_REGION = "us-east-1"   # AWS Marketplace Metering always us-east-1

    def __init__(
        self,
        product_code: str = PRODUCT_CODE,
        region: str = MARKETPLACE_REGION,
        dry_run: bool = False,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self.product_code = product_code
        self.region       = region
        self.dry_run      = dry_run
        self._aws_key     = aws_access_key_id
        self._aws_secret  = aws_secret_access_key
        self._client      = None   # Lazy-init

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve_customer(self, registration_token: str) -> dict[str, str]:
        """
        Exchange an AWS Marketplace registration token for a CustomerIdentifier.
        Called once during customer onboarding (SNS webhook handler).

        Returns:
            {"customer_id": "...", "product_code": "...", "dimension": "..."}

        In dry_run mode, returns a synthetic response.
        """
        if self.dry_run:
            fake_id = hashlib.sha256(registration_token.encode()).hexdigest()[:16]
            result = {
                "customer_id":  f"aws-cust-{fake_id}",
                "product_code": self.product_code,
                "dimension":    "active_sites",
            }
            log.info("marketplace.resolve_customer.dry_run", **result)
            return result

        client = self._get_client()
        resp = client.resolve_customer(RegistrationToken=registration_token)
        return {
            "customer_id":  resp["CustomerIdentifier"],
            "product_code": resp["ProductCode"],
            "dimension":    resp.get("EntitlementData", [{}])[0].get("Dimension", ""),
        }

    def meter_usage(self, record: UsageRecord) -> dict[str, Any]:
        """
        Report a single usage record to AWS Marketplace.

        Returns the AWS API response dict (or a dry-run stub).
        """
        if not record.is_valid:
            raise ValueError(f"Invalid usage record: {record}")

        if self.dry_run:
            result = {
                "MeteringRecordId": record.transaction_id,
                "Status":           "Success",
                "DryRun":           True,
            }
            log.info("marketplace.meter_usage.dry_run",
                     customer=record.customer_identifier,
                     dimension=record.dimension,
                     quantity=record.quantity)
            return result

        client = self._get_client()
        return client.meter_usage(
            ProductCode=self.product_code,
            Timestamp=record.timestamp,
            UsageDimension=record.dimension,
            UsageQuantity=record.quantity,
            DryRun=False,
        )

    def batch_meter_usage(self, records: list[UsageRecord]) -> MeteringResult:
        """
        Submit up to 25 usage records in one API call.
        AWS Marketplace BatchMeterUsage limit is 25 records per call.

        Automatically chunks larger batches.
        """
        if not records:
            return MeteringResult(success=True, records_submitted=0, records_failed=0,
                                   dry_run=self.dry_run)

        # Validate all records first
        valid   = [r for r in records if r.is_valid]
        invalid = [r for r in records if not r.is_valid]

        if self.dry_run:
            log.info("marketplace.batch_meter.dry_run",
                     total=len(records), valid=len(valid), invalid=len(invalid))
            return MeteringResult(
                success=True,
                records_submitted=len(valid),
                records_failed=len(invalid),
                failed_records=[{"record": str(r), "reason": "invalid"} for r in invalid],
                dry_run=True,
            )

        client = self._get_client()
        submitted, failed = 0, len(invalid)
        failed_records = [{"record": str(r), "reason": "invalid"} for r in invalid]

        # Chunk into 25-record batches
        chunk_size = 25
        for i in range(0, len(valid), chunk_size):
            chunk = valid[i : i + chunk_size]
            payload = [
                {
                    "CustomerIdentifier": r.customer_identifier,
                    "Dimensions": [{"Key": r.dimension, "Value": r.quantity}],
                    "Timestamp": r.timestamp,
                }
                for r in chunk
            ]
            try:
                resp = client.batch_meter_usage(
                    UsageRecords=payload,
                    ProductCode=self.product_code,
                )
                submitted += len(resp.get("Results", []))
                for unproc in resp.get("UnprocessedRecords", []):
                    failed += 1
                    failed_records.append({"record": unproc, "reason": "unprocessed"})
            except Exception as exc:
                log.error("marketplace.batch_meter_error", error=str(exc))
                failed += len(chunk)

        return MeteringResult(
            success=failed == 0,
            records_submitted=submitted,
            records_failed=failed,
            failed_records=failed_records,
            dry_run=False,
        )

    # ── Usage calculation helpers ─────────────────────────────────────────────

    @staticmethod
    def build_hourly_records(
        tenant_usage: list[dict[str, Any]],
        hour: datetime | None = None,
    ) -> list[UsageRecord]:
        """
        Build metering records for a billing hour from tenant usage data.

        Args:
            tenant_usage: List of dicts with keys:
                customer_id, sites_count, api_calls_hour, data_gb_hour, ml_calls_hour
            hour: The billing hour (defaults to current UTC hour).

        Returns:
            List of UsageRecord, one per tenant per dimension.
        """
        if hour is None:
            now = datetime.now(timezone.utc)
            hour = now.replace(minute=0, second=0, microsecond=0)

        records: list[UsageRecord] = []
        for usage in tenant_usage:
            cid = usage.get("customer_id", "")
            if not cid:
                continue

            # active_sites: integer count
            sites = int(usage.get("sites_count", 0))
            if sites > 0:
                records.append(UsageRecord(
                    customer_identifier=cid,
                    dimension="active_sites",
                    quantity=sites,
                    timestamp=hour,
                ))

            # api_calls: rounded up to nearest 1000
            api_calls = int(usage.get("api_calls_hour", 0))
            api_thousands = max(1, math.ceil(api_calls / 1000)) if api_calls > 0 else 0
            if api_thousands > 0:
                records.append(UsageRecord(
                    customer_identifier=cid,
                    dimension="api_calls",
                    quantity=api_thousands,
                    timestamp=hour,
                ))

            # data_gb: rounded up to nearest integer GB
            data_gb = math.ceil(float(usage.get("data_gb_hour", 0)))
            if data_gb > 0:
                records.append(UsageRecord(
                    customer_identifier=cid,
                    dimension="data_gb",
                    quantity=data_gb,
                    timestamp=hour,
                ))

            # ml_predictions: rounded up to nearest 1000
            ml_calls = int(usage.get("ml_calls_hour", 0))
            ml_thousands = max(1, math.ceil(ml_calls / 1000)) if ml_calls > 0 else 0
            if ml_thousands > 0:
                records.append(UsageRecord(
                    customer_identifier=cid,
                    dimension="ml_predictions",
                    quantity=ml_thousands,
                    timestamp=hour,
                ))

        return records

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            import boto3
            kwargs: dict[str, Any] = {"region_name": self.region}
            if self._aws_key:
                kwargs["aws_access_key_id"]     = self._aws_key
                kwargs["aws_secret_access_key"] = self._aws_secret
            self._client = boto3.client("meteringmarketplace", **kwargs)
        return self._client
