from collections.abc import Callable
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.federation_ingest import (
    FederationIngestPlan,
    FederationIngestPlanDocument,
    FederationIngestPlanEntry,
    FederationIngestValidationError,
    plan_federation_ingest,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.submission_registries import SubmissionRegistry, SubmissionRegistrySource


def test_federation_ingest_plan_derives_from_registry() -> None:
    registry = _registry()
    bundle = _publication_bundle_reference()

    plan = plan_federation_ingest(
        id=ProtocolIdentifier.parse("federation-ingest-plans.public-sources@0.1.0"),
        registry=registry,
        expected_publication_bundles=(bundle,),
    )

    assert plan == FederationIngestPlan(
        id=ProtocolIdentifier.parse("federation-ingest-plans.public-sources@0.1.0"),
        registry_digest=registry.digest,
        entries=(
            FederationIngestPlanEntry(
                source=SubmissionRegistrySource(
                    repository="maximumcats/disabled-source",
                    repository_type="dataset",
                    enabled=False,
                ),
                status="inactive",
                expected_publication_bundles=(),
            ),
            FederationIngestPlanEntry(
                source=SubmissionRegistrySource(
                    repository="maximumcats/leibniz-submissions",
                    repository_type="dataset",
                    enabled=True,
                ),
                status="would-inspect",
                expected_publication_bundles=(bundle,),
            ),
        ),
        dry_run=True,
    )
    assert plan.to_record() == {
        "id": "federation-ingest-plans.public-sources@0.1.0",
        "registry_digest": str(registry.digest),
        "dry_run": True,
        "entries": [
            {
                "repository": "maximumcats/disabled-source",
                "repository_type": "dataset",
                "enabled": False,
                "status": "inactive",
                "expected_publication_bundles": [],
            },
            {
                "repository": "maximumcats/leibniz-submissions",
                "repository_type": "dataset",
                "enabled": True,
                "status": "would-inspect",
                "expected_publication_bundles": [bundle.to_record()],
            },
        ],
    }
    assert plan.digest == ContentDigest.from_value(plan.to_record())


def test_federation_ingest_plan_document_loads_bytes_with_digest() -> None:
    registry = _registry()
    plan = plan_federation_ingest(
        id=ProtocolIdentifier.parse("federation-ingest-plans.public-sources@0.1.0"),
        registry=registry,
        expected_publication_bundles=(_publication_bundle_reference(),),
    )

    document = FederationIngestPlanDocument.from_bytes(
        canonical_document_bytes(plan.to_record()),
        registry=registry,
    )

    assert document.plan == plan
    assert document.digest == ContentDigest.from_value(plan.to_record())


def test_federation_ingest_plan_records_discovered_local_publication_bundles() -> None:
    source = SubmissionRegistrySource(
        repository="maximumcats/leibniz-submissions",
        repository_type="dataset",
        enabled=True,
    )
    expected = _publication_bundle_reference()
    discovered = ArtifactReference.from_record(
        {
            "kind": "publication-bundle",
            "record_digest": str(ContentDigest.from_value({"publication": "local"})),
        }
    )

    entry = FederationIngestPlanEntry(
        source=source,
        status="would-inspect",
        expected_publication_bundles=(expected,),
        discovered_publication_bundles=(discovered,),
    )

    assert entry.to_record() == {
        "repository": "maximumcats/leibniz-submissions",
        "repository_type": "dataset",
        "enabled": True,
        "status": "would-inspect",
        "expected_publication_bundles": [expected.to_record()],
        "discovered_publication_bundles": [discovered.to_record()],
    }


def test_federation_ingest_plan_validates_registry_sources() -> None:
    registry = _registry()
    plan = plan_federation_ingest(
        id=ProtocolIdentifier.parse("federation-ingest-plans.public-sources@0.1.0"),
        registry=registry,
        expected_publication_bundles=(_publication_bundle_reference(),),
    )
    altered_record = plan.to_record()
    altered_record["registry_digest"] = str(ContentDigest.from_value({"other": True}))
    assert str(
        capture_ingest_error(
            lambda: FederationIngestPlan.from_record(altered_record, registry=registry)
        )
    ) == "registry_digest does not match registry"

    altered_record = plan.to_record()
    entries = list(_entry_records(altered_record))
    entries[0]["repository"] = "maximumcats/other-source"
    altered_record["entries"] = entries
    assert str(
        capture_ingest_error(
            lambda: FederationIngestPlan.from_record(altered_record, registry=registry)
        )
    ) == "entries do not match registry sources"


def test_federation_ingest_plan_rejects_missing_and_duplicate_bundle_refs() -> None:
    assert str(
        capture_ingest_error(
            lambda: plan_federation_ingest(
                id=ProtocolIdentifier.parse("federation-ingest-plans.public-sources@0.1.0"),
                registry=_registry(),
                expected_publication_bundles=(),
            )
        )
    ) == "expected_publication_bundles must contain at least one reference"

    entry = _enabled_entry()
    assert str(
        capture_ingest_error(
            lambda: FederationIngestPlanEntry(
                source=entry.source,
                status="would-inspect",
                expected_publication_bundles=(),
            )
        )
    ) == "enabled entries must contain expected publication bundles"

    duplicate = _publication_bundle_reference()
    assert str(
        capture_ingest_error(
            lambda: FederationIngestPlanEntry(
                source=entry.source,
                status="would-inspect",
                expected_publication_bundles=(duplicate, duplicate),
            )
        )
    ).startswith("duplicate expected publication bundle reference: sha256:")


def test_federation_ingest_plan_rejects_invalid_sources_and_local_state() -> None:
    record = _plan_record()
    record["dry_run"] = False
    assert str(capture_ingest_error(lambda: FederationIngestPlan.from_record(record))) == (
        "federation ingest plans must be dry-run"
    )

    record = _plan_record()
    record["cache_dir"] = ".leibniz/ingest-cache"
    assert str(capture_ingest_error(lambda: FederationIngestPlan.from_record(record))) == (
        "cache_dir: unknown field"
    )

    record = _plan_record()
    entries = list(_entry_records(record))
    entries[0]["repository"] = ".leibniz/source"
    record["entries"] = entries
    assert str(capture_ingest_error(lambda: FederationIngestPlan.from_record(record))) == (
        "repository must not be a local path"
    )

    record = _plan_record()
    entries = list(_entry_records(record))
    entries[0]["status"] = "downloaded"
    record["entries"] = entries
    assert str(capture_ingest_error(lambda: FederationIngestPlan.from_record(record))) == (
        "unsupported status: downloaded"
    )

    record = _plan_record()
    entries = list(_entry_records(record))
    entries[0]["expected_publication_bundles"] = [
        {
            "kind": "model-manifest",
            "content_digest": str(ContentDigest.from_value({"model": True})),
        }
    ]
    record["entries"] = entries
    assert str(capture_ingest_error(lambda: FederationIngestPlan.from_record(record))) == (
        "expected_publication_bundles references must have kind publication-bundle"
    )

    assert str(
        capture_ingest_error(lambda: FederationIngestPlanDocument.from_bytes(b"[]"))
    ) == "federation ingest plan document must contain an object"


def _registry() -> SubmissionRegistry:
    return SubmissionRegistry.from_record(
        {
            "id": "submission-registries.public-sources@0.1.0",
            "sources": [
                {
                    "repository": "maximumcats/leibniz-submissions",
                    "repository_type": "dataset",
                    "enabled": True,
                },
                {
                    "repository": "maximumcats/disabled-source",
                    "repository_type": "dataset",
                    "enabled": False,
                },
            ],
        }
    )


def _plan_record() -> dict[str, object]:
    return plan_federation_ingest(
        id=ProtocolIdentifier.parse("federation-ingest-plans.public-sources@0.1.0"),
        registry=_registry(),
        expected_publication_bundles=(_publication_bundle_reference(),),
    ).to_record()


def _enabled_entry() -> FederationIngestPlanEntry:
    return FederationIngestPlanEntry(
        source=SubmissionRegistrySource(
            repository="maximumcats/leibniz-submissions",
            repository_type="dataset",
            enabled=True,
        ),
        status="would-inspect",
        expected_publication_bundles=(_publication_bundle_reference(),),
    )


def _publication_bundle_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "publication-bundle",
            "content_digest": str(ContentDigest.from_value({"publication": "bundle"})),
        }
    )


def _entry_records(record: dict[str, object]) -> list[dict[str, object]]:
    value = record["entries"]
    assert isinstance(value, list)
    items = cast(list[object], value)
    records: list[dict[str, object]] = []
    for item in items:
        assert isinstance(item, dict)
        records.append(cast(dict[str, object], item))
    return records


def capture_ingest_error(action: Callable[[], object]) -> FederationIngestValidationError:
    try:
        action()
    except FederationIngestValidationError as error:
        return error
    raise AssertionError("expected FederationIngestValidationError")
