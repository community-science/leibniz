import hashlib
import sys
from array import array

import pytest

from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.field_artifacts import (
    FieldArtifactError,
    FieldArtifactReference,
    field_content_digest,
    verify_field_artifact,
)


def test_field_content_digest_uses_raw_little_endian_float_buffers() -> None:
    values = (0.25, 1.5, -2.0)
    expected = array("f", values)
    if sys.byteorder == "big":
        expected.byteswap()

    digest = field_content_digest(values, dtype="float32")

    assert digest.algorithm == "sha256"
    assert digest.hex == hashlib.sha256(expected.tobytes()).hexdigest()


def test_field_artifact_reference_round_trips_and_verifies_values() -> None:
    values = (0.25, 1.5, -2.0, 4.0)
    reference = FieldArtifactReference(
        shape=(2, 2),
        dtype="float64",
        content_digest=field_content_digest(values, dtype="float64"),
        materialization={
            "seed": 123,
            "generator": {
                "kind": "field-generator",
                "content_digest": "sha256:" + "0" * 64,
            },
            "solver": "cpu-reference",
        },
    )

    record = load_object_document(
        canonical_document_bytes(reference.to_record()),
        description="field artifact reference",
    )
    parsed = FieldArtifactReference.from_record(record)

    assert parsed == reference
    assert verify_field_artifact(reference, values)
    assert not verify_field_artifact(reference, (0.25, 1.5, -2.0, 4.01))
    assert not verify_field_artifact(reference, values[:3])


@pytest.mark.parametrize(
    "record",
    [
        {
            "shape": [],
            "dtype": "float32",
            "content_digest": "sha256:" + "0" * 64,
            "materialization": {
                "seed": 1,
                "generator": {"kind": "field-generator"},
            },
        },
        {
            "shape": [1],
            "dtype": "int64",
            "content_digest": "sha256:" + "0" * 64,
            "materialization": {
                "seed": 1,
                "generator": {"kind": "field-generator"},
            },
        },
        {
            "shape": [1],
            "dtype": "float32",
            "content_digest": "sha256:" + "0" * 64,
            "materialization": {
                "generator": {"kind": "field-generator"},
            },
        },
        {
            "shape": [1],
            "dtype": "float32",
            "content_digest": "sha256:" + "0" * 64,
            "materialization": {
                "seed": 1,
                "generator": {},
            },
        },
    ],
)
def test_field_artifact_reference_rejects_malformed_records(
    record: dict[str, object],
) -> None:
    with pytest.raises(FieldArtifactError):
        FieldArtifactReference.from_record(record)
