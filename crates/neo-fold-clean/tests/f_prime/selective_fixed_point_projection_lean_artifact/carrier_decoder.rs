//! Complete public-coordinate decoder for the bounded fixed-point artifact.
//!
//! Owns: exact export of the verifier-written constant, direct source-field
//! coordinates, and compiler-inserted public padding from the same prepared
//! layout used by the projected emitter.
//!
//! Does not own: private assignment decoding, source-field semantics, CCS/CE
//! membership, commitment-key alignment, or permission to remove rows.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{SelectiveProjectedPublicCoordinateSource, SelectiveProjectedRowsAudit};

use super::GeneratedLeanFile;

const GENERATED_ROOT: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/Carrier270/Generated";
const IMPORT_ROOT: &str = "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Schema";
const NAMESPACE_ROOT: &str = "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated";
const SCHEMA_VERSION: usize = 1;
const LOGICAL_PUBLIC_WIDTH: usize = 257;
const ALIGNED_PUBLIC_WIDTH: usize = 270;
const CHUNK_SIZE: usize = 256;

fn source(source: SelectiveProjectedPublicCoordinateSource) -> String {
    match source {
        SelectiveProjectedPublicCoordinateSource::ConstantOne => ".constantOne".to_owned(),
        SelectiveProjectedPublicCoordinateSource::SourceField(field) => {
            format!(".sourceField {field}")
        }
        SelectiveProjectedPublicCoordinateSource::FixedZero => ".fixedZero".to_owned(),
    }
}

pub(super) fn render(projected: &SelectiveProjectedRowsAudit) -> Vec<GeneratedLeanFile> {
    let coordinates = projected.public_coordinates();
    assert_eq!(coordinates.len(), ALIGNED_PUBLIC_WIDTH, "complete public decoder");
    for (column, coordinate) in coordinates.iter().copied().enumerate() {
        assert_eq!(coordinate.column(), column, "public coordinate order");
        let expected = match column {
            0 => SelectiveProjectedPublicCoordinateSource::ConstantOne,
            1..LOGICAL_PUBLIC_WIDTH => SelectiveProjectedPublicCoordinateSource::SourceField(column),
            LOGICAL_PUBLIC_WIDTH..ALIGNED_PUBLIC_WIDTH => SelectiveProjectedPublicCoordinateSource::FixedZero,
            _ => unreachable!("validated public decoder width"),
        };
        assert_eq!(coordinate.source(), expected, "public coordinate owner");
    }

    let chunks = coordinates.chunks(CHUNK_SIZE).collect::<Vec<_>>();
    assert_eq!(chunks.len(), 2, "256 + 14 public decoder partition");
    assert_eq!(chunks[0].len(), 256, "first public decoder chunk");
    assert_eq!(chunks[1].len(), 14, "final public decoder chunk");

    chunks
        .into_iter()
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut contents = String::new();
            writeln!(
                contents,
                "import {IMPORT_ROOT}\n\n\
/-! Generated file: fixed-point public-coordinate decoder chunk.\n\n\
Owns: exact proof-free coordinate owners exported from the prepared selective\n\
layout used by the bounded fixed-point projected emitter.\n\n\
Does not own: source semantics, private coordinates, relation satisfaction,\n\
commitment alignment, or row removal. Do not hand-edit.\n\n\
Emits constraints: no.\n\n\
| Artifact field | Exact source | Meaning |\n\
|---|---|---|\n\
| `totalColumns` | final projected-emitter width | bounded profile only |\n\
| `rawCoordinates` | validated prepared-layout owners | public decoder data |\n-/\n\n\
namespace {NAMESPACE_ROOT}.Chunk{chunk_index}\n\n\
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Wire\n\n\
def totalColumns : Nat := {}\n\
def rawCoordinates : List RawCoordinate := [",
                projected.columns(),
            )
            .expect("render public decoder header");
            for (index, coordinate) in chunk.iter().copied().enumerate() {
                let separator = if index == 0 { "  " } else { ", " };
                writeln!(
                    contents,
                    "{separator}{{ schemaVersion := {SCHEMA_VERSION}, column := {}, source := {} }}",
                    coordinate.column(),
                    source(coordinate.source()),
                )
                .expect("render public decoder coordinate");
            }
            writeln!(contents, "]\n\nend {NAMESPACE_ROOT}.Chunk{chunk_index}").expect("render public decoder footer");
            GeneratedLeanFile {
                relative_path: format!("{GENERATED_ROOT}/PublicDecoderChunk{chunk_index}.lean"),
                contents,
            }
        })
        .collect()
}
