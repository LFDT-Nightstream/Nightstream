use std::fs;
use std::path::Path;

use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_math::F;
use p3_field::PrimeField64;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub fn source_hash(repo_root: &Path, relative: &str) -> Value {
    let bytes = fs::read(repo_root.join(relative)).unwrap_or_else(|error| panic!("read {relative}: {error}"));
    json!({ "path": relative, "sha256": format!("{:x}", Sha256::digest(bytes)) })
}

/// Hash the complete exact matrix range, including coefficients stored in
/// compact seeded Phi81 A blocks.
pub fn range_hash(builder: &R1csBuilder, range: &RowFamilyRange) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"nightstream/fprime-full-history-row-range/v2");
    hasher.update((range.row_end - range.row_start).to_le_bytes());
    let (a, b, c) = builder.sparse_triplets();
    for &(row, column, coefficient) in a {
        if row < range.row_start || row >= range.row_end {
            continue;
        }
        hasher.update(b"A");
        hasher.update((row - range.row_start).to_le_bytes());
        hasher.update(column.to_le_bytes());
        hasher.update(coefficient.as_canonical_u64().to_le_bytes());
    }
    for block in builder.seeded_phi81_a_blocks() {
        if block.row_end() <= range.row_start || range.row_end <= block.row_start() {
            continue;
        }
        block.for_each_term::<F, _>(|row, column, coefficient| {
            if range.row_start <= row && row < range.row_end {
                hasher.update(b"A");
                hasher.update((row - range.row_start).to_le_bytes());
                hasher.update(column.to_le_bytes());
                hasher.update(coefficient.as_canonical_u64().to_le_bytes());
            }
        });
    }
    for (tag, trips) in [(b'B', b), (b'C', c)] {
        for &(row, column, coefficient) in trips {
            if row < range.row_start || row >= range.row_end {
                continue;
            }
            hasher.update([tag]);
            hasher.update((row - range.row_start).to_le_bytes());
            hasher.update(column.to_le_bytes());
            hasher.update(coefficient.as_canonical_u64().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

pub fn range_nonzeros(builder: &R1csBuilder, range: &RowFamilyRange) -> usize {
    let (a, b, c) = builder.sparse_triplets();
    let explicit = a
        .iter()
        .chain(b)
        .chain(c)
        .filter(|&&(row, _, _)| row >= range.row_start && row < range.row_end)
        .count();
    let mut implicit = 0usize;
    for block in builder.seeded_phi81_a_blocks() {
        if block.row_end() <= range.row_start || range.row_end <= block.row_start() {
            continue;
        }
        block.for_each_term::<F, _>(|row, _, _| {
            if range.row_start <= row && row < range.row_end {
                implicit += 1;
            }
        });
    }
    explicit + implicit
}

pub fn range_json(builder: &R1csBuilder, range: &RowFamilyRange) -> Value {
    json!({
        "name": range.name,
        "row_start": range.row_start,
        "row_end": range.row_end,
        "row_count": range.row_end - range.row_start,
        "nonzero_entries": range_nonzeros(builder, range),
        "sha256": range_hash(builder, range),
    })
}

pub fn top_ranges<'a>(builder: &'a R1csBuilder, names: &[&str]) -> Vec<&'a RowFamilyRange> {
    let mut ranges = names
        .iter()
        .map(|name| {
            let matches = builder
                .row_family_ranges()
                .iter()
                .filter(|range| range.name == *name)
                .collect::<Vec<_>>();
            assert_eq!(matches.len(), 1, "expected one full-history owner {name}");
            matches[0]
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.row_start);
    ranges
}

pub fn assert_partition(builder: &R1csBuilder, ranges: &[&RowFamilyRange]) {
    let mut cursor = 0;
    for range in ranges {
        assert_eq!(range.row_start, cursor, "gap or overlap before {}", range.name);
        assert!(range.row_end >= range.row_start, "reversed range {}", range.name);
        cursor = range.row_end;
    }
    assert_eq!(cursor, builder.rows(), "top-level owners do not cover full history");
}
