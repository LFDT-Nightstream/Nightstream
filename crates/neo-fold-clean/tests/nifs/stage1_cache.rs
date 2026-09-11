//! Native cache checks on the selected package and its actual base witness.

use std::{fs, path::PathBuf, time::Instant};

use neo_ccs::Mat;
use neo_fold_clean::Poseidon2HashChainV1Package;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::{SuperneoCompactRowOffsets, SuperneoZBlocks};
use nightstream_fprime::{load_poseidon2_hash_chain_v1_package, LogicalMatrixEntry, PI_CCS_V1_1_ROUND_COUNT};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;
use sha2::{Digest, Sha256};

fn artifact(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

fn package_bytes() -> Vec<u8> {
    fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).expect("selected package")
}

#[derive(Default, Debug)]
struct BlockCensus {
    singleton: u64,
    dense: u64,
    coefficients: u64,
    capacities: [u64; 5],
}

struct CacheMemory {
    current: u64,
    peak_with_reallocation: u64,
}

impl BlockCensus {
    fn grow(&mut self, slot: usize, len: u64, additional: u64, bytes: u64, memory: &mut CacheMemory) {
        let old = self.capacities[slot];
        if len + additional > old {
            // Unreserved baseline on installed Rust1.94.1:
            // alloc/raw_vec/mod.rs grow_amortized, lines502–507.
            // These are Vec capacities, not an RSS guarantee.
            let capacity = (2 * old)
                .max(len + additional)
                .max(if bytes == 1 { 8 } else { 4 });
            memory.peak_with_reallocation = memory
                .peak_with_reallocation
                .max(memory.current + capacity * bytes);
            memory.current += (capacity - old) * bytes;
            self.capacities[slot] = capacity;
        }
    }

    fn row(&mut self, entries: &[LogicalMatrixEntry], rows: usize, memory: &mut CacheMemory) {
        let first_nonempty = !entries.is_empty() && self.singleton + self.dense == 0;
        let mut start = 0;
        while start < entries.len() {
            let block = entries[start].column() / D;
            let mut end = start + 1;
            while end < entries.len() && entries[end].column() / D == block {
                end += 1;
            }
            let coefficient = F::from_u64(entries[start].coefficient());
            if end == start + 1 && (coefficient == F::ONE || coefficient == -F::ONE) {
                self.grow(0, self.singleton + self.dense, 1, 4, memory);
                self.singleton += 1;
            } else {
                // Same append order as row_source::append_block: sparse
                // pattern, offset, side-table entry, packed row reference.
                let added = (end - start) as u64;
                self.grow(3, self.coefficients, added, 1, memory);
                self.grow(4, self.coefficients, added, 8, memory);
                self.grow(2, self.dense + 1, 1, 4, memory);
                self.grow(1, self.dense, 1, 8, memory);
                self.grow(0, self.singleton + self.dense, 1, 4, memory);
                self.dense += 1;
                self.coefficients += added;
            }
            start = end;
        }
        if first_nonempty {
            memory.current += (rows as u64 + 1) * 4;
            memory.peak_with_reallocation = memory.peak_with_reallocation.max(memory.current);
        }
    }

    fn payload_bytes(&self) -> u64 {
        4 * self.singleton + 16 * self.dense + 9 * self.coefficients + 4
    }
}

#[test]
#[ignore = "Full selected row census; run before allocating the selected cache, under the 300-second test cap."]
fn selected_cache_block_census() {
    let started = Instant::now();
    let package = Poseidon2HashChainV1Package::load(&package_bytes()).expect("selected package");
    let rows = package.structure().n;
    let mut census = (0..package.structure().t())
        .map(|_| BlockCensus {
            capacities: [0, 0, 1, 0, 0],
            ..BlockCensus::default()
        })
        .collect::<Vec<_>>();
    let initial_bytes = (rows * 2 + census.len() * 4) as u64;
    let mut memory = CacheMemory {
        current: initial_bytes,
        peak_with_reallocation: initial_bytes,
    };
    package
        .visit_matrix_rows(0..rows, |_, row| {
            for (matrix, counts) in census.iter_mut().enumerate() {
                counts.row(row.matrix(matrix).expect("selected matrix"), rows, &mut memory);
            }
            Ok(())
        })
        .expect("complete actual matrix row stream");
    let nonzeros = census
        .iter()
        .map(|counts| counts.singleton + counts.coefficients)
        .collect::<Vec<_>>();
    let payload = census.iter().map(BlockCensus::payload_bytes).sum::<u64>();
    let live = census
        .iter()
        .filter(|counts| counts.singleton + counts.dense != 0)
        .count();
    let offsets_and_masks = (live * (rows + 1) * 4 + rows * 2) as u64;
    println!("structural_identifier={:?}", package.structural_identifier());
    println!("package_identity={:?}", package.package_identity());
    println!(
        "rows={rows}, logical_columns={}, matrices={}",
        package.structure().m,
        census.len()
    );
    println!("nonzeros={nonzeros:?}");
    println!("block_census={census:?}");
    println!("compact_payload_bytes={payload}, u32_offsets_and_masks_bytes={offsets_and_masks}");
    println!("exact_reserved_capacity_bytes={}", payload + offsets_and_masks);
    println!(
        "vec_capacity_bytes={}, peak_capacity_with_reallocation_bytes={}",
        memory.current, memory.peak_with_reallocation
    );
    println!("elapsed={:?}", started.elapsed());
}

#[derive(Deserialize)]
struct BaseFixture(u64, [u64; 4], Vec<u64>, Vec<u64>, serde_json::Value);

#[derive(Deserialize)]
struct Expected {
    base_fixture_sha256: String,
    structural_identifier: [u64; 4],
    package_identity: [u64; 4],
    point: Vec<[u64; 2]>,
    evaluations: Vec<Vec<[u64; 2]>>,
}

fn equality_factor(point: &[K]) -> Vec<K> {
    let mut values = vec![K::ONE];
    for &coordinate in point {
        let width = values.len();
        values.resize(2 * width, K::ZERO);
        for index in 0..width {
            let value = values[index];
            values[index] = value * (K::ONE - coordinate);
            values[index + width] = value * coordinate;
        }
    }
    values
}

#[test]
#[ignore = "Full actual-witness comparison; run after the storage census, under the 300-second test cap."]
fn selected_cache_matches_all_actual_base_matrix_evaluations() {
    let started = Instant::now();
    let expected: Expected = serde_json::from_str(include_str!("fixtures/stage1_base_matrix_evaluations.json"))
        .expect("retained Lean expectation");
    let bytes = package_bytes();
    let producer = load_poseidon2_hash_chain_v1_package(&bytes).expect("selected witness producer");
    let fixture_bytes =
        fs::read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json")).expect("actual base fixture");
    // This hash identifies test input bytes. It is not a protocol binding.
    assert_eq!(
        format!("{:x}", Sha256::digest(&fixture_bytes)),
        expected.base_fixture_sha256
    );
    let BaseFixture(schema, context, private, public, base_result) = serde_json::from_slice(&fixture_bytes).unwrap();
    drop(base_result);
    assert_eq!(schema, 1);
    assert_eq!(
        context,
        producer
            .production_verifier_binding()
            .unwrap()
            .verifier_context()
            .digest()
    );
    let physical = producer
        .execute_witness(&private, &public)
        .expect("actual witness IR");
    let logical = producer
        .execute_logical_assignment(&physical)
        .expect("actual logical transport");
    let columns = logical.len().div_ceil(D);
    let mut positive = vec![0u64; columns];
    let mut negative = vec![0u64; columns];
    for (index, &value) in logical.balanced_values().iter().enumerate() {
        let mask = 1u64 << (index % D);
        match value {
            0 => {}
            1 => positive[index / D] |= mask,
            -1 => negative[index / D] |= mask,
            _ => panic!("actual logical witness is not signed-unit"),
        }
    }
    let witness = Mat::<F>::compact_signed_unit_from_column_masks(D, columns, &positive, &negative)
        .expect("actual packed witness, with zero final padding");
    let blocks = SuperneoZBlocks::from_witness_mat(&witness, logical.len()).expect("native witness blocks");
    drop((
        producer,
        physical,
        logical,
        witness,
        positive,
        negative,
        private,
        public,
        fixture_bytes,
    ));
    println!("actual_witness_elapsed={:?}", started.elapsed());

    let package = Poseidon2HashChainV1Package::load(&bytes).expect("selected cache owner");
    drop(bytes);
    assert_eq!(package.structural_identifier(), expected.structural_identifier);
    assert_eq!(package.package_identity(), expected.package_identity);
    let rows = package.structure().n;
    assert_eq!(expected.point.len(), PI_CCS_V1_1_ROUND_COUNT);
    assert!(rows <= 1usize << expected.point.len());
    let point = expected
        .point
        .iter()
        .map(|pair| K::from_coeffs(pair.map(F::from_u64)))
        .collect::<Vec<_>>();
    // These two factors have minimum combined size at this split. Only the
    // active row prefix is materialized; the 2^28 table is never allocated.
    let low_bits = point.len() / 2;
    let low = equality_factor(&point[..low_bits]);
    let high = equality_factor(&point[low_bits..]);
    let weights = (0..rows)
        .map(|row| low[row & (low.len() - 1)] * high[row >> low_bits])
        .collect::<Vec<_>>();
    let cache = package
        .build_superneo_cache()
        .expect("cache from actual selected rows");
    println!("actual_cache_elapsed={:?}", started.elapsed());
    assert_eq!(cache.matrix_caches().len(), expected.evaluations.len());
    let zero = cache.matrix(13).unwrap().compact_device_parts().unwrap();
    assert!(matches!(zero.row_offsets, SuperneoCompactRowOffsets::Empty));
    assert!(zero.row_blocks.is_empty());
    let actual = cache.eval_ring_linear_forms_for_real_z_blocks(&weights, rows, &[blocks]);
    for (matrix, wanted) in expected.evaluations.iter().enumerate() {
        assert_eq!(wanted.len(), D);
        for (coefficient, pair) in wanted.iter().enumerate() {
            assert_eq!(
                actual[0][matrix][coefficient]
                    .as_coeffs()
                    .map(|value| value.as_canonical_u64()),
                *pair,
                "matrix {matrix}, coefficient {coefficient}"
            );
        }
    }
    assert_eq!(actual[0][13], [K::ZERO; D]);
    println!("all_actual_matrix_evaluations_elapsed={:?}", started.elapsed());
}
