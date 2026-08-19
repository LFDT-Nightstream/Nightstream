//! Shared deterministic renderer for versioned Rust-to-Lean R1CS artifacts.
//!
//! This file is test support, not a test binary. Exporters include it with a
//! local `mod` declaration so every artifact uses one schema and one canonical
//! sparse-row serialization.

use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_math::F;
use p3_field::PrimeField64;
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: u64 = 1;

pub fn lean_terms(trips: &[(usize, usize, F)], row: usize) -> String {
    let terms: Vec<String> = trips
        .iter()
        .filter(|&&(r, _, _)| r == row)
        .map(|&(_, col, coeff)| format!("({}, {})", col, coeff.as_canonical_u64()))
        .collect();
    format!("[{}]", terms.join(", "))
}

pub fn lean_rows(builder: &R1csBuilder) -> String {
    let (a, b, c) = builder.sparse_triplets();
    let mut a_rows = (0..builder.rows())
        .map(|row| lean_terms(a, row))
        .collect::<Vec<_>>();
    for block in builder.seeded_phi81_a_blocks() {
        let mut rows = vec![Vec::new(); block.row_end() - block.row_start()];
        block.for_each_term::<F, _>(|row, column, coefficient| {
            rows[row - block.row_start()].push(format!("({}, {})", column, coefficient.as_canonical_u64()));
        });
        for (offset, terms) in rows.into_iter().enumerate() {
            let row = block.row_start() + offset;
            assert_eq!(
                a_rows[row], "[]",
                "seeded Phi81 A rows must not overlap explicit A rows"
            );
            a_rows[row] = format!("[{}]", terms.join(", "));
        }
    }
    let rows: Vec<String> = (0..builder.rows())
        .map(|row| format!("⟨{}, {}, {}⟩", a_rows[row], lean_terms(b, row), lean_terms(c, row)))
        .collect();
    format!("[{}]", rows.join(",\n   "))
}

pub fn lean_nat_list(values: impl IntoIterator<Item = usize>) -> String {
    let values: Vec<String> = values.into_iter().map(|value| value.to_string()).collect();
    format!("[{}]", values.join(", "))
}

pub fn lean_witness(name: &str, witness: &[F]) -> String {
    let values: Vec<String> = witness
        .iter()
        .map(|value| value.as_canonical_u64().to_string())
        .collect();
    format!("def {name} : List Nat :=\n  [{}]\n", values.join(", "))
}

/// Content identifier for the generated payload below the hash declaration.
/// This is assurance metadata only; it is not used in a protocol transcript.
pub fn sha256_hex(payload: &str) -> String {
    let digest = Sha256::digest(payload.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
