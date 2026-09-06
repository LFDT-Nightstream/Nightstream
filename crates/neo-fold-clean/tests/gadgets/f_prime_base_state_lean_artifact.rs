//! Exact verifier-owned F' base-state pin artifact and conformance gate.
//!
//! The fixture invokes the production `enforce_base_state_constants` owner
//! through its narrow test-isolation wrapper. The generated constants are
//! derived from preprocessing, never supplied by the proof witness.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_ccs::Mat;
use neo_fold_clean::engine::decider::__test_isolation::enforce_base_state_constants_against;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::f_prime::r1cs::FPrimePublicInputLayout;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeBase/FPrimeBaseStateArtifact.lean";

// This is the exact emission order in `enforce_base_state_constants`.
const PINNED_COLUMNS: [usize; 31] = [
    1, 2, 3, 4, // vk_fs_digest
    5, 6, 7, 8, // pi_ccs_header_bundle
    11, 12, 13, 14, // z_0
    15, 16, 17, 18, // z_i
    20, 21, 22, 23, // initial semantic state
    28, 29, 30, 31, // public trace seed
    24, 25, 26, 27, // empty accumulator
    9, 10, 19, // zero counters and pc=1
];

fn bit_carrier_r1cs() -> R1cs {
    let layout = FPrimePublicInputLayout::plain();
    let mut a = Mat::zero(layout.carrier_padding_len(), layout.total_len(), F::ZERO);
    let mut b = Mat::zero(layout.carrier_padding_len(), layout.total_len(), F::ZERO);
    for row in 0..layout.carrier_padding_len() {
        a[(row, layout.carrier_padding_offset() + row)] = F::ONE;
        b[(row, 0)] = F::ONE;
    }
    R1cs {
        a,
        b,
        c: Mat::zero(layout.carrier_padding_len(), layout.total_len(), F::ZERO),
        m_in: layout.total_len(),
    }
}

fn build() -> neo_fold_clean::engine::r1cs_circuit::R1csBuilder {
    let prep = direct_ccs::preprocess_seeded(&bit_carrier_r1cs(), 42).expect("base-state preprocessing");
    let (builder, _) = enforce_base_state_constants_against(&prep, [0u8; 32]);
    builder
}

fn pinned_values(builder: &neo_fold_clean::engine::r1cs_circuit::R1csBuilder) -> Vec<u64> {
    PINNED_COLUMNS
        .iter()
        .map(|&column| builder.witness()[column].as_canonical_u64())
        .collect()
}

fn lean_u64_list(values: &[u64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn without_whitespace(value: &str) -> String {
    value.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn artifact_hashes(builder: &neo_fold_clean::engine::r1cs_circuit::R1csBuilder) -> (String, String) {
    let values = pinned_values(builder);
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/f-prime-base-state\n\
         source=enforce_base_state_constants\ncolumns={}\nvalues={}\nrows={}\ncols={}\n{}",
        lean_nat_list(PINNED_COLUMNS),
        lean_u64_list(&values),
        builder.rows(),
        builder.cols(),
        lean_rows(builder),
    );
    let witness_payload = lean_witness("honestWitness", builder.witness());
    (sha256_hex(&row_payload), sha256_hex(&witness_payload))
}

#[test]
fn base_state_pins_accept_preprocessing_derived_witness() {
    let builder = build();
    assert_eq!(builder.rows(), 31, "base-state pin row count changed");
    assert_eq!(builder.cols(), 36, "base-state pin column count changed");
    // The isolation wrapper allocates four dummy `x_out` lanes after the
    // state. They are intentionally outside this row family's ownership.
    assert_eq!(builder.unconstrained_columns(), vec![32, 33, 34, 35]);
    assert!(builder.is_satisfied());
}

#[test]
fn base_state_pins_reject_every_authority_family() {
    // One representative from each independently owned coordinate family.
    for column in [1, 5, 9, 10, 11, 15, 19, 20, 24, 28] {
        let mut builder = build();
        builder.tamper_witness(column, builder.witness()[column] + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "base-state authority disconnected at column {column}"
        );
    }
}

#[test]
fn lean_base_state_artifact_matches_committed_file() {
    let builder = build();
    let values = pinned_values(&builder);
    let (row_hash, witness_hash) = artifact_hashes(&builder);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    let expected_rows = format!("def artifactSha256 : String := \"{row_hash}\"");
    let expected_witnesses = format!("def witnessSha256 : String := \"{witness_hash}\"");
    let expected_columns = format!("def pinnedColumns : List Nat :=\n  {}", lean_nat_list(PINNED_COLUMNS));
    let expected_values = format!("def pinnedValues : List Nat :=\n  {}", lean_u64_list(&values));
    let compact_committed = without_whitespace(&committed);
    if !committed.contains(&expected_rows)
        || !committed.contains(&expected_witnesses)
        || !compact_committed.contains(&without_whitespace(&expected_columns))
        || !compact_committed.contains(&without_whitespace(&expected_values))
    {
        panic!("frozen Lean reference differs: {path:?}");
    }
}
