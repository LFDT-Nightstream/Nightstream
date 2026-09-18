//! Exact full-history F' state-link artifact and conformance gate.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_fold_clean::engine::decider::__test_isolation::{enforce_state_link_against_self, StateLinkProbeWires};
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrime/FPrimeStateLinkArtifact.lean";

const COLUMN_PAIRS: [(usize, usize); 31] = [
    (1, 32),
    (2, 33),
    (3, 34),
    (4, 35),
    (5, 36),
    (6, 37),
    (7, 38),
    (8, 39),
    (9, 40),
    (10, 41),
    (11, 42),
    (12, 43),
    (13, 44),
    (14, 45),
    (15, 46),
    (16, 47),
    (17, 48),
    (18, 49),
    (19, 50),
    (20, 51),
    (21, 52),
    (22, 53),
    (23, 54),
    (24, 55),
    (25, 56),
    (26, 57),
    (27, 58),
    (28, 59),
    (29, 60),
    (30, 61),
    (31, 62),
];

fn build() -> (R1csBuilder, StateLinkProbeWires) {
    enforce_state_link_against_self()
}

fn tampered_witness(column: usize) -> Vec<F> {
    let (mut builder, _) = build();
    builder.tamper_witness(column, builder.witness()[column] + F::ONE);
    builder.witness().to_vec()
}

fn artifact_hashes(honest: &R1csBuilder, forged: &[F]) -> (String, String) {
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/f-prime-state-link\nsource=enforce_state_link\n\
         pairs={}\nrows={}\ncols={}\n{}",
        COLUMN_PAIRS
            .iter()
            .map(|&(left, right)| lean_nat_list([left, right]))
            .collect::<Vec<_>>()
            .join(";"),
        honest.rows(),
        honest.cols(),
        lean_rows(honest),
    );
    let witness_payload = format!(
        "{}\n{}",
        lean_witness("honestWitness", honest.witness()),
        lean_witness("forgedWitness", forged),
    );
    (sha256_hex(&row_payload), sha256_hex(&witness_payload))
}

#[test]
fn state_link_accepts_honest_witness() {
    let (builder, _) = build();
    assert_eq!(builder.rows(), 31, "plain state-link row count changed");
    assert_eq!(builder.cols(), 63, "plain state-link column count changed");
    assert!(builder.unconstrained_columns().is_empty());
    assert!(builder.is_satisfied());
}

#[test]
fn state_link_rejects_every_coordinate_family() {
    let selectors: [fn(&StateLinkProbeWires) -> usize; 10] = [
        |w| w.vk_fs0.col(),
        |w| w.structure0.col(),
        |w| w.chunk_count.col(),
        |w| w.step_count.col(),
        |w| w.z_0_0.col(),
        |w| w.z_i_0.col(),
        |w| w.pc.col(),
        |w| w.semantic0.col(),
        |w| w.acc0.col(),
        |w| w.public_trace0.col(),
    ];
    for select in selectors {
        let (mut builder, wires) = build();
        let column = select(&wires);
        builder.tamper_witness(column, builder.witness()[column] + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "state-link coordinate disconnected at column {column}"
        );
    }
}

#[test]
fn lean_state_link_artifact_matches_committed_file() {
    let (honest, probes) = build();
    let forged = tampered_witness(probes.step_count.col());
    let (row_hash, witness_hash) = artifact_hashes(&honest, &forged);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    let expected_rows = format!("def artifactSha256 : String := \"{row_hash}\"");
    let expected_witnesses = format!("def witnessSha256 : String := \"{witness_hash}\"");
    if !committed.contains(&expected_rows) || !committed.contains(&expected_witnesses) {
        let expected_path = format!("{path}.expected");
        std::fs::write(&expected_path, format!("{expected_rows}\n{expected_witnesses}\n"))
            .expect("write .expected artifact hashes");
        panic!("generated Lean state-link artifact drifted. Wrote {expected_path}; inspect and copy it over {path}");
    }
}
