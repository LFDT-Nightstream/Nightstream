//! Exact terminal delayed-link artifact and Rust conformance gate.
//!
//! The test-only decider isolation wrapper calls the private production owner
//! `enforce_terminal_latest_link`, so this artifact cannot drift into a
//! handwritten lookalike.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_fold_clean::engine::decider::__test_isolation::enforce_terminal_latest_link_against;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrime/FPrimeTerminalLinkArtifact.lean";
const TAMPERED_BIT: usize = 37;

struct BuiltLink {
    builder: neo_fold_clean::engine::r1cs_circuit::R1csBuilder,
    fresh_cols: Vec<Vec<usize>>,
    last_x_out_bit_cols: Vec<usize>,
}

fn canonical_inputs() -> (Vec<Vec<F>>, Vec<F>) {
    let bits = (0..256)
        .map(|bit| F::from_u64((bit % 2) as u64))
        .collect::<Vec<_>>();
    let mut fresh = Vec::with_capacity(257);
    fresh.push(F::ONE);
    fresh.extend(bits.iter().copied());
    (vec![fresh], bits)
}

fn build(last_bits: &[F], fresh: &[Vec<F>]) -> Result<BuiltLink, String> {
    let builder = enforce_terminal_latest_link_against(last_bits, fresh)?;
    Ok(BuiltLink {
        last_x_out_bit_cols: (1..=256).collect(),
        fresh_cols: vec![(257..=513).collect()],
        builder,
    })
}

fn build_honest() -> BuiltLink {
    let (fresh, bits) = canonical_inputs();
    build(&bits, &fresh).expect("emit terminal latest link")
}

fn artifact_hashes(honest: &BuiltLink, wrong_one_witness: &[F], wrong_bit_witness: &[F]) -> (String, String) {
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/f-prime-terminal-latest-link\n\
         source=enforce_terminal_latest_link\nfresh_cols={}\nlast_cols={}\nrows={}\ncols={}\n{}",
        honest
            .fresh_cols
            .iter()
            .map(|row| lean_nat_list(row.iter().copied()))
            .collect::<Vec<_>>()
            .join(";"),
        lean_nat_list(honest.last_x_out_bit_cols.iter().copied()),
        honest.builder.rows(),
        honest.builder.cols(),
        lean_rows(&honest.builder),
    );
    let witness_payload = format!(
        "{}\n{}\n{}",
        lean_witness("honestWitness", honest.builder.witness()),
        lean_witness("wrongOneWitness", wrong_one_witness),
        lean_witness("wrongBitWitness", wrong_bit_witness),
    );
    (sha256_hex(&row_payload), sha256_hex(&witness_payload))
}

#[test]
fn terminal_link_accepts_honest_witness() {
    let built = build_honest();
    assert_eq!(built.builder.rows(), 257, "terminal link row count changed");
    assert_eq!(built.builder.cols(), 514, "terminal link column count changed");
    assert!(built.builder.unconstrained_columns().is_empty());
    assert!(built.builder.is_satisfied());
}

#[test]
fn terminal_link_rejects_wrong_affine_one() {
    let (mut fresh, bits) = canonical_inputs();
    fresh[0][0] = F::ZERO;
    let built = build(&bits, &fresh).expect("emit");
    assert_eq!(built.builder.first_unsatisfied_row(), Some(0));
}

#[test]
fn terminal_link_rejects_wrong_public_bit() {
    let (mut fresh, bits) = canonical_inputs();
    fresh[0][1 + TAMPERED_BIT] = F::ONE - fresh[0][1 + TAMPERED_BIT];
    let built = build(&bits, &fresh).expect("emit");
    assert_eq!(built.builder.first_unsatisfied_row(), Some(1 + TAMPERED_BIT));
}

#[test]
fn terminal_link_rejects_host_shape_errors() {
    let (fresh, bits) = canonical_inputs();
    assert!(build(&bits, &[]).is_err());
    assert!(build(&bits[..255], &fresh).is_err());
    assert!(build(&bits, &[fresh[0][..256].to_vec()]).is_err());
}

#[test]
fn lean_terminal_link_artifact_matches_committed_file() {
    let honest = build_honest();
    let (mut wrong_one, bits) = canonical_inputs();
    wrong_one[0][0] = F::ZERO;
    let wrong_one = build(&bits, &wrong_one).expect("emit wrong one");
    let (mut wrong_bit, bits) = canonical_inputs();
    wrong_bit[0][1 + TAMPERED_BIT] = F::ONE - wrong_bit[0][1 + TAMPERED_BIT];
    let wrong_bit = build(&bits, &wrong_bit).expect("emit wrong bit");
    let (row_hash, witness_hash) = artifact_hashes(&honest, wrong_one.builder.witness(), wrong_bit.builder.witness());

    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    let expected_rows = format!("def artifactSha256 : String := \"{row_hash}\"");
    let expected_witnesses = format!("def witnessSha256 : String := \"{witness_hash}\"");
    if !committed.contains(&expected_rows) || !committed.contains(&expected_witnesses) {
        let expected_path = format!("{path}.expected");
        std::fs::write(&expected_path, format!("{expected_rows}\n{expected_witnesses}\n"))
            .expect("write .expected artifact hashes");
        panic!("generated Lean terminal-link artifact drifted. Wrote {expected_path}; inspect and copy it over {path}");
    }
}
