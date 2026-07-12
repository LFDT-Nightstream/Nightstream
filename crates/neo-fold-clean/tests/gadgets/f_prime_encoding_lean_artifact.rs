//! Exact `enc_inst(x_out)` artifact export and Rust conformance gate.
//!
//! The exported rows come from the same helper used for both the prior-link
//! and outgoing-public-input bindings in production F'. Lean proves that any
//! satisfying assignment carries exactly four canonical Goldilocks lanes as
//! 256 little-endian bits.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_x_out_public_bits, enforce_public_bits_encode_digest, F_PRIME_ENC_INST_BITS,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrime/FPrimeEncodingArtifact.lean";

const SAMPLE_LANES: [u64; 4] = [0, 5, 0x0123_4567_89AB_CDEF, 18_446_744_069_414_584_320];

#[derive(Clone, Debug)]
struct EncodingLayout {
    digest_cols: [usize; 4],
    public_bit_cols: [usize; F_PRIME_ENC_INST_BITS],
    canonical_maps: [Vec<usize>; 4],
}

struct BuiltEncoding {
    builder: R1csBuilder,
    layout: EncodingLayout,
}

fn canonical_map(digest: Var, aux_start: usize) -> Vec<usize> {
    std::iter::once(0)
        .chain(std::iter::once(digest.col()))
        .chain(aux_start..aux_start + 64)
        .chain([aux_start + 64, aux_start + 65])
        .collect()
}

fn build_encoding(lanes: [u64; 4]) -> BuiltEncoding {
    let mut builder = R1csBuilder::new();
    let digest: [Var; 4] = lanes.map(|lane| builder.alloc(F::from_u64(lane)));
    let encoded = encode_x_out_public_bits(lanes.map(F::from_u64));
    let public_bits: [Var; F_PRIME_ENC_INST_BITS] = encoded
        .into_iter()
        .map(|bit| builder.alloc(bit))
        .collect::<Vec<_>>()
        .try_into()
        .expect("enc_inst has fixed width");
    let aux_start = builder.cols();

    enforce_public_bits_encode_digest(&mut builder, &public_bits, &digest).expect("emit enc_inst binding");

    let layout = EncodingLayout {
        digest_cols: digest.map(Var::col),
        public_bit_cols: public_bits.map(Var::col),
        canonical_maps: std::array::from_fn(|lane| canonical_map(digest[lane], aux_start + 66 * lane)),
    };
    BuiltEncoding { builder, layout }
}

fn artifact_hashes(honest: &BuiltEncoding, wrong_bit_witness: &[F]) -> (String, String) {
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/f-prime-enc-inst\nsource=enforce_public_bits_encode_digest\n\
         digest_cols={}\npublic_bit_cols={}\ncanonical_maps={}\nrows={}\ncols={}\n{}",
        lean_nat_list(honest.layout.digest_cols),
        lean_nat_list(honest.layout.public_bit_cols),
        honest
            .layout
            .canonical_maps
            .iter()
            .map(|row| lean_nat_list(row.iter().copied()))
            .collect::<Vec<_>>()
            .join(";"),
        honest.builder.rows(),
        honest.builder.cols(),
        lean_rows(&honest.builder),
    );
    let witness_payload = format!(
        "{}\n{}",
        lean_witness("honestWitness", honest.builder.witness()),
        lean_witness("wrongBitWitness", wrong_bit_witness),
    );
    (sha256_hex(&row_payload), sha256_hex(&witness_payload))
}

#[test]
fn production_encoding_accepts_honest_witness() {
    let built = build_encoding(SAMPLE_LANES);
    assert_eq!(built.builder.rows(), 532, "F' enc_inst row count changed");
    assert_eq!(built.builder.cols(), 525, "F' enc_inst column count changed");
    assert!(built.builder.unconstrained_columns().is_empty());
    assert!(built.builder.is_satisfied());
}

#[test]
fn production_encoding_rejects_wrong_length_before_emitting_rows() {
    let mut builder = R1csBuilder::new();
    let digest = SAMPLE_LANES.map(|lane| builder.alloc(F::from_u64(lane)));
    let short_bits: Vec<_> = (0..F_PRIME_ENC_INST_BITS - 1)
        .map(|_| builder.alloc(F::ZERO))
        .collect();
    let error = enforce_public_bits_encode_digest(&mut builder, &short_bits, &digest)
        .expect_err("255-bit enc_inst must be rejected");
    assert!(error
        .to_string()
        .contains("enc_inst body length 255 != 256"));
    assert_eq!(builder.rows(), 0, "shape rejection must precede row emission");
}

#[test]
fn production_encoding_rejects_public_bit_flip() {
    let mut built = build_encoding(SAMPLE_LANES);
    let col = built.layout.public_bit_cols[0];
    built.builder.tamper_witness(col, F::ONE);
    assert_eq!(built.builder.first_unsatisfied_row(), Some(69));
}

#[test]
fn lean_f_prime_encoding_artifact_matches_committed_file() {
    let honest = build_encoding(SAMPLE_LANES);
    let mut wrong_bit = build_encoding(SAMPLE_LANES);
    wrong_bit
        .builder
        .tamper_witness(wrong_bit.layout.public_bit_cols[0], F::ONE);
    let (row_hash, witness_hash) = artifact_hashes(&honest, wrong_bit.builder.witness());

    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    let expected_rows = format!("def artifactSha256 : String := \"{row_hash}\"");
    let expected_witnesses = format!("def witnessSha256 : String := \"{witness_hash}\"");
    if !committed.contains(&expected_rows) || !committed.contains(&expected_witnesses) {
        let expected_path = format!("{path}.expected");
        std::fs::write(&expected_path, format!("{expected_rows}\n{expected_witnesses}\n"))
            .expect("write .expected artifact hashes");
        panic!("generated Lean F' encoding artifact drifted. Wrote {expected_path}; inspect and copy it over {path}");
    }
}
