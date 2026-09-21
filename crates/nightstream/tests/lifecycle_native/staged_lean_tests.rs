use super::super::super::super::{claim, commitment, fields as input_fields, frame, proof as fixture_proof};
use super::*;

fn fixture() -> (Value, CcsClaim, Vec<CeClaim>, NifsProof) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage1_actual_nifs/actual_result.json");
    let actual = read(path);
    let input = &actual["pi_ccs_input"];
    let fresh = CcsClaim {
        c: commitment(&input[1]),
        x: input_fields(&input[2]),
        m_in: 270,
        adv: None,
    };
    // The native C transcript absorbs the caller digest. Recover it from
    // the original public input, as load_claims does for staged execution.
    let prior_digest = std::array::from_fn(|lane| {
        (0..64).fold(0u64, |word, bit| {
            let digit = fresh.x[1 + lane * 64 + bit].as_canonical_u64();
            assert!(digit <= 1);
            word | (digit << bit)
        })
    });
    let running = (0..16)
        .map(|source| {
            claim(
                &json!([
                    input[6][1][source],
                    input[6][2][source],
                    input[6][0],
                    input[6][3][source],
                    input[6][4][source]
                ]),
                frame(prior_digest),
            )
        })
        .collect();
    let proof = fixture_proof(&actual);
    (actual, fresh, running, proof)
}

#[test]
fn export_matches_retained_lean_input_and_every_native_c_trace_field() {
    let (actual, fresh, running, proof) = fixture();
    assert_eq!(ccs_input(&fresh, &running, &proof.pi_ccs), actual["pi_ccs_input"]);
    assert_eq!(running_value(&proof.pi_dec.children), actual["children"]);
    assert_eq!(claim_value(&proof.pi_rlc.combined, true), actual["pi_rlc_parent"]);
    let loaded = nightstream_fprime::load_poseidon2_hash_chain_v1_package(
        &fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap(),
    )
    .unwrap();
    let structure = loaded.ccs_structure_header().unwrap();
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree()).unwrap();
    let (accepted, trace) = optimized_verify_with_trace(
        &mut Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0),
        params.inner(),
        &structure,
        &[fresh],
        &running,
        &proof.pi_ccs.outputs,
        &proof.pi_ccs.sumcheck,
    )
    .unwrap();
    assert!(accepted);
    assert_eq!(ccs_phase(&proof.pi_ccs, &trace), actual["pi_ccs_phase"]);
    let lean = read(artifact("nightstream-fprime-stage1-base-nifs-result-v1.json"));
    assert_eq!(actual["pi_ccs_input"], lean[1]);
    assert_eq!(actual["pi_ccs_phase"], lean[5]);
    assert_eq!(actual["children"], lean[9][16][1]);
}

#[test]
fn export_keeps_changed_messages_and_rejects_dropped_native_padding() {
    let (actual, fresh, running, mut proof) = fixture();
    proof.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;
    assert_ne!(ccs_input(&fresh, &running, &proof.pi_ccs), actual["pi_ccs_input"]);
    proof.pi_dec.children[0].eval_k[D] = K::ONE;
    assert!(std::panic::catch_unwind(|| running_value(&proof.pi_dec.children)).is_err());
}
