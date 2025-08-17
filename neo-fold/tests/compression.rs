use neo_ccs::{mv_poly, CcsStructure};
use neo_fields::{from_base, ExtF, F};
use neo_fold::{EvalInstance, FoldState};
use neo_sumcheck::oracle::{deserialize_fri_proof, FriOracle};
use p3_field::PrimeCharacteristicRing;

fn dummy_structure() -> CcsStructure {
    CcsStructure::new(vec![], mv_poly(|_: &[ExtF]| ExtF::ZERO, 1))
}

#[test]
fn test_fri_compress_final_real_roundtrip() {
    let structure = dummy_structure();
    let mut state = FoldState::new(structure);
    let ys = vec![from_base(F::from_u64(5))];
    let poly = neo_poly::Polynomial::new(ys.clone());
    let mut tmp_transcript = state.transcript.clone();
    tmp_transcript.extend(b"final_poly_hash");
    let tmp_oracle = FriOracle::new(vec![poly.clone()], &mut tmp_transcript);
    let point = vec![ExtF::ONE];
    let e_eval = poly.eval(point[0]) + tmp_oracle.blinds[0];
    let eval_instance = EvalInstance {
        commitment: vec![],
        r: point.clone(),
        ys: ys.clone(),
        u: ExtF::ZERO,
        e_eval,
        norm_bound: 0,
    };
    state.eval_instances.push(eval_instance);
    let (commit, proof) = state.fri_compress_final().expect("FRI");
    let coeff_len = state.eval_instances.last().unwrap().ys.len();
    let claimed = state.eval_instances.last().unwrap().e_eval;
    assert!(FoldState::fri_verify_compressed(
        &commit,
        &proof,
        &point,
        claimed,
        coeff_len,
    ));
    assert!(!FoldState::fri_verify_compressed(
        &commit,
        &proof,
        &point,
        claimed + ExtF::ONE,
        coeff_len,
    ));
}

#[test]
fn test_fri_compress_final_no_panic() {
    let structure = dummy_structure();
    let mut state = FoldState::new(structure);
    let ys = vec![ExtF::ZERO, ExtF::ONE];
    let poly = neo_poly::Polynomial::new(ys.clone());
    let mut tmp_transcript = state.transcript.clone();
    tmp_transcript.extend(b"final_poly_hash");
    let tmp_oracle = FriOracle::new(vec![poly.clone()], &mut tmp_transcript);
    let point = vec![ExtF::ONE];
    let e_eval = poly.eval(point[0]) + tmp_oracle.blinds[0];
    let eval_instance = EvalInstance {
        commitment: vec![],
        r: point,
        ys,
        u: ExtF::ZERO,
        e_eval,
        norm_bound: 0,
    };
    state.eval_instances.push(eval_instance);
    let res = state.fri_compress_final();
    assert!(res.is_ok());
}

#[test]
fn test_fri_compress_final_ys_as_coeffs() {
    let structure = dummy_structure();
    let mut state = FoldState::new(structure);
    let ys = vec![ExtF::ZERO, ExtF::ONE];
    let poly = neo_poly::Polynomial::new(ys.clone());
    let mut tmp_transcript = state.transcript.clone();
    tmp_transcript.extend(b"final_poly_hash");
    let tmp_oracle = FriOracle::new(vec![poly.clone()], &mut tmp_transcript);
    let point = vec![ExtF::ONE];
    let e_eval = poly.eval(point[0]) + tmp_oracle.blinds[0];
    let eval_instance = EvalInstance {
        commitment: vec![],
        r: point.clone(),
        ys,
        u: ExtF::ZERO,
        e_eval,
        norm_bound: 0,
    };
    state.eval_instances.push(eval_instance);
    let (commit, proof) = state.fri_compress_final().expect("FRI");
    let coeff_len = 2;
    let claimed = state.eval_instances.last().unwrap().e_eval;
    assert!(deserialize_fri_proof(&proof).is_ok());
    assert!(FoldState::fri_verify_compressed(&commit, &proof, &point, claimed, coeff_len));
}

#[test]
fn test_fri_verify_rejects_wrong_eval() {
    let commit = vec![1u8; 32];
    let proof = vec![0u8; 32];
    let claimed = ExtF::ONE;
    assert!(!FoldState::fri_verify_compressed(
        &commit,
        &proof,
        &[ExtF::ZERO],
        claimed,
        1,
    ));
}
