use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{D, F, K};
use neo_reductions::engines::utils::{bind_me_inputs, me_input_projection_digest_poseidon};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn claim_with_x_shape(rows: usize, cols: usize, m_in: usize) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: Commitment {
            d: D,
            kappa: 1,
            data: vec![F::from_u64(5); D],
        },
        X: Mat::zero(rows, cols, F::ZERO),
        r: vec![K::from(F::from_u64(7))],
        s_col: Vec::new(),
        y_ring: vec![vec![K::from(F::from_u64(11))]],
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in,
        fold_digest: [0; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

#[test]
fn me_input_projection_digest_requires_superneo_x_shape() {
    let valid = claim_with_x_shape(D, 2, 2);
    assert!(me_input_projection_digest_poseidon(&valid).is_ok());

    let bad_rows = claim_with_x_shape(1, 2, 2);
    assert!(me_input_projection_digest_poseidon(&bad_rows).is_err());

    let bad_cols = claim_with_x_shape(D, 3, 2);
    assert!(me_input_projection_digest_poseidon(&bad_cols).is_err());
}

#[test]
fn bind_me_inputs_rejects_malformed_superneo_x_shape_before_transcript_absorb() {
    let bad_rows = claim_with_x_shape(1, 2, 2);
    let mut transcript = Poseidon2Transcript::new(b"projection_digest_shape/bad_rows");
    assert!(bind_me_inputs(&mut transcript, &[bad_rows]).is_err());

    let bad_cols = claim_with_x_shape(D, 3, 2);
    let mut transcript = Poseidon2Transcript::new(b"projection_digest_shape/bad_cols");
    assert!(bind_me_inputs(&mut transcript, &[bad_cols]).is_err());
}
