use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_fold_next::core::proof::Carry;
use neo_fold_next::rv32im::{Rv32imAccumulatorHandle, Rv32imChunkFoldCarry};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

fn claim_with_paper_packed_x(seed: u64) -> CeClaim<Commitment, F, K> {
    let m_in = 3;
    let mut x = Mat::zero(D, m_in, F::ZERO);
    for col in 0..m_in {
        x[(col % D, col / D)] = F::from_u64(seed + 101 + col as u64);
    }
    x[(1, 1)] = F::from_u64(seed + 211);
    x[(2, 2)] = F::from_u64(seed + 223);

    CeClaim {
        c: Commitment {
            d: D,
            kappa: 1,
            data: vec![F::from_u64(seed + 5); D],
        },
        X: x,
        r: vec![K::from(F::from_u64(seed + 7)), K::from(F::from_u64(seed + 11))],
        s_col: Vec::new(),
        y_ring: vec![
            vec![K::from(F::from_u64(seed + 19)), K::from(F::from_u64(seed + 23))],
            vec![K::from(F::from_u64(seed + 29)), K::from(F::from_u64(seed + 31))],
        ],
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in,
        fold_digest: [seed as u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn carried_projection_digest(claim: CeClaim<Commitment, F, K>) -> [F; 4] {
    let carry = Rv32imChunkFoldCarry::from_main(
        Carry {
            claims: vec![claim],
            witnesses: Vec::new(),
        },
        Rv32imAccumulatorHandle([0u8; 32]),
    );
    carry.main_projection_digests[0]
}

#[test]
fn rv32im_carried_projection_digest_binds_paper_packed_x_lanes_not_diagonal_decoys() {
    let claim = claim_with_paper_packed_x(17);
    let baseline = carried_projection_digest(claim.clone());

    let mut authoritative_lane_tamper = claim.clone();
    authoritative_lane_tamper.X[(1, 0)] += F::ONE;
    assert_ne!(
        carried_projection_digest(authoritative_lane_tamper),
        baseline,
        "carried projection digest must bind logical x[1] at SuperNeo coordinate X[(1, 0)]"
    );

    let mut diagonal_decoy_tamper = claim;
    diagonal_decoy_tamper.X[(1, 1)] += F::ONE;
    assert_eq!(
        carried_projection_digest(diagonal_decoy_tamper),
        baseline,
        "carried projection digest must not bind the diagonal decoy X[(1, 1)] as logical x[1]"
    );
}
