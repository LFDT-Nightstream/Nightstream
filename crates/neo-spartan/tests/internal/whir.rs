//! The Plonky3 0.8 boundary: our Poseidon2 instance, the transcript handoff,
//! and WHIR openings at caller points.

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::F;
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing as _, PrimeField64 as _};
use p3_sumcheck_v08::OpeningBatch;
use p3_symmetric_v08::Permutation;

use crate::field::{gl, Ext, Gl};
use crate::hash::{challenger, permutation};
use crate::pcs::{Pcs, POINTS};
use crate::{outer_terms, Shape};

/// Deterministic pseudo-random words; the tests need variety, not secrecy.
fn word(seed: u64, index: u64) -> u64 {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ index.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 29)
}

fn ext(seed: u64) -> Ext {
    Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(seed, i as u64)))
}

/// The multilinear extension of `values` at `point`, low index bit first.
fn mle(values: &[Gl], point: &[Ext]) -> Ext {
    let mut layer: Vec<Ext> = values.iter().map(|&value| Ext::from(value)).collect();
    for &coordinate in point {
        layer = layer
            .chunks(2)
            .map(|pair| pair[0] + coordinate * (pair[1] - pair[0]))
            .collect();
    }
    assert_eq!(layer.len(), 1);
    layer[0]
}

fn fold_transcript(extra: u64) -> Poseidon2Transcript {
    let mut transcript = Poseidon2Transcript::new_v1_1();
    transcript.absorb_v1_1(&[F::from_u64(3), F::from_u64(extra)]);
    transcript
}

#[test]
fn poseidon2_matches_the_workspace_permutation() {
    for seed in 0..64 {
        let state: [F; p2::WIDTH] = std::array::from_fn(|lane| F::from_u64(word(seed, lane as u64)));
        let ours = p2::permute_state(state).map(|value| value.as_canonical_u64());
        let theirs = permutation()
            .permute(state.map(gl))
            .map(|value| value.as_canonical_u64());
        assert_eq!(ours, theirs, "state {seed}");
    }
}

#[test]
fn whir_opens_two_points_through_the_seeded_challenger() {
    let variables = 12;
    let values: Vec<Gl> = (0..1u64 << variables)
        .map(|i| Gl::from_u64(word(7, i)))
        .collect();
    let points: [Vec<Ext>; POINTS] = std::array::from_fn(|batch| {
        (0..variables)
            .map(|c| ext(100 * batch as u64 + c as u64))
            .collect()
    });
    let expected = points.each_ref().map(|point| mle(&values, point));
    let pcs = Pcs::new(variables, 100.0, &[]).unwrap();
    assert!(pcs.security_bits() >= 100.0);

    let mut prover = challenger(fold_transcript(1));
    let (commitment, data) = pcs.commit(values, &mut prover);
    let opening = pcs.open(data, &points, &mut prover);

    let verify = |transcript: Poseidon2Transcript, points: &[Vec<Ext>; POINTS], opening| {
        let mut verifier = challenger(transcript);
        pcs.observe(&commitment, &mut verifier);
        pcs.verify(&commitment, opening, points, &mut verifier)
    };
    assert_eq!(verify(fold_transcript(1), &points, &opening).unwrap(), expected);

    // A different fold transcript seeds a different challenger.
    assert!(verify(fold_transcript(2), &points, &opening).is_err());
    // A point the prover did not open.
    let mut moved = points.clone();
    moved[1][0] += Ext::ONE;
    assert!(verify(fold_transcript(1), &moved, &opening).is_err());
    // A claimed value that the commitment does not hold.
    let mut changed = opening.clone();
    let batch = &changed.evals[0];
    let mut current = batch.current().to_vec();
    current[0] += Ext::ONE;
    changed.evals[0] = OpeningBatch::new(current, batch.next().to_vec());
    assert!(verify(fold_transcript(1), &points, &changed).is_err());
}

#[test]
fn production_shape_reaches_the_required_security() {
    // The production shape at e402a6d99: 814,144 blocks (a 2^26 cube),
    // 1,004,131 rows, t = 4, 5 public blocks, a 28-variable point, H = 3,672.
    // 117 bits is the design's estimate of the layer-1 share of a 114-bit
    // total; the real target is derived at run time.
    let shape = Shape {
        blocks: 814_144,
        block_variables: 20,
        rows: 1_004_131,
        matrices: 4,
        kappa: 22,
        public_blocks: 5,
        point_variables: 28,
        norm_bound: 3_672,
    };
    let pcs = Pcs::new(shape.cube_variables(), 117.0, &outer_terms(&shape)).unwrap();
    assert!(pcs.security_bits() >= 117.0, "{}", pcs.security_bits());
}
