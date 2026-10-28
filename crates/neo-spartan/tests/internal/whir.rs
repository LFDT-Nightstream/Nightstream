//! The Plonky3 0.8 boundary: our Poseidon2 instance, the transcript handoff,
//! and WHIR openings at caller points.

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::F;
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing as _, PrimeField64 as _};
use p3_sumcheck_v08::OpeningBatch;
use p3_symmetric_v08::Permutation;

use crate::device::Accelerator;
use crate::field::{gl, Ext, Gl};
use crate::hash::{challenger, permutation, seed};
use crate::matrix::Structure;
use crate::pcs::{Pcs, TablePlan};
use crate::{Error, Key, Relation};

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
    let mut transcript = Poseidon2Transcript::new_v1_2();
    transcript.absorb_v1_2(&[F::from_u64(3), F::from_u64(extra)]);
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
fn whir_opens_two_tables_through_the_seeded_challenger() {
    // Table 0: one column of 2^12, opened at two points. Table 1: three
    // columns of 2^3 (below the first folding round), opened at one point.
    let plans = vec![
        TablePlan {
            variables: 12,
            width: 1,
            points: vec![vec![0]; 2],
        },
        TablePlan {
            variables: 3,
            width: 3,
            points: vec![vec![0, 2]],
        },
    ];
    let column = |seed: u64, variables: usize| -> Vec<Gl> {
        (0..1u64 << variables)
            .map(|i| Gl::from_u64(word(seed, i)))
            .collect()
    };
    let large = column(7, 12);
    let small: Vec<Vec<Gl>> = (0..3).map(|c| column(20 + c, 3)).collect();
    let points: Vec<Vec<Ext>> = [(0, 12), (1, 12), (2, 3)]
        .iter()
        .map(|&(batch, variables)| {
            (0..variables)
                .map(|c| ext(100 * batch + c as u64))
                .collect()
        })
        .collect();
    let expected = vec![
        vec![mle(&large, &points[0])],
        vec![mle(&large, &points[1])],
        vec![mle(&small[0], &points[2]), mle(&small[2], &points[2])],
    ];
    let pcs = Pcs::new(crate::pcs::LAYER1, plans, 100.0, &|_| Vec::new()).unwrap();
    assert!(pcs.security_bits() >= 100.0);

    let mut prover = challenger(seed(fold_transcript(1)));
    let (commitment, data) = pcs
        .commit(vec![large, small.concat()], &mut prover, &Accelerator::cpu())
        .unwrap();
    let opening = pcs
        .open(data, &points, &mut prover, &Accelerator::cpu())
        .unwrap();

    let verify = |transcript: Poseidon2Transcript, points: &[Vec<Ext>], opening| {
        let mut verifier = challenger(seed(transcript));
        pcs.observe(&commitment, &mut verifier);
        pcs.verify(&commitment, opening, points, &mut verifier)
    };
    assert_eq!(verify(fold_transcript(1), &points, &opening).unwrap(), expected);

    // A different fold transcript seeds a different challenger.
    assert!(verify(fold_transcript(2), &points, &opening).is_err());
    // A point the prover did not open.
    let mut moved = points.clone();
    moved[2][0] += Ext::ONE;
    assert!(verify(fold_transcript(1), &moved, &opening).is_err());
    // A claimed value that the commitment does not hold.
    let mut changed = opening.clone();
    let batch = &changed.evals[2];
    let mut current = batch.current().to_vec();
    current[1] += Ext::ONE;
    changed.evals[2] = OpeningBatch::new(current, batch.next().to_vec());
    assert!(verify(fold_transcript(1), &points, &changed).is_err());
}

/// A key of `2^block_variables` blocks and one run.
fn wide_key(block_variables: usize) -> Key {
    Key {
        root: [Gl::ZERO; 4],
        blocks: 1 << block_variables,
        rows: 1,
        structure: Structure {
            matrices: 1,
            row_variables: 1,
            run_variables: 1,
            slot_variables: 1,
            run_bounds: vec![0, 1],
            classes: Vec::new(),
            segments: Vec::new(),
            slots: 0,
        },
    }
}

/// `query_half` in `formal/compression-lean` needs at most 35 setup
/// variables. P1 stacks `6·2^(n-3)` cells, and WHIR refuses a first folded
/// domain above `2^32`, so no relation has more.
#[test]
fn relations_refuse_setups_past_31_variables() {
    let p1 = |key: &Key| {
        Pcs::new(crate::pcs::LAYER1, crate::relation::p1_plans(key, 0), 101.0, &|_| {
            Vec::new()
        })
        .is_ok()
    };
    let fits = wide_key(29);
    assert_eq!(fits.setup_variables(), 35);
    assert!(p1(&fits));
    let wide = wide_key(30);
    assert_eq!(wide.setup_variables(), 36);
    assert!(!p1(&wide));
    // WHIR's own bound is looser: the setup check needs a root of unity of
    // order 2^(setup variables + 1), and Goldilocks has none past 2^32.
    let past = wide_key(26);
    assert_eq!(past.setup_variables(), 32);
    assert!(p1(&past));
    assert!(matches!(
        Relation::new(&past, 1, 36, 1, 100.0),
        Err(Error::Setup("setup size"))
    ));
}

#[test]
fn production_shape_reaches_the_required_security() {
    // The production shape at e402a6d99: 814,144 blocks (a 2^26 cube),
    // 1,004,131 rows, t = 4, 5 public blocks, a 28-variable point, H = 3,672;
    // 38,533,993 runs and 1,070,525 slots (census, 2026-10-07). 117 bits is
    // the design's estimate of the layer-1 share of a 114-bit total; the
    // real target is derived at run time.
    let key = Key {
        root: [Gl::ZERO; 4],
        blocks: 814_144,
        rows: 1_004_131,
        structure: Structure {
            matrices: 4,
            row_variables: 20,
            run_variables: 26,
            slot_variables: 21,
            run_bounds: vec![0, 10_000_000, 20_000_000, 30_000_000, 38_533_993],
            classes: Vec::new(),
            segments: Vec::new(),
            slots: 1_070_525,
        },
    };
    let relation = Relation::new(&key, 5, 28, 3_672, 117.0).unwrap();
    assert!(relation.security_bits() >= 117.0, "{}", relation.security_bits());
    eprintln!(
        "P0 {:.1} bits, P1 {:.1} bits, {} setup queries",
        relation.p0.security_bits(),
        relation.p1.security_bits(),
        relation.queries
    );
}
