use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use p3_challenger_v08::{CanObserve, CanSample, CanSampleUniformBits, FieldChallenger};
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};

use crate::circuit::hash::Duplex;
use crate::circuit::record::{Recorder, Trace};
use crate::circuit::{algebra, Backend, Native};
use crate::field::{Ext, Gl};
use crate::hash::{permutation, Challenger};
use crate::whir::query_index;
use crate::Error;

const REJECTED: u64 = Gl::ORDER_U64 - 1;

fn state() -> [u64; WIDTH] {
    std::array::from_fn(|lane| 19 + 23 * lane as u64)
}

fn buffered<B: Backend>(b: &mut B, next: &[u64]) -> Duplex<B> {
    let mut duplex = Duplex::new(b);
    duplex.state = state().map(|word| b.private(word));
    duplex.output = next.iter().rev().map(|&word| b.private(word)).collect();
    duplex
}

fn challenger(next: &[u64]) -> Challenger {
    let mut challenger = Challenger::new(permutation().clone());
    challenger.sponge_state = state().map(Gl::from_u64);
    challenger.output_buffer = next.iter().rev().copied().map(Gl::from_u64).collect();
    challenger
}

fn compare_query(ours: &mut Duplex<Native>, theirs: &mut Challenger, width: usize) {
    let expected = theirs.sample_uniform_bits::<true>(width).unwrap();
    let (bits, index) = query_index(&mut Native, ours, width).unwrap();
    assert_eq!(index.as_canonical_u64(), expected as u64);
    assert_eq!(bits.len(), width);
    for (i, bit) in bits.into_iter().enumerate() {
        assert_eq!(bit, Gl::from_usize((expected >> i) & 1));
    }
}

fn compare_continuation(ours: &mut Duplex<Native>, theirs: &mut Challenger) {
    let extension: Ext = theirs.sample_algebra_element();
    assert_eq!(ours.sample_ext(&mut Native), extension);
    for _ in 0..29 {
        assert_eq!(ours.sample(&mut Native), CanSample::<Gl>::sample(theirs));
    }
    for count in [1, 12, 17] {
        let words: Vec<_> = (0..count).map(|i| Gl::from_u64(71 + i)).collect();
        ours.observe_slice(&mut Native, &words);
        theirs.observe_slice(&words);
        let extension: Ext = theirs.sample_algebra_element();
        assert_eq!(ours.sample_ext(&mut Native), extension);
    }
}

#[test]
fn whir_redraws_preserve_the_complete_transcript() {
    for rejected in 0..=2 {
        for width in [1, 19, 25, 32] {
            let mut next = vec![REJECTED; rejected];
            next.extend([7, 19, 31, 43]);
            let mut ours = buffered(&mut Native, &next);
            let mut theirs = challenger(&next);
            compare_query(&mut ours, &mut theirs, width);
            compare_continuation(&mut ours, &mut theirs);
        }
    }
}

#[test]
fn whir_redraws_cross_the_output_buffer_boundary() {
    for rejected in 1..=2 {
        let next = vec![REJECTED; rejected];
        let mut ours = buffered(&mut Native, &next);
        let mut theirs = challenger(&next);
        compare_query(&mut ours, &mut theirs, 25);
        compare_continuation(&mut ours, &mut theirs);
    }
}

#[test]
fn whir_redraw_limit_is_per_query() {
    let next = [REJECTED, 7, REJECTED, REJECTED, 9, 11, REJECTED, 13, 17, 19, 23, 29];
    let mut ours = buffered(&mut Native, &next);
    let mut theirs = challenger(&next);
    for _ in 0..6 {
        compare_query(&mut ours, &mut theirs, 25);
    }
    compare_continuation(&mut ours, &mut theirs);
}

#[test]
fn whir_rejects_after_two_redraws() {
    let next = [REJECTED, REJECTED, REJECTED, 7];
    let mut ours = buffered(&mut Native, &next);
    assert!(matches!(
        query_index(&mut Native, &mut ours, 25),
        Err(Error::Rejected("WHIR query retry limit"))
    ));
}

#[test]
fn observe_after_a_redraw_discards_only_unused_output() {
    for rejected in 0..=2 {
        let mut next = vec![REJECTED; rejected];
        next.extend([7, 19, 31]);
        let mut ours = buffered(&mut Native, &next);
        let mut theirs = challenger(&next);
        compare_query(&mut ours, &mut theirs, 25);
        let observed = Gl::from_u64(97);
        ours.observe(&mut Native, observed);
        theirs.observe(observed);
        compare_continuation(&mut ours, &mut theirs);
    }
}

#[test]
fn disabled_sampling_preserves_an_empty_buffer() {
    let mut ours = buffered(&mut Native, &[]);
    let mut theirs = challenger(&[]);
    assert_eq!(ours.sample_if(&mut Native, Gl::ZERO), Gl::ZERO);
    compare_continuation(&mut ours, &mut theirs);
}

fn record_query(next: &[u64]) -> (Trace, Option<&'static str>, Gl) {
    let mut recorder = Recorder::new(Trace::default());
    let mut duplex = buffered(&mut recorder, next);
    let (_, index) = query_index(&mut recorder, &mut duplex, 25).unwrap();
    let _ = duplex.sample_ext(&mut recorder);
    let observed = recorder.private(97);
    duplex.observe(&mut recorder, observed);
    let _ = duplex.sample_ext(&mut recorder);
    let (trace, failure) = recorder.finish();
    (trace, failure, index.value())
}

#[test]
fn redraw_rows_are_fixed_and_accept_the_native_values() {
    let cases = [
        [7, 11, 13, 17],
        [REJECTED, 7, 13, 17],
        [REJECTED, REJECTED, 7, 17],
        [REJECTED, REJECTED, REJECTED, 7],
    ];
    let (baseline, _, _) = record_query(&cases[0]);
    let digest = baseline.digest();
    for (rejected, next) in cases.iter().enumerate() {
        let (trace, failure, index) = record_query(next);
        assert_eq!(trace.rows, baseline.rows);
        assert_eq!(trace.entries, baseline.entries);
        assert_eq!(trace.digest(), digest);
        assert_eq!(trace.failing_block(), None);
        let failed = (0..trace.rows.len()).any(|row| trace.row_value(row) != Gl::ZERO);
        if rejected < 3 {
            assert_eq!(failure, None);
            assert_eq!(index, Gl::from_u64(7));
            assert!(!failed);
        } else {
            assert_eq!(failure, Some("WHIR query retry limit"));
            assert!(failed);
        }
    }
}

#[test]
fn zero_test_rejects_a_forged_inverse_and_matching_products() {
    let mut recorder = Recorder::new(Trace::default());
    let value = recorder.private(7);
    let result = algebra::is_zero(&mut recorder, value).unwrap();
    assert_eq!(result.value(), Gl::ZERO);
    let (mut trace, failure) = recorder.finish();
    assert_eq!(failure, None);
    assert_eq!(trace.glue.len(), 4);
    trace.glue[1] = Gl::ZERO;
    trace.glue[2] = Gl::ZERO;
    trace.glue[3] = Gl::from_u64(7);
    assert_eq!(trace.row_value(0), Gl::ZERO);
    assert_eq!(trace.row_value(1), Gl::ZERO);
    assert_ne!(trace.row_value(2), Gl::ZERO);
}
