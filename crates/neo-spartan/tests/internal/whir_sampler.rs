use p3_challenger_v08::CanSampleUniformBits;
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};

use crate::circuit::hash::Duplex;
use crate::circuit::Native;
use crate::field::Gl;
use crate::hash::{permutation, Challenger};

#[test]
fn honest_whir_query_redraws_p_minus_one() {
    let rejected = Gl::from_u64(Gl::ORDER_U64 - 1);
    let accepted = Gl::from_u64(7);
    let mut ours = Duplex::new(&mut Native);
    ours.output = vec![accepted, rejected];

    let mut prover = Challenger::new(permutation().clone());
    prover.output_buffer = vec![accepted, rejected];
    let expected = prover.sample_uniform_bits::<true>(25).unwrap();
    assert_eq!(expected, 7);

    let (_, actual) = super::query_index(&mut Native, &mut ours, 25)
        .expect("the verifier must redraw the same rejected query sample as the honest WHIR prover");
    assert_eq!(actual.as_canonical_u64(), expected as u64);
}
