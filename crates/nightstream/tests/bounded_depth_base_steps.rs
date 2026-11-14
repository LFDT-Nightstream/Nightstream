//! Both step circuits accept step 0 of the Poseidon2 hash chain with a zero
//! fold of the default slot (the zero proof that today's base step uses):
//! every row of the circuit's relation holds on the witness that the
//! package's program fills, on the source columns and through the slot
//! layout.

mod bounded_depth_support;

use bounded_depth_support::{assert_every_row_holds, canonical, packages, StepZero};
use neo_math::F;
use neo_params::nightstream_p64_k14::K_RHO;
use nightstream_fprime::{
    derive_light_transcript, derive_refresh_transcript, Child, FoldWords, COMMITMENT_WORDS, EVALUATIONS, PUBLIC_WORDS,
};
use p3_field::PrimeCharacteristicRing;

#[test]
fn light_step_zero_holds_every_row() {
    let packages = packages();
    let step = StepZero::new(0);
    let transcript = derive_light_transcript(
        &[0; COMMITMENT_WORDS],
        &canonical(&step.fresh_public_input),
        &vec![vec![[0u64; 2]; 5]; 28],
        &vec![0; 2 * 2 * EVALUATIONS],
    )
    .expect("light transcript");
    let folded = step.folded(&transcript, 1, 1);
    let fold = FoldWords::Light {
        outputs: [vec![[F::ZERO; 2]; EVALUATIONS], vec![[F::ZERO; 2]; EVALUATIONS]],
    };
    assert_every_row_holds(packages.light(), &step.words(folded, fold));
}

#[test]
fn refresh_step_zero_holds_every_row() {
    let packages = packages();
    let step = StepZero::new(1);
    let (children, sources) = (K_RHO as usize, 1 + K_RHO as usize);
    let transcript = derive_refresh_transcript(
        &[0; COMMITMENT_WORDS],
        &canonical(&step.fresh_public_input),
        &vec![0; children * (COMMITMENT_WORDS + 2 * EVALUATIONS)],
        &vec![vec![[0u64; 2]; 5]; 28],
        &vec![0; sources * 2 * EVALUATIONS],
    )
    .expect("refresh transcript");
    let folded = step.folded(&transcript, sources, 0);
    let child = Child {
        commitment: vec![F::ZERO; COMMITMENT_WORDS],
        evaluations: vec![[F::ZERO; 2]; EVALUATIONS],
        digits: vec![F::ZERO; PUBLIC_WORDS],
    };
    let fold = FoldWords::Refresh {
        children: vec![child; children],
        outputs: vec![vec![[F::ZERO; 2]; EVALUATIONS]; sources],
    };
    assert_every_row_holds(packages.refresh(), &step.words(folded, fold));
}
