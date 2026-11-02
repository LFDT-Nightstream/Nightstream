//! Both step circuits accept step 0 of the Poseidon2 hash chain with a zero
//! fold of the default slot (the zero proof that today's base step uses):
//! every row of the circuit's relation holds on the witness that the
//! package's program fills, on the source columns and through the slot
//! layout.

mod option_b_support;

use nightstream_fprime::{derive_light_transcript, derive_refresh_transcript, Child, FoldWords};
use option_b_support::{assert_every_row_holds, canonical, packages, StepZero};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as G;

#[test]
fn light_step_zero_holds_every_row() {
    let packages = packages();
    let step = StepZero::new(0);
    let transcript = derive_light_transcript(
        &[0; 1188],
        &canonical(&step.fresh_public_input),
        &vec![vec![[0u64; 2]; 9]; 28],
        &vec![0; 2 * 540],
    )
    .expect("light transcript");
    let folded = step.folded(&transcript, 1, 1);
    let fold = FoldWords::Light {
        outputs: [vec![[G::ZERO; 2]; 270], vec![[G::ZERO; 2]; 270]],
    };
    assert_every_row_holds(packages.light(), &step.words(folded, fold));
}

#[test]
fn refresh_step_zero_holds_every_row() {
    let packages = packages();
    let step = StepZero::new(1);
    let transcript = derive_refresh_transcript(
        &[0; 1188],
        &canonical(&step.fresh_public_input),
        &vec![0; 16 * (1188 + 540)],
        &vec![vec![[0u64; 2]; 9]; 28],
        &vec![0; 17 * 540],
    )
    .expect("refresh transcript");
    let folded = step.folded(&transcript, 17, 0);
    let child = Child {
        commitment: vec![G::ZERO; 1188],
        evaluations: vec![[G::ZERO; 2]; 270],
        digits: vec![G::ZERO; 270],
    };
    let fold = FoldWords::Refresh {
        children: vec![child; 16],
        outputs: vec![vec![[G::ZERO; 2]; 270]; 17],
    };
    assert_every_row_holds(packages.refresh(), &step.words(folded, fold));
}
