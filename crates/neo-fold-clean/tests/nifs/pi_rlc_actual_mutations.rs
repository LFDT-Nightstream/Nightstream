//! Rejection checks on the same accepted PiCCS-to-PiRLC input.

use neo_fold_clean::engine::transcript::{Poseidon2TranscriptSnapshot, Transcript};
use neo_fold_clean::paper::{pi_rlc, relations::ajtai_rlc_mixer};

use super::{CcsStructure, Params, Poseidon2Transcript, RunningClaim, D, F, K, MATRICES};
use p3_field::PrimeCharacteristicRing;

fn rejected(
    name: &str,
    params: &Params,
    structure: &CcsStructure<F>,
    inputs: &[RunningClaim],
    state: [F; 8],
    parent: &RunningClaim,
) {
    let mut verifier = Transcript::session();
    verifier.restore_snapshot(Poseidon2TranscriptSnapshot::from_state_and_absorbed(state, 0));
    let result = pi_rlc::verify(
        &mut verifier,
        params,
        structure,
        ajtai_rlc_mixer,
        inputs,
        &pi_rlc::Proof {
            combined: parent.clone(),
        },
    );
    assert!(result.is_err(), "native PiRLC accepted mutation {name}");
    println!("pi_rlc_mutation={name} rejected={:?}", result.unwrap_err());
}

pub(super) fn check(
    params: &Params,
    structure: &CcsStructure<F>,
    inputs: &[RunningClaim],
    initial: &Poseidon2Transcript,
    parent: &RunningClaim,
) {
    for family in 0..MATRICES + 4 {
        let mut changed = parent.clone();
        let name = match family {
            0 => {
                changed.c.data[0] += F::ONE;
                "parent_commitment".to_owned()
            }
            1 => {
                changed.X[(0, 0)] += F::ONE;
                "parent_public".to_owned()
            }
            2 => {
                changed.r[0] += K::ONE;
                "parent_point".to_owned()
            }
            3 => {
                changed.eval_k[0] += K::ONE;
                "parent_eval_K".to_owned()
            }
            matrix => {
                changed.eval_a[matrix - 4][0] += K::ONE;
                format!("parent_eval_A{}", matrix - 4)
            }
        };
        rejected(&name, params, structure, inputs, initial.state(), &changed);
    }
    for source in 0..inputs.len() {
        let mut changed = inputs.to_vec();
        changed[source].c.data[0] += F::ONE;
        rejected(
            &format!("source_{source}_commitment"),
            params,
            structure,
            &changed,
            initial.state(),
            parent,
        );
    }
    for lane in 0..initial.state().len() {
        let mut state = initial.state();
        state[lane] += F::ONE;
        rejected(
            &format!("transcript_lane_{lane}"),
            params,
            structure,
            inputs,
            state,
            parent,
        );
    }
    let mut changed = parent.clone();
    changed.r.pop();
    rejected("short_point", params, structure, inputs, initial.state(), &changed);
    let mut changed = parent.clone();
    changed.eval_k.pop();
    rejected("short_eval_K", params, structure, inputs, initial.state(), &changed);
    let mut changed = parent.clone();
    changed.eval_a.pop();
    rejected("short_eval_A", params, structure, inputs, initial.state(), &changed);
    let mut changed = parent.clone();
    changed.eval_k[D] = K::ONE;
    rejected(
        "nonzero_eval_K_padding",
        params,
        structure,
        inputs,
        initial.state(),
        &changed,
    );
    for matrix in 0..MATRICES {
        let mut changed = parent.clone();
        changed.eval_a[matrix][D] = K::ONE;
        rejected(
            &format!("nonzero_eval_A{matrix}_padding"),
            params,
            structure,
            inputs,
            initial.state(),
            &changed,
        );
    }
    let mut changed = parent.clone();
    changed.fold_digest[0] ^= 1;
    rejected(
        "parent_fold_digest",
        params,
        structure,
        inputs,
        initial.state(),
        &changed,
    );
}
