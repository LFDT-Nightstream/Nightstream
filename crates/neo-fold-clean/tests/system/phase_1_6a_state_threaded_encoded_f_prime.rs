//! Phase 1.6a — State-threaded encoded F' fixture.
//!
//! Phase 1.5c-b's encoded F' fixture was *same-shape synthetic
//! repeats*: each fold call built an independent step that ignored
//! prior steps' outputs. That proved the mechanism works but not that
//! the F' state machine threads realistically.
//!
//! This phase introduces [`honest_state_threaded_encoded_f_prime_records`]:
//! a sequence where step `i.state_out` is wired into step `i+1.state_in`
//! for the threaded fields (`z_i`, `public_trace`, `acc_digest`,
//! counters). Tests below pin three properties:
//!
//! 1. **Threading at the snapshot level**: every adjacent pair
//!    `(record[i], record[i+1])` agrees on the threaded fields
//!    (`state_out` of `i` = `state_in` of `i+1`).
//! 2. **Encoded image consistency**: each record's encoded image
//!    decodes to exactly the snapshot the fixture claims, so the
//!    encoder is in fact consuming the threaded state — not just the
//!    builder.
//! 3. **Lifecycle foldability**: the threaded sequence still folds
//!    through `lifecycle::prove`, demonstrating the realistic
//!    threading doesn't break the encoded F' lifecycle path.

#[path = "../support/mod.rs"]
mod support;

use support::fibonacci_f_prime;

use support::fibonacci_f_prime::{
    canonical_threaded_plan, honest_state_threaded_encoded_f_prime_records, honest_state_threaded_encoded_f_prime_steps,
};

#[test]
fn phase_1_6a_state_threaded_records_link_outputs_to_next_inputs() {
    let records = honest_state_threaded_encoded_f_prime_records(4);

    for (i, pair) in records.windows(2).enumerate() {
        let (cur, next) = (&pair[0], &pair[1]);
        assert_eq!(
            cur.state_out.z_i,
            next.state_in.z_i,
            "z_i must thread step {i} → step {}",
            i + 1
        );
        assert_eq!(
            cur.state_out.public_trace,
            next.state_in.public_trace,
            "public_trace must thread step {i} → step {}",
            i + 1
        );
        assert_eq!(
            cur.state_out.acc_digest,
            next.state_in.acc_digest,
            "acc_digest must thread step {i} → step {}",
            i + 1
        );
        assert_eq!(
            cur.state_out.chunk_count,
            next.state_in.chunk_count,
            "chunk_count must thread step {i} → step {}",
            i + 1
        );
        assert_eq!(
            cur.state_out.step_count,
            next.state_in.step_count,
            "step_count must thread step {i} → step {}",
            i + 1
        );
        // Headers stay constant across the chain.
        assert_eq!(cur.state_in.vk_fs_digest, next.state_in.vk_fs_digest);
        assert_eq!(cur.state_in.structure_digest, next.state_in.structure_digest);
        assert_eq!(cur.state_in.z_0, next.state_in.z_0);
        assert_eq!(cur.state_in.pc, next.state_in.pc);
    }
}

#[test]
fn phase_1_6a_encoded_images_decode_to_threaded_state() {
    let records = honest_state_threaded_encoded_f_prime_records(3);

    for (i, record) in records.iter().enumerate() {
        let decoded_in = record.encoded.image.decode_state_in();
        let decoded_out = record.encoded.image.decode_state_out();

        assert_eq!(
            decoded_in.z_i_in, record.state_in.z_i,
            "step {i} encoded state_in.z_i_in must match threaded record"
        );
        assert_eq!(
            decoded_in.public_trace_in, record.state_in.public_trace,
            "step {i} encoded state_in.public_trace_in must match threaded record"
        );
        assert_eq!(
            decoded_in.acc_digest_in, record.state_in.acc_digest,
            "step {i} encoded state_in.acc_digest_in must match threaded record"
        );

        assert_eq!(
            decoded_out.new_chunk_count, record.state_out.chunk_count,
            "step {i} encoded state_out.new_chunk_count must match threaded record"
        );
        assert_eq!(
            decoded_out.new_step_count, record.state_out.step_count,
            "step {i} encoded state_out.new_step_count must match threaded record"
        );
        assert_eq!(
            decoded_out.new_z_i, record.state_out.z_i,
            "step {i} encoded state_out.new_z_i must match threaded record"
        );
        assert_eq!(
            decoded_out.new_public_trace, record.state_out.public_trace,
            "step {i} encoded state_out.new_public_trace must match threaded record"
        );
        assert_eq!(
            decoded_out.new_acc_digest, record.state_out.acc_digest,
            "step {i} encoded state_out.new_acc_digest must match threaded record"
        );
    }
}

#[test]
fn phase_1_6a_state_threaded_encoded_steps_fold_through_lifecycle() {
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0x1F16_A004).expect("preprocess");
    // n = 2 is the minimum that exercises the recursive fold: the
    // first call produces `NoFold`, the second produces
    // `Recursive`. The original fixture used n = 4 under a stub
    // canonical plan; now that the plan matches the real post-fold
    // shape (and each fold is correspondingly expensive), n = 2 still
    // checks the same gate (`running.witnesses.is_empty() == false`
    // after at least one Recursive fold).
    let steps = honest_state_threaded_encoded_f_prime_steps(2);
    let proof = fibonacci_f_prime::prove_encoded_steps(&prep, &steps).expect("prove");

    let running = match &proof.proof.state.proof {
        neo_fold_clean::ProofState::Active { running, .. } => running
            .materialize()
            .expect("running materialization for shape check"),
        _ => panic!("expected ProofState::Active after folding encoded F' steps"),
    };
    assert!(
        !running.witnesses.is_empty(),
        "encoded F' lifecycle should leave running witnesses after >= 2 steps"
    );
}
