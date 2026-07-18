//! Structural regression for Π_RLC identity authority groups.
//!
//! Owns: the exact public/delayed-NC/Nebula partition and `y_zcol` limb split.
//! Does not own: identity arithmetic, costs, or constraint emission.
//! Emits constraints: no.
//!
//! | Stage | Obligation | Authority | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `nifs.pi_rlc.verify.identities` | Partition identity leaves by authority boundary | organizational | `pi_rlc_circuit::stage` | `FPrimeFullHistory.NifsPaper.PiRlc` |
//! | `nifs.pi_rlc.verify.identities.y_zcol.*` | Attribute each arithmetic phase to exactly one carried limb | diagnostic | `pi_rlc_circuit::stage` | exact lowering bridge open |

use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;

fn children(parent: &str) -> &'static [&'static str] {
    pi_rlc_stage::HIERARCHY
        .iter()
        .find_map(|(node, children)| (*node == parent).then_some(*children))
        .expect("PiRLC identity group")
}

#[test]
fn pi_rlc_identity_authority_groups_are_explicit() {
    assert_eq!(
        children(pi_rlc_stage::IDENTITIES),
        &[
            pi_rlc_stage::IDENTITIES_PUBLIC,
            pi_rlc_stage::IDENTITIES_DELAYED_NC,
            pi_rlc_stage::IDENTITIES_NEBULA,
        ]
    );
    assert_eq!(
        children(pi_rlc_stage::IDENTITIES_PUBLIC),
        &[
            pi_rlc_stage::IDENTITIES_COMMITMENT,
            pi_rlc_stage::IDENTITIES_X,
            pi_rlc_stage::IDENTITIES_Y_RING,
        ]
    );
    assert_eq!(
        children(pi_rlc_stage::IDENTITIES_DELAYED_NC),
        &[pi_rlc_stage::IDENTITIES_Y_ZCOL]
    );
    assert_eq!(
        children(pi_rlc_stage::IDENTITIES_NEBULA),
        &[pi_rlc_stage::IDENTITIES_ADV]
    );

    for group in [
        pi_rlc_stage::IDENTITIES_PUBLIC,
        pi_rlc_stage::IDENTITIES_DELAYED_NC,
        pi_rlc_stage::IDENTITIES_NEBULA,
    ] {
        assert!(pi_rlc_stage::ALL.contains(&group));
        assert!(pi_rlc_stage::IDENTITY_PHASE_NODES.contains(&group));
    }
}

#[test]
fn pi_rlc_y_zcol_phases_are_split_by_identity_limb() {
    use pi_rlc_stage as s;
    let limb0 = s::Y_ZCOL_LIMB0_IDENTITY_STAGES;
    let limb1 = s::Y_ZCOL_LIMB1_IDENTITY_STAGES;
    for (parent, limbs) in [
        (
            s::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS,
            [limb0.input_evaluations, limb1.input_evaluations],
        ),
        (
            s::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT,
            [limb0.output_evaluation, limb1.output_evaluation],
        ),
        (
            s::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT,
            [limb0.quotient_evaluation, limb1.quotient_evaluation],
        ),
        (
            s::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT,
            [limb0.rho_times_input, limb1.rho_times_input],
        ),
        (
            s::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI,
            [limb0.quotient_times_phi, limb1.quotient_times_phi],
        ),
        (
            s::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS,
            [limb0.final_limb_checks, limb1.final_limb_checks],
        ),
    ] {
        assert_eq!(children(parent), limbs);
        for limb in limbs {
            assert!(s::ALL.contains(&limb));
            assert!(s::IDENTITY_PHASE_NODES.contains(&limb));
        }
    }
}
