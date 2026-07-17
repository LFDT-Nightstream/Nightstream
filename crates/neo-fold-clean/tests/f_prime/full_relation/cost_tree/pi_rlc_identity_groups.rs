//! Structural regression for Π_RLC identity authority groups.
//!
//! Owns: the exact public/delayed-NC/Nebula parent-child partition.
//! Does not own: identity arithmetic, costs, or constraint emission.
//! Emits constraints: no.
//!
//! | Stage | Obligation | Authority | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `nifs.pi_rlc.verify.identities` | Partition identity leaves by authority boundary | organizational | `pi_rlc_circuit::stage` | `FPrimeFullHistory.NifsPaper.PiRlc` |

use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;

#[test]
fn pi_rlc_identity_authority_groups_are_explicit() {
    let children = |parent| {
        pi_rlc_stage::HIERARCHY
            .iter()
            .find_map(|(node, children)| (*node == parent).then_some(*children))
            .expect("PiRLC identity authority group")
    };

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
