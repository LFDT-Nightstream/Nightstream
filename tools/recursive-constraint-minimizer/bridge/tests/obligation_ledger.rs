use std::collections::BTreeSet;

use nightstream_constraint_exporter::{paper_obligation_ledger, ObligationState};

#[test]
fn obligation_ledger_keeps_the_reviewed_scope_and_open_checks() {
    let expected = BTreeSet::from([
        "superneo.fresh_ccs_validity",
        "superneo.carried_shared_point_evaluations",
        "superneo.norm_checks",
        "superneo.combined_sumcheck_target_and_separation",
        "superneo.rlc_commitment_public_evaluation_updates",
        "superneo.decomposition_digit_bounds",
        "superneo.decomposition_commitment_recomposition",
        "superneo.decomposition_public_recomposition",
        "superneo.decomposition_evaluation_recomposition",
        "hypernova.canonical_default_and_base",
        "hypernova.program_counter_and_selected_function",
        "hypernova.prior_state_binding",
        "hypernova.selected_nifs_and_unchanged_slots",
        "hypernova.fresh_and_running_relation_membership",
        "hypernova.canonical_encoding_and_decoding",
        "hypernova.transcript_schedule_and_statement_binding",
        "hypernova.compact_verifier_projection",
        "hypernova.poseidon2_state_binding",
        "hypernova.terminal_linkage",
        "hypernova.recursive_size_closure",
    ]);
    let actual = paper_obligation_ledger()
        .iter()
        .map(|obligation| obligation.id())
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);

    let open = paper_obligation_ledger()
        .iter()
        .filter(|obligation| obligation.state() == ObligationState::Open)
        .map(|obligation| obligation.id())
        .collect::<BTreeSet<_>>();
    assert_eq!(open, BTreeSet::from(["hypernova.recursive_size_closure"]));
}
