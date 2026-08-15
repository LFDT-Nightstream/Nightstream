//! Review map from paper obligations to exact enforcement owners.
//!
//! This map is a coverage gate. It does not prove that an owner implements
//! the cited paper equation and it does not authorize constraint removal.

use std::collections::BTreeSet;

use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    TERMINAL_CONTEXT_GUARD_NAMES, TERMINAL_PROOF_GUARD_NAMES, TERMINAL_R1CS_FAMILY_NAMES,
    TERMINAL_STATEMENT_GUARD_NAMES,
};
use neo_fold_clean::paper::construction2::TRIVIAL_PC;
use neo_fold_clean::paper::f_prime::stage as fprime_stage;
use neo_fold_clean::paper::nifs::circuit::stage as nifs_stage;
use neo_fold_clean::paper::reductions::pi_ccs_circuit::stage as pi_ccs_stage;
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;

use super::ExportError;

const FIXED_ONE_PROGRAM: &str = "nightstream.profile.fixed_one_program";
const COMBINED_PRE_FINAL_SPARTAN: &str = "nightstream.lifecycle.combined_pre_final_spartan";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Paper {
    SuperNeo,
    HyperNova,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObligationState {
    Mapped,
    Open,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceKind {
    BaseRowFamily,
    RecursiveRowFamily,
    TerminalRowFamily,
    TerminalNativeGuard,
    TerminalLifecycle,
    FixedProfileInvariant,
    OpenCheck,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObligationEvidence {
    kind: EvidenceKind,
    name: &'static str,
}

impl ObligationEvidence {
    pub fn kind(&self) -> EvidenceKind {
        self.kind
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaperObligation {
    paper: Paper,
    id: &'static str,
    statement: &'static str,
    state: ObligationState,
    evidence: &'static [ObligationEvidence],
}

impl PaperObligation {
    pub fn paper(&self) -> Paper {
        self.paper
    }

    pub fn id(&self) -> &'static str {
        self.id
    }

    pub fn statement(&self) -> &'static str {
        self.statement
    }

    pub fn state(&self) -> ObligationState {
        self.state
    }

    pub fn evidence(&self) -> &'static [ObligationEvidence] {
        self.evidence
    }
}

const OBLIGATIONS: &[PaperObligation] = &[
    PaperObligation {
        paper: Paper::SuperNeo,
        id: "superneo.fresh_ccs_validity",
        statement: "Fresh CCS claims satisfy the selected CCS relation.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::SUMCHECK,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::TERMINAL,
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.fresh.selected_relation",
            },
        ],
    },
    PaperObligation {
        paper: Paper::SuperNeo,
        id: "superneo.carried_shared_point_evaluations",
        statement: "Carried evaluations use the shared evaluation point and remain valid.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::CANONICALITY,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::TERMINAL,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: nifs_stage::POINT_BINDING,
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.running.evaluations",
            },
        ],
    },
    PaperObligation {
        paper: Paper::SuperNeo,
        id: "superneo.norm_checks",
        statement: "Fresh and running witness representations satisfy the selected norm bound.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::TERMINAL,
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.fresh.norm",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.running.norm",
            },
        ],
    },
    PaperObligation {
        paper: Paper::SuperNeo,
        id: "superneo.combined_sumcheck_target_and_separation",
        statement: "The combined sum-check target binds CCS, norm, and carried-evaluation terms with separate coefficients.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::CHALLENGES,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::SUMCHECK,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::TERMINAL,
            },
        ],
    },
    PaperObligation {
        paper: Paper::SuperNeo,
        id: "superneo.rlc_commitment_public_evaluation_updates",
        statement: "Pi_RLC applies one challenge vector to commitments, public inputs, and evaluation claims.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::IDENTITIES_COMMITMENT_FINAL_LIMB_CHECKS,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::IDENTITIES_X_FINAL_LIMB_CHECKS,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::IDENTITIES_Y_RING_FINAL_LIMB_CHECKS,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::PROJECTION_SHARED_RHO_EVALUATIONS,
            },
        ],
    },
    PaperObligation {
        paper: Paper::SuperNeo,
        id: "superneo.decomposition_digit_bounds",
        statement: "Pi_DEC children use the canonical bounded signed-digit alphabet.",
        state: ObligationState::Mapped,
        evidence: &[ObligationEvidence {
            kind: EvidenceKind::RecursiveRowFamily,
            name: nifs_stage::PI_DEC_VERIFY,
        }],
    },
    PaperObligation {
        paper: Paper::SuperNeo,
        id: "superneo.decomposition_commitment_recomposition",
        statement: "Pi_DEC child commitments recompose to the parent commitment.",
        state: ObligationState::Mapped,
        evidence: &[ObligationEvidence {
            kind: EvidenceKind::RecursiveRowFamily,
            name: nifs_stage::PI_DEC_VERIFY,
        }],
    },
    PaperObligation {
        paper: Paper::SuperNeo,
        id: "superneo.decomposition_public_recomposition",
        statement: "Pi_DEC child public inputs recompose to the parent public input.",
        state: ObligationState::Mapped,
        evidence: &[ObligationEvidence {
            kind: EvidenceKind::RecursiveRowFamily,
            name: nifs_stage::PI_DEC_VERIFY,
        }],
    },
    PaperObligation {
        paper: Paper::SuperNeo,
        id: "superneo.decomposition_evaluation_recomposition",
        statement: "Pi_DEC child evaluation claims recompose to the parent evaluation claims.",
        state: ObligationState::Mapped,
        evidence: &[ObligationEvidence {
            kind: EvidenceKind::RecursiveRowFamily,
            name: nifs_stage::PI_DEC_VERIFY,
        }],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.canonical_default_and_base",
        statement: "The zero-step state and base accumulator use canonical verifier-derived values.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::BaseRowFamily,
                name: fprime_stage::BASE_SOURCE,
            },
            ObligationEvidence {
                kind: EvidenceKind::BaseRowFamily,
                name: fprime_stage::BASE_INITIAL,
            },
            ObligationEvidence {
                kind: EvidenceKind::BaseRowFamily,
                name: fprime_stage::BASE_OUTPUT,
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.initial_semantic_state",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.initial_boundary",
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.program_counter_and_selected_function",
        statement: "The program counter is in range and selects the fixed F-prime function before use.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::BaseRowFamily,
                name: fprime_stage::BASE_PRELUDE,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_PRELUDE,
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.program_counter",
            },
            ObligationEvidence {
                kind: EvidenceKind::FixedProfileInvariant,
                name: FIXED_ONE_PROGRAM,
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.prior_state_binding",
        statement: "The fresh instance binds the verifier key, counters, initial state, current state, running accumulator, and program counter.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_TRANSCRIPT,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_PRIOR_LINK_DIGEST,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_PRIOR_LINK_ENC_INST,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_ACCUMULATOR_INPUT,
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.verifier_key",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.counters",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.fresh_boundary",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.running_accumulator",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.state_x_out",
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.selected_nifs_and_unchanged_slots",
        statement: "The selected running slot is updated by NIFS and every other slot is unchanged.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::TERMINAL,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::IDENTITIES_COMMITMENT_FINAL_LIMB_CHECKS,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: nifs_stage::PI_DEC_VERIFY,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_ACCUMULATOR_OUTPUT_CHILD_DIGESTS,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_ACCUMULATOR_OUTPUT_AGGREGATE,
            },
            ObligationEvidence {
                kind: EvidenceKind::FixedProfileInvariant,
                name: FIXED_ONE_PROGRAM,
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.fresh_and_running_relation_membership",
        statement: "Terminal acceptance checks the fresh CCS relation and every running CE relation.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.fresh.commitment",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.fresh.norm",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.fresh.public_projection",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.fresh.selected_relation",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.running.commitment",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.running.evaluations",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.running.norm",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalRowFamily,
                name: "terminal.running.public_projection",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.proof.spartan_verification",
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.canonical_encoding_and_decoding",
        statement: "Recursive public links use the canonical instance encoding and fixed carrier padding.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::BaseRowFamily,
                name: fprime_stage::BASE_SOURCE,
            },
            ObligationEvidence {
                kind: EvidenceKind::BaseRowFamily,
                name: fprime_stage::BASE_OUTPUT,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_PRIOR_LINK_ENC_INST,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_PRIOR_LINK_CARRIER_PADDING,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_OUTPUT,
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.fresh_public_link",
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.transcript_schedule_and_statement_binding",
        statement: "The transcript schedule binds the complete recursive statement before each challenge.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_TRANSCRIPT,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::PREFIX,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::CHALLENGES,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_ccs_stage::OUTPUT_TRANSCRIPT,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::PROJECTION_BINDING_TRANSCRIPT_BETA,
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.compact_verifier_projection",
        statement: "The recursive circuit uses fixed compact verifier projections and statement identifiers.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::PROJECTION_BINDING_SIS_DIGEST,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::PROJECTION_BINDING_TRANSCRIPT_BETA,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::PROJECTION_SHARED_BETA_LADDER,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: pi_rlc_stage::IDENTITIES_COMMITMENT_FINAL_LIMB_CHECKS,
            },
            ObligationEvidence {
                kind: EvidenceKind::FixedProfileInvariant,
                name: FIXED_ONE_PROGRAM,
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.poseidon2_state_binding",
        statement: "Protocol state and recursive links use Poseidon2 and recomputed authoritative inputs.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_PRIOR_LINK_DIGEST,
            },
            ObligationEvidence {
                kind: EvidenceKind::RecursiveRowFamily,
                name: fprime_stage::RECURSIVE_OUTPUT,
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.state_x_out",
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.terminal_linkage",
        statement: "The terminal verifier closes the exact combined Nebula running and fresh relations.",
        state: ObligationState::Mapped,
        evidence: &[
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.statement.fresh_public_link",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.proof.expected_public_image",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalNativeGuard,
                name: "terminal.proof.public_statement",
            },
            ObligationEvidence {
                kind: EvidenceKind::TerminalLifecycle,
                name: COMBINED_PRE_FINAL_SPARTAN,
            },
        ],
    },
    PaperObligation {
        paper: Paper::HyperNova,
        id: "hypernova.recursive_size_closure",
        statement: "The frozen production relation reaches a recursive fixed point after every accepted removal batch.",
        state: ObligationState::Open,
        evidence: &[ObligationEvidence {
            kind: EvidenceKind::OpenCheck,
            name: "freeze the production profile and solve the regenerated recursive fixed point",
        }],
    },
];

pub fn paper_obligation_ledger() -> &'static [PaperObligation] {
    OBLIGATIONS
}

/// Check ledger structure and bind every row-family name to a reviewed exact
/// source vocabulary. Open checks remain open and cannot pass as mapped.
pub fn validate_paper_obligation_ledger(
    base_families: &[&str],
    recursive_families: &[&str],
) -> Result<(), ExportError> {
    let base = base_families.iter().copied().collect::<BTreeSet<_>>();
    let recursive = recursive_families.iter().copied().collect::<BTreeSet<_>>();
    let terminal = TERMINAL_R1CS_FAMILY_NAMES
        .into_iter()
        .collect::<BTreeSet<_>>();
    let terminal_guards = TERMINAL_CONTEXT_GUARD_NAMES
        .into_iter()
        .chain(TERMINAL_STATEMENT_GUARD_NAMES)
        .chain(TERMINAL_PROOF_GUARD_NAMES)
        .collect::<BTreeSet<_>>();
    let mut ids = BTreeSet::new();
    let mut papers = BTreeSet::new();

    for obligation in OBLIGATIONS {
        if obligation.id.trim().is_empty()
            || obligation.statement.trim().is_empty()
            || obligation.evidence.is_empty()
            || !ids.insert(obligation.id)
        {
            return Err(ExportError::new(
                "paper obligation ledger has an invalid or duplicate entry",
            ));
        }
        papers.insert(obligation.paper);
        let mut evidence = BTreeSet::new();
        let mut has_open_check = false;
        for item in obligation.evidence {
            if item.name.trim().is_empty() || !evidence.insert((item.kind, item.name)) {
                return Err(ExportError::new(format!(
                    "paper obligation {:?} has empty or duplicate evidence",
                    obligation.id
                )));
            }
            let known = match item.kind {
                EvidenceKind::BaseRowFamily => base.contains(item.name),
                EvidenceKind::RecursiveRowFamily => recursive.contains(item.name),
                EvidenceKind::TerminalRowFamily => terminal.contains(item.name),
                EvidenceKind::TerminalNativeGuard => terminal_guards.contains(item.name),
                EvidenceKind::TerminalLifecycle => item.name == COMBINED_PRE_FINAL_SPARTAN,
                EvidenceKind::FixedProfileInvariant => item.name == FIXED_ONE_PROGRAM && TRIVIAL_PC == 1,
                EvidenceKind::OpenCheck => {
                    has_open_check = true;
                    true
                }
            };
            if !known {
                return Err(ExportError::new(format!(
                    "paper obligation {:?} cites unknown {:?} evidence {:?}",
                    obligation.id, item.kind, item.name
                )));
            }
        }
        if matches!(obligation.state, ObligationState::Open) != has_open_check {
            return Err(ExportError::new(format!(
                "paper obligation {:?} has an inconsistent open state",
                obligation.id
            )));
        }
    }
    if papers != BTreeSet::from([Paper::SuperNeo, Paper::HyperNova]) {
        return Err(ExportError::new("paper obligation ledger does not cover both papers"));
    }
    Ok(())
}
