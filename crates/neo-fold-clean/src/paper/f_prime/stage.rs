//! Stable diagnostic paths for the complete base and recursive F-prime trees.
//!
//! Owns: branch -> phase -> constraint-family names and immediate-child
//! ownership for the complete augmented relation.
//!
//! Does not own: constraint emission, verifier semantics, or measured totals.
//!
//! Emits constraints: no.
//!
//! Authority boundary: these labels are profiler metadata only. Source R1CS
//! rows and validated lowering traces remain the acceptance and cost authority.
//!
//! | Child phase | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | verifier key | Fix context and transcript parameters | yes | `full_relation` | full-relation bridge open |
//! | step prelude | Allocate and bind incoming state/source data | yes | `r1cs` | FPrime bridge open |
//! | recursive NIFS | Verify Pi_CCS, Pi_RLC, Pi_DEC, and point binding | yes | `paper/nifs/circuit` | SuperNeo bridge partial |
//! | prior/accumulator links | Bind recursive and accumulator authority | yes | `r1cs` | authority bridge open |
//! | counter/output | Advance state and derive exact `x_out` | yes | `r1cs` | FPrime bridge open |
//! | finalization | Bind context, application, and semantic state | yes | `full_relation` | full-relation bridge open |

use crate::paper::nifs::circuit::stage as nifs_stage;
use crate::paper::reductions::pi_ccs_circuit::stage as pi_ccs_stage;
use crate::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;

pub const BASE_ROOT: &str = "fprime.base";
pub const BASE_VERIFIER_KEY: &str = "fprime.base.verifier_key";
pub const BASE_STEP: &str = "fprime.base.step";
pub const BASE_PRELUDE: &str = "fprime.base.step.prelude";
pub const BASE_SOURCE: &str = "fprime.base.step.source";
pub const BASE_INITIAL: &str = "fprime.base.step.initial";
pub const BASE_ADVANCE: &str = "fprime.base.step.advance";
pub const BASE_OUTPUT: &str = "fprime.base.step.output";
pub const BASE_FINALIZE: &str = "fprime.base.finalize";
pub const BASE_CONTEXT_LINK: &str = "fprime.base.finalize.context_link";
pub const BASE_APPLICATION: &str = "fprime.base.finalize.application";
pub const BASE_SEMANTIC_LINKS: &str = "fprime.base.finalize.semantic_links";

pub const BASE_ALL: &[&str] = &[
    BASE_ROOT,
    BASE_VERIFIER_KEY,
    BASE_STEP,
    BASE_PRELUDE,
    BASE_SOURCE,
    BASE_INITIAL,
    BASE_ADVANCE,
    BASE_OUTPUT,
    BASE_FINALIZE,
    BASE_CONTEXT_LINK,
    BASE_APPLICATION,
    BASE_SEMANTIC_LINKS,
];

pub const BASE_HIERARCHY: &[(&str, &[&str])] = &[
    (BASE_ROOT, &[BASE_VERIFIER_KEY, BASE_STEP, BASE_FINALIZE]),
    (
        BASE_STEP,
        &[BASE_PRELUDE, BASE_SOURCE, BASE_INITIAL, BASE_ADVANCE, BASE_OUTPUT],
    ),
    (
        BASE_FINALIZE,
        &[BASE_CONTEXT_LINK, BASE_APPLICATION, BASE_SEMANTIC_LINKS],
    ),
];

pub const RECURSIVE_ROOT: &str = "fprime.recursive";
pub const RECURSIVE_VERIFIER_KEY: &str = "fprime.recursive.verifier_key";
pub const RECURSIVE_STEP: &str = "fprime.recursive.step";
pub const RECURSIVE_PRELUDE: &str = "fprime.recursive.step.prelude";
pub const RECURSIVE_TRANSCRIPT: &str = "fprime.recursive.step.transcript";
pub const RECURSIVE_NIFS: &str = "fprime.recursive.step.nifs";
pub const RECURSIVE_PRIOR_LINK: &str = "fprime.recursive.step.prior_link";
pub const RECURSIVE_PRIOR_LINK_DIGEST: &str = "fprime.recursive.step.prior_link.digest";
pub const RECURSIVE_PRIOR_LINK_ENC_INST: &str = "fprime.recursive.step.prior_link.enc_inst";
pub const RECURSIVE_PRIOR_LINK_CARRIER_PADDING: &str = "fprime.recursive.step.prior_link.carrier_padding";
pub const RECURSIVE_NEBULA: &str = "fprime.recursive.step.nebula";
pub const RECURSIVE_ACCUMULATOR: &str = "fprime.recursive.step.accumulator";
pub const RECURSIVE_ACCUMULATOR_INPUT: &str = "fprime.recursive.step.accumulator.input_link";
pub const RECURSIVE_ACCUMULATOR_OUTPUT: &str = "fprime.recursive.step.accumulator.output_authority";
pub const RECURSIVE_ACCUMULATOR_OUTPUT_CLAIM: &str =
    "fprime.recursive.step.accumulator.output_authority.claimed_digest";
pub const RECURSIVE_ACCUMULATOR_OUTPUT_CHILD_DIGESTS: &str =
    "fprime.recursive.step.accumulator.output_authority.child_digests";
pub const RECURSIVE_ACCUMULATOR_OUTPUT_AGGREGATE: &str = "fprime.recursive.step.accumulator.output_authority.aggregate";
pub const RECURSIVE_COUNTERS: &str = "fprime.recursive.step.counters";
pub const RECURSIVE_OUTPUT: &str = "fprime.recursive.step.output";
pub const RECURSIVE_FINALIZE: &str = "fprime.recursive.finalize";
pub const RECURSIVE_CONTEXT_LINK: &str = "fprime.recursive.finalize.context_link";
pub const RECURSIVE_APPLICATION: &str = "fprime.recursive.finalize.application";
pub const RECURSIVE_SEMANTIC_LINKS: &str = "fprime.recursive.finalize.semantic_links";

/// F-prime-owned nodes. The complete recursive audit additionally unions the
/// Pi_CCS, Pi_RLC, and NIFS-tail node sets referenced by `RECURSIVE_HIERARCHY`.
pub const RECURSIVE_ALL: &[&str] = &[
    RECURSIVE_ROOT,
    RECURSIVE_VERIFIER_KEY,
    RECURSIVE_STEP,
    RECURSIVE_PRELUDE,
    RECURSIVE_TRANSCRIPT,
    RECURSIVE_NIFS,
    RECURSIVE_PRIOR_LINK,
    RECURSIVE_PRIOR_LINK_DIGEST,
    RECURSIVE_PRIOR_LINK_ENC_INST,
    RECURSIVE_PRIOR_LINK_CARRIER_PADDING,
    RECURSIVE_NEBULA,
    RECURSIVE_ACCUMULATOR,
    RECURSIVE_ACCUMULATOR_INPUT,
    RECURSIVE_ACCUMULATOR_OUTPUT,
    RECURSIVE_ACCUMULATOR_OUTPUT_CLAIM,
    RECURSIVE_ACCUMULATOR_OUTPUT_CHILD_DIGESTS,
    RECURSIVE_ACCUMULATOR_OUTPUT_AGGREGATE,
    RECURSIVE_COUNTERS,
    RECURSIVE_OUTPUT,
    RECURSIVE_FINALIZE,
    RECURSIVE_CONTEXT_LINK,
    RECURSIVE_APPLICATION,
    RECURSIVE_SEMANTIC_LINKS,
];

pub const RECURSIVE_HIERARCHY: &[(&str, &[&str])] = &[
    (
        RECURSIVE_ROOT,
        &[RECURSIVE_VERIFIER_KEY, RECURSIVE_STEP, RECURSIVE_FINALIZE],
    ),
    (
        RECURSIVE_STEP,
        &[
            RECURSIVE_PRELUDE,
            RECURSIVE_TRANSCRIPT,
            RECURSIVE_NIFS,
            RECURSIVE_PRIOR_LINK,
            RECURSIVE_NEBULA,
            RECURSIVE_ACCUMULATOR,
            RECURSIVE_COUNTERS,
            RECURSIVE_OUTPUT,
        ],
    ),
    (
        RECURSIVE_NIFS,
        &[
            pi_ccs_stage::ROOT,
            pi_rlc_stage::ROOT,
            nifs_stage::RUNNING_PARENT_PI_DEC,
            nifs_stage::PI_DEC,
            nifs_stage::POINT_BINDING,
        ],
    ),
    (
        RECURSIVE_PRIOR_LINK,
        &[
            RECURSIVE_PRIOR_LINK_DIGEST,
            RECURSIVE_PRIOR_LINK_ENC_INST,
            RECURSIVE_PRIOR_LINK_CARRIER_PADDING,
        ],
    ),
    (
        RECURSIVE_ACCUMULATOR,
        &[RECURSIVE_ACCUMULATOR_INPUT, RECURSIVE_ACCUMULATOR_OUTPUT],
    ),
    (
        RECURSIVE_ACCUMULATOR_OUTPUT,
        &[
            RECURSIVE_ACCUMULATOR_OUTPUT_CLAIM,
            RECURSIVE_ACCUMULATOR_OUTPUT_CHILD_DIGESTS,
            RECURSIVE_ACCUMULATOR_OUTPUT_AGGREGATE,
        ],
    ),
    (
        RECURSIVE_FINALIZE,
        &[RECURSIVE_CONTEXT_LINK, RECURSIVE_APPLICATION, RECURSIVE_SEMANTIC_LINKS],
    ),
];
