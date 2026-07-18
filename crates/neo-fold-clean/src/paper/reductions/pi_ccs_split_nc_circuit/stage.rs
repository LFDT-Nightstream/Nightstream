//! Stable diagnostic paths for the in-circuit Pi_CCS verifier cost tree.
//!
//! Owns: the protocol -> phase -> constraint-family vocabulary and its
//! immediate-child hierarchy.
//!
//! Does not own: verifier semantics, constraint emission, or cost totals.
//!
//! Emits constraints: no.
//!
//! Authority boundary: these labels are diagnostic metadata. Production
//! constraints and validated gadget traces remain the cost authority.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `allocate_and_normalize` | Materialize fixed-shape claim wires | yes | `verifier` allocation helpers | concrete refinement open |
//! | `running_authority` | Check the Pi_DEC parent and canonical output views; old-point NC projection remains unbound | yes | `verifier` authority helpers | concrete laundering counterexample; delayed authority bridge open |
//! | `fresh_claim_hashes` | Bind fresh CCS claims to Poseidon2 digests | yes | `digests` | concrete digest bridge open |
//! | `running_parent_hash` | Bind the checked parent and shared point | yes | `digests`, `verifier` | authority bridge open |
//! | `instance_hash_and_absorb` | Derive and absorb the public-instance digest | yes | `digests`, `transcript` | transcript bridge open |
//! | `running_handle_hash_and_absorb` | Hash the exact ordered children and absorb their handle | yes | `digests`, `transcript` | exact-child bridge open |
//! | `engine_challenges` | Derive FE and NC challenges | yes | `transcript` | transcript bridge open |
//! | `fe_claim_and_sumcheck` | Check the FE initial claim and SumCheck | yes | `fe` | FE bridge open |
//! | `nc_sumcheck` | Check the NC SumCheck | yes | `nc` | NC bridge open |
//! | `output_binding_and_terminal_checks` | Bind outputs and both terminal identities | yes | `verifier`, `fe`, `nc` | terminal bridge open |
//! | `...nc_terminal.identity` | Keep terminal arithmetic in one physical lowering stage | yes | `nc` | range semantics partial; remaining terminal bridge open |
//! | `ROW_NC_TERMINAL_*` | Attribute equality, basis, mixing, range, and final-product source rows independently without changing lowering | no | `nc` row-family markers | range semantics partial; remaining terminal bridge open |
//! | `header_catch_up` | Replay the digest cursor and bind output headers | yes | `transcript`, `verifier` | transcript bridge open |
//! | `output_message_hashes` | Bind the Pi_CCS message passed to Pi_RLC | yes | `digests` | output projection bridge open |

pub const ROOT: &str = "nifs.pi_ccs";

pub const ALLOCATE_AND_NORMALIZE: &str = "nifs.pi_ccs.allocate_and_normalize";
pub const ALLOCATE_FRESH: &str = "nifs.pi_ccs.allocate_and_normalize.fresh";
pub const ALLOCATE_RUNNING: &str = "nifs.pi_ccs.allocate_and_normalize.running";
pub const ALLOCATE_RUNNING_PARENT: &str = "nifs.pi_ccs.allocate_and_normalize.running_parent";
pub const ALLOCATE_OUTPUTS: &str = "nifs.pi_ccs.allocate_and_normalize.outputs";

pub const RUNNING_AUTHORITY: &str = "nifs.pi_ccs.running_authority";
pub const RUNNING_AUTHORITY_PARENT_DEC: &str = "nifs.pi_ccs.running_authority.parent_dec";
pub const RUNNING_AUTHORITY_OUTPUT_CT: &str = "nifs.pi_ccs.running_authority.output_ct";
pub const RUNNING_AUTHORITY_OUTPUT_Y_RING_PADDING: &str = "nifs.pi_ccs.running_authority.output_y_ring_padding";
pub const RUNNING_AUTHORITY_OUTPUT_Y_ZCOL_PADDING: &str = "nifs.pi_ccs.running_authority.output_y_zcol_padding";

pub const FRESH_CLAIM_HASHES: &str = "nifs.pi_ccs.fresh_claim_hashes";
pub const FRESH_CLAIM_HASHES_DIGEST: &str = "nifs.pi_ccs.fresh_claim_hashes.digest";

pub const RUNNING_PARENT_HASH: &str = "nifs.pi_ccs.running_parent_hash";
pub const RUNNING_PARENT_HASH_SHARED_R: &str = "nifs.pi_ccs.running_parent_hash.shared_r";
pub const RUNNING_PARENT_HASH_DIGEST: &str = "nifs.pi_ccs.running_parent_hash.digest";

pub const INSTANCE_HASH_AND_ABSORB: &str = "nifs.pi_ccs.instance_hash_and_absorb";
pub const INSTANCE_HASH: &str = "nifs.pi_ccs.instance_hash_and_absorb.instance_digest";
pub const INSTANCE_HEADER_ABSORB: &str = "nifs.pi_ccs.instance_hash_and_absorb.header_absorb";

pub const RUNNING_HANDLE_HASH_AND_ABSORB: &str = "nifs.pi_ccs.running_handle_hash_and_absorb";
pub const RUNNING_HANDLE_CHILD_DIGESTS: &str = "nifs.pi_ccs.running_handle_hash_and_absorb.child_digests";
pub const RUNNING_HANDLE_AGGREGATE: &str = "nifs.pi_ccs.running_handle_hash_and_absorb.aggregate";
pub const RUNNING_HANDLE_ABSORB: &str = "nifs.pi_ccs.running_handle_hash_and_absorb.absorb";

pub const ENGINE_CHALLENGES: &str = "nifs.pi_ccs.engine_challenges";
pub const ENGINE_CHALLENGES_MAIN: &str = "nifs.pi_ccs.engine_challenges.main";
pub const ENGINE_CHALLENGES_BETA_M: &str = "nifs.pi_ccs.engine_challenges.beta_m";

pub const FE_CLAIM_AND_SUMCHECK: &str = "nifs.pi_ccs.fe_claim_and_sumcheck";
pub const FE_CLAIMED_INITIAL: &str = "nifs.pi_ccs.fe_claim_and_sumcheck.claimed_initial";
pub const FE_OPTIONAL_CLAIM: &str = "nifs.pi_ccs.fe_claim_and_sumcheck.optional_claim";
pub const FE_ROUNDS: &str = "nifs.pi_ccs.fe_claim_and_sumcheck.rounds";
pub const FE_SUMCHECK_DRIVER: &str = "nifs.pi_ccs.fe_claim_and_sumcheck.driver";

pub const NC_SUMCHECK: &str = "nifs.pi_ccs.nc_sumcheck";
pub const NC_SUMCHECK_ROUNDS: &str = "nifs.pi_ccs.nc_sumcheck.rounds";
pub const NC_SUMCHECK_DRIVER: &str = "nifs.pi_ccs.nc_sumcheck.driver";

pub const OUTPUT_BINDING_AND_TERMINAL_CHECKS: &str = "nifs.pi_ccs.output_binding_and_terminal_checks";
pub const OUTPUT_BINDING: &str = "nifs.pi_ccs.output_binding_and_terminal_checks.output_binding";
pub const FE_TERMINAL_IDENTITY: &str = "nifs.pi_ccs.output_binding_and_terminal_checks.fe_terminal.identity";
pub const FE_TERMINAL_FINAL_SUM: &str = "nifs.pi_ccs.output_binding_and_terminal_checks.fe_terminal.final_sum";
pub const NC_TERMINAL_IDENTITY: &str = "nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity";
pub const NC_TERMINAL_FINAL_SUM: &str = "nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.final_sum";

pub const HEADER_CATCH_UP: &str = "nifs.pi_ccs.header_catch_up";
pub const HEADER_FIELDS: &str = "nifs.pi_ccs.header_catch_up.fields";
pub const HEADER_TRANSCRIPT: &str = "nifs.pi_ccs.header_catch_up.transcript";
pub const HEADER_OUTPUT_BINDING: &str = "nifs.pi_ccs.header_catch_up.output_binding";

pub const OUTPUT_MESSAGE_HASHES: &str = "nifs.pi_ccs.output_message_hashes";
pub const OUTPUT_MESSAGE_DIGEST: &str = "nifs.pi_ccs.output_message_hashes.digest";
pub const OUTPUT_MESSAGE_CLAIM: &str = "nifs.pi_ccs.output_message_hashes.claim";
pub const OUTPUT_MESSAGE_BINDING: &str = "nifs.pi_ccs.output_message_hashes.binding";

/// Every protocol, phase, and constraint-family node, including zero-cost owners.
pub const ALL: &[&str] = &[
    ROOT,
    ALLOCATE_AND_NORMALIZE,
    ALLOCATE_FRESH,
    ALLOCATE_RUNNING,
    ALLOCATE_RUNNING_PARENT,
    ALLOCATE_OUTPUTS,
    RUNNING_AUTHORITY,
    RUNNING_AUTHORITY_PARENT_DEC,
    RUNNING_AUTHORITY_OUTPUT_CT,
    RUNNING_AUTHORITY_OUTPUT_Y_RING_PADDING,
    RUNNING_AUTHORITY_OUTPUT_Y_ZCOL_PADDING,
    FRESH_CLAIM_HASHES,
    FRESH_CLAIM_HASHES_DIGEST,
    RUNNING_PARENT_HASH,
    RUNNING_PARENT_HASH_SHARED_R,
    RUNNING_PARENT_HASH_DIGEST,
    INSTANCE_HASH_AND_ABSORB,
    INSTANCE_HASH,
    INSTANCE_HEADER_ABSORB,
    RUNNING_HANDLE_HASH_AND_ABSORB,
    RUNNING_HANDLE_CHILD_DIGESTS,
    RUNNING_HANDLE_AGGREGATE,
    RUNNING_HANDLE_ABSORB,
    ENGINE_CHALLENGES,
    ENGINE_CHALLENGES_MAIN,
    ENGINE_CHALLENGES_BETA_M,
    FE_CLAIM_AND_SUMCHECK,
    FE_CLAIMED_INITIAL,
    FE_OPTIONAL_CLAIM,
    FE_ROUNDS,
    FE_SUMCHECK_DRIVER,
    NC_SUMCHECK,
    NC_SUMCHECK_ROUNDS,
    NC_SUMCHECK_DRIVER,
    OUTPUT_BINDING_AND_TERMINAL_CHECKS,
    OUTPUT_BINDING,
    FE_TERMINAL_IDENTITY,
    FE_TERMINAL_FINAL_SUM,
    NC_TERMINAL_IDENTITY,
    NC_TERMINAL_FINAL_SUM,
    HEADER_CATCH_UP,
    HEADER_FIELDS,
    HEADER_TRANSCRIPT,
    HEADER_OUTPUT_BINDING,
    OUTPUT_MESSAGE_HASHES,
    OUTPUT_MESSAGE_DIGEST,
    OUTPUT_MESSAGE_CLAIM,
    OUTPUT_MESSAGE_BINDING,
];

/// Immediate-child ownership used by the exact source/lowered cost audit.
pub const HIERARCHY: &[(&str, &[&str])] = &[
    (
        ROOT,
        &[
            ALLOCATE_AND_NORMALIZE,
            RUNNING_AUTHORITY,
            FRESH_CLAIM_HASHES,
            RUNNING_PARENT_HASH,
            INSTANCE_HASH_AND_ABSORB,
            RUNNING_HANDLE_HASH_AND_ABSORB,
            ENGINE_CHALLENGES,
            FE_CLAIM_AND_SUMCHECK,
            NC_SUMCHECK,
            OUTPUT_BINDING_AND_TERMINAL_CHECKS,
            HEADER_CATCH_UP,
            OUTPUT_MESSAGE_HASHES,
        ],
    ),
    (
        ALLOCATE_AND_NORMALIZE,
        &[
            ALLOCATE_FRESH,
            ALLOCATE_RUNNING,
            ALLOCATE_RUNNING_PARENT,
            ALLOCATE_OUTPUTS,
        ],
    ),
    (
        RUNNING_AUTHORITY,
        &[
            RUNNING_AUTHORITY_PARENT_DEC,
            RUNNING_AUTHORITY_OUTPUT_CT,
            RUNNING_AUTHORITY_OUTPUT_Y_RING_PADDING,
            RUNNING_AUTHORITY_OUTPUT_Y_ZCOL_PADDING,
        ],
    ),
    (FRESH_CLAIM_HASHES, &[FRESH_CLAIM_HASHES_DIGEST]),
    (
        RUNNING_PARENT_HASH,
        &[RUNNING_PARENT_HASH_SHARED_R, RUNNING_PARENT_HASH_DIGEST],
    ),
    (INSTANCE_HASH_AND_ABSORB, &[INSTANCE_HASH, INSTANCE_HEADER_ABSORB]),
    (
        RUNNING_HANDLE_HASH_AND_ABSORB,
        &[
            RUNNING_HANDLE_CHILD_DIGESTS,
            RUNNING_HANDLE_AGGREGATE,
            RUNNING_HANDLE_ABSORB,
        ],
    ),
    (ENGINE_CHALLENGES, &[ENGINE_CHALLENGES_MAIN, ENGINE_CHALLENGES_BETA_M]),
    (
        FE_CLAIM_AND_SUMCHECK,
        &[FE_CLAIMED_INITIAL, FE_OPTIONAL_CLAIM, FE_ROUNDS, FE_SUMCHECK_DRIVER],
    ),
    (NC_SUMCHECK, &[NC_SUMCHECK_ROUNDS, NC_SUMCHECK_DRIVER]),
    (
        OUTPUT_BINDING_AND_TERMINAL_CHECKS,
        &[
            OUTPUT_BINDING,
            FE_TERMINAL_IDENTITY,
            FE_TERMINAL_FINAL_SUM,
            NC_TERMINAL_IDENTITY,
            NC_TERMINAL_FINAL_SUM,
        ],
    ),
    (
        HEADER_CATCH_UP,
        &[HEADER_FIELDS, HEADER_TRANSCRIPT, HEADER_OUTPUT_BINDING],
    ),
    (
        OUTPUT_MESSAGE_HASHES,
        &[OUTPUT_MESSAGE_DIGEST, OUTPUT_MESSAGE_CLAIM, OUTPUT_MESSAGE_BINDING],
    ),
];

pub const ROW_ALLOCATION: &str = "nifs.pi_ccs.allocation";
pub const ROW_AUTHORITY: &str = "nifs.pi_ccs.authority";
pub const ROW_FRESH_DIGESTS: &str = "nifs.pi_ccs.fresh_digests";
pub const ROW_RUNNING_AUTHORITY: &str = "nifs.pi_ccs.running_authority";
pub const ROW_TRANSCRIPT: &str = "nifs.pi_ccs.transcript";
pub const ROW_FE_INITIAL: &str = "nifs.pi_ccs.fe_initial";
pub const ROW_FE_OPTIONAL_CLAIM: &str = FE_OPTIONAL_CLAIM;
pub const ROW_FE_SUMCHECK: &str = "nifs.pi_ccs.fe_sumcheck";
pub const ROW_NC_SUMCHECK: &str = "nifs.pi_ccs.nc_sumcheck";
pub const ROW_OUTPUT_BINDING: &str = "nifs.pi_ccs.output_binding";
pub const ROW_FE_TERMINAL: &str = "nifs.pi_ccs.fe_terminal";
pub const ROW_NC_TERMINAL: &str = "nifs.pi_ccs.nc_terminal";
pub const ROW_NC_TERMINAL_EQUALITY_FACTORS: &str =
    "nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity.equality_factors";
pub const ROW_NC_TERMINAL_CHI_ALPHA: &str =
    "nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity.chi_alpha";
pub const ROW_NC_TERMINAL_GAMMA_POWERS: &str =
    "nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity.gamma_powers";
pub const ROW_NC_TERMINAL_OUTPUT_EVALUATIONS: &str =
    "nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity.output_evaluations";
pub const ROW_NC_TERMINAL_RANGE_PRODUCTS: &str =
    "nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity.range_products";
pub const ROW_NC_TERMINAL_WEIGHTED_SUM: &str =
    "nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity.weighted_sum";
pub const ROW_NC_TERMINAL_FINAL_PRODUCT: &str =
    "nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity.final_product";
pub const ROW_CATCH_UP: &str = "nifs.pi_ccs.catchup";

/// Constraint-family children inside the single physical NC identity stage.
/// These names are a non-mutating source-row overlay, not encoding stages.
pub const ROW_NC_TERMINAL_IDENTITY_CHILDREN: &[&str] = &[
    ROW_NC_TERMINAL_EQUALITY_FACTORS,
    ROW_NC_TERMINAL_CHI_ALPHA,
    ROW_NC_TERMINAL_GAMMA_POWERS,
    ROW_NC_TERMINAL_OUTPUT_EVALUATIONS,
    ROW_NC_TERMINAL_RANGE_PRODUCTS,
    ROW_NC_TERMINAL_WEIGHTED_SUM,
    ROW_NC_TERMINAL_FINAL_PRODUCT,
];

/// Parent/child structure for diagnostic row-family overlays. Keep this
/// separate from [`HIERARCHY`], whose nodes are physical lowering stages.
pub const ROW_HIERARCHY: &[(&str, &[&str])] = &[(NC_TERMINAL_IDENTITY, ROW_NC_TERMINAL_IDENTITY_CHILDREN)];
