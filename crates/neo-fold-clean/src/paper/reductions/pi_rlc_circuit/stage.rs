//! Stable diagnostic paths for the in-circuit Π_RLC lifecycle and algebra cost tree.
//!
//! Owns: the stable protocol → phase → constraint-family path vocabulary and
//! immediate-child hierarchy.
//!
//! Does not own: transcript state, witness allocation, or constraint emission.
//!
//! Emits constraints: no.
//!
//! Authority boundary: labels are diagnostic metadata only; profiler totals
//! must be derived from production row/column ranges and never trusted from
//! handwritten counts.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `challenge` | Bind outputs and derive rho coefficients | yes | `alphabet_sampling` | `ChallengeWiringArtifact` proves static sharing only; terminal source binding is conditional and recursive source binding is open |
//! | `shape.allocate_parent_and_children` | Allocate Π_DEC parent/children; row overlays separate inactive-X sentinels, digest rejection, and five metadata pins per claim | yes | `pi_dec_circuit` | allocation refinement open |
//! | `fold_wires` | Typed views of Π_CCS inputs and Π_DEC parent | no | `nifs::circuit::pi_rlc::fold_wires` | claim parameters |
//! | `consistency` | Pin implementation NC/transcript sidecars `s_col` and fold digest; not a paper CE equation | yes | `pi_rlc_circuit::consistency` | separate authority proof open |
//! | `projection_binding` | Bind combined/advice values before beta | yes | `nifs::circuit::pi_rlc::projection::binding` | exact-or-bad-root bridge open |
//! | `projection_shared` | Build beta ladder and evaluate each rho | yes | `ring_action` | projection refinement open |
//! | `identities` | Group the public, delayed-NC, and Nebula projection identities | yes | arithmetic claim leaves | ownership split by the three children below |
//! | `identities.public` | Check 29 paper-public Phi81 projections: commitment 18, packed X 5, and y_ring 6 | yes | `commitment`/`x`/`padded_k` | `NifsPaper.PiRlc.equations_of_refinement`, exact or `BatchBadRoot` |
//! | `identities.delayed_nc` | Check two `y_zcol` delayed-NC sidecar projections; these are not paper CE equations | yes | `padded_k` | delayed-NC production authority open |
//! | `identities.nebula` | Check optional advice/product-commitment projections; absent in the fixed profile | yes when present | `commitment` | separate Nebula refinement open |
//! | `identities.*.evaluations` | Evaluate inputs, output, and quotient at beta | yes | `ring_action::enforce_eval_at_beta` | bounded evaluation refinement |
//! | `identities.*.k_products` | Multiply rho/input evaluations and quotient/Phi | yes | `field_ext::enforce_k_mul` | exact Karatsuba refinement |
//! | `identities.*.final_limb_checks` | Equate the two extension-field limbs | yes | `ring_action` | projection identity soundness |
//! | `padding` | Canonical implementation encoding for inactive X and y tails; not paper CE arithmetic | yes | `x`/`padded_k` | encoding/sidecar refinement open |

use crate::engine::r1cs_circuit::ring_action::ProjectionIdentityStageLabels;

pub const ROOT: &str = "nifs.pi_rlc";
pub const CHALLENGE: &str = "nifs.pi_rlc.challenge";

pub const SHAPE: &str = "nifs.pi_rlc.shape";
pub const SHAPE_ALLOCATE: &str = "nifs.pi_rlc.shape.allocate_parent_and_children";
pub const SHAPE_OUTPUT_PARITY: &str = "nifs.pi_rlc.shape.output_parity";
pub const SHAPE_PARENT: &str = "nifs.pi_rlc.shape.parent";
pub const SHAPE_D_PAD: &str = "nifs.pi_rlc.shape.d_pad";

pub const ROW_SHAPE_ALLOCATE_INACTIVE_X_SENTINEL: &str =
    "nifs.pi_rlc.shape.allocate_parent_and_children.inactive_x_sentinel";
pub const ROW_SHAPE_ALLOCATE_FOLD_DIGEST_CANONICALITY: &str =
    "nifs.pi_rlc.shape.allocate_parent_and_children.fold_digest_canonicality";
pub const ROW_SHAPE_ALLOCATE_METADATA: &str = "nifs.pi_rlc.shape.allocate_parent_and_children.metadata";
pub const ROW_SHAPE_ALLOCATE_CHILDREN: &[&str] = &[
    ROW_SHAPE_ALLOCATE_INACTIVE_X_SENTINEL,
    ROW_SHAPE_ALLOCATE_FOLD_DIGEST_CANONICALITY,
    ROW_SHAPE_ALLOCATE_METADATA,
];

/// Row-only children of the single physical allocation stage. Keeping this
/// separate from [`LIFECYCLE_HIERARCHY`] avoids changing stage-local lowering.
pub const ROW_HIERARCHY: &[(&str, &[&str])] = &[(SHAPE_ALLOCATE, ROW_SHAPE_ALLOCATE_CHILDREN)];

pub const VERIFY: &str = "nifs.pi_rlc.verify";

pub const FOLD_WIRES: &str = "nifs.pi_rlc.verify.fold_wires";
pub const FOLD_WIRES_COMMITMENT: &str = "nifs.pi_rlc.verify.fold_wires.commitment";
pub const FOLD_WIRES_ADV: &str = "nifs.pi_rlc.verify.fold_wires.adv";
pub const FOLD_WIRES_X: &str = "nifs.pi_rlc.verify.fold_wires.x";
pub const FOLD_WIRES_Y_RING: &str = "nifs.pi_rlc.verify.fold_wires.y_ring";
pub const FOLD_WIRES_Y_ZCOL: &str = "nifs.pi_rlc.verify.fold_wires.y_zcol";

pub const CONSISTENCY: &str = "nifs.pi_rlc.verify.consistency";
pub const CONSISTENCY_S_COL: &str = "nifs.pi_rlc.verify.consistency.s_col";
pub const CONSISTENCY_FOLD_DIGEST: &str = "nifs.pi_rlc.verify.consistency.fold_digest";

pub const PROJECTION_BINDING: &str = "nifs.pi_rlc.verify.projection_binding";
pub const PROJECTION_BINDING_DOMAIN: &str = "nifs.pi_rlc.verify.projection_binding.domain";
pub const PROJECTION_BINDING_COMBINED: &str = "nifs.pi_rlc.verify.projection_binding.combined";
pub const PROJECTION_BINDING_COMBINED_COMMITMENT: &str = "nifs.pi_rlc.verify.projection_binding.combined.commitment";
pub const PROJECTION_BINDING_COMBINED_ADV: &str = "nifs.pi_rlc.verify.projection_binding.combined.adv";
pub const PROJECTION_BINDING_COMBINED_X: &str = "nifs.pi_rlc.verify.projection_binding.combined.x";
pub const PROJECTION_BINDING_COMBINED_Y_RING: &str = "nifs.pi_rlc.verify.projection_binding.combined.y_ring";
pub const PROJECTION_BINDING_COMBINED_Y_ZCOL: &str = "nifs.pi_rlc.verify.projection_binding.combined.y_zcol";
pub const PROJECTION_BINDING_QUOTIENT: &str = "nifs.pi_rlc.verify.projection_binding.quotient";
pub const PROJECTION_BINDING_QUOTIENT_COMMITMENT: &str = "nifs.pi_rlc.verify.projection_binding.quotient.commitment";
pub const PROJECTION_BINDING_QUOTIENT_ADV: &str = "nifs.pi_rlc.verify.projection_binding.quotient.adv";
pub const PROJECTION_BINDING_QUOTIENT_X: &str = "nifs.pi_rlc.verify.projection_binding.quotient.x";
pub const PROJECTION_BINDING_QUOTIENT_Y_RING: &str = "nifs.pi_rlc.verify.projection_binding.quotient.y_ring";
pub const PROJECTION_BINDING_QUOTIENT_Y_ZCOL: &str = "nifs.pi_rlc.verify.projection_binding.quotient.y_zcol";
pub const PROJECTION_BINDING_SIS_DIGEST: &str = "nifs.pi_rlc.verify.projection_binding.sis_digest";
pub const PROJECTION_BINDING_TRANSCRIPT_BETA: &str = "nifs.pi_rlc.verify.projection_binding.transcript_beta";

pub const PROJECTION_SHARED: &str = "nifs.pi_rlc.verify.projection_shared";
pub const PROJECTION_SHARED_BETA_LADDER: &str = "nifs.pi_rlc.verify.projection_shared.beta_ladder";
pub const PROJECTION_SHARED_RHO_EVALUATIONS: &str = "nifs.pi_rlc.verify.projection_shared.rho_evaluations";

pub const IDENTITIES: &str = "nifs.pi_rlc.verify.identities";
pub const IDENTITIES_PUBLIC: &str = "nifs.pi_rlc.verify.identities.public";
pub const IDENTITIES_DELAYED_NC: &str = "nifs.pi_rlc.verify.identities.delayed_nc";
pub const IDENTITIES_NEBULA: &str = "nifs.pi_rlc.verify.identities.nebula";
pub const IDENTITIES_COMMITMENT: &str = "nifs.pi_rlc.verify.identities.commitment";
pub const IDENTITIES_ADV: &str = "nifs.pi_rlc.verify.identities.adv";
pub const IDENTITIES_X: &str = "nifs.pi_rlc.verify.identities.x";
pub const IDENTITIES_Y_RING: &str = "nifs.pi_rlc.verify.identities.y_ring";
pub const IDENTITIES_Y_ZCOL: &str = "nifs.pi_rlc.verify.identities.y_zcol";

macro_rules! identity_phase_paths {
    ($prefix:literal, $evaluations:ident, $inputs:ident, $output:ident, $quotient:ident,
     $k_products:ident, $rho_input:ident, $quotient_phi:ident, $final_checks:ident) => {
        pub const $evaluations: &str = concat!($prefix, ".evaluations");
        pub const $inputs: &str = concat!($prefix, ".evaluations.inputs");
        pub const $output: &str = concat!($prefix, ".evaluations.output");
        pub const $quotient: &str = concat!($prefix, ".evaluations.quotient");
        pub const $k_products: &str = concat!($prefix, ".k_products");
        pub const $rho_input: &str = concat!($prefix, ".k_products.rho_times_input");
        pub const $quotient_phi: &str = concat!($prefix, ".k_products.quotient_times_phi");
        pub const $final_checks: &str = concat!($prefix, ".final_limb_checks");
    };
}

identity_phase_paths!(
    "nifs.pi_rlc.verify.identities.commitment",
    IDENTITIES_COMMITMENT_EVALUATIONS,
    IDENTITIES_COMMITMENT_EVALUATIONS_INPUTS,
    IDENTITIES_COMMITMENT_EVALUATIONS_OUTPUT,
    IDENTITIES_COMMITMENT_EVALUATIONS_QUOTIENT,
    IDENTITIES_COMMITMENT_K_PRODUCTS,
    IDENTITIES_COMMITMENT_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_COMMITMENT_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_COMMITMENT_FINAL_LIMB_CHECKS
);
identity_phase_paths!(
    "nifs.pi_rlc.verify.identities.adv",
    IDENTITIES_ADV_EVALUATIONS,
    IDENTITIES_ADV_EVALUATIONS_INPUTS,
    IDENTITIES_ADV_EVALUATIONS_OUTPUT,
    IDENTITIES_ADV_EVALUATIONS_QUOTIENT,
    IDENTITIES_ADV_K_PRODUCTS,
    IDENTITIES_ADV_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_ADV_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_ADV_FINAL_LIMB_CHECKS
);
identity_phase_paths!(
    "nifs.pi_rlc.verify.identities.x",
    IDENTITIES_X_EVALUATIONS,
    IDENTITIES_X_EVALUATIONS_INPUTS,
    IDENTITIES_X_EVALUATIONS_OUTPUT,
    IDENTITIES_X_EVALUATIONS_QUOTIENT,
    IDENTITIES_X_K_PRODUCTS,
    IDENTITIES_X_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_X_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_X_FINAL_LIMB_CHECKS
);
identity_phase_paths!(
    "nifs.pi_rlc.verify.identities.y_ring",
    IDENTITIES_Y_RING_EVALUATIONS,
    IDENTITIES_Y_RING_EVALUATIONS_INPUTS,
    IDENTITIES_Y_RING_EVALUATIONS_OUTPUT,
    IDENTITIES_Y_RING_EVALUATIONS_QUOTIENT,
    IDENTITIES_Y_RING_K_PRODUCTS,
    IDENTITIES_Y_RING_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_Y_RING_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_Y_RING_FINAL_LIMB_CHECKS
);
identity_phase_paths!(
    "nifs.pi_rlc.verify.identities.y_zcol",
    IDENTITIES_Y_ZCOL_EVALUATIONS,
    IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS,
    IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT,
    IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT,
    IDENTITIES_Y_ZCOL_K_PRODUCTS,
    IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS
);

pub const COMMITMENT_IDENTITY_STAGES: ProjectionIdentityStageLabels = ProjectionIdentityStageLabels {
    input_evaluations: IDENTITIES_COMMITMENT_EVALUATIONS_INPUTS,
    rho_times_input: IDENTITIES_COMMITMENT_K_PRODUCTS_RHO_TIMES_INPUT,
    output_evaluation: IDENTITIES_COMMITMENT_EVALUATIONS_OUTPUT,
    quotient_evaluation: IDENTITIES_COMMITMENT_EVALUATIONS_QUOTIENT,
    quotient_times_phi: IDENTITIES_COMMITMENT_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    final_limb_checks: IDENTITIES_COMMITMENT_FINAL_LIMB_CHECKS,
};
pub const ADV_IDENTITY_STAGES: ProjectionIdentityStageLabels = ProjectionIdentityStageLabels {
    input_evaluations: IDENTITIES_ADV_EVALUATIONS_INPUTS,
    rho_times_input: IDENTITIES_ADV_K_PRODUCTS_RHO_TIMES_INPUT,
    output_evaluation: IDENTITIES_ADV_EVALUATIONS_OUTPUT,
    quotient_evaluation: IDENTITIES_ADV_EVALUATIONS_QUOTIENT,
    quotient_times_phi: IDENTITIES_ADV_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    final_limb_checks: IDENTITIES_ADV_FINAL_LIMB_CHECKS,
};
pub const X_IDENTITY_STAGES: ProjectionIdentityStageLabels = ProjectionIdentityStageLabels {
    input_evaluations: IDENTITIES_X_EVALUATIONS_INPUTS,
    rho_times_input: IDENTITIES_X_K_PRODUCTS_RHO_TIMES_INPUT,
    output_evaluation: IDENTITIES_X_EVALUATIONS_OUTPUT,
    quotient_evaluation: IDENTITIES_X_EVALUATIONS_QUOTIENT,
    quotient_times_phi: IDENTITIES_X_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    final_limb_checks: IDENTITIES_X_FINAL_LIMB_CHECKS,
};
pub const Y_RING_IDENTITY_STAGES: ProjectionIdentityStageLabels = ProjectionIdentityStageLabels {
    input_evaluations: IDENTITIES_Y_RING_EVALUATIONS_INPUTS,
    rho_times_input: IDENTITIES_Y_RING_K_PRODUCTS_RHO_TIMES_INPUT,
    output_evaluation: IDENTITIES_Y_RING_EVALUATIONS_OUTPUT,
    quotient_evaluation: IDENTITIES_Y_RING_EVALUATIONS_QUOTIENT,
    quotient_times_phi: IDENTITIES_Y_RING_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    final_limb_checks: IDENTITIES_Y_RING_FINAL_LIMB_CHECKS,
};
pub const Y_ZCOL_IDENTITY_STAGES: ProjectionIdentityStageLabels = ProjectionIdentityStageLabels {
    input_evaluations: IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS,
    rho_times_input: IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT,
    output_evaluation: IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT,
    quotient_evaluation: IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT,
    quotient_times_phi: IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    final_limb_checks: IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS,
};

pub const IDENTITY_PHASE_NODES: &[&str] = &[
    IDENTITIES_PUBLIC,
    IDENTITIES_DELAYED_NC,
    IDENTITIES_NEBULA,
    IDENTITIES_COMMITMENT_EVALUATIONS,
    IDENTITIES_COMMITMENT_EVALUATIONS_INPUTS,
    IDENTITIES_COMMITMENT_EVALUATIONS_OUTPUT,
    IDENTITIES_COMMITMENT_EVALUATIONS_QUOTIENT,
    IDENTITIES_COMMITMENT_K_PRODUCTS,
    IDENTITIES_COMMITMENT_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_COMMITMENT_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_COMMITMENT_FINAL_LIMB_CHECKS,
    IDENTITIES_ADV_EVALUATIONS,
    IDENTITIES_ADV_EVALUATIONS_INPUTS,
    IDENTITIES_ADV_EVALUATIONS_OUTPUT,
    IDENTITIES_ADV_EVALUATIONS_QUOTIENT,
    IDENTITIES_ADV_K_PRODUCTS,
    IDENTITIES_ADV_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_ADV_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_ADV_FINAL_LIMB_CHECKS,
    IDENTITIES_X_EVALUATIONS,
    IDENTITIES_X_EVALUATIONS_INPUTS,
    IDENTITIES_X_EVALUATIONS_OUTPUT,
    IDENTITIES_X_EVALUATIONS_QUOTIENT,
    IDENTITIES_X_K_PRODUCTS,
    IDENTITIES_X_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_X_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_X_FINAL_LIMB_CHECKS,
    IDENTITIES_Y_RING_EVALUATIONS,
    IDENTITIES_Y_RING_EVALUATIONS_INPUTS,
    IDENTITIES_Y_RING_EVALUATIONS_OUTPUT,
    IDENTITIES_Y_RING_EVALUATIONS_QUOTIENT,
    IDENTITIES_Y_RING_K_PRODUCTS,
    IDENTITIES_Y_RING_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_Y_RING_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_Y_RING_FINAL_LIMB_CHECKS,
    IDENTITIES_Y_ZCOL_EVALUATIONS,
    IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS,
    IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT,
    IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT,
    IDENTITIES_Y_ZCOL_K_PRODUCTS,
    IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS,
];

pub const PADDING: &str = "nifs.pi_rlc.verify.padding";
pub const PADDING_X: &str = "nifs.pi_rlc.verify.padding.x";
pub const PADDING_Y_RING: &str = "nifs.pi_rlc.verify.padding.y_ring";
pub const PADDING_Y_ZCOL: &str = "nifs.pi_rlc.verify.padding.y_zcol";

/// Lifecycle nodes owned by NIFS orchestration. Challenge and verifier
/// descendants are owned by their respective stage modules.
pub const LIFECYCLE_ALL: &[&str] = &[
    ROOT,
    SHAPE,
    SHAPE_ALLOCATE,
    SHAPE_OUTPUT_PARITY,
    SHAPE_PARENT,
    SHAPE_D_PAD,
];

pub const LIFECYCLE_HIERARCHY: &[(&str, &[&str])] = &[
    (ROOT, &[CHALLENGE, SHAPE, VERIFY]),
    (SHAPE, &[SHAPE_ALLOCATE, SHAPE_OUTPUT_PARITY, SHAPE_PARENT, SHAPE_D_PAD]),
];

/// Every stable node in the Π_RLC verifier-algebra tree, including zero-cost owners.
pub const ALL: &[&str] = &[
    VERIFY,
    FOLD_WIRES,
    FOLD_WIRES_COMMITMENT,
    FOLD_WIRES_ADV,
    FOLD_WIRES_X,
    FOLD_WIRES_Y_RING,
    FOLD_WIRES_Y_ZCOL,
    CONSISTENCY,
    CONSISTENCY_S_COL,
    CONSISTENCY_FOLD_DIGEST,
    PROJECTION_BINDING,
    PROJECTION_BINDING_DOMAIN,
    PROJECTION_BINDING_COMBINED,
    PROJECTION_BINDING_COMBINED_COMMITMENT,
    PROJECTION_BINDING_COMBINED_ADV,
    PROJECTION_BINDING_COMBINED_X,
    PROJECTION_BINDING_COMBINED_Y_RING,
    PROJECTION_BINDING_COMBINED_Y_ZCOL,
    PROJECTION_BINDING_QUOTIENT,
    PROJECTION_BINDING_QUOTIENT_COMMITMENT,
    PROJECTION_BINDING_QUOTIENT_ADV,
    PROJECTION_BINDING_QUOTIENT_X,
    PROJECTION_BINDING_QUOTIENT_Y_RING,
    PROJECTION_BINDING_QUOTIENT_Y_ZCOL,
    PROJECTION_BINDING_SIS_DIGEST,
    PROJECTION_BINDING_TRANSCRIPT_BETA,
    PROJECTION_SHARED,
    PROJECTION_SHARED_BETA_LADDER,
    PROJECTION_SHARED_RHO_EVALUATIONS,
    IDENTITIES,
    IDENTITIES_PUBLIC,
    IDENTITIES_DELAYED_NC,
    IDENTITIES_NEBULA,
    IDENTITIES_COMMITMENT,
    IDENTITIES_ADV,
    IDENTITIES_X,
    IDENTITIES_Y_RING,
    IDENTITIES_Y_ZCOL,
    IDENTITIES_COMMITMENT_EVALUATIONS,
    IDENTITIES_COMMITMENT_EVALUATIONS_INPUTS,
    IDENTITIES_COMMITMENT_EVALUATIONS_OUTPUT,
    IDENTITIES_COMMITMENT_EVALUATIONS_QUOTIENT,
    IDENTITIES_COMMITMENT_K_PRODUCTS,
    IDENTITIES_COMMITMENT_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_COMMITMENT_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_COMMITMENT_FINAL_LIMB_CHECKS,
    IDENTITIES_ADV_EVALUATIONS,
    IDENTITIES_ADV_EVALUATIONS_INPUTS,
    IDENTITIES_ADV_EVALUATIONS_OUTPUT,
    IDENTITIES_ADV_EVALUATIONS_QUOTIENT,
    IDENTITIES_ADV_K_PRODUCTS,
    IDENTITIES_ADV_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_ADV_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_ADV_FINAL_LIMB_CHECKS,
    IDENTITIES_X_EVALUATIONS,
    IDENTITIES_X_EVALUATIONS_INPUTS,
    IDENTITIES_X_EVALUATIONS_OUTPUT,
    IDENTITIES_X_EVALUATIONS_QUOTIENT,
    IDENTITIES_X_K_PRODUCTS,
    IDENTITIES_X_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_X_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_X_FINAL_LIMB_CHECKS,
    IDENTITIES_Y_RING_EVALUATIONS,
    IDENTITIES_Y_RING_EVALUATIONS_INPUTS,
    IDENTITIES_Y_RING_EVALUATIONS_OUTPUT,
    IDENTITIES_Y_RING_EVALUATIONS_QUOTIENT,
    IDENTITIES_Y_RING_K_PRODUCTS,
    IDENTITIES_Y_RING_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_Y_RING_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_Y_RING_FINAL_LIMB_CHECKS,
    IDENTITIES_Y_ZCOL_EVALUATIONS,
    IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS,
    IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT,
    IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT,
    IDENTITIES_Y_ZCOL_K_PRODUCTS,
    IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT,
    IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI,
    IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS,
    PADDING,
    PADDING_X,
    PADDING_Y_RING,
    PADDING_Y_ZCOL,
];

/// Immediate children for every organizational node. A stage profile can use
/// this table to assert that each parent's cost is exactly its child sum.
pub const HIERARCHY: &[(&str, &[&str])] = &[
    (
        VERIFY,
        &[
            FOLD_WIRES,
            CONSISTENCY,
            PROJECTION_BINDING,
            PROJECTION_SHARED,
            IDENTITIES,
            PADDING,
        ],
    ),
    (
        FOLD_WIRES,
        &[
            FOLD_WIRES_COMMITMENT,
            FOLD_WIRES_ADV,
            FOLD_WIRES_X,
            FOLD_WIRES_Y_RING,
            FOLD_WIRES_Y_ZCOL,
        ],
    ),
    (CONSISTENCY, &[CONSISTENCY_S_COL, CONSISTENCY_FOLD_DIGEST]),
    (
        PROJECTION_BINDING,
        &[
            PROJECTION_BINDING_DOMAIN,
            PROJECTION_BINDING_COMBINED,
            PROJECTION_BINDING_QUOTIENT,
            PROJECTION_BINDING_SIS_DIGEST,
            PROJECTION_BINDING_TRANSCRIPT_BETA,
        ],
    ),
    (
        PROJECTION_BINDING_COMBINED,
        &[
            PROJECTION_BINDING_COMBINED_COMMITMENT,
            PROJECTION_BINDING_COMBINED_ADV,
            PROJECTION_BINDING_COMBINED_X,
            PROJECTION_BINDING_COMBINED_Y_RING,
            PROJECTION_BINDING_COMBINED_Y_ZCOL,
        ],
    ),
    (
        PROJECTION_BINDING_QUOTIENT,
        &[
            PROJECTION_BINDING_QUOTIENT_COMMITMENT,
            PROJECTION_BINDING_QUOTIENT_ADV,
            PROJECTION_BINDING_QUOTIENT_X,
            PROJECTION_BINDING_QUOTIENT_Y_RING,
            PROJECTION_BINDING_QUOTIENT_Y_ZCOL,
        ],
    ),
    (
        PROJECTION_SHARED,
        &[PROJECTION_SHARED_BETA_LADDER, PROJECTION_SHARED_RHO_EVALUATIONS],
    ),
    (
        IDENTITIES,
        &[IDENTITIES_PUBLIC, IDENTITIES_DELAYED_NC, IDENTITIES_NEBULA],
    ),
    (
        IDENTITIES_PUBLIC,
        &[IDENTITIES_COMMITMENT, IDENTITIES_X, IDENTITIES_Y_RING],
    ),
    (IDENTITIES_DELAYED_NC, &[IDENTITIES_Y_ZCOL]),
    (IDENTITIES_NEBULA, &[IDENTITIES_ADV]),
    (
        IDENTITIES_COMMITMENT,
        &[
            IDENTITIES_COMMITMENT_EVALUATIONS,
            IDENTITIES_COMMITMENT_K_PRODUCTS,
            IDENTITIES_COMMITMENT_FINAL_LIMB_CHECKS,
        ],
    ),
    (
        IDENTITIES_COMMITMENT_EVALUATIONS,
        &[
            IDENTITIES_COMMITMENT_EVALUATIONS_INPUTS,
            IDENTITIES_COMMITMENT_EVALUATIONS_OUTPUT,
            IDENTITIES_COMMITMENT_EVALUATIONS_QUOTIENT,
        ],
    ),
    (
        IDENTITIES_COMMITMENT_K_PRODUCTS,
        &[
            IDENTITIES_COMMITMENT_K_PRODUCTS_RHO_TIMES_INPUT,
            IDENTITIES_COMMITMENT_K_PRODUCTS_QUOTIENT_TIMES_PHI,
        ],
    ),
    (
        IDENTITIES_ADV,
        &[
            IDENTITIES_ADV_EVALUATIONS,
            IDENTITIES_ADV_K_PRODUCTS,
            IDENTITIES_ADV_FINAL_LIMB_CHECKS,
        ],
    ),
    (
        IDENTITIES_ADV_EVALUATIONS,
        &[
            IDENTITIES_ADV_EVALUATIONS_INPUTS,
            IDENTITIES_ADV_EVALUATIONS_OUTPUT,
            IDENTITIES_ADV_EVALUATIONS_QUOTIENT,
        ],
    ),
    (
        IDENTITIES_ADV_K_PRODUCTS,
        &[
            IDENTITIES_ADV_K_PRODUCTS_RHO_TIMES_INPUT,
            IDENTITIES_ADV_K_PRODUCTS_QUOTIENT_TIMES_PHI,
        ],
    ),
    (
        IDENTITIES_X,
        &[
            IDENTITIES_X_EVALUATIONS,
            IDENTITIES_X_K_PRODUCTS,
            IDENTITIES_X_FINAL_LIMB_CHECKS,
        ],
    ),
    (
        IDENTITIES_X_EVALUATIONS,
        &[
            IDENTITIES_X_EVALUATIONS_INPUTS,
            IDENTITIES_X_EVALUATIONS_OUTPUT,
            IDENTITIES_X_EVALUATIONS_QUOTIENT,
        ],
    ),
    (
        IDENTITIES_X_K_PRODUCTS,
        &[
            IDENTITIES_X_K_PRODUCTS_RHO_TIMES_INPUT,
            IDENTITIES_X_K_PRODUCTS_QUOTIENT_TIMES_PHI,
        ],
    ),
    (
        IDENTITIES_Y_RING,
        &[
            IDENTITIES_Y_RING_EVALUATIONS,
            IDENTITIES_Y_RING_K_PRODUCTS,
            IDENTITIES_Y_RING_FINAL_LIMB_CHECKS,
        ],
    ),
    (
        IDENTITIES_Y_RING_EVALUATIONS,
        &[
            IDENTITIES_Y_RING_EVALUATIONS_INPUTS,
            IDENTITIES_Y_RING_EVALUATIONS_OUTPUT,
            IDENTITIES_Y_RING_EVALUATIONS_QUOTIENT,
        ],
    ),
    (
        IDENTITIES_Y_RING_K_PRODUCTS,
        &[
            IDENTITIES_Y_RING_K_PRODUCTS_RHO_TIMES_INPUT,
            IDENTITIES_Y_RING_K_PRODUCTS_QUOTIENT_TIMES_PHI,
        ],
    ),
    (
        IDENTITIES_Y_ZCOL,
        &[
            IDENTITIES_Y_ZCOL_EVALUATIONS,
            IDENTITIES_Y_ZCOL_K_PRODUCTS,
            IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS,
        ],
    ),
    (
        IDENTITIES_Y_ZCOL_EVALUATIONS,
        &[
            IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS,
            IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT,
            IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT,
        ],
    ),
    (
        IDENTITIES_Y_ZCOL_K_PRODUCTS,
        &[
            IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT,
            IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI,
        ],
    ),
    (PADDING, &[PADDING_X, PADDING_Y_RING, PADDING_Y_ZCOL]),
];
