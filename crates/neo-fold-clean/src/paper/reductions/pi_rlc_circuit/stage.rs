//! Stable diagnostic paths for the selected in-circuit Π_RLC verifier.
//!
//! Owns: stable diagnostic labels and their row hierarchy.
//!
//! Does not own: protocol authority, constraints, or cost totals.
//!
//! Emits constraints: no.

use crate::engine::r1cs_circuit::ring_action::ProjectionIdentityStageLabels;

pub const ROOT: &str = "nifs.pi_rlc";
pub const CHALLENGE: &str = "nifs.pi_rlc.challenge";

pub const SHAPE: &str = "nifs.pi_rlc.shape";
pub const SHAPE_ALLOCATE: &str = "nifs.pi_rlc.shape.allocate_parent_and_children";
pub const SHAPE_OUTPUT_PARITY: &str = "nifs.pi_rlc.shape.output_parity";
pub const SHAPE_PARENT: &str = "nifs.pi_rlc.shape.parent";
pub const SHAPE_D_PAD: &str = "nifs.pi_rlc.shape.d_pad";

pub const ROW_SHAPE_ALLOCATE_FOLD_DIGEST_CANONICALITY: &str =
    "nifs.pi_rlc.shape.allocate_parent_and_children.fold_digest_canonicality";
pub const ROW_SHAPE_ALLOCATE_METADATA: &str = "nifs.pi_rlc.shape.allocate_parent_and_children.metadata";
pub const ROW_SHAPE_ALLOCATE_CHILDREN: &[&str] =
    &[ROW_SHAPE_ALLOCATE_FOLD_DIGEST_CANONICALITY, ROW_SHAPE_ALLOCATE_METADATA];
pub const ROW_HIERARCHY: &[(&str, &[&str])] = &[(SHAPE_ALLOCATE, ROW_SHAPE_ALLOCATE_CHILDREN)];

pub const VERIFY: &str = "nifs.pi_rlc.verify";

pub const FOLD_WIRES: &str = "nifs.pi_rlc.verify.fold_wires";
pub const FOLD_WIRES_COMMITMENT: &str = "nifs.pi_rlc.verify.fold_wires.commitment";
pub const FOLD_WIRES_ADV: &str = "nifs.pi_rlc.verify.fold_wires.adv";
pub const FOLD_WIRES_X: &str = "nifs.pi_rlc.verify.fold_wires.x";
pub const FOLD_WIRES_Y_RING: &str = "nifs.pi_rlc.verify.fold_wires.y_ring";

pub const CONSISTENCY: &str = "nifs.pi_rlc.verify.consistency";
pub const CONSISTENCY_FOLD_DIGEST: &str = "nifs.pi_rlc.verify.consistency.fold_digest";

pub const PROJECTION_BINDING: &str = "nifs.pi_rlc.verify.projection_binding";
pub const PROJECTION_BINDING_DOMAIN: &str = "nifs.pi_rlc.verify.projection_binding.domain";
pub const PROJECTION_BINDING_COMBINED: &str = "nifs.pi_rlc.verify.projection_binding.combined";
pub const PROJECTION_BINDING_COMBINED_COMMITMENT: &str = "nifs.pi_rlc.verify.projection_binding.combined.commitment";
pub const PROJECTION_BINDING_COMBINED_ADV: &str = "nifs.pi_rlc.verify.projection_binding.combined.adv";
pub const PROJECTION_BINDING_COMBINED_X: &str = "nifs.pi_rlc.verify.projection_binding.combined.x";
pub const PROJECTION_BINDING_COMBINED_Y_RING: &str = "nifs.pi_rlc.verify.projection_binding.combined.y_ring";
pub const PROJECTION_BINDING_QUOTIENT: &str = "nifs.pi_rlc.verify.projection_binding.quotient";
pub const PROJECTION_BINDING_QUOTIENT_COMMITMENT: &str = "nifs.pi_rlc.verify.projection_binding.quotient.commitment";
pub const PROJECTION_BINDING_QUOTIENT_ADV: &str = "nifs.pi_rlc.verify.projection_binding.quotient.adv";
pub const PROJECTION_BINDING_QUOTIENT_X: &str = "nifs.pi_rlc.verify.projection_binding.quotient.x";
pub const PROJECTION_BINDING_QUOTIENT_Y_RING: &str = "nifs.pi_rlc.verify.projection_binding.quotient.y_ring";
pub const PROJECTION_BINDING_SIS_DIGEST: &str = "nifs.pi_rlc.verify.projection_binding.sis_digest";
pub const PROJECTION_BINDING_TRANSCRIPT_BETA: &str = "nifs.pi_rlc.verify.projection_binding.transcript_beta";

pub const PROJECTION_SHARED: &str = "nifs.pi_rlc.verify.projection_shared";
pub const PROJECTION_SHARED_BETA_LADDER: &str = "nifs.pi_rlc.verify.projection_shared.beta_ladder";
pub const PROJECTION_SHARED_RHO_EVALUATIONS: &str = "nifs.pi_rlc.verify.projection_shared.rho_evaluations";

pub const IDENTITIES: &str = "nifs.pi_rlc.verify.identities";
pub const IDENTITIES_PUBLIC: &str = "nifs.pi_rlc.verify.identities.public";
pub const IDENTITIES_NEBULA: &str = "nifs.pi_rlc.verify.identities.nebula";
pub const IDENTITIES_COMMITMENT: &str = "nifs.pi_rlc.verify.identities.commitment";
pub const IDENTITIES_ADV: &str = "nifs.pi_rlc.verify.identities.adv";
pub const IDENTITIES_X: &str = "nifs.pi_rlc.verify.identities.x";
pub const IDENTITIES_Y_RING: &str = "nifs.pi_rlc.verify.identities.y_ring";

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

pub const IDENTITY_PHASE_NODES: &[&str] = &[
    IDENTITIES_PUBLIC,
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
];

pub const PADDING: &str = "nifs.pi_rlc.verify.padding";
pub const PADDING_Y_RING: &str = "nifs.pi_rlc.verify.padding.y_ring";

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

pub const ALL: &[&str] = &[
    VERIFY,
    FOLD_WIRES,
    FOLD_WIRES_COMMITMENT,
    FOLD_WIRES_ADV,
    FOLD_WIRES_X,
    FOLD_WIRES_Y_RING,
    CONSISTENCY,
    CONSISTENCY_FOLD_DIGEST,
    PROJECTION_BINDING,
    PROJECTION_BINDING_DOMAIN,
    PROJECTION_BINDING_COMBINED,
    PROJECTION_BINDING_COMBINED_COMMITMENT,
    PROJECTION_BINDING_COMBINED_ADV,
    PROJECTION_BINDING_COMBINED_X,
    PROJECTION_BINDING_COMBINED_Y_RING,
    PROJECTION_BINDING_QUOTIENT,
    PROJECTION_BINDING_QUOTIENT_COMMITMENT,
    PROJECTION_BINDING_QUOTIENT_ADV,
    PROJECTION_BINDING_QUOTIENT_X,
    PROJECTION_BINDING_QUOTIENT_Y_RING,
    PROJECTION_BINDING_SIS_DIGEST,
    PROJECTION_BINDING_TRANSCRIPT_BETA,
    PROJECTION_SHARED,
    PROJECTION_SHARED_BETA_LADDER,
    PROJECTION_SHARED_RHO_EVALUATIONS,
    IDENTITIES,
    IDENTITIES_PUBLIC,
    IDENTITIES_NEBULA,
    IDENTITIES_COMMITMENT,
    IDENTITIES_ADV,
    IDENTITIES_X,
    IDENTITIES_Y_RING,
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
    PADDING,
    PADDING_Y_RING,
];

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
        &[FOLD_WIRES_COMMITMENT, FOLD_WIRES_ADV, FOLD_WIRES_X, FOLD_WIRES_Y_RING],
    ),
    (CONSISTENCY, &[CONSISTENCY_FOLD_DIGEST]),
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
        ],
    ),
    (
        PROJECTION_BINDING_QUOTIENT,
        &[
            PROJECTION_BINDING_QUOTIENT_COMMITMENT,
            PROJECTION_BINDING_QUOTIENT_ADV,
            PROJECTION_BINDING_QUOTIENT_X,
            PROJECTION_BINDING_QUOTIENT_Y_RING,
        ],
    ),
    (
        PROJECTION_SHARED,
        &[PROJECTION_SHARED_BETA_LADDER, PROJECTION_SHARED_RHO_EVALUATIONS],
    ),
    (IDENTITIES, &[IDENTITIES_PUBLIC, IDENTITIES_NEBULA]),
    (
        IDENTITIES_PUBLIC,
        &[IDENTITIES_COMMITMENT, IDENTITIES_X, IDENTITIES_Y_RING],
    ),
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
    (PADDING, &[PADDING_Y_RING]),
];
