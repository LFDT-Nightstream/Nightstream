//! Stable diagnostic paths for the strict in-circuit Pi_DEC verifier.
//!
//! Owns: the verifier -> constraint-family row-overlay vocabulary and
//! immediate-child hierarchy used by source-row audits.
//!
//! Does not own: Pi_DEC semantics, constraint emission, or cost totals.
//!
//! Emits constraints: no.
//!
//! Authority boundary: these paths are provenance metadata. The parent/child
//! equations emitted by `pi_dec_circuit` remain the acceptance authority.
//! Leaves deliberately remain row-family overlays: gadget-native pairing is
//! physical-stage-local, so turning each leaf into a physical stage would
//! change the encoded relation merely for diagnostics.
//!
//! | Child path | Mathematical obligation | Multiplicity | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `recomposition.commitment` | `C_parent = sum_i b^i C_i` coordinate-wise | commitment width | `enforce_dec_v_inner` | typed Pi_DEC commitment homomorphism; row refinement open |
//! | `recomposition.advice` | Apply the same radix map to optional product-commitment advice | advice width when present | `enforce_adv_recomposition` | concrete row refinement open |
//! | `recomposition.x` | `X_parent = sum_i b^i X_i` on active public coordinates | active X width | `enforce_active_x_combination` | typed Pi_DEC public-input homomorphism; row refinement open |
//! | `recomposition.y_ring` | `y_parent = sum_i b^i y_i` on the semantic ring prefix | matrices x D x extension limbs | `enforce_lane_combination_y` | typed Pi_DEC evaluation homomorphism; row refinement open |
//! | `shape` | Pin verifier-visible carrier dimensions | parent plus children | `enforce_shape_metadata_consistency` | concrete shape refinement open |
//! | `r` | Parent and children share the CE evaluation point | children x point coordinates x extension limbs | `enforce_r_consistency` | paper Pi_DEC shared-point obligation |
//! | `inactive_x` | Canonical inactive X coordinates are zero | inactive child coordinates | `enforce_inactive_x_zero` | encoding refinement open |
//! | `alphabet` | Binary child X coordinates are the uniform-sign canonical split; wider radices retain centered CE(b) membership | active logical coordinates x (`k+2`) for b=2 | `enforce_child_x_canonical_split` | uniform-signed-digit refinement |
//! | `ct` | Cached constant terms equal lane zero of `y_ring` | claims x matrices x extension limbs | `enforce_ct_consistency` | evaluation bridge partial |
//! | `y_ring_padding` | Padded `y_ring` lanes are zero | padded claims x lanes x extension limbs | `enforce_y_ring_padding_zero` | encoding refinement open |
//! | `fold_digest` | Children carry the parent's verifier-owned fold digest | children x four digest lanes | `enforce_fold_digest_consistency` | transcript authority bridge open |

pub const ROOT: &str = "nifs.pi_dec";
pub const VERIFY: &str = "nifs.pi_dec.verify";

pub const RECOMPOSITION: &str = "nifs.pi_dec.verify.recomposition";
pub const RECOMPOSITION_COMMITMENT: &str = "nifs.pi_dec.verify.recomposition.commitment";
pub const RECOMPOSITION_ADVICE: &str = "nifs.pi_dec.verify.recomposition.advice";
pub const RECOMPOSITION_X: &str = "nifs.pi_dec.verify.recomposition.x";
pub const RECOMPOSITION_Y_RING: &str = "nifs.pi_dec.verify.recomposition.y_ring";

pub const SHAPE: &str = "nifs.pi_dec.verify.shape";
pub const R: &str = "nifs.pi_dec.verify.r";
pub const INACTIVE_X: &str = "nifs.pi_dec.verify.inactive_x";
pub const ALPHABET: &str = "nifs.pi_dec.verify.alphabet";
pub const CT: &str = "nifs.pi_dec.verify.ct";
pub const Y_RING_PADDING: &str = "nifs.pi_dec.verify.y_ring_padding";
pub const FOLD_DIGEST: &str = "nifs.pi_dec.verify.fold_digest";

/// Constraint-emitting leaves in production emission order.
pub const LEAVES: &[&str] = &[
    RECOMPOSITION_COMMITMENT,
    RECOMPOSITION_ADVICE,
    RECOMPOSITION_X,
    RECOMPOSITION_Y_RING,
    SHAPE,
    R,
    INACTIVE_X,
    ALPHABET,
    CT,
    Y_RING_PADDING,
    FOLD_DIGEST,
];

/// Every verifier row-overlay node, including its zero-cost organizational
/// owner. The protocol root remains a physical NIFS stage.
pub const ROW_ALL: &[&str] = &[
    VERIFY,
    RECOMPOSITION,
    RECOMPOSITION_COMMITMENT,
    RECOMPOSITION_ADVICE,
    RECOMPOSITION_X,
    RECOMPOSITION_Y_RING,
    SHAPE,
    R,
    INACTIVE_X,
    ALPHABET,
    CT,
    Y_RING_PADDING,
    FOLD_DIGEST,
];

/// Immediate-child ownership for exact source-row reconciliation.
pub const ROW_HIERARCHY: &[(&str, &[&str])] = &[
    (
        VERIFY,
        &[
            RECOMPOSITION,
            SHAPE,
            R,
            INACTIVE_X,
            ALPHABET,
            CT,
            Y_RING_PADDING,
            FOLD_DIGEST,
        ],
    ),
    (
        RECOMPOSITION,
        &[
            RECOMPOSITION_COMMITMENT,
            RECOMPOSITION_ADVICE,
            RECOMPOSITION_X,
            RECOMPOSITION_Y_RING,
        ],
    ),
];
