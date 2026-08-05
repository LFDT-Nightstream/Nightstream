//! Stable diagnostic paths for the selected one-joint PiCCS circuit.
//!
//! Owns: stable diagnostic labels and their parent-child hierarchy.
//!
//! Does not own: verifier equations, transcript state, or cost totals.
//!
//! Emits constraints: no.

pub const ROOT: &str = "nifs.pi_ccs";
pub const ALLOCATIONS: &str = "nifs.pi_ccs.padded_row.allocations";
pub const CANONICALITY: &str = "nifs.pi_ccs.padded_row.canonicality";
pub const BINDING: &str = "nifs.pi_ccs.padded_row.binding";
pub const PREFIX: &str = "nifs.pi_ccs.padded_row.prefix";
pub const CHALLENGES: &str = "nifs.pi_ccs.padded_row.challenges";
pub const SUMCHECK: &str = "nifs.pi_ccs.padded_row.sumcheck";
pub const TERMINAL: &str = "nifs.pi_ccs.padded_row.terminal";
pub const OUTPUT_TRANSCRIPT: &str = "nifs.pi_ccs.padded_row.output_transcript";
pub const OUTPUT_DIGEST: &str = "nifs.pi_ccs.padded_row.output_digest";

pub const OUTPUT_MESSAGE_PREIMAGE: &str = "nifs.pi_ccs.padded_row.output_digest.preimage";
pub const OUTPUT_MESSAGE_PREIMAGE_OUTER_HEADER: &str = "nifs.pi_ccs.padded_row.output_digest.preimage.outer_header";
pub const OUTPUT_MESSAGE_PREIMAGE_SOURCE_HEADERS: &str = "nifs.pi_ccs.padded_row.output_digest.preimage.source_headers";
pub const OUTPUT_MESSAGE_PREIMAGE_Y_RING: &str = "nifs.pi_ccs.padded_row.output_digest.preimage.y_ring";
pub const OUTPUT_MESSAGE_SIS: &str = "nifs.pi_ccs.padded_row.output_digest.sis";
pub const OUTPUT_MESSAGE_CLAIM: &str = "nifs.pi_ccs.padded_row.output_digest.claim";

pub const ALL: &[&str] = &[
    ROOT,
    ALLOCATIONS,
    CANONICALITY,
    BINDING,
    PREFIX,
    CHALLENGES,
    SUMCHECK,
    TERMINAL,
    OUTPUT_TRANSCRIPT,
    OUTPUT_DIGEST,
    OUTPUT_MESSAGE_PREIMAGE,
    OUTPUT_MESSAGE_PREIMAGE_OUTER_HEADER,
    OUTPUT_MESSAGE_PREIMAGE_SOURCE_HEADERS,
    OUTPUT_MESSAGE_PREIMAGE_Y_RING,
    OUTPUT_MESSAGE_SIS,
    OUTPUT_MESSAGE_CLAIM,
];

pub const HIERARCHY: &[(&str, &[&str])] = &[
    (
        ROOT,
        &[
            ALLOCATIONS,
            CANONICALITY,
            BINDING,
            PREFIX,
            CHALLENGES,
            SUMCHECK,
            TERMINAL,
            OUTPUT_TRANSCRIPT,
            OUTPUT_DIGEST,
        ],
    ),
    (
        OUTPUT_DIGEST,
        &[OUTPUT_MESSAGE_PREIMAGE, OUTPUT_MESSAGE_SIS, OUTPUT_MESSAGE_CLAIM],
    ),
    (
        OUTPUT_MESSAGE_PREIMAGE,
        &[
            OUTPUT_MESSAGE_PREIMAGE_OUTER_HEADER,
            OUTPUT_MESSAGE_PREIMAGE_SOURCE_HEADERS,
            OUTPUT_MESSAGE_PREIMAGE_Y_RING,
        ],
    ),
];

pub const ROW_HIERARCHY: &[(&str, &[&str])] = &[];
