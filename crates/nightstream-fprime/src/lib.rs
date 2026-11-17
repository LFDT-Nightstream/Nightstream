#[cfg(test)]
extern crate self as nightstream_fprime;

mod application_records;
mod bounded_depth;
mod components;
mod identity;
mod package;
mod witness;

pub use application_records::{
    ApplicationForm, ApplicationRecipeNode, ApplicationRecords, ApplicationRecordsWriter, ApplicationRowHeader,
    ApplicationTerm, ApplicationWalk,
};
pub use bounded_depth::{
    bounded_depth_identity,
    layout::{Kind as SlotKind, LogicalLayout, SlotSpans},
    load_bounded_depth_packages, load_pinned_bounded_depth_packages, parse_atoms,
    words::{encode_digest, Child, FoldWords, KWords, Slot, State as BoundedDepthState, StepWords},
    BoundedDepthPackages, ExpandedPackage, ExpandedPackages, Linear, NativeBlock, NativeShape, PlanRows, StepPackage,
    BOUNDED_DEPTH_PACKAGES_IDENTITY,
};
pub use package::{
    derive_light_transcript, derive_refresh_transcript, IdentityKind, PackageError, PiCcsV1_2Transcript, BLOCK_LOCALS,
    COMMITMENT_WORDS, EVALUATIONS, LIGHT_DOMAIN, PI_CCS_V1_2_MATRIX_COUNT, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT,
    PI_CCS_V1_2_ROUND_COUNT, PUBLIC_WORDS, REFRESH_DOMAIN,
};
pub(crate) use package::{DECOMPOSITION_BASE, DECOMPOSITION_K_RHO, DECOMPOSITION_NORM_BOUND, RING_DEGREE};
