//! Loader for the Lean-emitted bounded-depth step packages (`bounded_depth`), with the
//! application records and the PiCCS transcripts of the light and refresh
//! folds. It does not own the F′ relation, its circuit structure, or a proof
//! backend.

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
    layout::{LogicalLayout, SlotSpans},
    load_bounded_depth_packages, load_pinned_bounded_depth_packages, parse_atoms,
    words::{encode_digest, Child, FoldWords, KWords, Slot, State as BoundedDepthState, StepWords},
    BoundedDepthPackages, Linear, NativeBlock, NativeShape, PlanRows, StepPackage, BOUNDED_DEPTH_PACKAGES_IDENTITY,
};
pub use package::{
    derive_light_transcript, derive_refresh_transcript, IdentityKind, PackageError, PiCcsV1_2Transcript, LIGHT_DOMAIN,
    PI_CCS_V1_2_ROUND_COUNT, PUBLIC_WORDS, REFRESH_DOMAIN,
};
pub(crate) use package::{
    DECOMPOSITION_BASE, DECOMPOSITION_K_RHO, DECOMPOSITION_NORM_BOUND, PI_CCS_V1_2_MATRIX_COUNT, RING_DEGREE,
};
