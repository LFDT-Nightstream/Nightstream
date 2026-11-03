//! Loader for the Lean-emitted option-B step packages (`option_b`), with the
//! application records and the PiCCS transcripts of the light and refresh
//! folds. It does not own the F′ relation, its circuit structure, or a proof
//! backend.

#[cfg(test)]
extern crate self as nightstream_fprime;

mod application_records;
mod components;
mod identity;
mod option_b;
mod package;
mod witness;

pub use application_records::{
    ApplicationForm, ApplicationRecipeNode, ApplicationRecords, ApplicationRecordsWriter, ApplicationRowHeader,
    ApplicationTerm,
};
pub use option_b::{
    layout::LogicalLayout,
    load_option_b_finish, load_option_b_packages, option_b_identity, parse_atoms,
    words::{encode_digest, Child, FoldWords, KWords, Slot, State as OptionBState, StepWords, PUBLIC_WORDS},
    Linear, NativeBlock, NativeShape, OptionBPackages, PlanRows, StepPackage, OPTION_B_FINISH_IDENTITY,
    OPTION_B_PACKAGES_IDENTITY,
};
pub use package::{
    derive_light_transcript, derive_refresh_transcript, IdentityKind, PackageError, PiCcsV1_2Transcript, LIGHT_DOMAIN,
    PI_CCS_V1_2_ROUND_COUNT, REFRESH_DOMAIN,
};
pub(crate) use package::{
    DECOMPOSITION_BASE, DECOMPOSITION_K_RHO, DECOMPOSITION_NORM_BOUND, PI_CCS_V1_2_COEFFICIENT_COUNT,
    PI_CCS_V1_2_MATRIX_COUNT,
};
