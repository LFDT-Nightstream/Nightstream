//! Shared F' shell used by concrete app frontends.
//!
//! This module owns the app-agnostic encoded-F' image: boundary,
//! lifecycle state, optional source-image NIFS payloads, accumulator
//! handles/selectors, Poseidon traces, recursive-step image plan, and
//! the base shell CCS rows. App frontends such as [`fibonacci_f_prime`] and
//! [`crate::frontends::r1cs_f_prime`] add their own app semantics on
//! top of this shell.

pub mod compiler;
pub mod encoder;
pub mod image;
pub mod projection_structure;
pub mod recursive_plan;
pub mod structure;

pub use encoder::{encode_f_prime_step, EncodedFPrimeStep, FPrimeStepInput, NifsPayloadInput};
pub use image::{
    FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, KMulView, NifsCcsClaimView, NifsCeClaimShape, NifsCeClaimView,
    NifsPayloadShape, StateIn, StateOut,
};
pub use recursive_plan::{
    build_recursive_step_image_config, AccumulatorPlanOptions, RecursiveStepImagePlan, StateXOutPlanOptions,
};
pub use structure::{build_f_prime_structure, FPrimeStructure};
