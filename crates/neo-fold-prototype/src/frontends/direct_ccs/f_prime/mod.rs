//! Owns direct-CCS F' authority material.
//!
//! This folder separates the native latest-step image, low-norm source
//! encoding, R1CS shell, verifier-shaped body, and folded prior chain.
//! Terminal source packing and final CE checks remain terminal responsibilities.

pub(crate) mod chain;
mod native;
pub(crate) mod r1cs;
mod source;
mod verifier_body;

pub use native::{
    DirectCcsCompactFPrimeImage, DirectCcsFPrimeNifsPayloadShape, DirectCcsNativeFPrimeAdvice,
    DirectCcsNativeFPrimeStepImage,
};
pub use r1cs::{DirectCcsFPrimeLowNormSourceR1cs, DirectCcsFPrimeLowNormSourceR1csShape};
pub use source::DirectCcsFPrimeLowNormSourceImage;
pub use verifier_body::{
    export_latest_direct_ccs_f_prime_verifier_body_r1cs, measure_latest_direct_ccs_f_prime_verifier_body,
    DirectCcsFPrimeVerifierBodyNifsShape, DirectCcsFPrimeVerifierBodyShape,
};
