//! Owns the compact native F' image for direct CCS.
//!
//! This is the paper-shaped Construction-2 boundary that a future low-norm
//! `enc(F')` relation must prove. It intentionally carries only compact
//! counters, handles, and digests; terminal source packing and final CE checks
//! remain terminal-compression responsibilities.

mod advice;
mod image;
mod low_norm;
mod nifs;

pub use advice::{DirectCcsNativeFPrimeAdvice, DirectCcsNativeFPrimeStepImage};
pub use image::DirectCcsCompactFPrimeImage;
pub use low_norm::DirectCcsFPrimeLowNormSourceImage;
pub use nifs::DirectCcsFPrimeNifsPayloadShape;
