//! Internal IVC utilities
//!
//! This module contains the internal implementation details of IVC that are
//! not exposed publicly but may be used by NIVC through the engine interface.

#[allow(dead_code)]
mod prelude;

#[allow(dead_code)]
pub(crate) mod ev;
#[allow(dead_code)]
pub(crate) mod augmented;
#[allow(dead_code)]
pub(crate) mod transcript;
#[allow(dead_code)]
pub(crate) mod commit;
#[allow(dead_code)]
pub(crate) mod basecase;
#[allow(dead_code)]
pub(crate) mod tie;

