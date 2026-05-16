//! Owns the outer-compression bridge contract between theorem-facing Nightstream
//! public-boundary exports from `neo-fold-prototype` and Midnight-ledger-compatible
//! proving artifacts.
//!
//! This crate does not own the inner SuperNeo folding relation, ISA frontend
//! arithmetization, or proof-complete bridge-private witness carriers.

pub mod rv64im;
