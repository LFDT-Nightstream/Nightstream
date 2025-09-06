//! Neo Protocol Red-Team Security Tests
//!
//! This crate contains security-focused tests designed to detect vulnerabilities
//! and verify that exploit attempts are properly rejected by the Neo protocol.
//!
//! ## Test Categories
//!
//! - **E2E Tests**: End-to-end tamper detection
//! - **Parameter Guard Tests**: Validation of protocol parameters
//! - **Bridge Layer Tests**: Public-IO binding and cross-proof validation
//! - **Spartan-Bridge Tests**: Ajtai commitment binding vulnerabilities
//! - **Folding Layer Tests**: Matrix evaluation consistency checks
//!
//! ## Running Red-Team Tests
//!
//! ```bash
//! cargo test --package neo-redteam-tests
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Red-team test utilities and common functions
pub mod utils {
    use p3_goldilocks::Goldilocks as F;
    use p3_field::PrimeCharacteristicRing;
    use neo_ccs::{Mat, r1cs::r1cs_to_ccs, CcsStructure};

    /// Create a tiny CCS for testing: constraint (z0 - z1) = 0
    pub fn tiny_ccs() -> CcsStructure<F> {
        let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
        let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
        let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
        r1cs_to_ccs(a, b, c)
    }

    /// Create a witness that satisfies the tiny CCS constraint
    pub fn satisfying_witness() -> Vec<F> {
        vec![F::from_u64(5), F::from_u64(5)] // z0 = z1 = 5 satisfies (z0 - z1) = 0
    }
}
