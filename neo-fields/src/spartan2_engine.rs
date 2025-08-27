//! Spartan2 Engine implementation using Goldilocks field
//! 
//! This module provides a proper Spartan2 Engine that uses Goldilocks field
//! and can be configured with different PCS backends (FRI or Hyrax).

// This keeps the path neo_fields::spartan2_engine::GoldilocksEngine valid in all builds.
// In NARK/default builds it's an empty marker type.
#[cfg(not(feature = "snark_spartan2"))]
pub struct GoldilocksEngine;

// Real Spartan2 Engine implementation when snark_spartan2 is enabled
#[cfg(feature = "snark_spartan2")]
pub use goldilocks_engine_impl::*;

#[cfg(feature = "snark_spartan2")]
mod goldilocks_engine_impl {
    use super::super::*;
    use spartan2::traits::Engine;
    use ff::Field;

    /// Goldilocks-based Spartan2 Engine
    /// This engine uses Goldilocks field for both Base and Scalar
    /// The PCS is determined by the type parameter P
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GoldilocksEngine<P> {
        _phantom: core::marker::PhantomData<P>,
    }

    impl<P> Engine for GoldilocksEngine<P>
    where
        P: spartan2::traits::pcs::PCSEngineTrait<Self> + Clone + Send + Sync,
    {
        type Base = F;
        type Scalar = F;
        type GE = GoldilocksGroupElement;
        type TE = GoldilocksTranscript;
        type PCS = P;
    }

    /// Group element for Goldilocks engine
    /// Since we're using a prime field, we simulate a group using field arithmetic
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GoldilocksGroupElement(pub F);

    impl GoldilocksGroupElement {
        pub fn generator() -> Self {
            Self(F::from_u64(7)) // Use 7 as a generator
        }
    }

    impl std::ops::Mul<F> for GoldilocksGroupElement {
        type Output = Self;
        
        fn mul(self, scalar: F) -> Self::Output {
            Self(self.0 * scalar)
        }
    }

    impl group::Group for GoldilocksGroupElement {
        type Scalar = F;

        fn random(mut rng: impl rand::RngCore) -> Self {
            let val = F::from_u64(rng.next_u64());
            Self(val)
        }

        fn identity() -> Self {
            Self(F::ONE)
        }

        fn generator() -> Self {
            Self::generator()
        }

        fn is_identity(&self) -> subtle::Choice {
            use subtle::ConstantTimeEq;
            self.0.ct_eq(&F::ONE)
        }

        fn double(&self) -> Self {
            Self(self.0 + self.0)
        }
    }

    impl std::ops::Add<GoldilocksGroupElement> for GoldilocksGroupElement {
        type Output = Self;
        
        fn add(self, other: Self) -> Self::Output {
            Self(self.0 + other.0)
        }
    }

    impl std::ops::Sub<GoldilocksGroupElement> for GoldilocksGroupElement {
        type Output = Self;
        
        fn sub(self, other: Self) -> Self::Output {
            Self(self.0 - other.0)
        }
    }

    impl std::ops::Neg for GoldilocksGroupElement {
        type Output = Self;
        
        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    /// Transcript engine for Goldilocks
    #[derive(Clone, Debug)]
    pub struct GoldilocksTranscript {
        state: Vec<u8>,
    }

    impl GoldilocksTranscript {
        pub fn new(label: &[u8]) -> Self {
            Self {
                state: label.to_vec(),
            }
        }
    }

    impl spartan2::traits::transcript::TranscriptEngineTrait for GoldilocksTranscript {
        type GE = GoldilocksGroupElement;

        fn new(label: &'static [u8]) -> Self {
            Self::new(label)
        }

        fn absorb(&mut self, label: &'static [u8], o: &[u8]) {
            self.state.extend_from_slice(label);
            self.state.extend_from_slice(o);
        }

        fn absorb_ge(&mut self, _label: &'static [u8], o: &Self::GE) {
            let bytes = o.0.as_canonical_u64().to_le_bytes();
            self.state.extend_from_slice(&bytes);
        }

        fn challenge_scalar(&mut self, label: &'static [u8]) -> F {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&self.state);
            hasher.update(label);
            let hash = hasher.finalize();
            
            // Convert first 8 bytes to u64
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&hash[0..8]);
            F::from_u64(u64::from_le_bytes(bytes))
        }

        fn challenge_vector(&mut self, label: &'static [u8], len: usize) -> Vec<F> {
            (0..len).map(|i| {
                let extended_label = format!("{}_{}", std::str::from_utf8(label).unwrap_or(""), i);
                self.challenge_scalar(extended_label.as_bytes())
            }).collect()
        }
    }

    // Type aliases for convenience - will be defined in neo-commit to avoid circular deps
}
