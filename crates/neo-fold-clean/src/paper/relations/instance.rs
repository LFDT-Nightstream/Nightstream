//! `CcsInstance` — the (claim, witness) pair the caller hands to NIFS as
//! one fresh fold input.
//!
//! Owns the user-facing constructor `from_low_norm_assignment` plus its
//! private validation and ring-matrix packing helpers. This is the only
//! file in `paper/relations/` with substantive code; the others are
//! type aliases.

use neo_ajtai::AjtaiSModule;
use neo_ccs::{matrix::Mat as NeoCcsMat, traits::SModuleHomomorphism};
use neo_math::{D, F};

use crate::paper::params::Params;
use crate::paper::relations::ccs::{CcsClaim, CcsWitness, Structure};
use crate::paper::relations::RelationError;

/// One CCS (instance, witness) pair — paper-layer convenience for "u_i with
/// its w_i" that callers hand to Π_CCS as one fresh step.
#[derive(Clone, Debug)]
pub struct CcsInstance {
    pub claim: CcsClaim,
    pub witness: CcsWitness,
}

impl CcsInstance {
    /// Build a fresh CCS instance (Definition 12) from a low-norm full
    /// assignment z = [x, w] of length `structure.m`.
    ///
    /// The assignment must already satisfy ‖z‖_∞ < b. This function:
    ///
    /// 1. validates length and norm,
    /// 2. packs z into the D × cols ring-matrix `Z` that Ajtai expects,
    /// 3. computes `c = log.commit(&Z)`,
    /// 4. splits z into `x = z[..m_in]` (public) and `w = z[m_in..]` (private),
    /// 5. returns the `(claim, witness)` pair.
    ///
    /// `m_in` (paper: `n_𝔽,in`) is the public-input length the caller chose.
    pub fn from_low_norm_assignment(
        pp: &Params,
        log: &AjtaiSModule,
        structure: &Structure,
        z: &[F],
        m_in: usize,
    ) -> Result<Self, RelationError> {
        validate_assignment_shape(structure, z, m_in)?;
        validate_low_norm(pp, z)?;
        let z_mat = pack_assignment_into_ring_matrix(structure, z);
        let c = log.commit(&z_mat);
        Ok(Self {
            claim: CcsClaim {
                c,
                x: z[..m_in].to_vec(),
                m_in,
            },
            witness: CcsWitness {
                w: z[m_in..].to_vec(),
                Z: z_mat,
            },
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Step bodies — private validation + packing
// ──────────────────────────────────────────────────────────────────────────

fn validate_assignment_shape(structure: &Structure, z: &[F], m_in: usize) -> Result<(), RelationError> {
    if z.len() != structure.m {
        return Err(RelationError::AssignmentLength {
            got: z.len(),
            expected: structure.m,
        });
    }
    if m_in > z.len() {
        return Err(RelationError::MInOutOfRange { m_in, len: z.len() });
    }
    Ok(())
}

/// Definition 12 norm bound: ‖z‖_∞ < b (centered representative).
fn validate_low_norm(pp: &Params, z: &[F]) -> Result<(), RelationError> {
    let b = pp.b();
    for (idx, v) in z.iter().enumerate() {
        if !neo_math::balanced::within_nc_bound(*v, b) {
            return Err(RelationError::NormBoundViolated { idx, b });
        }
    }
    Ok(())
}

/// Pack the assignment z ∈ 𝔽^m into the D × cols ring matrix that Ajtai
/// commits. The layout is column-major over D-blocks: entry `(rho, block)` is
/// `z[block * D + rho]`. The verifier uses the same layout, so prover and
/// verifier commit to the same value of `Z`.
fn pack_assignment_into_ring_matrix(structure: &Structure, z: &[F]) -> NeoCcsMat<F> {
    let cols = structure.m.div_ceil(D);
    let mut out = NeoCcsMat::zero(D, cols, F::default());
    for column in 0..z.len() {
        let block = column / D;
        let rho = column % D;
        out[(rho, block)] = z[column];
    }
    out
}
