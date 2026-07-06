//! `LaneScheme` — the Nebula lane-commitment context (spec §5.1/§5.2).
//!
//! Owns: the two dedicated Ajtai matrices (`A_ops`; `A_mem` shared by the
//! `is` and `fs` lanes — load-bearing for cross-segment boundary equality)
//! and the whole-ring-column lane ranges of the witness matrix (L-ALIGN).
//! Consumers: the Π_DEC prover (child tuples, spec R2), the terminal
//! decider (slice openings, spec R3), and the segment prover (fresh-claim
//! tuples, §13 step 5).
//!
//! Does not own: lane *content* semantics (the `S_mem` rows do), the
//! fold-time mixing (`commitment_ops::mix_adv`/`recompose_adv`), or the
//! F′ carried chains (spec §6).
//!
//! Ranges are in ring-column units of the packed witness `Z ∈ F^{D×(m/D)}`,
//! so L-ALIGN (lanes on whole ring columns) is inherent to the type — a
//! misaligned lane is unrepresentable, not merely rejected.

use std::ops::Range;
use std::sync::Arc;

use neo_ajtai::{setup_par, AjtaiSModule, Commitment};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{LaneCommitments, Mat};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LaneSchemeError {
    #[error("lane scheme: {0}")]
    Invalid(&'static str),
    #[error("lane scheme: Ajtai setup failed: {0}")]
    Setup(String),
    #[error("lane scheme: witness has {got} columns, lanes need at least {need}")]
    WitnessWidth { need: usize, got: usize },
}

/// Ring-column ranges of the three memory lanes inside the witness matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneRanges {
    pub ops: Range<usize>,
    pub is: Range<usize>,
    pub fs: Range<usize>,
}

/// The lane-commitment context: dedicated matrices + lane geometry.
#[derive(Clone)]
pub struct LaneScheme {
    a_ops: Arc<AjtaiSModule>,
    a_mem: Arc<AjtaiSModule>,
    ranges: LaneRanges,
}

impl LaneScheme {
    /// Build from plan-provided seeds. `A_ops` and `A_mem` are fresh
    /// matrices, independent of each other and of the engine's full-`z`
    /// matrix `A` (distinct seeds — security-note A2 assumes independence).
    /// `A_mem`'s width is the shared `is`/`fs` lane width: one matrix, two
    /// lanes, which is what makes `c_fs(k) = c_is(k+1)` meaningful.
    pub fn from_seeds(
        kappa: usize,
        ranges: LaneRanges,
        ops_seed: [u8; 32],
        mem_seed: [u8; 32],
    ) -> Result<Self, LaneSchemeError> {
        if ranges.ops.is_empty() || ranges.is.is_empty() || ranges.fs.is_empty() {
            return Err(LaneSchemeError::Invalid("lane ranges must be non-empty"));
        }
        if ranges.is.len() != ranges.fs.len() {
            return Err(LaneSchemeError::Invalid(
                "is/fs lanes must have identical width (byte-identical layout, spec §3.3)",
            ));
        }
        if ranges.ops.end > ranges.is.start || ranges.is.end > ranges.fs.start {
            return Err(LaneSchemeError::Invalid(
                "lane ranges must be disjoint and ordered ops < is < fs (spec §5.1 layout)",
            ));
        }
        let setup = |seed: [u8; 32], m: usize| -> Result<Arc<AjtaiSModule>, LaneSchemeError> {
            let mut rng = ChaCha8Rng::from_seed(seed);
            let pp = setup_par(&mut rng, D, kappa, m).map_err(|e| LaneSchemeError::Setup(e.to_string()))?;
            Ok(Arc::new(AjtaiSModule::new(Arc::new(pp))))
        };
        Ok(Self {
            a_ops: setup(ops_seed, ranges.ops.len())?,
            a_mem: setup(mem_seed, ranges.is.len())?,
            ranges,
        })
    }

    /// Commit the three lane slices of a witness matrix — the prover side
    /// of R2 (Π_DEC child tuples) and of fresh-claim construction (§13
    /// step 5).
    pub fn commit(&self, z: &Mat<F>) -> Result<LaneCommitments<Commitment>, LaneSchemeError> {
        self.check_width(z)?;
        Ok(LaneCommitments {
            ops: self.a_ops.commit(&column_slice(z, &self.ranges.ops)),
            is: self.a_mem.commit(&column_slice(z, &self.ranges.is)),
            fs: self.a_mem.commit(&column_slice(z, &self.ranges.fs)),
        })
    }

    /// Commit the three lanes from their bit vectors directly — the
    /// pre-γ path of the two-pass prover (spec §1): lane contents exist
    /// before any `x` (hence any full witness) does. Packs each lane
    /// column-major exactly as `CcsInstance::from_low_norm_assignment`
    /// packs `z`, so the tuple equals [`Self::commit`] of the eventual
    /// full witness (pinned by test).
    pub fn commit_bits(&self, ops: &[F], is: &[F], fs: &[F]) -> Result<LaneCommitments<Commitment>, LaneSchemeError> {
        Ok(LaneCommitments {
            ops: self
                .a_ops
                .commit(&pack_lane_bits(ops, self.ranges.ops.len())?),
            is: self
                .a_mem
                .commit(&pack_lane_bits(is, self.ranges.is.len())?),
            fs: self
                .a_mem
                .commit(&pack_lane_bits(fs, self.ranges.fs.len())?),
        })
    }

    /// Commit one mem-domain lane (IS or FS layout) from its bits — the
    /// plan generator's path for `D_init` (spec §7): the initial-memory
    /// scan lanes are committed under `A_mem` with no witness in sight.
    pub fn commit_mem_lane_bits(&self, bits: &[F]) -> Result<Commitment, LaneSchemeError> {
        Ok(self
            .a_mem
            .commit(&pack_lane_bits(bits, self.ranges.is.len())?))
    }

    /// Terminal decider slice-opening (R3): does each published component
    /// open to its lane slice of this witness? Recomputes; never trusts.
    pub fn open_matches(&self, adv: &LaneCommitments<Commitment>, z: &Mat<F>) -> Result<bool, LaneSchemeError> {
        Ok(self.commit(z)? == *adv)
    }

    fn check_width(&self, z: &Mat<F>) -> Result<(), LaneSchemeError> {
        let need = self.ranges.fs.end;
        if z.cols() < need {
            return Err(LaneSchemeError::WitnessWidth { need, got: z.cols() });
        }
        Ok(())
    }
}

impl std::fmt::Debug for LaneScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaneScheme")
            .field("ranges", &self.ranges)
            .finish()
    }
}

/// Column-major pack of one lane's bits into its `D × cols` sub-message —
/// the same layout `pack_assignment_into_ring_matrix` gives the full `z`
/// (bit `k` lands at row `k % D`, column `k / D`), so bit-level commits
/// and witness-slice commits agree.
fn pack_lane_bits(bits: &[F], cols: usize) -> Result<Mat<F>, LaneSchemeError> {
    if bits.len() != cols * D {
        return Err(LaneSchemeError::WitnessWidth {
            need: cols * D,
            got: bits.len(),
        });
    }
    let mut out = Mat::zero(D, cols, F::ZERO);
    for (k, &bit) in bits.iter().enumerate() {
        out[(k % D, k / D)] = bit;
    }
    Ok(out)
}

/// Row-major copy of `z[.., cols]`; lanes are whole ring columns, so the
/// slice is exactly the sub-message the lane matrix commits.
fn column_slice(z: &Mat<F>, cols: &Range<usize>) -> Mat<F> {
    let width = cols.len();
    let mut data = Vec::with_capacity(z.rows() * width);
    for r in 0..z.rows() {
        data.extend_from_slice(&z.row(r)[cols.clone()]);
    }
    Mat::from_row_major(z.rows(), width, data)
}
