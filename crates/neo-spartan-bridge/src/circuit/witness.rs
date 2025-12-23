//! Witness and instance types for FoldRun circuit
//!
//! These types describe the public inputs and private witness that the
//! FoldRun circuit expects. They mirror the structures used by neo-fold
//! and neo-reductions, but are tailored for circuit synthesis.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{Mat, MeInstance};
use neo_fold::shard::ShardProof as FoldRun;
use neo_math::{F, K};
use neo_reductions::PiCcsProof;

/// Public inputs to the FoldRun circuit
#[derive(Clone, Debug)]
pub struct FoldRunInstance {
    /// Hash of NeoParams
    pub params_digest: [u8; 32],

    /// Hash of CCS structure
    pub ccs_digest: [u8; 32],

    /// Hash of MCS instances
    pub mcs_digest: [u8; 32],

    /// Initial accumulator (public)
    pub initial_accumulator: Vec<MeInstance<Cmt, F, K>>,

    /// Final accumulator (public claim)
    pub final_accumulator: Vec<MeInstance<Cmt, F, K>>,

    /// Π-CCS challenges (from the Π-CCS Fiat–Shamir transcript).
    ///
    /// These are conceptually public with respect to the Π-CCS protocol
    /// itself, but in the Spartan circuit they are treated as ordinary
    /// witness data (not R1CS public inputs). This lets the outer system
    /// decide whether and how to expose the transcript.
    pub pi_ccs_challenges: Vec<PiCcsChallenges>,
}

/// Challenges used in a single Π-CCS proof
#[derive(Clone, Debug)]
pub struct PiCcsChallenges {
    /// α challenge
    pub alpha: Vec<K>,

    /// β_a, β_r challenges
    pub beta_a: Vec<K>,
    pub beta_r: Vec<K>,

    /// γ challenge
    pub gamma: K,

    /// r' challenge (sumcheck point, row part)
    pub r_prime: Vec<K>,

    /// α' challenge (sumcheck point, Ajtai part)
    pub alpha_prime: Vec<K>,

    /// Per-round sumcheck challenges
    pub sumcheck_challenges: Vec<K>,
}

/// Private witness for the FoldRun circuit
#[derive(Clone, Debug)]
pub struct FoldRunWitness {
    /// The complete fold run
    pub fold_run: FoldRun,

    /// Π-CCS proofs for each step
    pub pi_ccs_proofs: Vec<PiCcsProof>,

    /// Z witnesses for each ME instance in each step
    /// witnesses[step_idx][me_idx] is the Z matrix for that ME instance
    pub witnesses: Vec<Vec<Mat<F>>>,

    /// RLC ρ matrices for each step
    pub rlc_rhos: Vec<Vec<Mat<F>>>,

    /// DEC children Z matrices for each step
    pub dec_children_z: Vec<Vec<Mat<F>>>,
}

impl FoldRunInstance {
    /// Create a public instance from a FoldRun
    pub fn from_fold_run(
        run: &FoldRun,
        params_digest: [u8; 32],
        ccs_digest: [u8; 32],
        mcs_digest: [u8; 32],
        initial_accumulator: Vec<MeInstance<Cmt, F, K>>,
        pi_ccs_challenges: Vec<PiCcsChallenges>,
    ) -> Self {
        let final_accumulator = run.compute_final_outputs(&initial_accumulator);

        Self {
            params_digest,
            ccs_digest,
            mcs_digest,
            initial_accumulator,
            final_accumulator,
            pi_ccs_challenges,
        }
    }
}

impl FoldRunWitness {
    /// Create a witness from a FoldRun and its components
    pub fn from_fold_run(
        fold_run: FoldRun,
        pi_ccs_proofs: Vec<PiCcsProof>,
        witnesses: Vec<Vec<Mat<F>>>,
        rlc_rhos: Vec<Vec<Mat<F>>>,
        dec_children_z: Vec<Vec<Mat<F>>>,
    ) -> Self {
        Self {
            fold_run,
            pi_ccs_proofs,
            witnesses,
            rlc_rhos,
            dec_children_z,
        }
    }
}
