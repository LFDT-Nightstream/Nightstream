//! Π_CCS engine trait and implementations.
//!
//! This module defines the core trait for Π_CCS proving/verification engines
//! and provides concrete implementations (optimized, paper-exact, cross-check).
//!
//! For the public API, see the facade module which delegates to these engines.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

use crate::error::PiCcsError;

// Re-export proof type from optimized_engine
pub use super::optimized_engine::PiCcsProof;

#[cfg(feature = "paper-exact")]
pub use super::crosscheck_engine::{CrossCheckEngine, CrosscheckCfg};

/// A minimal trait implemented by each Π_CCS engine.
pub trait PiCcsEngine {
    fn prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        mcs_witnesses: &[McsWitness<F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_witnesses: &[Mat<F>],
        log: &L,
    ) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError>;

    fn verify(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_outputs: &[MeInstance<Cmt, F, K>],
        proof: &PiCcsProof,
    ) -> Result<bool, PiCcsError>;
}

/// Engine backed by the optimized implementation.
#[derive(Clone, Debug, Default, Copy)]
pub struct OptimizedEngine;

impl PiCcsEngine for OptimizedEngine {
    fn prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        mcs_witnesses: &[McsWitness<F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_witnesses: &[Mat<F>],
        log: &L,
    ) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
        super::optimized_engine::pi_ccs_prove(tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log)
    }

    fn verify(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_outputs: &[MeInstance<Cmt, F, K>],
        proof: &PiCcsProof,
    ) -> Result<bool, PiCcsError> {
        super::optimized_engine::pi_ccs_verify(tr, params, s, mcs_list, me_inputs, me_outputs, proof)
    }
}

/// Reference engine for correctness-only runs (paper exact).
#[cfg(feature = "paper-exact")]
#[derive(Clone, Debug, Default, Copy)]
pub struct PaperExactEngine;

#[cfg(feature = "paper-exact")]
impl PiCcsEngine for PaperExactEngine {
    fn prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        mcs_witnesses: &[McsWitness<F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_witnesses: &[Mat<F>],
        log: &L,
    ) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
        super::paper_exact_engine::paper_exact_prove(
            tr,
            params,
            s,
            mcs_list,
            mcs_witnesses,
            me_inputs,
            me_witnesses,
            log,
        )
    }

    fn verify(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_outputs: &[MeInstance<Cmt, F, K>],
        proof: &PiCcsProof,
    ) -> Result<bool, PiCcsError> {
        super::paper_exact_engine::paper_exact_verify(tr, params, s, mcs_list, me_inputs, me_outputs, proof)
    }
}

/// Cross-check engine trait implementation.
#[cfg(feature = "paper-exact")]
impl<I: PiCcsEngine, R: PiCcsEngine> PiCcsEngine for CrossCheckEngine<I, R> {
    fn prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        mcs_witnesses: &[McsWitness<F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_witnesses: &[Mat<F>],
        log: &L,
    ) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
        super::crosscheck_engine::crosscheck_prove(
            &self.inner,
            &self.cfg,
            tr,
            params,
            s,
            mcs_list,
            mcs_witnesses,
            me_inputs,
            me_witnesses,
            log,
        )
    }

    fn verify(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_outputs: &[MeInstance<Cmt, F, K>],
        proof: &PiCcsProof,
    ) -> Result<bool, PiCcsError> {
        super::crosscheck_engine::crosscheck_verify(&self.inner, tr, params, s, mcs_list, me_inputs, me_outputs, proof)
    }
}
