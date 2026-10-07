//! WHIR for the one committed witness column.
//!
//! Owns: the WHIR configuration, the commitment, and openings at points the
//! layer-1 protocol fixes. The required security comes from the caller; the
//! settings below are performance choices, not security parameters.
//! Plonky3 0.8 PCS types do not leave this file.

use p3_commit_v08::MultilinearPcs;
use p3_dft_v08::Radix2DFTSmallBatch;
use p3_field_v08::PrimeCharacteristicRing;
use p3_matrix_v08::dense::RowMajorMatrix;
use p3_security_v08::SecurityTerm;
use p3_sumcheck_v08::layout::{Layout as _, SuffixProver, Table};
use p3_sumcheck_v08::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::prover::WhirProver;

use crate::field::{p3_point, Ext, Gl};
use crate::hash::{mmcs, Challenger, Mmcs};
use crate::Error;

type Dft = Radix2DFTSmallBatch<Gl>;
type Layout = SuffixProver<Gl, Ext>;
type Whir = WhirProver<Ext, Gl, Dft, Mmcs, Challenger, Layout>;
pub(crate) type Commitment = <Whir as MultilinearPcs<Ext, Challenger>>::Commitment;
pub(crate) type ProverData = <Whir as MultilinearPcs<Ext, Challenger>>::ProverData;
pub(crate) type Opening = <Whir as MultilinearPcs<Ext, Challenger>>::Proof;

/// Initial code rate 1/2: the smallest codeword, so the fastest commitment.
const LOG_INV_RATE: usize = 1;
/// Variables folded per WHIR round, as in the WHIR paper's benchmarks (§6).
const FOLDING: usize = 4;
/// No proof-of-work grinding until the owner sets a value.
const POW_BITS: usize = 0;
/// Openings per proof: the GKR leaf point and the linear sum-check point.
pub(crate) const POINTS: usize = 2;

/// WHIR over one committed column of `2^num_variables` base-field values,
/// opened at `POINTS` caller-fixed points.
pub(crate) struct Pcs {
    whir: Whir,
    protocol: OpeningProtocol,
    level: usize,
    security_bits: f64,
}

impl Pcs {
    /// The cheapest configuration whose composed error, the opening plus every
    /// `outer` draw charged over the commitment's candidates, is at most
    /// `2^-required_bits`, under the proven Johnson-bound assumption.
    pub(crate) fn new(num_variables: usize, required_bits: f64, outer: &[SecurityTerm]) -> Result<Self, Error> {
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
            vec![OpeningBatch::new(vec![0], vec![]); POINTS],
        )])
        .pad_to_min_num_variables(FOLDING);
        // Each step raises every per-round level by one bit; the field size
        // bounds what any configuration can reach.
        let first = required_bits.ceil() as usize;
        for level in first..=<Ext as p3_field_v08::Field>::bits() {
            let config = WhirConfig::new(num_variables, parameters(num_variables, level)?)
                .map_err(|_| Error::Setup("WHIR configuration"))?;
            if !config.check_pow_bits() {
                return Err(Error::Setup("WHIR configuration needs proof-of-work grinding"));
            }
            let whir = Whir::new(config, Dft::default(), mmcs());
            let mut report = whir
                .prescribed_security(&protocol)
                .ok_or(Error::Setup("WHIR security report"))?;
            for &term in outer {
                report.charge_reduction(term);
            }
            let security_bits = report.error().bits();
            if security_bits >= required_bits {
                return Ok(Self {
                    whir,
                    protocol,
                    level,
                    security_bits,
                });
            }
        }
        Err(Error::Setup("WHIR cannot reach the required security"))
    }

    /// `-log2` of the composed error, as Plonky3 reports it.
    pub(crate) fn security_bits(&self) -> f64 {
        self.security_bits
    }

    /// The configuration as transcript words: per-round level, rate, folding,
    /// grinding and the soundness assumption (1 = Johnson bound).
    pub(crate) fn profile_words(&self) -> Vec<Gl> {
        [self.level, LOG_INV_RATE, FOLDING, POW_BITS, 1]
            .map(Gl::from_usize)
            .to_vec()
    }

    /// Commit to `values` (length `2^num_variables`, low index bit first) and
    /// bind the root in `challenger`.
    pub(crate) fn commit(&self, values: Vec<Gl>, challenger: &mut Challenger) -> (Commitment, ProverData) {
        let width = values.len();
        let witness = Layout::new_witness(vec![Table::new(RowMajorMatrix::new(values, width))], FOLDING);
        self.whir
            .commit(witness, challenger)
            .expect("the configuration fixes the witness size")
    }

    /// Bind a received root at the same transcript position as `commit`.
    pub(crate) fn observe(&self, commitment: &Commitment, challenger: &mut Challenger) {
        self.whir.observe_commitment(commitment, challenger);
    }

    /// Open the committed column at `points` (low index bit first).
    pub(crate) fn open(&self, data: ProverData, points: &[Vec<Ext>; POINTS], challenger: &mut Challenger) -> Opening {
        let points = points.each_ref().map(|point| p3_point(point));
        self.whir
            .open_at(data, &self.protocol, &points, challenger)
            .expect("the configuration fixes the claim count")
    }

    /// The committed column's values at `points`, if the opening verifies.
    pub(crate) fn verify(
        &self,
        commitment: &Commitment,
        opening: &Opening,
        points: &[Vec<Ext>; POINTS],
        challenger: &mut Challenger,
    ) -> Result<[Ext; POINTS], Error> {
        let points = points.each_ref().map(|point| p3_point(point));
        let evals = self
            .whir
            .verify_at(commitment, opening, &self.protocol, &points, challenger)
            .map_err(|_| Error::Rejected("WHIR opening"))?;
        let mut values = [Ext::default(); POINTS];
        for (value, batch) in values.iter_mut().zip(&evals) {
            *value = *batch
                .current()
                .first()
                .ok_or(Error::Rejected("WHIR opening shape"))?;
        }
        Ok(values)
    }
}

fn parameters(num_variables: usize, security_level: usize) -> Result<ProtocolParameters, Error> {
    let folding_factor = FoldingFactor::Constant(FOLDING);
    let schedule = folding_factor
        .compute_folding_schedule(num_variables)
        .map_err(|_| Error::Setup("WHIR folding schedule"))?;
    let mut rate = LOG_INV_RATE;
    let round_log_inv_rates = schedule[..schedule.len() - 1]
        .iter()
        .map(|folding| {
            rate += folding - 1;
            rate
        })
        .collect();
    Ok(ProtocolParameters {
        security_level,
        pow_bits: POW_BITS,
        folding_factor,
        soundness_type: SecurityAssumption::JohnsonBound,
        starting_log_inv_rate: LOG_INV_RATE,
        round_log_inv_rates,
    })
}
