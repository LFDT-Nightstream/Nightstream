//! WHIR for the per-proof tables.
//!
//! Owns: the WHIR configuration, the commitment to a list of base-field
//! tables, and openings at points the layer-1 protocol fixes. The required
//! security comes from the caller; the settings below are performance
//! choices, not security parameters. Plonky3 0.8 PCS types do not leave this
//! file.

use p3_commit_v08::MultilinearPcs;
use p3_dft_v08::Radix2DFTSmallBatch;
use p3_field_v08::PrimeCharacteristicRing;
use p3_matrix_v08::dense::RowMajorMatrix;
use p3_security_v08::SecurityTerm;
use p3_sumcheck_v08::layout::{plan_stacked_layout, Layout as _, SuffixProver, Table};
use p3_sumcheck_v08::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::prover::WhirProver;

use crate::field::{p3_point, Ext, Gl};
use crate::hash::{mmcs, Challenger, Mmcs};
use crate::Error;

type Dft = Radix2DFTSmallBatch<Gl>;
type Layout = SuffixProver<Gl, Ext>;
type Whir = WhirProver<Ext, Gl, Dft, Mmcs, Challenger, Layout>;
pub(crate) type Config = WhirConfig<Ext, Gl, Challenger>;
pub(crate) type Commitment = <Whir as MultilinearPcs<Ext, Challenger>>::Commitment;
pub(crate) type ProverData = <Whir as MultilinearPcs<Ext, Challenger>>::ProverData;
pub(crate) type Opening = <Whir as MultilinearPcs<Ext, Challenger>>::Proof;
pub(crate) type MultiProof = <Mmcs as p3_commit_v08::Mmcs<Gl>>::MultiProof;

/// Initial code rate 1/2: the smallest codeword, so the fastest commitment.
const LOG_INV_RATE: usize = 1;
/// Variables folded per WHIR round, as in the WHIR paper's benchmarks (§6).
const FOLDING: usize = 4;
/// No proof-of-work grinding until the owner sets a value.
const POW_BITS: usize = 0;

/// One committed table: `width` columns of `2^variables` base-field values,
/// and the columns opened at each of its points, in transcript order.
#[derive(Clone, Debug)]
pub(crate) struct TablePlan {
    pub(crate) variables: usize,
    pub(crate) width: usize,
    pub(crate) points: Vec<Vec<usize>>,
}

/// WHIR over a fixed list of tables, opened at caller-fixed points.
pub(crate) struct Pcs {
    whir: Whir,
    config: Config,
    protocol: OpeningProtocol,
    plans: Vec<TablePlan>,
    level: usize,
    security_bits: f64,
    log2_candidates: f64,
}

impl Pcs {
    /// The cheapest configuration whose composed error, the opening plus every
    /// `outer` draw charged over the commitment's candidates, is at most
    /// `2^-required_bits`, under the proven Johnson-bound assumption. `outer`
    /// receives `log2` of the candidate count, for terms that depend on it.
    pub(crate) fn new(
        plans: Vec<TablePlan>,
        required_bits: f64,
        outer: &dyn Fn(f64) -> Vec<SecurityTerm>,
    ) -> Result<Self, Error> {
        let protocol = OpeningProtocol::new(
            plans
                .iter()
                .map(|plan| {
                    TableSpec::new(
                        TableShape::new(plan.variables, plan.width),
                        plan.points
                            .iter()
                            .map(|columns| OpeningBatch::new(columns.clone(), Vec::new()))
                            .collect(),
                    )
                })
                .collect(),
        )
        .pad_to_min_num_variables(FOLDING);
        let (num_variables, _) = plan_stacked_layout(&protocol.table_shapes());
        // Each step raises every per-round level by one bit; the field size
        // bounds what any configuration can reach.
        let first = required_bits.ceil() as usize;
        for level in first..=<Ext as p3_field_v08::Field>::bits() {
            let config = WhirConfig::new(num_variables, parameters(num_variables, level)?)
                .map_err(|_| Error::Setup("WHIR configuration"))?;
            if !config.check_pow_bits() {
                return Err(Error::Setup("WHIR configuration needs proof-of-work grinding"));
            }
            let whir = Whir::new(config.clone(), Dft::default(), mmcs());
            let mut report = whir
                .prescribed_security(&protocol)
                .ok_or(Error::Setup("WHIR security report"))?;
            let log2_candidates = report.log2_max_candidates;
            for term in outer(log2_candidates) {
                report.charge_reduction(term);
            }
            let security_bits = report.error().bits();
            if security_bits >= required_bits {
                return Ok(Self {
                    whir,
                    config,
                    protocol,
                    plans,
                    level,
                    security_bits,
                    log2_candidates,
                });
            }
        }
        Err(Error::Setup("WHIR cannot reach the required security"))
    }

    /// `-log2` of the composed error, as Plonky3 reports it.
    pub(crate) fn security_bits(&self) -> f64 {
        self.security_bits
    }

    /// The WHIR schedule, for our own verifier (`whir/`).
    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    /// The opening protocol, padded to the first folding round.
    pub(crate) fn protocol(&self) -> &OpeningProtocol {
        &self.protocol
    }

    /// `log2` of the candidate polynomials the commitment leaves open.
    pub(crate) fn log2_candidates(&self) -> f64 {
        self.log2_candidates
    }

    /// The configuration as transcript words: per-round level, rate, folding,
    /// grinding and the soundness assumption (1 = Johnson bound).
    pub(crate) fn profile_words(&self) -> Vec<Gl> {
        [self.level, LOG_INV_RATE, FOLDING, POW_BITS, 1]
            .map(Gl::from_usize)
            .to_vec()
    }

    /// Commit to `tables` (each its columns one after another, low index bit
    /// first) and bind the root in `challenger`.
    pub(crate) fn commit(&self, tables: Vec<Vec<Gl>>, challenger: &mut Challenger) -> (Commitment, ProverData) {
        assert_eq!(tables.len(), self.plans.len());
        let tables = tables
            .into_iter()
            .zip(&self.plans)
            .map(|(values, plan)| {
                assert_eq!(values.len(), plan.width << plan.variables);
                Table::new(RowMajorMatrix::new(values, 1 << plan.variables))
            })
            .collect();
        let witness = Layout::new_witness(tables, FOLDING);
        self.whir
            .commit(witness, challenger)
            .expect("the configuration fixes the witness size")
    }

    /// Bind a received root at the same transcript position as `commit`.
    pub(crate) fn observe(&self, commitment: &Commitment, challenger: &mut Challenger) {
        self.whir.observe_commitment(commitment, challenger);
    }

    /// Open every planned batch at its point (low index bit first), in plan
    /// order: tables in order, then each table's points.
    pub(crate) fn open(&self, data: ProverData, points: &[Vec<Ext>], challenger: &mut Challenger) -> Opening {
        let points = self.p3_points(points);
        self.whir
            .open_at(data, &self.protocol, &points, challenger)
            .expect("the plan fixes the claims")
    }

    /// The opened columns' values per batch, if the opening verifies.
    pub(crate) fn verify(
        &self,
        commitment: &Commitment,
        opening: &Opening,
        points: &[Vec<Ext>],
        challenger: &mut Challenger,
    ) -> Result<Vec<Vec<Ext>>, Error> {
        if points.len() != self.protocol.num_openings() {
            return Err(Error::Rejected("WHIR point count"));
        }
        let points = self.p3_points(points);
        let evals = self
            .whir
            .verify_at(commitment, opening, &self.protocol, &points, challenger)
            .map_err(|_| Error::Rejected("WHIR opening"))?;
        Ok(evals.iter().map(|batch| batch.current().to_vec()).collect())
    }

    /// Points in Plonky3 order, padded with zeros for tables the protocol
    /// padded to the first folding round.
    fn p3_points(&self, points: &[Vec<Ext>]) -> Vec<p3_multilinear_util_v08::point::Point<Ext>> {
        let shapes = self.protocol.table_shapes();
        self.protocol
            .iter_openings()
            .zip(points)
            .map(|((table, _), point)| {
                assert_eq!(point.len(), self.plans[table].variables);
                let mut padded = point.clone();
                padded.resize(shapes[table].num_variables(), Ext::ZERO);
                p3_point(&padded)
            })
            .collect()
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
