//! WHIR for the per-proof tables.
//!
//! Owns: the WHIR configuration, the commitment to a list of base-field
//! tables, and openings at points the caller fixes. The required security
//! comes from the caller; a `Profile` holds performance choices, not
//! security parameters. Plonky3 0.8 PCS types do not leave this file and
//! `whir/`.

use std::sync::{Arc, OnceLock};

use p3_commit_v08::MultilinearPcs;
use p3_dft_v08::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_matrix_v08::dense::RowMajorMatrix;
use p3_matrix_v08::Matrix;
use p3_security_v08::SecurityTerm;
use p3_sumcheck_v08::layout::{plan_stacked_layout, Layout as _, SuffixProver, Table, Witness};
use p3_sumcheck_v08::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::prover::WhirProver;

use crate::device::Accelerator;
use crate::field::{p3_point, Ext, Gl};
use crate::hash::Challenger;
use crate::merkle::Mmcs;
use crate::whir::prover;
use crate::Error;

pub(crate) type Layout = SuffixProver<Gl, Ext>;
type Whir = WhirProver<Ext, Gl, Dft, Mmcs, Challenger, Layout>;
pub(crate) type Config = WhirConfig<Ext, Gl, Challenger>;
pub(crate) type Commitment = <Whir as MultilinearPcs<Ext, Challenger>>::Commitment;
pub(crate) use crate::whir::prover::ProverData;
pub(crate) type Opening = <Whir as MultilinearPcs<Ext, Challenger>>::Proof;
pub(crate) type MultiProof = <Mmcs as p3_commit_v08::Mmcs<Gl>>::MultiProof;

/// The code rate, the folding schedule and the grinding of one WHIR use.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Profile {
    log_inv_rate: usize,
    first_folding: usize,
    folding: usize,
    pow_bits: usize,
}

/// Layer 1: initial rate 1/2 (the smallest codeword, so the fastest
/// commitment), four variables per round as in the WHIR paper's benchmarks
/// (§6), no grinding.
pub(crate) const LAYER1: Profile = Profile {
    log_inv_rate: 1,
    first_folding: 4,
    folding: 4,
    pow_bits: 0,
};

/// The shrink layer: rate 1/8, a first fold of five variables, 20 bits of
/// grinding (owner limit, 2026-10-07). Owner choice (2026-10-09) after a
/// production sweep at 2^24: 1/8 gives a 129.5 KB final proof and an 8 s
/// shrink; 1/64 gives 97.6 KB and 36 s; lower rates save under 3 s more.
pub(crate) const SHRINK: Profile = Profile {
    log_inv_rate: 3,
    first_folding: 5,
    folding: 4,
    pow_bits: 20,
};

impl Profile {
    fn folding_factor(&self) -> FoldingFactor {
        if self.first_folding == self.folding {
            FoldingFactor::Constant(self.folding)
        } else {
            FoldingFactor::ConstantFromSecondRound(self.first_folding, self.folding)
        }
    }
}

/// One committed table: `width` columns of `2^variables` base-field values,
/// and the columns opened at each of its points, in transcript order.
#[derive(Clone, Debug)]
pub(crate) struct TablePlan {
    pub(crate) variables: usize,
    pub(crate) width: usize,
    pub(crate) points: Vec<Vec<usize>>,
}

/// The DFT of the WHIR provers: on the accelerator's device when it runs
/// it, otherwise Plonky3's. Both give the same evaluations. Plonky3's DFT
/// returns no error, so a device failure is recorded here and fails the
/// commitment or opening that ran it (`check`).
#[derive(Clone, Default)]
pub(crate) struct Dft {
    /// Shares its twiddle cache with every clone.
    cpu: Radix2DFTSmallBatch<Gl>,
    accelerator: Accelerator,
    /// The first device failure, shared with every clone of one prover.
    failure: Arc<OnceLock<String>>,
}

impl Dft {
    fn on(&self, accelerator: &Accelerator) -> Self {
        Self {
            cpu: self.cpu.clone(),
            accelerator: accelerator.clone(),
            failure: Arc::default(),
        }
    }

    /// Fail when the device failed during a DFT of this prover.
    fn check(&self) -> Result<(), Error> {
        self.failure
            .get()
            .map_or(Ok(()), |reason| Err(Error::Device(reason.clone())))
    }
}

impl TwoAdicSubgroupDft<Gl> for Dft {
    type Evaluations = RowMajorMatrix<Gl>;

    fn dft_batch(&self, mut matrix: RowMajorMatrix<Gl>) -> RowMajorMatrix<Gl> {
        let (rows, width) = (matrix.height(), matrix.width());
        if let Some(device) = self.accelerator.device().filter(|_| rows.is_power_of_two()) {
            let generator = Gl::two_adic_generator(rows.trailing_zeros() as usize).as_canonical_u64();
            // SAFETY: `Goldilocks` is `repr(transparent)` over `u64`; the
            // device leaves canonical words.
            let words = unsafe {
                std::slice::from_raw_parts_mut(matrix.values.as_mut_ptr().cast::<u64>(), matrix.values.len())
            };
            match device.dft_columns(words, width, generator) {
                Ok(true) => return matrix,
                Ok(false) => {}
                Err(reason) => {
                    let _ = self.failure.set(reason);
                    return matrix;
                }
            }
        }
        self.cpu.dft_batch(matrix).to_row_major_matrix()
    }
}

/// WHIR over a fixed list of tables, opened at caller-fixed points.
pub(crate) struct Pcs {
    /// The CPU instance: binds and verifies roots.
    whir: Whir,
    /// Shares its twiddle cache with every prover instance.
    dft: Dft,
    profile: Profile,
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
        profile: Profile,
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
        .pad_to_min_num_variables(profile.first_folding);
        let (num_variables, _) = plan_stacked_layout(&protocol.table_shapes());
        // Each step raises every per-round level by one bit; the field size
        // bounds what any configuration can reach.
        let first = required_bits.ceil() as usize;
        for level in first..=<Ext as p3_field_v08::Field>::bits() {
            let config = WhirConfig::new(num_variables, parameters(&profile, num_variables, level)?)
                .map_err(|_| Error::Setup("WHIR configuration"))?;
            if !config.check_pow_bits() {
                return Err(Error::Setup("WHIR configuration needs proof-of-work grinding"));
            }
            let dft = Dft::default();
            let whir = Whir::new(config.clone(), dft.clone(), Mmcs::new(&Accelerator::cpu()));
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
                    dft,
                    profile,
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

    /// The configuration as transcript words: per-round level, rate, first
    /// and later folding, grinding and the soundness assumption (1 = Johnson
    /// bound).
    pub(crate) fn profile_words(&self) -> Vec<Gl> {
        let p = &self.profile;
        [self.level, p.log_inv_rate, p.first_folding, p.folding, p.pow_bits, 1]
            .map(Gl::from_usize)
            .to_vec()
    }

    /// The DFT and the Merkle tree of a prover on `accelerator`; each records
    /// its own failures, for `check`.
    fn prover(&self, accelerator: &Accelerator) -> (Dft, Mmcs) {
        (self.dft.on(accelerator), Mmcs::new(accelerator))
    }

    /// Commit to `tables` (each its columns one after another, low index bit
    /// first) and bind the root in `challenger`.
    pub(crate) fn commit(
        &self,
        tables: Vec<Vec<Gl>>,
        challenger: &mut Challenger,
        accelerator: &Accelerator,
    ) -> Result<(Commitment, ProverData), Error> {
        let (dft, mmcs) = self.prover(accelerator);
        let committed = prover::commit(&self.config, &dft, &mmcs, self.witness(tables), challenger);
        dft.check()?;
        mmcs.check()?;
        Ok(committed)
    }

    /// The layout that `commit` leaves in its prover data for `tables`: the
    /// opening needs it, the commitment does not.
    pub(crate) fn layout(&self, tables: Vec<Vec<Gl>>) -> Layout {
        Layout::from_witness(self.witness(tables))
    }

    pub(crate) fn witness(&self, tables: Vec<Vec<Gl>>) -> Witness<Gl> {
        assert_eq!(tables.len(), self.plans.len());
        let tables = tables
            .into_iter()
            .zip(&self.plans)
            .map(|(values, plan)| {
                assert_eq!(values.len(), plan.width << plan.variables);
                Table::new(RowMajorMatrix::new(values, 1 << plan.variables))
            })
            .collect();
        Layout::new_witness(tables, self.profile.first_folding)
    }

    /// Plonky3's own root binding: the shrink layer's, and the reference for
    /// `whir::observe_root`.
    pub(crate) fn observe(&self, commitment: &Commitment, challenger: &mut Challenger) {
        self.whir.observe_commitment(commitment, challenger);
    }

    /// Open every planned batch at its point (low index bit first), in plan
    /// order: tables in order, then each table's points.
    pub(crate) fn open(
        &self,
        data: ProverData,
        points: &[Vec<Ext>],
        challenger: &mut Challenger,
        accelerator: &Accelerator,
    ) -> Result<Opening, Error> {
        let points = self.p3_points(points);
        let (dft, mmcs) = self.prover(accelerator);
        let opening = prover::open(&self.config, &dft, &mmcs, data, &self.protocol, &points, challenger);
        dft.check()?;
        mmcs.check()?;
        Ok(opening)
    }

    /// Plonky3's own verifier, the opened columns' values per batch if the
    /// opening verifies. The shrink layer uses it (it is never recursed, and
    /// it grinds); it is also the reference for `whir::verify`.
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
    pub(crate) fn p3_points(&self, points: &[Vec<Ext>]) -> Vec<p3_multilinear_util_v08::point::Point<Ext>> {
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

fn parameters(profile: &Profile, num_variables: usize, security_level: usize) -> Result<ProtocolParameters, Error> {
    let folding_factor = profile.folding_factor();
    let schedule = folding_factor
        .compute_folding_schedule(num_variables)
        .map_err(|_| Error::Setup("WHIR folding schedule"))?;
    let mut rate = profile.log_inv_rate;
    let round_log_inv_rates = schedule[..schedule.len() - 1]
        .iter()
        .map(|folding| {
            rate += folding - 1;
            rate
        })
        .collect();
    Ok(ProtocolParameters {
        security_level,
        pow_bits: profile.pow_bits,
        folding_factor,
        soundness_type: SecurityAssumption::JohnsonBound,
        starting_log_inv_rate: profile.log_inv_rate,
        round_log_inv_rates,
    })
}
