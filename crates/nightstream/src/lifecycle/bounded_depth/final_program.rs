//! The bounded-depth scheme's final program (CW-6.4), written once over `neo_spartan::Backend`:
//! the finish package's rows (`FinishRelation.finish_holds`) on the words of
//! the last fold, then the layer-1 verifier of each slot's claim. The shrink
//! prover records it; `verify_final` runs it again.
//!
//! The rows are the package's relation rows as one R1CS block, each port in
//! Lean's form (`side`), and each Poseidon2 block as a permutation of its
//! input lanes into its output locals, which is what the block's S-box
//! rows encode. Lean's copy is `Ports/BoundedDepth/Program.lean` in
//! `formal/compression-lean`. The
//! statement and the state keys are bound to their shared words; the slot
//! claims that layer 1 reads are the next slots that the slot select fixes.

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_math::{D, F};
use neo_spartan::{Backend, ClaimWords, Error, FoldTranscript, R1cs, Relation};
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use nightstream_fprime::{PlanRows, StepPackage, PUBLIC_WORDS};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use super::compression::SEED_DOMAIN;

const BLOCK_LANES: usize = 16;
const KAPPA_WORDS: usize = 1188;
const EVALUATION_WORDS: usize = 2 * 270;
const POINT_WORDS: usize = 2 * 28;
const SLOT_WORDS: usize = KAPPA_WORDS + EVALUATION_WORDS + POINT_WORDS + PUBLIC_WORDS + 1;
const ITERATION: usize = PUBLIC_WORDS;
const KEYS: usize = PUBLIC_WORDS + 5;
const Z0: usize = KEYS + 8;
const CURRENT: usize = Z0 + 4;
const WITNESS: usize = CURRENT + 8;

/// The finish package, its source rows, and the R1CS block of those rows on
/// the source columns, without the blocks' S-box rows.
pub(crate) struct Finish {
    package: StepPackage,
    rows: PlanRows,
    system: R1cs,
    /// The columns below `sbox_start` that the final program reads. The
    /// others are the blocks' dead locals and the columns that only the
    /// S-box rows read (SPEC.md section 7.7): no row of the program reads
    /// them, so the program gives them the value 0 and no cell.
    live: Vec<bool>,
}

impl Finish {
    pub(crate) fn new(package: StepPackage) -> Result<Self, nightstream_fprime::PackageError> {
        let rows: PlanRows = package.source_rows()?;
        let mut system = R1cs::new(package.sbox_start());
        let mut live = vec![false; package.sbox_start()];
        for row in 0..rows.ordinary_row_count() {
            let sides = [0, 1, 2].map(|port| side(&rows, row, port));
            for &(column, _) in sides.iter().flatten() {
                live[column] = true;
            }
            system.push(&sides[0], &sides[1], &sides[2]);
        }
        for block in package.blocks() {
            for (_, terms) in block.lane_forms() {
                for &(column, _) in terms {
                    live[column] = true;
                }
            }
            live[block.outputs()..block.outputs() + BLOCK_LANES].fill(true);
        }
        let finish = Self {
            package,
            rows,
            system,
            live: Vec::new(),
        };
        for range in finish.program_reads() {
            live[range].fill(true);
        }
        Ok(Self { live, ..finish })
    }

    /// The words of slot `slot` among the step's seven slots.
    fn slot_start(&self, slot: usize) -> usize {
        WITNESS + self.witness_words() + slot * SLOT_WORDS
    }

    fn fresh_public_start(&self) -> usize {
        self.slot_start(7) + 20 + KAPPA_WORDS
    }

    /// The columns that `FinalProgram::run` reads outside the rows and the
    /// blocks: the constant, the statement and key words, the two slot
    /// claims of layer 1 and the bits of the state hash.
    fn program_reads(&self) -> Vec<std::ops::Range<usize>> {
        let mut reads = vec![
            0..1,
            ITERATION..ITERATION + 1,
            KEYS..KEYS + 8,
            Z0..Z0 + 4,
            CURRENT..CURRENT + 4,
        ];
        reads.extend((0..2).map(|slot| {
            let start = self.slot_start(5 + slot);
            start..start + SLOT_WORDS - 1
        }));
        let hash_bits = self.fresh_public_start() + 1;
        reads.push(hash_bits..hash_bits + 4 * 64);
        reads
    }

    pub(crate) fn live(&self) -> &[bool] {
        &self.live
    }

    pub(crate) fn package(&self) -> &StepPackage {
        &self.package
    }

    /// The witness words of the live columns, in column order.
    pub(crate) fn columns(&self, words: &nightstream_fprime::StepWords) -> Result<Vec<u64>, super::BoundedDepthError> {
        let columns = self.package.circuit_columns(&words.columns()?)?;
        let columns = self.package.relation_columns(columns, &self.rows)?;
        Ok(columns[..self.package.sbox_start()]
            .iter()
            .zip(&self.live)
            .filter(|&(_, &live)| live)
            .map(|(word, _)| word.as_canonical_u64())
            .collect())
    }

    fn witness_words(&self) -> usize {
        self.package.witness_words()
    }
}

/// One port of a plan row in Lean's form (`Terminal.Package.side`): sorted by
/// column, equal columns summed, zero sums dropped.
fn side(rows: &PlanRows, row: usize, port: usize) -> Vec<(usize, u64)> {
    let mut entries: Vec<(usize, F)> = rows.port(row, port).collect();
    entries.sort_by_key(|&(column, _)| column);
    let mut merged: Vec<(usize, F)> = Vec::with_capacity(entries.len());
    for (column, value) in entries {
        match merged.last_mut() {
            Some(last) if last.0 == column => last.1 += value,
            _ => merged.push((column, value)),
        }
    }
    merged
        .into_iter()
        .filter(|&(_, value)| value != F::ZERO)
        .map(|(column, value)| (column, value.as_canonical_u64()))
        .collect()
}

/// What the prover adds to a run: the finish columns and both layer-1 proofs.
pub(crate) struct FinalWitness {
    pub(crate) columns: Vec<u64>,
    pub(crate) layer1: [neo_spartan::Proof; 2],
}

pub(crate) struct FinalProgram<'a> {
    pub(crate) keys: [[F; 4]; 2],
    pub(crate) finish: &'a Finish,
    pub(crate) relations: [&'a Relation<'a>; 2],
    pub(crate) statement: (u64, [F; 4], [F; 4]),
    pub(crate) witness: Option<&'a FinalWitness>,
}

/// The finish columns of one run: a cell for each live column and the
/// constant 0 for each other column.
struct Columns<'a, F> {
    forms: Vec<F>,
    live: &'a [bool],
}

impl<F: Copy> Columns<'_, F> {
    /// Column `column`, which the program reads. A read of a column that
    /// `Finish::live` leaves out is a fault of that list.
    fn at(&self, column: usize) -> F {
        assert!(
            self.live[column],
            "the final program reads a finish column that has no cell"
        );
        self.forms[column]
    }
}

impl FinalProgram<'_> {
    fn claim<B: Backend>(columns: &Columns<'_, B::F>, start: usize) -> ClaimWords<B::F> {
        let pairs = |from: usize, count: usize| -> Vec<[B::F; 2]> {
            (0..count)
                .map(|index| [columns.at(from + 2 * index), columns.at(from + 2 * index + 1)])
                .collect()
        };
        let ring = |from: usize| -> [[B::F; 2]; D] {
            std::array::from_fn(|coefficient| {
                [
                    columns.at(from + 2 * coefficient),
                    columns.at(from + 2 * coefficient + 1),
                ]
            })
        };
        let evaluations = start + KAPPA_WORDS;
        let point = evaluations + EVALUATION_WORDS;
        let public = point + POINT_WORDS;
        ClaimWords {
            commitment: (0..KAPPA_WORDS / D)
                .map(|row| std::array::from_fn(|lane| columns.at(start + row * D + lane)))
                .collect(),
            public: (0..PUBLIC_WORDS / D)
                .map(|block| std::array::from_fn(|lane| columns.at(public + block * D + lane)))
                .collect(),
            point: pairs(point, POINT_WORDS / 2),
            eval_k: ring(evaluations),
            eval_a: (1..5)
                .map(|family| ring(evaluations + family * 2 * D))
                .collect(),
        }
    }

    /// Layer 1's transcript for `slot`: the domain chunk, then the state hash
    /// that the fresh public input's bits encode and the slot
    /// (`BoundedDepth::layer1_seed`).
    fn seed<B: Backend>(&self, b: &mut B, columns: &Columns<'_, B::F>, slot: usize) -> FoldTranscript<B> {
        let mut domain = Poseidon2Transcript::new_v1_2();
        domain.absorb_v1_2(&domain_chunk_v1_2(SEED_DOMAIN));
        let mut state: [B::F; WIDTH] = domain
            .state()
            .map(|lane| b.constant(lane.as_canonical_u64()));
        let bits = self.finish.fresh_public_start() + 1;
        for word in 0..4 {
            let mut value = b.constant(0);
            for bit in 0..64 {
                let scaled = b.scale(columns.at(bits + 64 * word + bit), 1 << bit);
                value = b.add(value, scaled);
            }
            state[word] = b.add(state[word], value);
        }
        let index = b.constant(slot as u64);
        state[4] = b.add(state[4], index);
        FoldTranscript::from_state(b.permute(state))
    }
}

impl neo_spartan::shrink::Program for FinalProgram<'_> {
    /// Both layer-1 relations, the state keys and the finish package's
    /// identity.
    fn key(&self) -> Vec<u64> {
        let mut words = self.relations[0].key_words();
        words.extend(self.relations[1].key_words());
        words.extend(
            self.keys
                .iter()
                .flatten()
                .map(|word| word.as_canonical_u64()),
        );
        words
    }

    /// The iteration, `z0` and the current state.
    fn statement(&self) -> Vec<u64> {
        let (iteration, z0, current) = self.statement;
        let mut words = vec![iteration];
        words.extend(
            z0.iter()
                .chain(&current)
                .map(|word| word.as_canonical_u64()),
        );
        words
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let statement: Vec<B::F> = self
            .statement()
            .into_iter()
            .map(|word| b.public(word))
            .collect();
        let inverse = b.hint(&[statement[0]], 1, &|values| {
            vec![F::from_u64(values[0])
                .try_inverse()
                .map_or(0, |value| value.as_canonical_u64())]
        })[0];
        let product = b.mul(statement[0], inverse);
        let one = b.constant(1);
        b.assert_equal(product, one, "positive iteration")?;

        let live = self.finish.live();
        let count = live.iter().filter(|&&live| live).count();
        let mut cells = b
            .hint(&[], count, &|_| {
                self.witness
                    .map_or_else(|| vec![0; count], |witness| witness.columns.clone())
            })
            .into_iter();
        let zero = b.constant(0);
        let columns = Columns {
            forms: live
                .iter()
                .map(|&live| {
                    if live {
                        cells.next().expect("one cell for each live column")
                    } else {
                        zero
                    }
                })
                .collect(),
            live,
        };
        b.assert_equal(columns.at(0), one, "constant column")?;
        b.assert_equal(columns.at(ITERATION), statement[0], "iteration word")?;
        for lane in 0..4 {
            b.assert_equal(columns.at(Z0 + lane), statement[1 + lane], "z0 word")?;
            b.assert_equal(columns.at(CURRENT + lane), statement[5 + lane], "current word")?;
        }
        for (index, word) in self.keys.iter().flatten().enumerate() {
            let key = b.constant(word.as_canonical_u64());
            b.assert_equal(columns.at(KEYS + index), key, "key word")?;
        }
        b.r1cs(&self.finish.system, &columns.forms, "finish package")?;
        for block in self.finish.package().blocks() {
            let lanes: Vec<B::F> = block
                .lane_forms()
                .map(|(constant, terms)| {
                    terms.iter().fold(
                        b.constant(constant.as_canonical_u64()),
                        |sum, &(column, coefficient)| {
                            let term = b.scale(columns.at(column), coefficient.as_canonical_u64());
                            b.add(sum, term)
                        },
                    )
                })
                .collect();
            let state: [B::F; WIDTH] = lanes.try_into().map_err(|_| Error::Shape("block lanes"))?;
            for (lane, output) in b.permute(state).into_iter().enumerate().take(BLOCK_LANES) {
                b.assert_equal(columns.at(block.outputs() + lane), output, "Poseidon2 block")?;
            }
        }
        for slot in 0..2 {
            let claim = Self::claim::<B>(&columns, self.finish.slot_start(5 + slot));
            let transcript = self.seed(b, &columns, slot);
            neo_spartan::verify_with(
                b,
                self.relations[slot],
                transcript,
                &claim,
                self.witness.map(|witness| &witness.layer1[slot]),
            )?;
        }
        Ok(())
    }
}
