//! The shrink layer: an argument that a program written over `Backend`
//! accepts its statement, with one WHIR commitment.
//!
//! Owns: the program contract, the shrink relation (layout and WHIR), the
//! proof, its prover and its verifier. The argument is SuperSpartan (ePrint
//! 2023/552) for the CCS of `neo_verified::layout`: a zero-check sum-check of
//! degree 8 over the rows, a batching draw, a product sum-check over `z`, and
//! one WHIR opening of `w`. The verifier rebuilds the glue rows by a shape run
//! of the program and computes the block rows in closed form; it stores no
//! rows.
//!
//! Invariants: the transcript binds the shape, the WHIR profile and the
//! statement before the commitment; the verifier's shape run must emit the
//! shape's counts and the statement it bound.

mod evaluate;
pub(crate) mod layout;

use neo_verified::circuit::{Kind, Wire, KINDS, MATRICES};
use neo_verified::degrees::{self, DEGREE};
use neo_verified::layout::{block_cell, block_row, cell_variables, column, entry_column, glue_row, one, row_variables};
use p3_challenger_v08::{CanObserve, FieldChallenger};
use p3_field_v08::{Algebra, Field, PrimeCharacteristicRing};
use p3_security_v08::{ErrorBits, SecurityTerm};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub use self::layout::ShapeDef;
pub use neo_verified::layout::Shape;

use self::evaluate::Evaluate;
use self::layout::{block_part, cell_parts, row_parts, words};
use neo_verified::field as verified;
use neo_verified::r1cs::R1cs;
use neo_verified::sumcheck::replay_claim;

use crate::circuit::block::{entries, Entry, C, X};
use crate::circuit::record::{Form, Recorder, Sink, Term, Trace};
use crate::circuit::{algebra, Backend, Native};
use crate::device::Accelerator;
use crate::field::{
    eq_table, ext_from_verified, ext_to_verified, exts_to_verified, fold_low, from_verified, gl, Ext, Gl,
};
use crate::hash::{permutation, Challenger};
use crate::pcs::{self, Pcs, TablePlan};
use crate::{sumcheck, Error};

/// Names the shrink protocol at the start of its transcript.
const DOMAIN: &[u8] = b"Nightstream/SuperNeo/shrink/v1";

/// A verifier written once over `Backend`.
pub trait Program {
    /// The statement words, canonical; `run` emits exactly these as
    /// `public`, first.
    fn statement(&self) -> Vec<u64>;
    /// Canonical words of everything else that fixes the checks (a key, a
    /// context). The shrink transcript binds them with the statement, so
    /// two programs of one shape never share challenges.
    fn key(&self) -> Vec<u64>;
    /// Run the checks. A shape run reads its proof as zeros.
    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error>;
}

/// Counts what a run records.
struct Count(Shape);

impl Sink for Count {
    fn row(&mut self, _: &[(Vec<Term>, Vec<Term>)], _: &[Term]) {
        self.0.glue_rows += 1;
    }

    fn cell(&mut self, _: verified::Gl) {
        self.0.glue_cells += 1;
    }

    fn block(&mut self, kind: Kind, _: &[verified::Gl]) -> Vec<verified::Gl> {
        self.0.blocks[kind.index()] += 1;
        Vec::new()
    }

    fn public(&mut self, _: verified::Gl) {
        self.0.publics += 1;
    }
    fn r1cs(&mut self, system: &R1cs, _: &[Form]) {
        self.0.glue_rows += system.rows();
    }
    fn values(&self) -> bool {
        false
    }
}

/// The counts of one run of `program`.
pub fn shape_of(program: &impl Program) -> Result<Shape, Error> {
    let mut recorder = Recorder::new(Count(Shape {
        blocks: [0; KINDS.len()],
        glue_rows: 0,
        glue_cells: 0,
        publics: 0,
    }));
    program.run(&mut recorder)?;
    Ok(recorder.finish().0 .0)
}

/// The run digest of `program`: a Poseidon2 digest of the counts and every
/// row of one recorded run, without values, so every run of one program has
/// one digest. `formal/compression-lean` computes it from the Lean builder's
/// program (`runDigest`), so a test can pin a program's rows without Lean.
pub fn run_digest(program: &impl Program) -> Result<[u64; 4], Error> {
    let mut recorder = Recorder::new(Trace::default());
    program.run(&mut recorder)?;
    Ok(recorder.finish().0.digest())
}

/// The shrink relation of one shape: the shape and the WHIR configuration.
pub struct Shrink {
    shape: Shape,
    pcs: Pcs,
}

impl Shrink {
    /// `security_bits` is `-log2` of the soundness error this layer may add.
    pub fn new(shape: Shape, security_bits: f64) -> Result<Self, Error> {
        let plans = vec![TablePlan {
            variables: cell_variables(&shape),
            width: 1,
            points: vec![vec![0]],
        }];
        let terms = terms(&shape);
        let pcs = Pcs::new(pcs::SHRINK, plans, security_bits, &|_| terms.clone())?;
        Ok(Self { shape, pcs })
    }

    pub fn security_bits(&self) -> f64 {
        self.pcs.security_bits()
    }

    fn start(&self, key: &[Gl], statement: &[Gl]) -> Challenger {
        let mut challenger = Challenger::new(permutation().clone());
        challenger.observe_slice(&neo_transcript::domain_chunk_v1_2(DOMAIN).map(gl));
        challenger.observe_slice(&words(&self.shape));
        challenger.observe_slice(&self.pcs.profile_words());
        challenger.observe(Gl::from_usize(key.len()));
        challenger.observe_slice(key);
        challenger.observe_slice(statement);
        challenger
    }
}

/// The sum-check draws charged over WHIR's candidates, as degrees over |Ext|.
fn terms(shape: &Shape) -> Vec<SecurityTerm> {
    let field_bits = (<Ext as Field>::bits() - 1) as f64;
    let term =
        |label, degree: usize| SecurityTerm::new(label, ErrorBits::from_log2(field_bits - (degree as f64).log2()));
    let [point, outer, batching, inner] = degrees::shrink_terms(shape);
    vec![
        term("zero-check point", point),
        term("outer sum-check", outer),
        term("matrix batching", batching),
        term("inner sum-check", inner),
    ]
}

/// A shrink proof. It contains no witness cell.
#[derive(Clone, Serialize, Deserialize)]
pub struct ShrinkProof {
    pub(crate) commitment: pcs::Commitment,
    pub(crate) outer: Vec<Vec<Ext>>,
    /// `M̃_j·z` at the outer point, in matrix order.
    pub(crate) values: [Ext; MATRICES],
    pub(crate) inner: Vec<Vec<Ext>>,
    pub(crate) opening: pcs::Opening,
}

impl ShrinkProof {
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("an in-memory proof always encodes")
    }

    /// Strict: the bytes must be the canonical encoding of the decoded proof.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        crate::decode_canonical(bytes, "shrink proof")
    }
}

/// The statement words as layer-1 field values; they must be canonical.
fn statement_of(program: &impl Program) -> Result<Vec<Gl>, Error> {
    canonical(program.statement(), "statement words must be canonical")
}

fn key_of(program: &impl Program) -> Result<Vec<Gl>, Error> {
    canonical(program.key(), "key words must be canonical")
}

fn canonical(words: Vec<u64>, what: &'static str) -> Result<Vec<Gl>, Error> {
    words
        .into_iter()
        .map(|word| neo_verified::field::canonical(word).map(from_verified))
        .collect::<Option<_>>()
        .ok_or(Error::Shape(what))
}

/// Prove that `program` accepts its statement. The Merkle leaves hash on
/// `accelerator`.
pub fn prove(shrink: &Shrink, program: &impl Program, accelerator: &Accelerator) -> Result<ShrinkProof, Error> {
    let shape = &shrink.shape;
    let statement = statement_of(program)?;
    let mut recorder = Recorder::new(Trace::default());
    program.run(&mut recorder)?;
    let (trace, failure) = recorder.finish();
    if let Some(what) = failure {
        return Err(Error::Witness(what));
    }
    let counts = Shape {
        blocks: KINDS.map(|kind| trace.count(kind)),
        glue_rows: trace.rows.len(),
        glue_cells: trace.glue.len(),
        publics: trace.public.len(),
    };
    if counts != *shape || trace.public != statement {
        return Err(Error::Shape("the program's run differs from the shrink shape"));
    }
    let z = z_table(shape, &trace);
    let products = products(shape, &trace, &z);

    let mut challenger = shrink.start(&key_of(program)?, &statement);
    let n = cell_variables(shape);
    let (commitment, data) = shrink
        .pcs
        .commit(vec![z[..1 << n].to_vec()], &mut challenger, accelerator)?;
    let tau: Vec<Ext> = (0..row_variables(shape))
        .map(|_| challenger.sample_algebra_element())
        .collect();
    let (outer, rx, values) = prove_outer(products, &tau, &mut challenger);
    challenger.observe_algebra_slice(&values);
    let rho = powers(challenger.sample_algebra_element());
    let weights = weights(shape, &trace, &rx, &rho);
    let z: Vec<Ext> = z.into_par_iter().map(Ext::from).collect();
    let (inner, ry, _) = sumcheck::prove_product(weights, z, &mut challenger);
    let opening = shrink
        .pcs
        .open(data, &[ry[..n].to_vec()], &mut challenger, accelerator)?;
    Ok(ShrinkProof {
        commitment,
        outer,
        values,
        inner,
        opening,
    })
}

/// Verify that `program` accepts its statement.
pub fn verify(shrink: &Shrink, program: &impl Program, proof: &ShrinkProof) -> Result<(), Error> {
    let shape = &shrink.shape;
    let (m, n) = (row_variables(shape), cell_variables(shape));
    if proof.outer.len() != m
        || proof.outer.iter().any(|message| message.len() != DEGREE)
        || proof.inner.len() != n + 1
        || proof.inner.iter().any(|message| message.len() != 2)
    {
        return Err(Error::Rejected("shrink proof shape"));
    }
    let statement = statement_of(program)?;
    let mut challenger = shrink.start(&key_of(program)?, &statement);
    shrink.pcs.observe(&proof.commitment, &mut challenger);
    let tau: Vec<Ext> = (0..m)
        .map(|_| challenger.sample_algebra_element())
        .collect();
    let (rx, last) = replay(DEGREE, &proof.outer, Ext::ZERO, &mut challenger);
    let v = proof.values;
    let eq = algebra::eq_eval(&mut Native, &exts_to_verified(&tau), &exts_to_verified(&rx));
    if last != ext_from_verified(eq) * constraint(&v) {
        return Err(Error::Rejected("shrink outer sum-check"));
    }
    challenger.observe_algebra_slice(&v);
    let rho = powers(challenger.sample_algebra_element());
    let claim = rho.iter().zip(&v).map(|(&r, &value)| r * value).sum();
    let (ry, last) = replay(2, &proof.inner, claim, &mut challenger);
    let opened = shrink
        .pcs
        .verify(&proof.commitment, &proof.opening, &[ry[..n].to_vec()], &mut challenger)?;
    let w = opened[0][0];

    let mut recorder = Recorder::new(Evaluate::new(shape, &rx, &ry, rho));
    program.run(&mut recorder)?;
    let (sink, _) = recorder.finish();
    let counts = Shape {
        blocks: sink.blocks,
        glue_rows: sink.glue_rows,
        glue_cells: sink.glue_cells,
        publics: sink.publics.len(),
    };
    if counts != *shape || sink.publics != statement {
        return Err(Error::Rejected("the program's shape run differs from the shrink shape"));
    }
    let matrices = sink.total + block_part(shape, &rx, &ry, &rho);
    let mut public = vec![Gl::ONE];
    public.extend(&statement);
    let cells = exts_to_verified(&ry[..n]);
    let x: Ext = public
        .iter()
        .enumerate()
        .map(|(index, &value)| ext_from_verified(algebra::eq_at(&mut Native, &cells, index)) * value)
        .sum();
    let z = (Ext::ONE - ry[n]) * w + ry[n] * x;
    if last != matrices * z {
        return Err(Error::Rejected("shrink inner sum-check"));
    }
    Ok(())
}

/// `ρ^j` for every matrix.
fn powers(rho: Ext) -> [Ext; MATRICES] {
    let mut power = Ext::ONE;
    std::array::from_fn(|_| {
        let value = power;
        power *= rho;
        value
    })
}

/// Replay sum-check rounds (`h(0), h(2), ..., h(degree)`) on the native challenger. The
/// challenges depend only on the messages, so they are drawn first.
fn replay(degree: usize, rounds: &[Vec<Ext>], claim: Ext, challenger: &mut Challenger) -> (Vec<Ext>, Ext) {
    let point: Vec<Ext> = rounds
        .iter()
        .map(|message| sumcheck::send(message, challenger))
        .collect();
    let messages: Vec<Vec<verified::Ext>> = rounds
        .iter()
        .map(|message| exts_to_verified(message))
        .collect();
    let claim = replay_claim(degree, &messages, ext_to_verified(claim), &exts_to_verified(&point));
    (point, ext_from_verified(claim))
}

/// `z = (w, 1, x)` on `2^(n+1)` cells.
fn z_table(shape: &Shape, trace: &Trace) -> Vec<Gl> {
    let n = cell_variables(shape);
    let mut z = vec![Gl::ZERO; 2 << n];
    for (cell, &value) in trace.glue.iter().enumerate() {
        z[column(shape, Wire::Glue(cell as u32))] = value;
    }
    for kind in KINDS {
        for (block, cells) in trace.blocks[kind.index()]
            .chunks_exact(kind.cells())
            .enumerate()
        {
            for (cell, &value) in cells.iter().enumerate() {
                z[block_cell(shape, kind, block, cell)] = value;
            }
        }
    }
    let constant = one(shape);
    z[constant] = Gl::ONE;
    z[constant + 1..constant + 1 + trace.public.len()].copy_from_slice(&trace.public);
    z
}

/// `M_j·z` on the row cube for every matrix.
fn products(shape: &Shape, trace: &Trace, z: &[Gl]) -> Vec<Vec<Gl>> {
    let rows = 1usize << row_variables(shape);
    let mut tables = vec![vec![Gl::ZERO; rows]; MATRICES];
    for (matrix, table) in tables.iter_mut().enumerate() {
        for kind in KINDS {
            let mut by_row = vec![Vec::new(); kind.rows()];
            for entry in entries(kind).iter().filter(|entry| entry.matrix == matrix) {
                by_row[entry.row].push(entry);
            }
            // Every block owns its row slots, so the blocks fill them in parallel.
            for (start, stride, first) in row_parts(shape, kind) {
                table[start..start + stride * shape.blocks[kind.index()]]
                    .par_chunks_mut(stride)
                    .enumerate()
                    .for_each(|(block, slot)| {
                        for (value, entries) in slot.iter_mut().zip(by_row.iter().skip(first)) {
                            for entry in entries {
                                *value += entry.coefficient * z[entry_column(shape, kind, block, entry.cell)];
                            }
                        }
                    });
            }
        }
        if matrix == X {
            continue;
        }
        let side = if matrix == C { 6 } else { matrix - 1 };
        let start = glue_row(shape, 0);
        table[start..start + trace.rows.len()]
            .par_iter_mut()
            .zip(&trace.rows)
            .for_each(|(value, offsets)| {
                *value = trace.entries[offsets[side] as usize..offsets[side + 1] as usize]
                    .iter()
                    .map(|&(wire, coefficient)| coefficient * z[column(shape, wire)])
                    .sum();
            });
    }
    tables
}

/// `Σ_j ρ^j·M_j(rx, y)` for every column `y` of `z`.
fn weights(shape: &Shape, trace: &Trace, rx: &[Ext], rho: &[Ext; MATRICES]) -> Vec<Ext> {
    let eq_rows = eq_table(rx);
    let mut weights = vec![Ext::ZERO; 2 << cell_variables(shape)];
    for kind in KINDS {
        let blocks = shape.blocks[kind.index()];
        let mut by_cell = vec![Vec::new(); kind.cells()];
        let mut constant = Vec::new();
        for entry in entries(kind) {
            match entry.cell {
                Some(cell) => by_cell[cell].push(entry),
                None => constant.push(entry),
            }
        }
        let term = |block: usize, entry: &Entry| {
            rho[entry.matrix] * eq_rows[block_row(shape, kind, block, entry.row)] * entry.coefficient
        };
        // Every block owns its cell slots, so the blocks fill them in parallel.
        for (start, stride, first) in cell_parts(shape, kind) {
            weights[start..start + stride * blocks]
                .par_chunks_mut(stride)
                .enumerate()
                .for_each(|(block, slot)| {
                    for (weight, entries) in slot.iter_mut().zip(by_cell.iter().skip(first)) {
                        for entry in entries {
                            *weight += term(block, entry);
                        }
                    }
                });
        }
        weights[one(shape)] += (0..blocks)
            .into_par_iter()
            .map(|block| constant.iter().map(|entry| term(block, entry)).sum::<Ext>())
            .sum::<Ext>();
    }
    for (g, offsets) in trace.rows.iter().enumerate() {
        let weight = eq_rows[glue_row(shape, g)];
        for side in 0..7 {
            let matrix = if side == 6 { C } else { 1 + side };
            let scaled = rho[matrix] * weight;
            for &(wire, coefficient) in &trace.entries[offsets[side] as usize..offsets[side + 1] as usize] {
                weights[column(shape, wire)] += scaled * coefficient;
            }
        }
    }
    weights
}

/// The constraint `P = X^7 + Σ_i A_i·B_i − C` on one row's matrix values.
fn constraint<T: PrimeCharacteristicRing + Copy>(v: &[T; MATRICES]) -> T {
    v[X].exp_u64(7) + v[1] * v[2] + v[3] * v[4] + v[5] * v[6] - v[C]
}

/// The abscissas of an outer round message: `h(0), h(2), ..., h(DEGREE)`.
const ABSCISSAS: [usize; DEGREE] = [0, 2, 3, 4, 5, 6, 7, 8];

/// The outer sum-check of `Σ_x eq(τ, x)·P(x) = 0`, low bit first. Returns the
/// rounds, the point and every table at the point.
///
/// Round `i` writes `eq` as `c·ℓ(X)·eq(τ_{>i}, k)` (Gruen, ePrint 2024/108):
/// `c` is the product over the bound coordinates and `ℓ(X) = eq(τ_i, X)`.
/// The prover sums only `q(X) = Σ_k eq(τ_{>i}, k)·P_k(X)` and sends
/// `s(t) = c·ℓ(t)·q(t)`: the values the direct sum of `eq·P` gives.
/// Round 0 reads the Goldilocks tables; its fold moves them to `Ext`.
pub(crate) fn prove_outer(
    tables: Vec<Vec<Gl>>,
    tau: &[Ext],
    challenger: &mut Challenger,
) -> (Vec<Vec<Ext>>, Vec<Ext>, [Ext; MATRICES]) {
    let mut rounds = Vec::with_capacity(tau.len());
    let mut point = Vec::with_capacity(tau.len());
    let mut base = Some(tables);
    let mut folded: Vec<Vec<Ext>> = Vec::new();
    let mut bound = Ext::ONE;
    for (i, &tau_i) in tau.iter().enumerate() {
        let rest = &tau[i + 1..];
        let q = match &base {
            Some(tables) => line_sums(tables, rest),
            None => line_sums(&folded, rest),
        };
        let message: Vec<Ext> = q
            .iter()
            .zip(ABSCISSAS)
            .map(|(&value, t)| {
                let t = Ext::from_usize(t);
                bound * ((Ext::ONE - tau_i) * (Ext::ONE - t) + tau_i * t) * value
            })
            .collect();
        let r = sumcheck::send(&message, challenger);
        bound *= tau_i * r + (Ext::ONE - tau_i) * (Ext::ONE - r);
        match base.take() {
            Some(tables) => {
                folded = tables
                    .par_iter()
                    .map(|table| {
                        table
                            .par_chunks_exact(2)
                            .map(|pair| r * (pair[1] - pair[0]) + pair[0])
                            .collect()
                    })
                    .collect();
            }
            None => folded.par_iter_mut().for_each(|table| fold_low(table, r)),
        }
        rounds.push(message);
        point.push(r);
    }
    let values = match base {
        Some(tables) => std::array::from_fn(|j| Ext::from(tables[j][0])),
        None => std::array::from_fn(|j| folded[j][0]),
    };
    (rounds, point, values)
}

/// `q(t) = Σ_k eq(rest, k)·P_k(t)` at each of `ABSCISSAS`, where `P_k(t)` is
/// the constraint on the line through rows `2k` and `2k + 1`. `eq(rest, ·)`
/// is the product of two half-size tables (Dao and Thaler, ePrint
/// 2024/1210); a `T` value meets `Ext` only there.
fn line_sums<T>(tables: &[Vec<T>], rest: &[Ext]) -> [Ext; DEGREE]
where
    T: PrimeCharacteristicRing + Copy + Send + Sync,
    Ext: Algebra<T>,
{
    let split = rest.len() / 2;
    let (low, high) = (eq_table(&rest[..split]), eq_table(&rest[split..]));
    high.par_iter()
        .enumerate()
        .map(|(h, &outer)| {
            let mut sums = [Ext::ZERO; DEGREE];
            for (l, &inner) in low.iter().enumerate() {
                let k = (h << split) | l;
                let mut at: [T; MATRICES] = std::array::from_fn(|j| tables[j][2 * k]);
                let step: [T; MATRICES] = std::array::from_fn(|j| tables[j][2 * k + 1] - at[j]);
                let mut t = 0;
                for (sum, target) in sums.iter_mut().zip(ABSCISSAS) {
                    while t < target {
                        for j in 0..MATRICES {
                            at[j] += step[j];
                        }
                        t += 1;
                    }
                    *sum += inner * constraint(&at);
                }
            }
            sums.map(|sum| outer * sum)
        })
        .reduce(|| [Ext::ZERO; DEGREE], |a, b| std::array::from_fn(|s| a[s] + b[s]))
}
