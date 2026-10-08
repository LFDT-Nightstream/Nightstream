//! The shrink layer: an argument that a program written over `Backend`
//! accepts its statement, with one WHIR commitment.
//!
//! Owns: the program contract, the shrink relation (layout and WHIR), the
//! proof, its prover and its verifier. The argument is SuperSpartan (ePrint
//! 2023/552) for the CCS of `layout.rs`: a zero-check sum-check of degree 8
//! over the rows, a batching draw, a product sum-check over `z`, and one WHIR
//! opening of `w`. The verifier rebuilds the glue rows by a shape run of the
//! program and computes the block rows in closed form; it stores no rows.
//!
//! Invariants: the transcript binds the shape, the WHIR profile and the
//! statement before the commitment; the verifier's shape run must emit the
//! shape's counts and the statement it bound.

mod evaluate;
pub(crate) mod layout;

use p3_challenger_v08::{CanObserve, FieldChallenger};
use p3_field_v08::{Field, PrimeCharacteristicRing};
use p3_security_v08::{ErrorBits, SecurityTerm};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub use self::layout::Shape;

use self::evaluate::Evaluate;
use self::layout::Layout;
use crate::circuit::block::{Kind, C, KINDS, MATRICES, X};
use crate::circuit::record::{Recorder, Sink, Term, Trace, Wire};
use crate::circuit::{algebra, Backend, Native};
use crate::field::{eq_table, fold_low, gl, Ext, Gl};
use crate::hash::{permutation, Challenger};
use crate::pcs::{self, Pcs, TablePlan};
use crate::{sumcheck, Error};

/// Names the shrink protocol at the start of its transcript.
const DOMAIN: &[u8] = b"Nightstream/SuperNeo/shrink/v1";
/// Outer sum-check degree: `eq` times the S-box term `(X·z)^7`.
const DEGREE: usize = 8;

/// A verifier written once over `Backend`.
pub trait Program {
    /// The statement words, canonical; `run` emits exactly these as
    /// `public`, first.
    fn statement(&self) -> Vec<u64>;
    /// Run the checks. A shape run reads its proof as zeros.
    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error>;
}

/// Counts what a run records.
#[derive(Default)]
struct Count(Shape);

impl Sink for Count {
    fn row(&mut self, _: &[(Vec<Term>, Vec<Term>)], _: &[Term]) {
        self.0.glue_rows += 1;
    }

    fn cell(&mut self, _: Gl) {
        self.0.glue_cells += 1;
    }

    fn block(&mut self, kind: Kind, _: &[Gl]) {
        self.0.blocks[kind.index()] += 1;
    }

    fn public(&mut self, _: Gl) {
        self.0.publics += 1;
    }
}

impl Shape {
    /// The counts of one run of `program`.
    pub fn derive(program: &impl Program) -> Result<Self, Error> {
        let mut recorder = Recorder::new(Count::default());
        program.run(&mut recorder)?;
        Ok(recorder.finish().0 .0)
    }
}

/// The shrink relation of one shape: the layout and the WHIR configuration.
pub struct Shrink {
    layout: Layout,
    pcs: Pcs,
}

impl Shrink {
    /// `security_bits` is `-log2` of the soundness error this layer may add.
    pub fn new(shape: Shape, security_bits: f64) -> Result<Self, Error> {
        let layout = Layout::new(shape);
        let plans = vec![TablePlan {
            variables: layout.cell_variables,
            width: 1,
            points: vec![vec![0]],
        }];
        let terms = terms(&layout);
        let pcs = Pcs::new(pcs::SHRINK, plans, security_bits, &|_| terms.clone())?;
        Ok(Self { layout, pcs })
    }

    pub fn security_bits(&self) -> f64 {
        self.pcs.security_bits()
    }

    fn start(&self, statement: &[Gl]) -> Challenger {
        let mut challenger = Challenger::new(permutation().clone());
        challenger.observe_slice(&neo_transcript::domain_chunk_v1_1(DOMAIN).map(gl));
        challenger.observe_slice(&self.layout.shape.words());
        challenger.observe_slice(&self.pcs.profile_words());
        challenger.observe_slice(statement);
        challenger
    }
}

/// The sum-check draws charged over WHIR's candidates, as degrees over |Ext|.
fn terms(layout: &Layout) -> Vec<SecurityTerm> {
    let field_bits = (<Ext as Field>::bits() - 1) as f64;
    let term = |label, degree: f64| SecurityTerm::new(label, ErrorBits::from_log2(field_bits - degree.log2()));
    let (m, n) = (layout.row_variables as f64, layout.cell_variables as f64);
    vec![
        term("zero-check point", m),
        term("outer sum-check", DEGREE as f64 * m),
        term("matrix batching", (MATRICES - 1) as f64),
        term("inner sum-check", 2.0 * (n + 1.0)),
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
        let proof: Self = bincode::deserialize(bytes).map_err(|_| Error::Codec("shrink proof"))?;
        if proof.to_bytes() != bytes {
            return Err(Error::Codec("non-canonical shrink proof"));
        }
        Ok(proof)
    }
}

/// The statement words as layer-1 field values; they must be canonical.
fn statement_of(program: &impl Program) -> Result<Vec<Gl>, Error> {
    let words = program.statement();
    if words
        .iter()
        .any(|&word| word >= <Gl as p3_field_v08::PrimeField64>::ORDER_U64)
    {
        return Err(Error::Shape("statement words must be canonical"));
    }
    Ok(words.into_iter().map(Gl::new).collect())
}

/// Prove that `program` accepts its statement.
pub fn prove(shrink: &Shrink, program: &impl Program) -> Result<ShrinkProof, Error> {
    let layout = &shrink.layout;
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
    if counts != layout.shape || trace.public != statement {
        return Err(Error::Shape("the program's run differs from the shrink shape"));
    }
    let z = z_table(layout, &trace);
    let products = products(layout, &trace, &z);

    let mut challenger = shrink.start(&statement);
    let n = layout.cell_variables;
    let (commitment, data) = shrink
        .pcs
        .commit(vec![z[..1 << n].to_vec()], &mut challenger);
    let tau: Vec<Ext> = (0..layout.row_variables)
        .map(|_| challenger.sample_algebra_element())
        .collect();
    let (outer, rx, values) = prove_outer(products, &tau, &mut challenger);
    challenger.observe_algebra_slice(&values);
    let rho = powers(challenger.sample_algebra_element());
    let weights = weights(layout, &trace, &rx, &rho);
    let z: Vec<Ext> = z.into_par_iter().map(Ext::from).collect();
    let (inner, ry, _) = sumcheck::prove_product(weights, z, &mut challenger);
    let opening = shrink.pcs.open(data, &[ry[..n].to_vec()], &mut challenger);
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
    let layout = &shrink.layout;
    let (m, n) = (layout.row_variables, layout.cell_variables);
    if proof.outer.len() != m
        || proof.outer.iter().any(|message| message.len() != DEGREE)
        || proof.inner.len() != n + 1
        || proof.inner.iter().any(|message| message.len() != 2)
    {
        return Err(Error::Rejected("shrink proof shape"));
    }
    let statement = statement_of(program)?;
    let mut challenger = shrink.start(&statement);
    shrink.pcs.observe(&proof.commitment, &mut challenger);
    let tau: Vec<Ext> = (0..m)
        .map(|_| challenger.sample_algebra_element())
        .collect();
    let (rx, last) = replay(&proof.outer, Ext::ZERO, &mut challenger);
    let v = proof.values;
    let constraint = v[X].exp_u64(7) + v[1] * v[2] + v[3] * v[4] + v[5] * v[6] - v[C];
    if last != eq(&tau, &rx) * constraint {
        return Err(Error::Rejected("shrink outer sum-check"));
    }
    challenger.observe_algebra_slice(&v);
    let rho = powers(challenger.sample_algebra_element());
    let claim = rho.iter().zip(&v).map(|(&r, &value)| r * value).sum();
    let (ry, last) = replay(&proof.inner, claim, &mut challenger);
    let opened = shrink
        .pcs
        .verify(&proof.commitment, &proof.opening, &[ry[..n].to_vec()], &mut challenger)?;
    let w = opened[0][0];

    let mut recorder = Recorder::new(Evaluate::new(layout, &rx, &ry, rho));
    program.run(&mut recorder)?;
    let (sink, _) = recorder.finish();
    let counts = Shape {
        blocks: sink.blocks,
        glue_rows: sink.glue_rows,
        glue_cells: sink.glue_cells,
        publics: sink.publics.len(),
    };
    if counts != layout.shape || sink.publics != statement {
        return Err(Error::Rejected("the program's shape run differs from the shrink shape"));
    }
    let matrices = sink.total + layout.block_part(&rx, &ry, &rho);
    let mut public = vec![Gl::ONE];
    public.extend(&statement);
    let public_eq = eq_table(&ry[..n]);
    let x: Ext = public
        .iter()
        .zip(&public_eq)
        .map(|(&value, &weight)| weight * value)
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

fn eq(a: &[Ext], b: &[Ext]) -> Ext {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x * y + (Ext::ONE - x) * (Ext::ONE - y))
        .product()
}

/// Replay sum-check rounds (`h(0), h(2), ..., h(d)`) on the native challenger.
fn replay(rounds: &[Vec<Ext>], mut claim: Ext, challenger: &mut Challenger) -> (Vec<Ext>, Ext) {
    let mut point = Vec::with_capacity(rounds.len());
    for message in rounds {
        let mut values = vec![message[0], claim - message[0]];
        values.extend_from_slice(&message[1..]);
        let r = sumcheck::send(message, challenger);
        claim = algebra::interpolate(&mut Native, &values, r);
        point.push(r);
    }
    (point, claim)
}

/// `z = (w, 1, x)` on `2^(n+1)` cells.
fn z_table(layout: &Layout, trace: &Trace) -> Vec<Gl> {
    let n = layout.cell_variables;
    let mut z = vec![Gl::ZERO; 2 << n];
    for (cell, &value) in trace.glue.iter().enumerate() {
        z[layout.column(Wire::Glue(cell as u32))] = value;
    }
    for kind in KINDS {
        for (block, cells) in trace.blocks[kind.index()]
            .chunks_exact(kind.cells())
            .enumerate()
        {
            for (cell, &value) in cells.iter().enumerate() {
                z[layout.block_cell(kind, block, cell)] = value;
            }
        }
    }
    let one = layout.one();
    z[one] = Gl::ONE;
    z[one + 1..one + 1 + trace.public.len()].copy_from_slice(&trace.public);
    z
}

/// `M_j·z` on the row cube for every matrix.
fn products(layout: &Layout, trace: &Trace, z: &[Gl]) -> Vec<Vec<Ext>> {
    let rows = 1usize << layout.row_variables;
    let mut tables = vec![vec![Gl::ZERO; rows]; MATRICES];
    for kind in KINDS {
        for block in 0..layout.shape.blocks[kind.index()] {
            for entry in kind.entries() {
                tables[entry.matrix][layout.block_row(kind, block, entry.row)] +=
                    entry.coefficient * z[layout.entry_column(kind, block, entry.cell)];
            }
        }
    }
    for (g, offsets) in trace.rows.iter().enumerate() {
        let row = layout.glue_row(g);
        for side in 0..7 {
            let matrix = if side == 6 { C } else { 1 + side };
            let terms = &trace.entries[offsets[side] as usize..offsets[side + 1] as usize];
            tables[matrix][row] = terms
                .iter()
                .map(|&(wire, coefficient)| coefficient * z[layout.column(wire)])
                .sum();
        }
    }
    tables
        .into_par_iter()
        .map(|table| table.into_iter().map(Ext::from).collect())
        .collect()
}

/// `Σ_j ρ^j·M_j(rx, y)` for every column `y` of `z`.
fn weights(layout: &Layout, trace: &Trace, rx: &[Ext], rho: &[Ext; MATRICES]) -> Vec<Ext> {
    let eq_rows = eq_table(rx);
    let mut weights = vec![Ext::ZERO; 2 << layout.cell_variables];
    for kind in KINDS {
        for block in 0..layout.shape.blocks[kind.index()] {
            for entry in kind.entries() {
                weights[layout.entry_column(kind, block, entry.cell)] +=
                    rho[entry.matrix] * eq_rows[layout.block_row(kind, block, entry.row)] * entry.coefficient;
            }
        }
    }
    for (g, offsets) in trace.rows.iter().enumerate() {
        let weight = eq_rows[layout.glue_row(g)];
        for side in 0..7 {
            let matrix = if side == 6 { C } else { 1 + side };
            let scaled = rho[matrix] * weight;
            for &(wire, coefficient) in &trace.entries[offsets[side] as usize..offsets[side + 1] as usize] {
                weights[layout.column(wire)] += scaled * coefficient;
            }
        }
    }
    weights
}

/// The outer sum-check of `Σ_x eq(τ, x)·(X^7 + Σ_i A_i·B_i − C)(x) = 0`, low
/// bit first. Returns the rounds, the point and every table at the point.
fn prove_outer(
    mut tables: Vec<Vec<Ext>>,
    tau: &[Ext],
    challenger: &mut Challenger,
) -> (Vec<Vec<Ext>>, Vec<Ext>, [Ext; MATRICES]) {
    let mut eq = eq_table(tau);
    let mut rounds = Vec::with_capacity(tau.len());
    let mut point = Vec::with_capacity(tau.len());
    while eq.len() > 1 {
        let half = eq.len() / 2;
        let sums = (0..half)
            .into_par_iter()
            .fold(
                || [Ext::ZERO; DEGREE + 1],
                |mut acc, k| {
                    let mut e = eq[2 * k];
                    let de = eq[2 * k + 1] - e;
                    let mut at: [Ext; MATRICES] = std::array::from_fn(|j| tables[j][2 * k]);
                    let step: [Ext; MATRICES] = std::array::from_fn(|j| tables[j][2 * k + 1] - at[j]);
                    for (t, sum) in acc.iter_mut().enumerate() {
                        if t != 1 {
                            *sum += e * (at[X].exp_u64(7) + at[1] * at[2] + at[3] * at[4] + at[5] * at[6] - at[C]);
                        }
                        e += de;
                        for j in 0..MATRICES {
                            at[j] += step[j];
                        }
                    }
                    acc
                },
            )
            .reduce(|| [Ext::ZERO; DEGREE + 1], |a, b| std::array::from_fn(|t| a[t] + b[t]));
        let mut message = vec![sums[0]];
        message.extend_from_slice(&sums[2..]);
        let r = sumcheck::send(&message, challenger);
        fold_low(&mut eq, r);
        tables.par_iter_mut().for_each(|table| fold_low(table, r));
        rounds.push(message);
        point.push(r);
    }
    (rounds, point, std::array::from_fn(|j| tables[j][0]))
}
