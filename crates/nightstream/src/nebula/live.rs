use std::ops::Range;

use neo_application::{ColumnRegistry, ColumnWidth};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::app::Application;
use super::expr::{direct_row, Row};
use super::rows::{Sym, View};
use super::words::Layout;
use super::{Geometry, Program, R1csApplication, SegmentError};
use crate::application::{private_variables, Affine};

/// Application rows and their positions in a Nebula invocation.
pub struct Circuit {
    pub rows: Vec<[Affine; 3]>,
    pub variables: usize,
    pub app_in: Range<usize>,
    pub app_out: Range<usize>,
    pub app_aux: Range<usize>,
    pub op_slots: Range<usize>,
    pub op_words: usize,
    pub op_fields: [usize; 8],
    pub state: Vec<String>,
    pub declared_boolean: Vec<bool>,
}

/// Extract the application rows and witness layout for a trusted application definition.
/// Supply its R1CS and column registry. Invalid geometry, incompatible dimensions,
/// and rows that need generated variables return a circuit error.
pub fn circuit<A: Application, Id: Clone + Eq>(
    geometry: &Geometry,
    app: &A,
    r1cs: &R1csApplication<Id>,
    columns: &ColumnRegistry,
) -> Result<Circuit, crate::Error> {
    if !geometry.is_valid() || !geometry.is_secure() {
        return Err(crate::Error::Circuit("the memory geometry breaks SPEC.md section 7.1"));
    }
    super::check_port_count(geometry, app)?;
    if app.width() != r1cs.width() || app.aux() != r1cs.aux() || app.port_count() != r1cs.port_count() {
        return Err(crate::Error::Circuit(
            "live: R1CS dimensions differ from the application",
        ));
    }
    let layout = Layout::new(geometry, app.width(), app.aux());
    let wires = private_variables(layout.count());
    let sym = Sym::new(geometry, layout, &wires, [F::ZERO; 12]);
    let view = View::new(&sym);
    let rows = app
        .rows(&view)?
        .into_iter()
        .map(|row| {
            let sides = match &row {
                Row::Expr(e) => direct_row(e),
                Row::RankOne(_) => row.rank_one(),
            }
            .ok_or(crate::Error::Circuit("live: application row needs generated variables"))?;
            Ok([sides.0, sides.1, sides.2])
        })
        .collect::<Result<Vec<_>, crate::Error>>()?;
    let variables = 4 + layout.count() + 4;
    let range = |start, len| 4 + start..4 + start + len;
    let app_in = range(layout.app_in(0), app.width());
    let app_out = range(layout.app_out(0), app.width());
    let app_aux = range(layout.aux(0), app.aux());
    let op_words = layout.op(1, 0) - layout.op(0, 0);
    let op_slots = range(layout.op(0, 0), geometry.b_ops * op_words);
    let op_fields = [
        0,
        1,
        2,
        layout.op_addr(0),
        layout.op_vr(0),
        layout.op_vw(0),
        layout.op_rt(0),
        layout.op_diff(0),
    ];
    let mut state = Vec::with_capacity(r1cs.links().len());
    for &(_, column) in r1cs.links() {
        let family = columns
            .family_for_column(column)
            .ok_or(crate::Error::Circuit("live: state column has no family"))?;
        let name = family
            .name
            .strip_suffix("_BEFORE")
            .ok_or(crate::Error::Circuit("live: state column name has no BEFORE suffix"))?;
        state.push(if family.len > 1 {
            format!("{name}[{}]", column - family.start)
        } else {
            name.to_owned()
        });
    }
    let mut declared_boolean = vec![false; variables];
    let mut boolean_columns = vec![false; columns.column_count()];
    for family in columns.families() {
        if family.width == ColumnWidth::Boolean {
            boolean_columns[family.start..family.end()].fill(true);
        }
    }
    let mut seen = vec![false; app.aux()];
    for t in 0..r1cs.trace_rows() {
        let mut c = 0;
        while let Some(k) = r1cs.witness_column(t, c) {
            if !seen[k] {
                seen[k] = true;
                declared_boolean[app_aux.start + k] = boolean_columns.get(c).copied().unwrap_or(false);
            }
            c += 1;
        }
    }
    Ok(Circuit {
        rows,
        variables,
        app_in,
        app_out,
        app_aux,
        op_slots,
        op_words,
        op_fields,
        state,
        declared_boolean,
    })
}

/// The program images in ROM and RAM order.
pub fn program_images(program: &Program) -> (&[u32], &[u32]) {
    (&program.rom, &program.ram)
}

/// One invocation and its decoded memory records.
pub struct Step {
    pub values: Vec<F>,
    pub ops: Vec<[u64; 8]>,
    pub initial: Vec<[u64; 2]>,
    pub final_: Vec<[u64; 2]>,
    pub state_in: [F; 4],
    pub state_out: [F; 4],
    pub carry_in: Vec<F>,
    pub carry_out: Vec<F>,
}

/// The current run statement.
pub struct Statement {
    pub steps: u64,
    pub segments: u64,
    pub final_ts: u64,
    pub final_root: [F; 4],
}

/// An incremental native memory run.
pub struct Run<A: Application> {
    inner: super::Run<A>,
    geometry: Geometry,
    layout: Layout,
}

impl<A: Application> Run<A> {
    /// Start a run from the program images and initial state.
    pub fn new(app: A, program: &Program, initial: A::State) -> Self {
        let layout = Layout::new(&program.geometry, app.width(), app.aux());
        Self {
            geometry: program.geometry,
            layout,
            inner: super::Run::new(app, program, initial),
        }
    }

    /// The statement after the segments run so far.
    pub fn statement(&self) -> Statement {
        Statement {
            steps: self.inner.steps,
            segments: self.inner.carry.seg_idx,
            final_ts: self.inner.carry.ts,
            final_root: self.inner.carry.mem_root,
        }
    }

    /// Execute the next segment once and return its records.
    pub fn segment(&mut self, inputs: &[A::Input]) -> Result<Vec<Step>, SegmentError> {
        let mut state = self
            .inner
            .context
            .state(&self.inner.state, &self.inner.carry);
        let mut carry = self.inner.carry;
        let invocations = self.inner.segment(inputs)?;
        let g = &self.geometry;
        let l = &self.layout;
        Ok(invocations
            .into_iter()
            .map(|inv| {
                let bits = |start: usize, len: usize| {
                    (0..len).fold(0, |v, k| v | (inv.words[start + k].as_canonical_u64() << k))
                };
                let ops = (0..g.b_ops)
                    .map(|j| {
                        let base = l.op(j, 0);
                        [
                            inv.words[base].as_canonical_u64(),
                            inv.words[base + 1].as_canonical_u64(),
                            inv.words[base + 2].as_canonical_u64(),
                            bits(base + l.op_addr(0), g.addr_width()),
                            bits(base + l.op_vr(0), 32),
                            bits(base + l.op_vw(0), 32),
                            bits(base + l.op_rt(0), g.w_ts),
                            bits(base + l.op_diff(0), g.w_ts),
                        ]
                    })
                    .collect();
                let initial = (0..g.b_scan)
                    .map(|j| [bits(l.initial(j, 0), 32), bits(l.initial(j, 32), g.w_ts)])
                    .collect();
                let final_ = (0..g.b_scan)
                    .map(|j| [bits(l.final_(j, 0), 32), bits(l.final_(j, 32), g.w_ts)])
                    .collect();
                let mut values = Vec::with_capacity(8 + inv.words.len());
                values.extend(state);
                values.extend_from_slice(&inv.words);
                values.extend(inv.output);
                let step = Step {
                    values,
                    ops,
                    initial,
                    final_,
                    state_in: state,
                    state_out: inv.output,
                    carry_in: carry.words().to_vec(),
                    carry_out: inv.carry.words().to_vec(),
                };
                state = inv.output;
                carry = inv.carry;
                step
            })
            .collect())
    }
}
