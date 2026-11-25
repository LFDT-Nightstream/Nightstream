//! A `Row::RankOne` lowers as its Lean `Expr` (`directConstraint`), and a
//! row renamed in place equals its renamed copy (`AuxRelocation.rename`).

use std::{collections::BTreeMap, num::NonZeroUsize};

use super::*;
use crate::application::ApplicationCircuit;
use crate::nebula::{
    batched_slots, circuit::memory_circuit, pack_regions, Application, Geometry, Memory, Poseidon2Hint,
    R1csApplication, RowRelation, SharedColumns, Size, Transition, View,
};
use crate::relation::{
    ColumnFamilySpec, ColumnRegistry, ColumnWidth, ContinuityCatalog, ContinuityGroup, ContinuityLink, MemoryCatalog,
    MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec,
};

fn assert_same_circuit(new: &ApplicationCircuit, old: &ApplicationCircuit) {
    let records = |circuit: &ApplicationCircuit| {
        let mut bytes = Vec::new();
        circuit.records().write_to(&mut bytes).unwrap();
        bytes
    };
    assert_eq!(records(new), records(old));
    assert_eq!(new.variable_count(), old.variable_count());
    assert_eq!(new.prepared_output_rows(), old.prepared_output_rows());
    assert_eq!(new.shape(), old.shape());
}

/// FNV-1a of the records, the counts, and the output rows.
fn digest(circuit: &ApplicationCircuit) -> u64 {
    let mut bytes = Vec::new();
    circuit.records().write_to(&mut bytes).unwrap();
    for word in [
        circuit.variable_count(),
        circuit.private_input_count(),
        circuit.shape().blocks.len(),
    ]
    .into_iter()
    .chain(circuit.prepared_output_rows())
    {
        bytes.extend((word as u64).to_le_bytes());
    }
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

fn var(index: usize) -> Expr {
    Expr::Var(Variable(index))
}

/// Empty sides, repeated and cancelling terms, zero coefficients, constant
/// words, affine words, and product words that need the generic lowering.
fn edge_forms() -> Vec<[Vec<(Expr, F)>; 3]> {
    let [x, y, w, u] = [4, 5, 6, 7].map(var);
    let f = F::from_u64;
    vec![
        [
            vec![(x.clone(), f(2)), (y.clone(), f(3))],
            vec![(w.clone(), F::ONE)],
            vec![(u.clone(), F::ONE)],
        ],
        [
            vec![],
            vec![(x.clone(), F::ONE), (y.clone(), f(5))],
            vec![(w.clone(), f(7))],
        ],
        [vec![(x.clone(), F::ONE)], vec![], vec![(y.clone(), F::ONE)]],
        [vec![], vec![], vec![(x.clone(), F::NEG_ONE)]],
        [vec![(x.clone(), F::ONE)], vec![(y.clone(), F::ONE)], vec![]],
        [vec![], vec![], vec![]],
        [
            vec![(x.clone(), F::ONE), (x.clone(), F::NEG_ONE)],
            vec![(y.clone(), f(2)), (y.clone(), f(3))],
            vec![(w.clone(), F::ZERO)],
        ],
        [
            vec![(Expr::constant(1), f(4)), (x.clone(), F::ONE)],
            vec![(Expr::constant(0), f(9))],
            vec![(Expr::constant(1), F::ONE)],
        ],
        [
            vec![(x.clone() + Expr::constant(2), f(3))],
            vec![(y.clone(), F::ONE)],
            vec![(Expr::constant(3) * w.clone(), F::ONE)],
        ],
        [vec![], vec![(Expr::constant(2) * u.clone(), F::ONE)], vec![]],
        [vec![(x.clone() * y.clone(), F::ONE)], vec![(w.clone(), F::ONE)], vec![]],
        [vec![], vec![(x.clone() * y.clone(), F::ONE)], vec![(w.clone(), F::ONE)]],
        [vec![(x, F::ONE)], vec![], vec![(y * w, F::NEG_ONE)]],
    ]
}

fn lowered(
    lower: impl FnOnce(&mut ApplicationBuilder) -> Result<usize, ApplicationError>,
) -> (usize, ApplicationCircuit) {
    let mut builder = ApplicationBuilder::new(4).unwrap();
    let row = lower(&mut builder).unwrap();
    let circuit = builder
        .finish(std::array::from_fn(|_| Affine::constant(F::ZERO)))
        .unwrap();
    (row, circuit)
}

#[test]
fn a_rank_one_row_lowers_as_its_expr() {
    let scope = BTreeSet::from([Variable(4), Variable(6)]);
    for forms in edge_forms() {
        let row = Row::RankOne(forms);
        let expression = row.expr();
        assert_eq!(row.lower_affine(), lower_affine(&expression));
        assert_eq!(row.rank_one(), rank_one_row(&expression));
        assert_eq!(row.variable_outside(&scope), expression.variable_outside(&scope));
        let (new_row, new) = lowered(|builder| row.lower(builder));
        let (old_row, old) = lowered(|builder| lower_constraint(builder, &expression));
        assert_eq!(new_row, old_row, "{expression:?}");
        assert_same_circuit(&new, &old);
    }
}

#[test]
fn renaming_in_place_matches_a_renamed_copy() {
    fn renamed(expression: &Expr, aliases: &BTreeMap<Variable, Variable>) -> Expr {
        match expression {
            Expr::Var(variable) => Expr::Var(*aliases.get(variable).unwrap_or(variable)),
            Expr::Const(value) => Expr::Const(*value),
            Expr::Add(left, right) => renamed(left, aliases) + renamed(right, aliases),
            Expr::Mul(left, right) => renamed(left, aliases) * renamed(right, aliases),
        }
    }
    let [x, y, w] = [4, 5, 6].map(var);
    let aliases = BTreeMap::from([(Variable(4), Variable(6)), (Variable(6), Variable(5))]);
    let sum = x.clone() + y.clone();
    let shared = sum.clone() * (sum - w.clone());
    let mut rows = vec![
        Row::Expr(shared.clone()),
        Row::Expr(shared.clone() + x.clone()),
        Row::RankOne([
            vec![(x.clone(), F::TWO)],
            vec![],
            vec![(w, F::ONE), (y, F::ONE), (x, F::ONE)],
        ]),
    ];
    let text = |expression: &Expr| format!("{expression:?}");
    let before = text(&shared);
    let expected: Vec<_> = rows
        .iter()
        .map(|row| text(&renamed(&row.expr(), &aliases)))
        .collect();
    for row in &mut rows {
        row.rename(&aliases);
    }
    assert_eq!(rows.iter().map(|row| text(&row.expr())).collect::<Vec<_>>(), expected);
    assert_eq!(
        text(&shared),
        before,
        "a node that another row shares keeps its variables"
    );
    assert_ne!(text(&rows[0].expr()), before);
}

const ONE: usize = 0;
const BEFORE: usize = 1;
const AFTER: usize = 2;
const VALUE: usize = 3;

fn registry() -> ColumnRegistry {
    ColumnRegistry::new([ColumnFamilySpec {
        region: "counter",
        start: 0,
        len: 4,
        name: "words".into(),
        role: "test",
        width: ColumnWidth::U32,
    }])
    .unwrap()
}

/// A ROM read port and a RAM write port at `BEFORE`, and one link.
fn application(rows: usize, shared: Vec<usize>) -> R1csApplication<u32> {
    let f = F::from_u64;
    let relation = RowRelation {
        rows: vec![
            [
                vec![(AFTER, F::ONE), (BEFORE, -F::ONE), (ONE, -F::ONE)],
                vec![(ONE, F::ONE)],
                vec![],
            ],
            [vec![], vec![(VALUE, F::ONE), (BEFORE, f(2))], vec![(AFTER, f(3))]],
            [vec![(VALUE, F::ONE)], vec![], vec![(BEFORE, F::ONE), (BEFORE, -F::ONE)]],
            [vec![], vec![], vec![]],
            [
                vec![(VALUE, F::ONE), (VALUE, F::ONE)],
                vec![(BEFORE, F::ONE), (ONE, f(5))],
                vec![(AFTER, F::ZERO)],
            ],
            [vec![(4, F::ONE)], vec![(4, F::ONE)], vec![(5, F::ONE)]],
            [vec![(VALUE, F::ONE)], vec![(4, F::ONE)], vec![(5, F::ONE)]],
        ],
        width: 6,
        one: ONE,
        range_checks: Vec::new(),
    };
    let port = |kind| MemoryPortSpec {
        address_columns: vec![BEFORE],
        value_column: VALUE,
        kind,
        activation: MemoryPortActivation::Always,
    };
    let catalog = MemoryCatalog::new(
        [
            MemorySpec {
                id: 0,
                kind: MemoryKind::Rom,
                ports: vec![port(MemoryPortKind::Read)],
            },
            MemorySpec {
                id: 1,
                kind: MemoryKind::Ram,
                ports: vec![port(MemoryPortKind::Write {
                    value_before_column: None,
                })],
            },
        ],
        &registry(),
    )
    .unwrap();
    let continuity = ContinuityCatalog::new(
        [ContinuityGroup {
            name: "count",
            role: "test",
            links: vec![ContinuityLink {
                previous_step_column: AFTER,
                next_step_column: BEFORE,
            }],
        }],
        &registry(),
    )
    .unwrap();
    let (regions, _) = pack_regions(
        catalog
            .entries()
            .iter()
            .map(|memory| (memory.id, memory.kind, vec![2])),
    )
    .unwrap();
    let count = NonZeroUsize::new(rows).unwrap();
    R1csApplication::with_shared(
        relation,
        &continuity,
        &catalog,
        Size {
            geometry: geometry(rows),
            rows: count,
            regions,
            slots: batched_slots(&catalog, &[vec![0], vec![1]], count).unwrap(),
        },
        SharedColumns {
            columns: shared,
            rows: NonZeroUsize::new(2).unwrap(),
        },
    )
    .unwrap()
}

fn geometry(rows: usize) -> Geometry {
    Geometry::new(2, 2, 2 * rows, 1, 1).unwrap()
}

/// The application with each row as its `Expr`, as the rows were before
/// `Row::RankOne`.
struct AsExprs<A>(A);

impl<A: Application> Application for AsExprs<A> {
    type Input = A::Input;
    type State = A::State;
    type Error = A::Error;

    fn width(&self) -> usize {
        self.0.width()
    }

    fn aux(&self) -> usize {
        self.0.aux()
    }

    fn port_count(&self) -> usize {
        self.0.port_count()
    }

    fn words(&self, state: &A::State) -> Vec<u64> {
        self.0.words(state)
    }

    fn rows(&self, view: &View) -> Result<Vec<Row>, crate::Error> {
        Ok(self
            .0
            .rows(view)?
            .iter()
            .map(|row| row.expr().into())
            .collect())
    }

    fn poseidon2_hints(&self, view: &View) -> Result<Vec<Poseidon2Hint>, crate::Error> {
        self.0.poseidon2_hints(view)
    }

    fn step(&self, state: &A::State, input: &A::Input, memory: &Memory) -> Result<Transition<A::State>, A::Error> {
        self.0.step(state, input, memory)
    }
}

/// SPEC.md section 7.6: the rank-one rows of one row, of linked rows, and of
/// a short final group of shared columns, with a read-only port. The pinned
/// digests are those of the build before `Row::RankOne`.
#[test]
fn an_r1cs_application_compiles_as_its_exprs() {
    let cases = [
        (1, Vec::new(), 0x7bd5_95ad_becf_6bae),
        (2, Vec::new(), 0x74c9_2013_3955_af32),
        (3, vec![4, 5], 0xfadc_4230_e57b_8d2a),
    ];
    for (rows, shared, pinned) in cases {
        let app = application(rows, shared);
        let new = memory_circuit(&geometry(rows), &app).unwrap();
        let old = memory_circuit(&geometry(rows), &AsExprs(app)).unwrap();
        assert_same_circuit(&new, &old);
        assert_eq!(digest(&new), pinned);
    }
}
