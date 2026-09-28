//! Independent raw decoder for the Lean-owned physical source-row custody.
//!
//! Only `witnessInstructions` and `assertionRows` own source rows. Other
//! package schedules are skipped and cannot synthesize an expected row.

use serde::{de::IgnoredAny, Deserialize};
use serde_json::Value;

use super::{checked_add, Entry, Field, Result, GOLDILOCKS_MODULUS};

const MAX_DOMAIN: usize = 1 << 28;

#[derive(Deserialize)]
struct RawSealed(u64, RawPackage, Value, IgnoredAny, IgnoredAny, RawRange, u64);

#[derive(Deserialize)]
struct RawRange(u64, u64);

#[derive(Deserialize)]
struct RawPackage(
    u64,
    RawProfile,
    RawPoseidon,
    RawLayout,
    RawRelation,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    Vec<RawWitnessInstruction>,
    Vec<RawSparseRow>,
    IgnoredAny,
);

#[derive(Deserialize)]
struct RawProfile(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Deserialize)]
struct RawPoseidon(u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Deserialize)]
struct RawSegment(u64, u64, u64);

#[derive(Deserialize)]
struct RawLayout(u64, u64, u64, u64, u64, Vec<RawSegment>, Vec<RawSegment>);

#[derive(Deserialize)]
struct RawRelation(u64, u64, u64, Vec<u64>, u64, IgnoredAny);

#[derive(Deserialize)]
struct RawSparseTerm(u64, u64);

#[derive(Deserialize)]
struct RawSparseCombination(u64, Vec<RawSparseTerm>);

#[derive(Deserialize)]
struct RawWitnessInstruction(u64, u64, RawSparseCombination, RawSparseCombination);

#[derive(Deserialize)]
struct RawSparseRow(u64, RawSparseCombination, RawSparseCombination, RawSparseCombination);

#[derive(Clone, Copy, Debug)]
struct Segment {
    role: u64,
    start: usize,
    length: usize,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub row_count: usize,
    pub private_columns: usize,
    pub constant_column: usize,
    pub public_columns: usize,
    pub total_columns: usize,
    private_segments: Vec<Segment>,
}

#[derive(Clone, Debug)]
struct SparseCombination {
    constant: Field,
    terms: Vec<Entry>,
}

#[derive(Clone, Debug)]
enum OwnedRow {
    Witness {
        target: usize,
        a: SparseCombination,
        b: SparseCombination,
    },
    Assertion {
        a: SparseCombination,
        b: SparseCombination,
        c: SparseCombination,
    },
}

#[derive(Clone, Debug)]
struct IndexedRow {
    index: usize,
    row: OwnedRow,
}

#[derive(Clone, Debug)]
pub struct SourceCombination {
    pub constant: Field,
    pub terms: Vec<Entry>,
}

#[derive(Clone, Debug)]
pub struct SourceRow {
    pub a: SourceCombination,
    pub b: SourceCombination,
    pub c: SourceCombination,
}

pub struct Artifact {
    pub sealed_schema: u64,
    pub sources: SourcePackage,
    pub matrix_program: Value,
    pub logical_rows: usize,
    pub logical_columns: usize,
    pub cube_variables: usize,
    pub logical_public_inputs: usize,
}

pub struct SourcePackage {
    pub layout: Layout,
    rows: Vec<IndexedRow>,
}

impl SourcePackage {
    pub fn decode(bytes: &[u8]) -> Result<Artifact> {
        if bytes.last() != Some(&b'\n') {
            return Err("sealed package is not newline terminated".into());
        }
        let RawSealed(outer_schema, raw, matrix_program, _, _, raw_range, logical_public) =
            serde_json::from_slice(bytes).map_err(|error| format!("independent sealed decode: {error}"))?;
        if outer_schema != 6 {
            return Err("independent sealed schema is not 6".into());
        }
        let RawRange(next_start, next_count) = raw_range;
        if next_start.checked_add(next_count).is_none() || logical_public == 0 {
            return Err("invalid sealed suffix metadata".into());
        }
        let RawPackage(schema, profile, poseidon, raw_layout, relation, _, _, _, _, _, _, witnesses, assertions, _) =
            raw;
        if schema != 8 {
            return Err("independent inner package schema is not 8".into());
        }
        validate_profile(profile)?;
        validate_poseidon(poseidon)?;
        let layout = decode_layout(raw_layout)?;
        let (logical_rows, logical_columns, cube_variables) = decode_relation(relation)?;
        let witness_start = layout
            .private_segments
            .iter()
            .filter(|segment| segment.role == 3)
            .map(|segment| segment.start)
            .collect::<Vec<_>>();
        if witness_start.len() != 1 {
            return Err("expected one physical witness segment".into());
        }

        let mut rows = Vec::with_capacity(witnesses.len() + assertions.len());
        for RawWitnessInstruction(index, target, a, b) in witnesses {
            let index = to_usize(index, "witness source row")?;
            let target = to_usize(target, "witness target")?;
            if index >= layout.row_count || target < witness_start[0] || target >= layout.private_columns {
                return Err("invalid witness source-row ownership".into());
            }
            let a = decode_sparse(a, &layout)?;
            let b = decode_sparse(b, &layout)?;
            if a.terms
                .iter()
                .chain(&b.terms)
                .any(|term| term.column < layout.constant_column && term.column >= target)
            {
                return Err("noncausal witness source row".into());
            }
            rows.push(IndexedRow {
                index,
                row: OwnedRow::Witness { target, a, b },
            });
        }
        for RawSparseRow(index, a, b, c) in assertions {
            let index = to_usize(index, "assertion source row")?;
            if index >= layout.row_count {
                return Err("assertion source row is out of range".into());
            }
            rows.push(IndexedRow {
                index,
                row: OwnedRow::Assertion {
                    a: decode_sparse(a, &layout)?,
                    b: decode_sparse(b, &layout)?,
                    c: decode_sparse(c, &layout)?,
                },
            });
        }
        rows.sort_unstable_by_key(|row| row.index);
        if rows.windows(2).any(|pair| pair[0].index == pair[1].index) {
            return Err("duplicate physical source-row owner".into());
        }

        Ok(Artifact {
            sealed_schema: outer_schema,
            sources: Self { layout, rows },
            matrix_program,
            logical_rows,
            logical_columns,
            cube_variables,
            logical_public_inputs: to_usize(logical_public, "logical public input count")?,
        })
    }

    pub fn row(&self, index: usize) -> Result<SourceRow> {
        let row = self
            .rows
            .binary_search_by_key(&index, |row| row.index)
            .ok()
            .and_then(|position| self.rows.get(position))
            .ok_or_else(|| format!("physical source row {index} has no Lean-owned source entry"))?;
        Ok(match &row.row {
            OwnedRow::Witness { target, a, b } => SourceRow {
                a: copy_sparse(a),
                b: copy_sparse(b),
                c: SourceCombination {
                    constant: Field::ZERO,
                    terms: vec![Entry {
                        column: *target,
                        coefficient: Field::ONE,
                    }],
                },
            },
            OwnedRow::Assertion { a, b, c } => SourceRow {
                a: copy_sparse(a),
                b: copy_sparse(b),
                c: copy_sparse(c),
            },
        })
    }
}

fn validate_profile(raw: RawProfile) -> Result<()> {
    let RawProfile(modulus, base, digits, bound, fresh, running, rlc, children, matrices, cube) = raw;
    if (
        modulus, base, digits, bound, fresh, running, rlc, children, matrices, cube,
    ) != (GOLDILOCKS_MODULUS, 2, 16, 65_536, 1, 16, 17, 16, 14, 28)
    {
        return Err("unexpected independent production profile".into());
    }
    Ok(())
}

fn validate_poseidon(raw: RawPoseidon) -> Result<()> {
    let RawPoseidon(width, rate, digest, initial, partial, terminal, recipes, output) = raw;
    if (width, rate, digest, initial, partial, terminal, recipes, output) != (8, 4, 4, 4, 22, 4, 592, 584) {
        return Err("unexpected independent Poseidon2 schedule".into());
    }
    Ok(())
}

fn decode_layout(raw: RawLayout) -> Result<Layout> {
    let RawLayout(rows, private, constant, public, total, private_segments, public_segments) = raw;
    let private_columns = to_usize(private, "private column count")?;
    let constant_column = to_usize(constant, "constant column")?;
    let public_columns = to_usize(public, "public column count")?;
    let total_columns = to_usize(total, "total column count")?;
    let private_segments = decode_segments(private_segments, 0, private_columns)?;
    let public_start = checked_add(constant_column, 1, "public column start")?;
    let decoded_public = decode_segments(public_segments, public_start, total_columns)?;
    let layout = Layout {
        row_count: to_usize(rows, "physical row count")?,
        private_columns,
        constant_column,
        public_columns,
        total_columns,
        private_segments,
    };
    if layout.private_columns != layout.constant_column
        || checked_add(
            checked_add(layout.private_columns, 1, "layout columns")?,
            layout.public_columns,
            "layout columns",
        )? != layout.total_columns
        || layout.row_count.max(layout.total_columns.saturating_sub(1)) > MAX_DOMAIN
        || decoded_public
            .iter()
            .map(|segment| segment.length)
            .sum::<usize>()
            != layout.public_columns
    {
        return Err("invalid independent constant/public column mapping".into());
    }
    Ok(layout)
}

fn decode_segments(raw: Vec<RawSegment>, first: usize, expected_end: usize) -> Result<Vec<Segment>> {
    let mut cursor = first;
    let mut result = Vec::with_capacity(raw.len());
    for RawSegment(role, start, length) in raw {
        let start = to_usize(start, "segment start")?;
        let length = to_usize(length, "segment length")?;
        if start != cursor {
            return Err("unordered layout segment".into());
        }
        cursor = checked_add(cursor, length, "segment end")?;
        if cursor > expected_end {
            return Err("layout segment is out of range".into());
        }
        result.push(Segment { role, start, length });
    }
    if cursor != expected_end {
        return Err("layout segments do not cover their columns".into());
    }
    Ok(result)
}

fn decode_relation(raw: RawRelation) -> Result<(usize, usize, usize)> {
    let RawRelation(rows, columns, cube, sources, degree, _) = raw;
    if sources.len() != 14 || !sources.iter().copied().eq(0u64..14) || degree == 0 {
        return Err("invalid independent logical relation header".into());
    }
    Ok((
        to_usize(rows, "logical row count")?,
        to_usize(columns, "logical column count")?,
        to_usize(cube, "cube variable count")?,
    ))
}

fn decode_sparse(raw: RawSparseCombination, layout: &Layout) -> Result<SparseCombination> {
    let RawSparseCombination(constant, terms) = raw;
    let terms = terms
        .into_iter()
        .map(|RawSparseTerm(column, coefficient)| {
            let column = to_usize(column, "source sparse column")?;
            if column >= layout.total_columns || column == layout.constant_column {
                return Err("source sparse term uses an invalid or constant column".into());
            }
            Ok(Entry {
                column,
                coefficient: Field::checked(coefficient, "source sparse coefficient")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SparseCombination {
        constant: Field::checked(constant, "source sparse constant")?,
        terms,
    })
}

fn copy_sparse(source: &SparseCombination) -> SourceCombination {
    SourceCombination {
        constant: source.constant,
        terms: source.terms.clone(),
    }
}

fn to_usize(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| format!("{label} exceeds usize"))
}
