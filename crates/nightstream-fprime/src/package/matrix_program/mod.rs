//! Generic execution of the Lean-authored compact 14-matrix program.
//!
//! This module owns wire decoding and row interpretation. It does not select
//! phases, applications, row schedules, or matrix formulas.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use rayon::prelude::*;
use serde_json::Value;

use super::{PackageError, GOLDILOCKS_MODULUS};

mod affine;
mod phi81;
mod poseidon;
mod poseidon_input;

#[cfg(test)]
#[path = "../../../tests/unit/matrix_program.rs"]
mod matrix_program_tests;

use affine::{AffineProgram, Coordinate};

pub(super) const MEANINGFUL_PORTS: usize = 13;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub(super) column: usize,
    pub(super) coefficient: Goldilocks,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) enum Form {
    #[default]
    Empty,
    One(Entry),
    Many(Vec<Entry>),
}

impl Form {
    pub(super) fn singleton(column: usize, coefficient: Goldilocks) -> Self {
        if coefficient == Goldilocks::ZERO {
            Self::default()
        } else {
            Self::One(Entry { column, coefficient })
        }
    }

    fn from_canonical_entries(mut entries: Vec<Entry>) -> Self {
        match entries.len() {
            0 => Self::Empty,
            1 => Self::One(entries.pop().expect("one form entry")),
            _ => Self::Many(entries),
        }
    }

    fn from_entries(mut entries: Vec<Entry>) -> Self {
        entries.sort_unstable_by_key(|entry| entry.column);
        let mut combined: Vec<Entry> = Vec::with_capacity(entries.len());
        for entry in entries {
            if let Some(last) = combined.last_mut() {
                if last.column == entry.column {
                    last.coefficient += entry.coefficient;
                    if last.coefficient == Goldilocks::ZERO {
                        combined.pop();
                    }
                    continue;
                }
            }
            if entry.coefficient != Goldilocks::ZERO {
                combined.push(entry);
            }
        }
        Self::from_canonical_entries(combined)
    }

    pub(super) fn entries(&self) -> &[Entry] {
        match self {
            Self::Empty => &[],
            Self::One(entry) => std::slice::from_ref(entry),
            Self::Many(entries) => entries,
        }
    }

    pub(super) fn into_entries(self) -> Vec<Entry> {
        match self {
            Self::Empty => Vec::new(),
            Self::One(entry) => vec![entry],
            Self::Many(entries) => entries,
        }
    }

    pub(super) fn append(self, other: Self) -> Self {
        let (left, right) = match (self, other) {
            (Self::Empty, other) => return other,
            (form, Self::Empty) => return form,
            (Self::One(left), Self::One(right)) => {
                return match left.column.cmp(&right.column) {
                    std::cmp::Ordering::Less => Self::Many(vec![left, right]),
                    std::cmp::Ordering::Greater => Self::Many(vec![right, left]),
                    std::cmp::Ordering::Equal => Self::singleton(left.column, left.coefficient + right.coefficient),
                };
            }
            (Self::Many(mut left), Self::Many(right))
                if left.last().expect("nonempty form").column < right.first().expect("nonempty form").column =>
            {
                left.extend(right);
                return Self::Many(left);
            }
            (Self::Many(left), Self::Many(mut right))
                if right.last().expect("nonempty form").column < left.first().expect("nonempty form").column =>
            {
                right.extend(left);
                return Self::Many(right);
            }
            (Self::Many(mut entries), Self::One(entry))
                if entries.last().expect("nonempty form").column < entry.column =>
            {
                entries.push(entry);
                return Self::Many(entries);
            }
            (Self::One(entry), Self::Many(mut entries))
                if entries.last().expect("nonempty form").column < entry.column =>
            {
                entries.push(entry);
                return Self::Many(entries);
            }
            (left, right) => (left.into_entries(), right.into_entries()),
        };
        let mut left = left.into_iter().peekable();
        let mut right = right.into_iter().peekable();
        let mut entries = Vec::with_capacity(left.len() + right.len());
        while let (Some(left_entry), Some(right_entry)) = (left.peek(), right.peek()) {
            match left_entry.column.cmp(&right_entry.column) {
                std::cmp::Ordering::Less => {
                    entries.push(left.next().expect("peeked left entry"));
                }
                std::cmp::Ordering::Greater => {
                    entries.push(right.next().expect("peeked right entry"));
                }
                std::cmp::Ordering::Equal => {
                    let left_entry = left.next().expect("peeked left entry");
                    let right_entry = right.next().expect("peeked right entry");
                    let coefficient = left_entry.coefficient + right_entry.coefficient;
                    if coefficient != Goldilocks::ZERO {
                        entries.push(Entry {
                            column: left_entry.column,
                            coefficient,
                        });
                    }
                }
            }
        }
        entries.extend(left);
        entries.extend(right);
        Self::from_canonical_entries(entries)
    }

    pub(super) fn scaled(mut self, scalar: Goldilocks) -> Self {
        if scalar == Goldilocks::ZERO {
            return Self::default();
        }
        for entry in match &mut self {
            Self::Empty => &mut [],
            Self::One(entry) => std::slice::from_mut(entry),
            Self::Many(entries) => entries,
        } {
            entry.coefficient *= scalar;
        }
        self
    }
}

pub(super) type RowForms = [Form; MEANINGFUL_PORTS];

pub(super) fn empty_row() -> RowForms {
    std::array::from_fn(|_| Form::default())
}

pub(super) fn decode_form(value: &Value) -> Result<Form, PackageError> {
    let entries = array(value, "matrix sparse form")?;
    let mut decoded = Vec::with_capacity(entries.len());
    for entry in entries {
        let fields = exact_array(entry, 2, "matrix sparse entry")?;
        decoded.push(Entry {
            column: usize_atom(&fields[0], "matrix sparse column")?,
            coefficient: field_atom(&fields[1], "matrix sparse coefficient")?,
        });
    }
    Ok(Form::from_entries(decoded))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RetainedKind {
    Bit,
    Centered,
    Field,
}

impl RetainedKind {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        match usize_atom(value, "retained kind")? {
            0 => Ok(Self::Bit),
            1 => Ok(Self::Centered),
            2 => Ok(Self::Field),
            _ => Err(PackageError::Invalid("retained kind")),
        }
    }

    fn width(self) -> usize {
        match self {
            Self::Bit | Self::Centered => 1,
            Self::Field => 41,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RetainedBlock {
    kind: RetainedKind,
    slot_count: usize,
    start: usize,
}

impl RetainedBlock {
    pub(super) fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 3, "retained block")?;
        Ok(Self {
            kind: RetainedKind::decode(&fields[0])?,
            slot_count: usize_atom(&fields[1], "retained slot count")?,
            start: usize_atom(&fields[2], "retained start")?,
        })
    }

    pub(super) fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub(super) fn kind(&self) -> RetainedKind {
        self.kind
    }

    pub(super) fn coordinate_count(&self) -> Result<usize, PackageError> {
        self.slot_count
            .checked_mul(self.kind.width())
            .ok_or(PackageError::Invalid("retained coordinate count"))
    }

    pub(super) fn fits(&self, logical_width: usize) -> Result<bool, PackageError> {
        Ok(self
            .start
            .checked_add(self.coordinate_count()?)
            .ok_or(PackageError::Invalid("retained column range"))?
            <= logical_width)
    }

    pub(super) fn form(&self, logical_width: usize, slot: usize) -> Result<Form, PackageError> {
        if slot >= self.slot_count || !self.fits(logical_width)? {
            return Err(PackageError::Invalid("retained slot"));
        }
        let width = self.kind.width();
        let first = self
            .start
            .checked_add(
                slot.checked_mul(width)
                    .ok_or(PackageError::Invalid("retained slot offset"))?,
            )
            .ok_or(PackageError::Invalid("retained slot offset"))?;
        let mut entries = Vec::with_capacity(width);
        let mut weight = Goldilocks::ONE;
        let radix = Goldilocks::from_u64(3);
        for coordinate in 0..width {
            entries.push(Entry {
                column: first + coordinate,
                coefficient: weight,
            });
            weight *= radix;
        }
        Ok(Form::from_canonical_entries(entries))
    }

    pub(super) fn external_form(
        &self,
        logical_width: usize,
        slot_base: usize,
        lane: usize,
    ) -> Result<Form, PackageError> {
        if lane >= 8 {
            return Err(PackageError::Invalid("retained external lane"));
        }
        let mut state = Vec::with_capacity(8);
        for selected in 0..8 {
            state.push(self.form(
                logical_width,
                checked_add(slot_base, selected, "retained external slot")?,
            )?);
        }
        Ok(external_layer(&state)?[lane].clone())
    }
}

#[derive(Clone, Debug)]
struct SourceRange {
    source_start: usize,
    source_count: usize,
    retained: RetainedBlock,
    slot_start: usize,
}

impl SourceRange {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 4, "matrix source range")?;
        Ok(Self {
            source_start: usize_atom(&fields[0], "source range start")?,
            source_count: usize_atom(&fields[1], "source range count")?,
            retained: RetainedBlock::decode(&fields[2])?,
            slot_start: usize_atom(&fields[3], "source range slot")?,
        })
    }

    fn form(&self, logical_width: usize, source: usize) -> Result<Option<Form>, PackageError> {
        if source < self.source_start {
            return Ok(None);
        }
        let offset = source - self.source_start;
        if offset >= self.source_count {
            return Ok(None);
        }
        Ok(Some(self.retained.form(
            logical_width,
            checked_add(self.slot_start, offset, "source range slot")?,
        )?))
    }
}

#[derive(Clone, Copy, Debug)]
enum SourceGridMode {
    Direct,
    External8,
}

#[derive(Clone, Debug)]
struct SourceGrid {
    source_start: usize,
    major_count: usize,
    major_source_stride: usize,
    minor_count: usize,
    minor_source_stride: usize,
    run_count: usize,
    retained: RetainedBlock,
    mode: SourceGridMode,
    slot_start: usize,
    major_slot_stride: usize,
    minor_slot_stride: usize,
}

impl SourceGrid {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 11, "matrix source grid")?;
        let mode = match usize_atom(&fields[7], "matrix source grid mode")? {
            0 => SourceGridMode::Direct,
            1 => SourceGridMode::External8,
            _ => return Err(PackageError::Invalid("matrix source grid mode")),
        };
        Ok(Self {
            source_start: usize_atom(&fields[0], "source grid start")?,
            major_count: usize_atom(&fields[1], "source grid major count")?,
            major_source_stride: usize_atom(&fields[2], "source grid major stride")?,
            minor_count: usize_atom(&fields[3], "source grid minor count")?,
            minor_source_stride: usize_atom(&fields[4], "source grid minor stride")?,
            run_count: usize_atom(&fields[5], "source grid run count")?,
            retained: RetainedBlock::decode(&fields[6])?,
            mode,
            slot_start: usize_atom(&fields[8], "source grid slot start")?,
            major_slot_stride: usize_atom(&fields[9], "source grid major slot stride")?,
            minor_slot_stride: usize_atom(&fields[10], "source grid minor slot stride")?,
        })
    }

    fn form(&self, logical_width: usize, source: usize) -> Result<Option<Form>, PackageError> {
        if source < self.source_start || self.major_source_stride == 0 || self.minor_source_stride == 0 {
            return Ok(None);
        }
        let delta = source - self.source_start;
        let major = delta / self.major_source_stride;
        let major_offset = delta % self.major_source_stride;
        if major >= self.major_count {
            return Ok(None);
        }
        let minor = major_offset / self.minor_source_stride;
        let offset = major_offset % self.minor_source_stride;
        if minor >= self.minor_count || offset >= self.run_count {
            return Ok(None);
        }
        let slot_base = checked_add(
            checked_add(
                self.slot_start,
                checked_mul(major, self.major_slot_stride, "source grid slot")?,
                "source grid slot",
            )?,
            checked_mul(minor, self.minor_slot_stride, "source grid slot")?,
            "source grid slot",
        )?;
        let form = match self.mode {
            SourceGridMode::Direct => self
                .retained
                .form(logical_width, checked_add(slot_base, offset, "source grid slot")?)?,
            SourceGridMode::External8 => self
                .retained
                .external_form(logical_width, slot_base, offset)?,
        };
        Ok(Some(form))
    }
}

#[derive(Clone, Debug)]
struct SourceSubstitution {
    ranges: Vec<SourceRange>,
    grids: Vec<SourceGrid>,
}

impl SourceSubstitution {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 2, "matrix source substitution")?;
        Ok(Self {
            ranges: decode_list(&fields[0], SourceRange::decode)?,
            grids: decode_list(&fields[1], SourceGrid::decode)?,
        })
    }

    fn form(&self, logical_width: usize, source: usize) -> Result<Form, PackageError> {
        let mut selected = None;
        for candidate in self
            .ranges
            .iter()
            .map(|range| range.form(logical_width, source))
            .chain(
                self.grids
                    .iter()
                    .map(|grid| grid.form(logical_width, source)),
            )
        {
            if let Some(form) = candidate? {
                if selected.replace(form).is_some() {
                    return Err(PackageError::Invalid("overlapping matrix source substitution"));
                }
            }
        }
        selected.ok_or(PackageError::Invalid("missing matrix source substitution"))
    }

    fn compile_combination(
        &self,
        logical_width: usize,
        one_column: usize,
        combination: &SourceCombination,
    ) -> Result<Form, PackageError> {
        if one_column >= logical_width {
            return Err(PackageError::Invalid("matrix affine one column"));
        }
        let mut form = Form::singleton(one_column, combination.constant);
        for term in &combination.terms {
            form = form.append(
                self.form(logical_width, term.column)?
                    .scaled(term.coefficient),
            );
        }
        Ok(form)
    }
}

#[derive(Clone, Debug)]
enum IndexSchedule {
    Ranges(Vec<IndexRange>),
    Table(Vec<usize>),
}

#[derive(Clone, Copy, Debug)]
struct IndexRange {
    start: usize,
    count: usize,
}

impl IndexSchedule {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 2, "matrix index schedule")?;
        match usize_atom(&fields[0], "matrix index schedule tag")? {
            0 => {
                let ranges = array(&fields[1], "matrix index ranges")?;
                let mut decoded = Vec::with_capacity(ranges.len());
                for range in ranges {
                    let fields = exact_array(range, 2, "matrix index range")?;
                    decoded.push(IndexRange {
                        start: usize_atom(&fields[0], "matrix index range start")?,
                        count: usize_atom(&fields[1], "matrix index range count")?,
                    });
                }
                Ok(Self::Ranges(decoded))
            }
            1 => Ok(Self::Table(decode_usize_list(&fields[1], "matrix index table")?)),
            _ => Err(PackageError::Invalid("matrix index schedule tag")),
        }
    }

    fn count(&self) -> Result<usize, PackageError> {
        match self {
            Self::Ranges(ranges) => ranges.iter().try_fold(0usize, |sum, range| {
                sum.checked_add(range.count)
                    .ok_or(PackageError::Invalid("matrix index count"))
            }),
            Self::Table(indices) => Ok(indices.len()),
        }
    }

    fn validate(&self, limit: usize) -> Result<(), PackageError> {
        match self {
            Self::Ranges(ranges) => {
                let mut minimum = 0;
                for range in ranges {
                    let end = checked_add(range.start, range.count, "matrix index range")?;
                    if range.count == 0 || range.start < minimum || end > limit {
                        return Err(PackageError::Invalid("matrix index range"));
                    }
                    minimum = end;
                }
            }
            Self::Table(indices) => {
                if indices.iter().any(|index| *index >= limit) {
                    return Err(PackageError::Invalid("matrix index table"));
                }
            }
        }
        Ok(())
    }

    fn index(&self, mut ordinal: usize) -> Option<usize> {
        match self {
            Self::Ranges(ranges) => {
                for range in ranges {
                    if ordinal < range.count {
                        return range.start.checked_add(ordinal);
                    }
                    ordinal -= range.count;
                }
                None
            }
            Self::Table(indices) => indices.get(ordinal).copied(),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct SourceCombination {
    pub(super) constant: Goldilocks,
    pub(super) terms: Vec<Entry>,
}

impl SourceCombination {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 2, "affine source form")?;
        Ok(Self {
            constant: field_atom(&fields[0], "affine source constant")?,
            terms: decode_list(&fields[1], |value| {
                let term = exact_array(value, 2, "affine source term")?;
                Ok(Entry {
                    column: usize_atom(&term[0], "affine source column")?,
                    coefficient: field_atom(&term[1], "affine source coefficient")?,
                })
            })?,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct SourceRow {
    pub(super) a: SourceCombination,
    pub(super) b: SourceCombination,
    pub(super) c: SourceCombination,
}

#[derive(Clone, Copy, Debug)]
struct SourceProjectionRange {
    package_start: usize,
    source_start: usize,
    count: usize,
}

impl SourceProjectionRange {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 3, "matrix source projection range")?;
        Ok(Self {
            package_start: usize_atom(&fields[0], "matrix source projection package start")?,
            source_start: usize_atom(&fields[1], "matrix source projection source start")?,
            count: usize_atom(&fields[2], "matrix source projection count")?,
        })
    }

    fn column(&self, column: usize) -> Result<Option<usize>, PackageError> {
        if column < self.package_start {
            return Ok(None);
        }
        let offset = column - self.package_start;
        if offset >= self.count {
            return Ok(None);
        }
        Ok(Some(checked_add(
            self.source_start,
            offset,
            "matrix source projection column",
        )?))
    }
}

#[derive(Clone, Debug)]
enum SourceProjection {
    Identity,
    Mapped(Vec<SourceProjectionRange>),
}

impl SourceProjection {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = array(value, "matrix source projection")?;
        match fields {
            [tag] if usize_atom(tag, "matrix source projection tag")? == 0 => Ok(Self::Identity),
            [tag, ranges] if usize_atom(tag, "matrix source projection tag")? == 1 => {
                Ok(Self::Mapped(decode_list(ranges, SourceProjectionRange::decode)?))
            }
            _ => Err(PackageError::Invalid("matrix source projection")),
        }
    }

    fn column(&self, column: usize) -> Result<usize, PackageError> {
        match self {
            Self::Identity => Ok(column),
            Self::Mapped(ranges) => {
                let mut selected = None;
                for range in ranges {
                    if let Some(source) = range.column(column)? {
                        if selected.replace(source).is_some() {
                            return Err(PackageError::Invalid("missing or overlapping matrix source projection"));
                        }
                    }
                }
                selected.ok_or(PackageError::Invalid("missing or overlapping matrix source projection"))
            }
        }
    }

    fn combination(&self, combination: &SourceCombination) -> Result<SourceCombination, PackageError> {
        let terms = combination
            .terms
            .iter()
            .map(|term| {
                Ok(Entry {
                    column: self.column(term.column)?,
                    coefficient: term.coefficient,
                })
            })
            .collect::<Result<_, PackageError>>()?;
        Ok(SourceCombination {
            constant: combination.constant,
            terms,
        })
    }

    fn row(&self, row: &SourceRow) -> Result<SourceRow, PackageError> {
        Ok(SourceRow {
            a: self.combination(&row.a)?,
            b: self.combination(&row.b)?,
            c: self.combination(&row.c)?,
        })
    }
}

#[derive(Clone, Debug)]
struct OrdinaryBlock {
    rows: IndexSchedule,
    one_column: usize,
    substitution: SourceSubstitution,
    projection: SourceProjection,
}

impl OrdinaryBlock {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 4, "ordinary matrix block")?;
        Ok(Self {
            rows: IndexSchedule::decode(&fields[0])?,
            one_column: usize_atom(&fields[1], "ordinary one column")?,
            substitution: SourceSubstitution::decode(&fields[2])?,
            projection: SourceProjection::decode(&fields[3])?,
        })
    }

    fn row_count(&self) -> Result<usize, PackageError> {
        self.rows.count()
    }

    fn validate(&self, source_limit: usize) -> Result<(), PackageError> {
        self.rows.validate(source_limit)
    }

    fn row(
        &self,
        logical_width: usize,
        ordinal: usize,
        source_row: &impl Fn(usize) -> Result<SourceRow, PackageError>,
    ) -> Result<RowForms, PackageError> {
        let source_index = self
            .rows
            .index(ordinal)
            .ok_or(PackageError::Invalid("ordinary row ordinal"))?;
        let source = self.projection.row(&source_row(source_index)?)?;
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Goldilocks::ONE);
        row[2] = self
            .substitution
            .compile_combination(logical_width, self.one_column, &source.a)?;
        row[3] = self
            .substitution
            .compile_combination(logical_width, self.one_column, &source.b)?;
        row[4] = self
            .substitution
            .compile_combination(logical_width, self.one_column, &source.c)?;
        Ok(row)
    }
}

#[derive(Clone, Debug)]
struct PinBlock {
    one_column: usize,
    values: Vec<Form>,
}

impl PinBlock {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 2, "pin matrix block")?;
        Ok(Self {
            one_column: usize_atom(&fields[0], "pin one column")?,
            values: decode_list(&fields[1], decode_form)?,
        })
    }

    fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms, PackageError> {
        if self.one_column >= logical_width {
            return Err(PackageError::Invalid("pin one column"));
        }
        let value = self
            .values
            .get(ordinal)
            .ok_or(PackageError::Invalid("pin row ordinal"))?
            .clone();
        validate_form(&value, logical_width)?;
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Goldilocks::ONE);
        row[4] = value;
        Ok(row)
    }
}

#[derive(Clone, Copy, Debug)]
struct MultiplicationShape {
    major_count: usize,
    middle_count: usize,
    minor_count: usize,
}

impl MultiplicationShape {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 3, "multiplication shape")?;
        Ok(Self {
            major_count: usize_atom(&fields[0], "multiplication major count")?,
            middle_count: usize_atom(&fields[1], "multiplication middle count")?,
            minor_count: usize_atom(&fields[2], "multiplication minor count")?,
        })
    }

    fn row_count(&self) -> Result<usize, PackageError> {
        checked_mul(
            self.major_count,
            checked_mul(self.middle_count, self.minor_count, "multiplication row count")?,
            "multiplication row count",
        )
    }

    fn coordinate(&self, ordinal: usize) -> Result<Coordinate, PackageError> {
        let inner = checked_mul(self.middle_count, self.minor_count, "multiplication coordinate")?;
        if ordinal >= self.row_count()? || inner == 0 || self.minor_count == 0 {
            return Err(PackageError::Invalid("multiplication row ordinal"));
        }
        let major = ordinal / inner;
        let remainder = ordinal % inner;
        Ok(Coordinate {
            major,
            middle: remainder / self.minor_count,
            minor: remainder % self.minor_count,
        })
    }
}

#[derive(Clone, Debug)]
struct MultiplicationBlock {
    shape: MultiplicationShape,
    one_column: usize,
    left: AffineProgram,
    right: AffineProgram,
    output: AffineProgram,
}

impl MultiplicationBlock {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 5, "multiplication matrix block")?;
        Ok(Self {
            shape: MultiplicationShape::decode(&fields[0])?,
            one_column: usize_atom(&fields[1], "multiplication one column")?,
            left: AffineProgram::decode(&fields[2])?,
            right: AffineProgram::decode(&fields[3])?,
            output: AffineProgram::decode(&fields[4])?,
        })
    }

    fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms, PackageError> {
        if self.one_column >= logical_width {
            return Err(PackageError::Invalid("multiplication one column"));
        }
        let coordinate = self.shape.coordinate(ordinal)?;
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Goldilocks::ONE);
        row[2] = self.left.form(logical_width, self.one_column, coordinate)?;
        row[3] = self
            .right
            .form(logical_width, self.one_column, coordinate)?;
        row[4] = self
            .output
            .form(logical_width, self.one_column, coordinate)?;
        Ok(row)
    }
}

#[derive(Clone, Debug)]
enum Block {
    Ordinary(OrdinaryBlock),
    Multiplication(MultiplicationBlock),
    Phi81(phi81::Block),
    Pin(PinBlock),
    Poseidon(poseidon::Block),
}

impl Block {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 2, "production matrix block")?;
        match usize_atom(&fields[0], "production matrix block tag")? {
            0 => Ok(Self::Ordinary(OrdinaryBlock::decode(&fields[1])?)),
            1 => Ok(Self::Pin(PinBlock::decode(&fields[1])?)),
            2 => Ok(Self::Poseidon(poseidon::Block::decode(&fields[1])?)),
            3 => Ok(Self::Phi81(phi81::Block::decode(&fields[1])?)),
            4 => Ok(Self::Multiplication(MultiplicationBlock::decode(&fields[1])?)),
            _ => Err(PackageError::Invalid("production matrix block tag")),
        }
    }

    fn row_count(&self) -> Result<usize, PackageError> {
        match self {
            Self::Ordinary(block) => block.row_count(),
            Self::Multiplication(block) => block.shape.row_count(),
            Self::Phi81(block) => block.row_count(),
            Self::Pin(block) => Ok(block.values.len()),
            Self::Poseidon(block) => block.row_count(),
        }
    }

    fn validate(&self, source_limit: usize) -> Result<(), PackageError> {
        if let Self::Ordinary(block) = self {
            block.validate(source_limit)?;
        }
        Ok(())
    }

    fn row(
        &self,
        logical_width: usize,
        ordinal: usize,
        source_row: &impl Fn(usize) -> Result<SourceRow, PackageError>,
    ) -> Result<RowForms, PackageError> {
        match self {
            Self::Ordinary(block) => block.row(logical_width, ordinal, source_row),
            Self::Multiplication(block) => block.row(logical_width, ordinal),
            Self::Phi81(block) => block.row(logical_width, ordinal),
            Self::Pin(block) => block.row(logical_width, ordinal),
            Self::Poseidon(block) => block.row(logical_width, ordinal),
        }
    }

    fn visit_rows(
        &self,
        logical_width: usize,
        start: usize,
        end: usize,
        source_row: &impl Fn(usize) -> Result<SourceRow, PackageError>,
        mut visit: impl FnMut(RowForms) -> Result<(), PackageError>,
    ) -> Result<(), PackageError> {
        match self {
            Self::Poseidon(block) => return block.visit_rows(logical_width, start, end, visit),
            Self::Phi81(block) => return block.visit_rows(logical_width, start, end, visit),
            _ => {}
        }
        if start > end || end > self.row_count()? {
            return Err(PackageError::Invalid("matrix block row range"));
        }
        for ordinal in start..end {
            visit(self.row(logical_width, ordinal, source_row)?)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct MatrixProgram {
    blocks: Vec<Block>,
}

impl MatrixProgram {
    pub(super) fn decode(value: &Value) -> Result<Self, PackageError> {
        Ok(Self {
            blocks: decode_list(value, Block::decode)?,
        })
    }

    pub(super) fn row_count(&self) -> Result<usize, PackageError> {
        self.blocks.iter().try_fold(0usize, |sum, block| {
            sum.checked_add(block.row_count()?)
                .ok_or(PackageError::Invalid("matrix program row count"))
        })
    }

    pub(super) fn validate(&self, source_limit: usize) -> Result<(), PackageError> {
        for block in &self.blocks {
            block.validate(source_limit)?;
        }
        Ok(())
    }

    pub(super) fn visit_rows(
        &self,
        logical_width: usize,
        start: usize,
        end: usize,
        source_row: &impl Fn(usize) -> Result<SourceRow, PackageError>,
        mut visit: impl FnMut(usize, RowForms) -> Result<(), PackageError>,
    ) -> Result<(), PackageError> {
        let row_count = self.row_count()?;
        if start > end || end > row_count {
            return Err(PackageError::Invalid("matrix program row range"));
        }

        let mut block_start = 0usize;
        let mut next = start;
        for block in &self.blocks {
            let block_end = checked_add(block_start, block.row_count()?, "matrix program block end")?;
            if start < block_end && block_start < end {
                let local_start = start.max(block_start) - block_start;
                let local_end = end.min(block_end) - block_start;
                let expected_start = checked_add(block_start, local_start, "matrix program visit start")?;
                let expected_end = checked_add(block_start, local_end, "matrix program visit end")?;
                if next != expected_start {
                    return Err(PackageError::Invalid("non-contiguous matrix program visit"));
                }
                block.visit_rows(logical_width, local_start, local_end, source_row, |forms| {
                    if next >= expected_end {
                        return Err(PackageError::Invalid("extra matrix program row"));
                    }
                    let ordinal = next;
                    visit(ordinal, forms)?;
                    next = next
                        .checked_add(1)
                        .ok_or(PackageError::Invalid("matrix program visit ordinal"))?;
                    Ok(())
                })?;
                if next != expected_end {
                    return Err(PackageError::Invalid("missing matrix program row"));
                }
            }
            block_start = block_end;
        }
        if block_start != row_count || next != end {
            return Err(PackageError::Invalid("incomplete matrix program visit"));
        }
        Ok(())
    }

    pub(super) fn validate_all_rows(
        &self,
        logical_width: usize,
        source_row: &(impl Fn(usize) -> Result<SourceRow, PackageError> + Sync),
    ) -> Result<[u64; MEANINGFUL_PORTS], PackageError> {
        const ROWS_PER_CHUNK: usize = 4_096;

        let mut chunks = Vec::new();
        for block in &self.blocks {
            let count = block.row_count()?;
            let mut start = 0;
            while start < count {
                let end = start.saturating_add(ROWS_PER_CHUNK).min(count);
                chunks.push((block, start, end));
                start = end;
            }
        }

        chunks
            .into_par_iter()
            .try_fold(
                || [0u64; MEANINGFUL_PORTS],
                |mut counts, (block, start, end)| {
                    block.visit_rows(logical_width, start, end, source_row, |forms| {
                        for (matrix, form) in forms.iter().enumerate() {
                            let mut previous = None;
                            for entry in form.entries() {
                                if entry.column >= logical_width
                                    || entry.coefficient == Goldilocks::ZERO
                                    || previous.is_some_and(|previous| previous >= entry.column)
                                {
                                    return Err(PackageError::Invalid("non-canonical logical matrix row"));
                                }
                                previous = Some(entry.column);
                            }
                            counts[matrix] = counts[matrix]
                                .checked_add(
                                    u64::try_from(form.entries().len())
                                        .map_err(|_| PackageError::Invalid("logical matrix nonzero count"))?,
                                )
                                .ok_or(PackageError::Invalid("logical matrix nonzero count"))?;
                        }
                        Ok(())
                    })?;
                    Ok(counts)
                },
            )
            .try_reduce(
                || [0u64; MEANINGFUL_PORTS],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left = left
                            .checked_add(right)
                            .ok_or(PackageError::Invalid("logical matrix nonzero count"))?;
                    }
                    Ok(left)
                },
            )
    }

    #[cfg(test)]
    pub(super) fn row(
        &self,
        logical_width: usize,
        mut ordinal: usize,
        source_row: &impl Fn(usize) -> Result<SourceRow, PackageError>,
    ) -> Result<RowForms, PackageError> {
        for block in &self.blocks {
            let count = block.row_count()?;
            if ordinal < count {
                return block.row(logical_width, ordinal, source_row);
            }
            ordinal -= count;
        }
        Err(PackageError::Invalid("matrix program row ordinal"))
    }
}

pub(super) fn external_layer(state: &[Form]) -> Result<Vec<Form>, PackageError> {
    if state.len() != 8 {
        return Err(PackageError::Invalid("matrix Poseidon2 state"));
    }
    let mut blocks = Vec::with_capacity(8);
    for base in [0usize, 4] {
        for lane in 0..4 {
            let coefficients = match lane {
                0 => [2, 3, 1, 1],
                1 => [1, 2, 3, 1],
                2 => [1, 1, 2, 3],
                _ => [3, 1, 1, 2],
            };
            let mut form = Form::default();
            for (offset, coefficient) in coefficients.into_iter().enumerate() {
                form = form.append(
                    state[base + offset]
                        .clone()
                        .scaled(Goldilocks::from_u64(coefficient)),
                );
            }
            blocks.push(form);
        }
    }
    let mut output = Vec::with_capacity(8);
    for lane in 0..8 {
        output.push(
            blocks[lane]
                .clone()
                .append(blocks[lane % 4].clone())
                .append(blocks[lane % 4 + 4].clone()),
        );
    }
    Ok(output)
}

pub(super) fn validate_form(form: &Form, logical_width: usize) -> Result<(), PackageError> {
    if form
        .entries()
        .iter()
        .any(|entry| entry.column >= logical_width)
    {
        return Err(PackageError::Invalid("matrix sparse column"));
    }
    Ok(())
}

pub(super) fn field(value: u64, location: &'static str) -> Result<Goldilocks, PackageError> {
    if value >= GOLDILOCKS_MODULUS {
        return Err(PackageError::NonCanonicalField { location, value });
    }
    Ok(Goldilocks::from_u64(value))
}

pub(super) fn array<'a>(value: &'a Value, location: &'static str) -> Result<&'a [Value], PackageError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or(PackageError::Invalid(location))
}

pub(super) fn exact_array<'a>(
    value: &'a Value,
    length: usize,
    location: &'static str,
) -> Result<&'a [Value], PackageError> {
    let values = array(value, location)?;
    if values.len() != length {
        return Err(PackageError::Invalid(location));
    }
    Ok(values)
}

pub(super) fn usize_atom(value: &Value, location: &'static str) -> Result<usize, PackageError> {
    let value = value.as_u64().ok_or(PackageError::Invalid(location))?;
    usize::try_from(value).map_err(|_| PackageError::Invalid(location))
}

pub(super) fn field_atom(value: &Value, location: &'static str) -> Result<Goldilocks, PackageError> {
    field(value.as_u64().ok_or(PackageError::Invalid(location))?, location)
}

pub(super) fn decode_list<T>(
    value: &Value,
    mut decode: impl FnMut(&Value) -> Result<T, PackageError>,
) -> Result<Vec<T>, PackageError> {
    array(value, "matrix list")?
        .iter()
        .map(&mut decode)
        .collect()
}

fn decode_usize_list(value: &Value, location: &'static str) -> Result<Vec<usize>, PackageError> {
    array(value, location)?
        .iter()
        .map(|item| usize_atom(item, location))
        .collect()
}

pub(super) fn checked_add(left: usize, right: usize, location: &'static str) -> Result<usize, PackageError> {
    left.checked_add(right)
        .ok_or(PackageError::Invalid(location))
}

pub(super) fn checked_mul(left: usize, right: usize, location: &'static str) -> Result<usize, PackageError> {
    left.checked_mul(right)
        .ok_or(PackageError::Invalid(location))
}
