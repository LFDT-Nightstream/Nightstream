//! Independent decoder and dispatcher for every compact matrix block opcode.

use serde_json::Value;

use super::affine::{AffineProgram, Coordinate};
use super::source::{SourceCombination, SourcePackage, SourceRow};
use super::{
    array, checked_add, checked_mul, decode_form, decode_list, empty_row, exact_array, word, Entry, Field, Form,
    Result, RetainedBlock, RowForms,
};

#[derive(Clone, Debug)]
struct SourceRange {
    source_start: usize,
    source_count: usize,
    retained: RetainedBlock,
    slot_start: usize,
}

impl SourceRange {
    fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 4, "matrix source range")?;
        let range = Self {
            source_start: word(&fields[0], "source range start")?,
            source_count: word(&fields[1], "source range count")?,
            retained: RetainedBlock::decode(&fields[2])?,
            slot_start: word(&fields[3], "source range slot")?,
        };
        range.retained.validate(logical_width)?;
        if range.source_count == 0
            || checked_add(range.slot_start, range.source_count, "source range slots")? > range.retained.slot_count
        {
            return Err("invalid source range slot mapping".into());
        }
        Ok(range)
    }

    fn form(&self, logical_width: usize, source: usize) -> Result<Option<Form>> {
        let Some(offset) = source.checked_sub(self.source_start) else {
            return Ok(None);
        };
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
enum GridMode {
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
    mode: GridMode,
    slot_start: usize,
    major_slot_stride: usize,
    minor_slot_stride: usize,
}

impl SourceGrid {
    fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 11, "matrix source grid")?;
        let mode = match word(&fields[7], "source grid mode")? {
            0 => GridMode::Direct,
            1 => GridMode::External8,
            _ => return Err("unknown source grid mode".into()),
        };
        let grid = Self {
            source_start: word(&fields[0], "source grid start")?,
            major_count: word(&fields[1], "source grid major count")?,
            major_source_stride: word(&fields[2], "source grid major stride")?,
            minor_count: word(&fields[3], "source grid minor count")?,
            minor_source_stride: word(&fields[4], "source grid minor stride")?,
            run_count: word(&fields[5], "source grid run count")?,
            retained: RetainedBlock::decode(&fields[6])?,
            mode,
            slot_start: word(&fields[8], "source grid slot start")?,
            major_slot_stride: word(&fields[9], "source grid major slot stride")?,
            minor_slot_stride: word(&fields[10], "source grid minor slot stride")?,
        };
        grid.retained.validate(logical_width)?;
        if grid.major_source_stride == 0 || grid.minor_source_stride == 0 {
            return Err("zero source-grid stride".into());
        }
        if grid.major_count == 0 || grid.minor_count == 0 || grid.run_count == 0 {
            return Err("zero source-grid extent".into());
        }
        let final_slot_base = checked_add(
            checked_add(
                grid.slot_start,
                checked_mul(grid.major_count - 1, grid.major_slot_stride, "source grid slot bound")?,
                "source grid slot bound",
            )?,
            checked_mul(grid.minor_count - 1, grid.minor_slot_stride, "source grid slot bound")?,
            "source grid slot bound",
        )?;
        let used_slots = match grid.mode {
            GridMode::Direct => grid.run_count,
            GridMode::External8 => 8,
        };
        if checked_add(final_slot_base, used_slots, "source grid slot bound")? > grid.retained.slot_count
            || matches!(grid.mode, GridMode::External8) && grid.run_count > 8
        {
            return Err("invalid source-grid retained mapping".into());
        }
        Ok(grid)
    }

    fn form(&self, logical_width: usize, source: usize) -> Result<Option<Form>> {
        if self.major_source_stride == 0 || self.minor_source_stride == 0 {
            return Err("zero source-grid stride".into());
        }
        let Some(delta) = source.checked_sub(self.source_start) else {
            return Ok(None);
        };
        let major = delta / self.major_source_stride;
        let within_major = delta % self.major_source_stride;
        let minor = within_major / self.minor_source_stride;
        let run = within_major % self.minor_source_stride;
        if major >= self.major_count || minor >= self.minor_count || run >= self.run_count {
            return Ok(None);
        }
        let slot_base = checked_add(
            checked_add(
                self.slot_start,
                checked_mul(major, self.major_slot_stride, "source grid major slot")?,
                "source grid slot",
            )?,
            checked_mul(minor, self.minor_slot_stride, "source grid minor slot")?,
            "source grid slot",
        )?;
        let form = match self.mode {
            GridMode::Direct => self
                .retained
                .form(logical_width, checked_add(slot_base, run, "source grid direct slot")?)?,
            GridMode::External8 => self.retained.external_form(logical_width, slot_base, run)?,
        };
        Ok(Some(form))
    }
}

#[derive(Clone, Debug)]
pub(super) struct SourceSubstitution {
    ranges: Vec<SourceRange>,
    grids: Vec<SourceGrid>,
}

impl SourceSubstitution {
    pub(super) fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 2, "matrix source substitution")?;
        Ok(Self {
            ranges: decode_list(
                &fields[0],
                |range| SourceRange::decode(range, logical_width),
                "source ranges",
            )?,
            grids: decode_list(
                &fields[1],
                |grid| SourceGrid::decode(grid, logical_width),
                "source grids",
            )?,
        })
    }

    fn form(&self, logical_width: usize, source: usize) -> Result<Form> {
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
                    return Err("overlapping matrix source substitution".into());
                }
            }
        }
        selected.ok_or_else(|| "missing matrix source substitution".into())
    }

    pub(super) fn compile(
        &self,
        logical_width: usize,
        one_column: usize,
        combination: &SourceCombination,
    ) -> Result<Form> {
        if one_column >= logical_width {
            return Err("affine constant column is out of range".into());
        }
        let mut form = Form::singleton(one_column, combination.constant);
        for entry in &combination.terms {
            form = form.append(
                self.form(logical_width, entry.column)?
                    .scaled(entry.coefficient),
            );
        }
        Ok(form)
    }
}

#[derive(Clone, Copy, Debug)]
struct ProjectionRange {
    package_start: usize,
    source_start: usize,
    count: usize,
}

impl ProjectionRange {
    fn decode(value: &Value, physical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 3, "matrix source projection range")?;
        let range = Self {
            package_start: word(&fields[0], "projection package start")?,
            source_start: word(&fields[1], "projection source start")?,
            count: word(&fields[2], "projection count")?,
        };
        if range.count == 0
            || checked_add(range.package_start, range.count, "projection package range")? > physical_width
            || checked_add(range.source_start, range.count, "projection source range").is_err()
        {
            return Err("invalid matrix source projection range".into());
        }
        Ok(range)
    }

    fn column(self, column: usize) -> Result<Option<usize>> {
        let Some(offset) = column.checked_sub(self.package_start) else {
            return Ok(None);
        };
        if offset >= self.count {
            return Ok(None);
        }
        Ok(Some(checked_add(self.source_start, offset, "projected source column")?))
    }
}

#[derive(Clone, Debug)]
enum SourceProjection {
    Identity,
    Mapped(Vec<ProjectionRange>),
}

impl SourceProjection {
    fn decode(value: &Value, physical_width: usize) -> Result<Self> {
        let fields = array(value, "matrix source projection")?;
        match fields {
            [tag] if word(tag, "projection tag")? == 0 => Ok(Self::Identity),
            [tag, ranges] if word(tag, "projection tag")? == 1 => {
                let ranges = decode_list(
                    ranges,
                    |range| ProjectionRange::decode(range, physical_width),
                    "projection ranges",
                )?;
                for left in 0..ranges.len() {
                    for right in left + 1..ranges.len() {
                        let left_end = ranges[left].package_start + ranges[left].count;
                        let right_end = ranges[right].package_start + ranges[right].count;
                        if ranges[left].package_start < right_end && ranges[right].package_start < left_end {
                            return Err("overlapping matrix source projection".into());
                        }
                    }
                }
                Ok(Self::Mapped(ranges))
            }
            _ => Err("unknown matrix source projection".into()),
        }
    }

    fn column(&self, column: usize) -> Result<usize> {
        match self {
            Self::Identity => Ok(column),
            Self::Mapped(ranges) => {
                let mut selected = None;
                for range in ranges {
                    if let Some(column) = range.column(column)? {
                        if selected.replace(column).is_some() {
                            return Err("overlapping projected source column".into());
                        }
                    }
                }
                selected.ok_or_else(|| "missing projected source column".into())
            }
        }
    }

    fn combination(&self, input: &SourceCombination) -> Result<SourceCombination> {
        Ok(SourceCombination {
            constant: input.constant,
            terms: input
                .terms
                .iter()
                .map(|entry| {
                    Ok(Entry {
                        column: self.column(entry.column)?,
                        coefficient: entry.coefficient,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn row(&self, input: &SourceRow) -> Result<SourceRow> {
        Ok(SourceRow {
            a: self.combination(&input.a)?,
            b: self.combination(&input.b)?,
            c: self.combination(&input.c)?,
        })
    }
}

#[derive(Clone, Debug)]
enum IndexSchedule {
    Ranges(Vec<(usize, usize)>),
    Table(Vec<usize>),
}

impl IndexSchedule {
    fn decode(value: &Value, source_limit: usize) -> Result<Self> {
        let fields = exact_array(value, 2, "matrix index schedule")?;
        let result = match word(&fields[0], "matrix index schedule tag")? {
            0 => {
                let ranges = array(&fields[1], "matrix index ranges")?
                    .iter()
                    .map(|range| {
                        let fields = exact_array(range, 2, "matrix index range")?;
                        Ok((
                            word(&fields[0], "matrix index range start")?,
                            word(&fields[1], "matrix index range count")?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut minimum = 0usize;
                for &(start, count) in &ranges {
                    let end = checked_add(start, count, "matrix index range")?;
                    if count == 0 || start < minimum || end > source_limit {
                        return Err("invalid or unordered matrix index range".into());
                    }
                    minimum = end;
                }
                Self::Ranges(ranges)
            }
            1 => {
                let table = array(&fields[1], "matrix index table")?
                    .iter()
                    .map(|index| word(index, "matrix source row"))
                    .collect::<Result<Vec<_>>>()?;
                if table.iter().any(|index| *index >= source_limit) {
                    return Err("matrix source row is out of range".into());
                }
                Self::Table(table)
            }
            _ => return Err("unknown matrix index schedule".into()),
        };
        Ok(result)
    }

    fn count(&self) -> Result<usize> {
        match self {
            Self::Ranges(ranges) => ranges.iter().try_fold(0usize, |count, (_, length)| {
                checked_add(count, *length, "matrix index count")
            }),
            Self::Table(indices) => Ok(indices.len()),
        }
    }

    fn index(&self, mut ordinal: usize) -> Option<usize> {
        match self {
            Self::Ranges(ranges) => {
                for &(start, count) in ranges {
                    if ordinal < count {
                        return start.checked_add(ordinal);
                    }
                    ordinal -= count;
                }
                None
            }
            Self::Table(indices) => indices.get(ordinal).copied(),
        }
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
    fn decode(value: &Value, source_limit: usize, physical_width: usize, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 4, "ordinary matrix block")?;
        let block = Self {
            rows: IndexSchedule::decode(&fields[0], source_limit)?,
            one_column: word(&fields[1], "ordinary one column")?,
            substitution: SourceSubstitution::decode(&fields[2], logical_width)?,
            projection: SourceProjection::decode(&fields[3], physical_width)?,
        };
        require_one_column(block.one_column, logical_width)?;
        Ok(block)
    }

    fn row_count(&self) -> Result<usize> {
        self.rows.count()
    }

    fn row(&self, logical_width: usize, ordinal: usize, sources: &SourcePackage) -> Result<RowForms> {
        let source_index = self
            .rows
            .index(ordinal)
            .ok_or_else(|| "ordinary row ordinal is out of range".to_string())?;
        let source = self.projection.row(&sources.row(source_index)?)?;
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Field::ONE);
        row[2] = self
            .substitution
            .compile(logical_width, self.one_column, &source.a)?;
        row[3] = self
            .substitution
            .compile(logical_width, self.one_column, &source.b)?;
        row[4] = self
            .substitution
            .compile(logical_width, self.one_column, &source.c)?;
        Ok(row)
    }
}

#[derive(Clone, Debug)]
struct PinBlock {
    one_column: usize,
    values: Vec<Form>,
}

impl PinBlock {
    fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 2, "pin matrix block")?;
        let block = Self {
            one_column: word(&fields[0], "pin one column")?,
            values: decode_list(&fields[1], |value| decode_form(value, logical_width), "pin values")?,
        };
        require_one_column(block.one_column, logical_width)?;
        for form in &block.values {
            form.validate(logical_width)?;
        }
        Ok(block)
    }

    fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms> {
        if self.one_column >= logical_width {
            return Err("pin one column is out of range".into());
        }
        let value = self
            .values
            .get(ordinal)
            .ok_or_else(|| "pin row ordinal is out of range".to_string())?
            .clone();
        value.validate(logical_width)?;
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Field::ONE);
        row[4] = value;
        Ok(row)
    }
}

#[derive(Clone, Copy, Debug)]
struct MultiplicationShape {
    major: usize,
    middle: usize,
    minor: usize,
}

impl MultiplicationShape {
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 3, "multiplication shape")?;
        Ok(Self {
            major: word(&fields[0], "multiplication major count")?,
            middle: word(&fields[1], "multiplication middle count")?,
            minor: word(&fields[2], "multiplication minor count")?,
        })
    }

    fn row_count(self) -> Result<usize> {
        checked_mul(
            self.major,
            checked_mul(self.middle, self.minor, "multiplication inner rows")?,
            "multiplication rows",
        )
    }

    fn coordinate(self, ordinal: usize) -> Result<Coordinate> {
        let inner = checked_mul(self.middle, self.minor, "multiplication coordinate")?;
        if inner == 0 || self.minor == 0 || ordinal >= self.row_count()? {
            return Err("multiplication row ordinal is out of range".into());
        }
        Ok(Coordinate {
            major: ordinal / inner,
            middle: (ordinal % inner) / self.minor,
            minor: ordinal % self.minor,
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
    fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 5, "multiplication matrix block")?;
        let block = Self {
            shape: MultiplicationShape::decode(&fields[0])?,
            one_column: word(&fields[1], "multiplication one column")?,
            left: AffineProgram::decode(&fields[2], logical_width)?,
            right: AffineProgram::decode(&fields[3], logical_width)?,
            output: AffineProgram::decode(&fields[4], logical_width)?,
        };
        require_one_column(block.one_column, logical_width)?;
        Ok(block)
    }

    fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms> {
        if self.one_column >= logical_width {
            return Err("multiplication one column is out of range".into());
        }
        let coordinate = self.shape.coordinate(ordinal)?;
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Field::ONE);
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
    Pin(PinBlock),
    Poseidon(super::poseidon::Block),
    Phi81(super::phi81::Block),
    Multiplication(MultiplicationBlock),
}

impl Block {
    fn decode(value: &Value, source_limit: usize, physical_width: usize, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 2, "matrix block")?;
        match word(&fields[0], "matrix block opcode")? {
            0 => Ok(Self::Ordinary(OrdinaryBlock::decode(
                &fields[1],
                source_limit,
                physical_width,
                logical_width,
            )?)),
            1 => Ok(Self::Pin(PinBlock::decode(&fields[1], logical_width)?)),
            2 => Ok(Self::Poseidon(super::poseidon::Block::decode(
                &fields[1],
                logical_width,
            )?)),
            3 => Ok(Self::Phi81(super::phi81::Block::decode(&fields[1], logical_width)?)),
            4 => Ok(Self::Multiplication(MultiplicationBlock::decode(
                &fields[1],
                logical_width,
            )?)),
            _ => Err("unknown matrix block opcode".into()),
        }
    }

    fn row_count(&self) -> Result<usize> {
        match self {
            Self::Ordinary(block) => block.row_count(),
            Self::Pin(block) => Ok(block.values.len()),
            Self::Poseidon(block) => block.row_count(),
            Self::Phi81(block) => block.row_count(),
            Self::Multiplication(block) => block.shape.row_count(),
        }
    }

    #[allow(dead_code)]
    fn opcode(&self) -> usize {
        match self {
            Self::Ordinary(_) => 0,
            Self::Pin(_) => 1,
            Self::Poseidon(_) => 2,
            Self::Phi81(_) => 3,
            Self::Multiplication(_) => 4,
        }
    }

    fn row(&self, logical_width: usize, ordinal: usize, sources: &SourcePackage) -> Result<RowForms> {
        match self {
            Self::Ordinary(block) => block.row(logical_width, ordinal, sources),
            Self::Pin(block) => block.row(logical_width, ordinal),
            Self::Poseidon(block) => block.row(logical_width, ordinal),
            Self::Phi81(block) => block.row(logical_width, ordinal),
            Self::Multiplication(block) => block.row(logical_width, ordinal),
        }
    }

    fn visit_rows(
        &self,
        logical_width: usize,
        start: usize,
        end: usize,
        sources: &SourcePackage,
        mut visit: impl FnMut(usize, RowForms) -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Poseidon(block) => block.visit_rows(logical_width, start, end, visit),
            Self::Phi81(block) => block.visit_rows(logical_width, start, end, visit),
            _ => {
                if start > end || end > self.row_count()? {
                    return Err("matrix block row range is out of bounds".into());
                }
                for ordinal in start..end {
                    visit(ordinal, self.row(logical_width, ordinal, sources)?)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug)]
struct ScheduledBlock {
    start: usize,
    end: usize,
    block: Block,
}

#[derive(Clone, Debug)]
pub struct MatrixProgram {
    blocks: Vec<ScheduledBlock>,
    logical_width: usize,
    row_count: usize,
}

impl MatrixProgram {
    pub fn decode(value: &Value, sources: &SourcePackage, logical_width: usize, expected_rows: usize) -> Result<Self> {
        let mut cursor = 0usize;
        let blocks = array(value, "matrix program")?
            .iter()
            .map(|value| {
                let block = Block::decode(
                    value,
                    sources.layout.row_count,
                    sources.layout.total_columns,
                    logical_width,
                )?;
                let count = block.row_count()?;
                if count == 0 {
                    return Err("zero-length matrix block".into());
                }
                let start = cursor;
                cursor = checked_add(cursor, count, "matrix program row count")?;
                Ok(ScheduledBlock {
                    start,
                    end: cursor,
                    block,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if cursor != expected_rows {
            return Err(format!("matrix program has {cursor} rows, expected {expected_rows}"));
        }
        Ok(Self {
            blocks,
            logical_width,
            row_count: cursor,
        })
    }

    #[allow(dead_code)]
    pub fn block_ends(&self) -> impl Iterator<Item = usize> + '_ {
        self.blocks.iter().map(|block| block.end)
    }

    #[allow(dead_code)]
    pub fn block_opcodes(&self) -> impl Iterator<Item = usize> + '_ {
        self.blocks.iter().map(|block| block.block.opcode())
    }

    pub fn row(&self, ordinal: usize, sources: &SourcePackage) -> Result<RowForms> {
        if ordinal >= self.row_count {
            return Err("logical matrix row ordinal is out of range".into());
        }
        let position = self
            .blocks
            .partition_point(|block| block.start <= ordinal)
            .checked_sub(1)
            .ok_or_else(|| "missing logical matrix block".to_string())?;
        let block = &self.blocks[position];
        if ordinal >= block.end {
            return Err("logical matrix block gap".into());
        }
        let row = block
            .block
            .row(self.logical_width, ordinal - block.start, sources)?;
        for form in &row {
            form.validate(self.logical_width)?;
        }
        Ok(row)
    }

    pub fn visit_rows(
        &self,
        start: usize,
        end: usize,
        sources: &SourcePackage,
        mut visit: impl FnMut(usize, RowForms) -> Result<()>,
    ) -> Result<()> {
        if start > end || end > self.row_count {
            return Err("logical matrix row range is out of bounds".into());
        }
        for scheduled in &self.blocks {
            if scheduled.end <= start {
                continue;
            }
            if scheduled.start >= end {
                break;
            }
            let local_start = start.saturating_sub(scheduled.start);
            let local_end = end.min(scheduled.end) - scheduled.start;
            scheduled.block.visit_rows(
                self.logical_width,
                local_start,
                local_end,
                sources,
                |local_ordinal, row| {
                    for form in &row {
                        form.validate(self.logical_width)?;
                    }
                    visit(scheduled.start + local_ordinal, row)
                },
            )?;
        }
        Ok(())
    }
}

fn require_one_column(one_column: usize, logical_width: usize) -> Result<()> {
    if one_column != 0 || one_column >= logical_width {
        return Err("matrix block does not map one to logical column zero".into());
    }
    Ok(())
}
