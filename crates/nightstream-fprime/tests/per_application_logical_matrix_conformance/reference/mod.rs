//! Independent sparse-form arithmetic and compact matrix-program decoding.
//!
//! This test-only module does not import the production matrix interpreter.

use std::sync::OnceLock;

use serde_json::Value;

pub mod affine;
#[allow(dead_code)]
pub mod assignment;
#[allow(dead_code)]
pub mod evaluation;
pub mod matrix;
#[allow(dead_code)]
pub mod mutation;
pub mod phi81;
pub mod poseidon;
pub mod poseidon_input;
#[allow(dead_code)]
pub mod relation;
pub mod source;

pub const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;
pub const MATRIX_COUNT: usize = 14;

pub type Result<T> = std::result::Result<T, String>;
pub type RowForms = [Form; MATRIX_COUNT];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field(u64);

impl Field {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    pub fn checked(value: u64, label: &str) -> Result<Self> {
        if value < GOLDILOCKS_MODULUS {
            Ok(Self(value))
        } else {
            Err(format!("noncanonical {label}: {value}"))
        }
    }

    pub fn canonical(self) -> u64 {
        self.0
    }
}

impl std::ops::Add for Field {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(((u128::from(self.0) + u128::from(rhs.0)) % u128::from(GOLDILOCKS_MODULUS)) as u64)
    }
}

impl std::ops::AddAssign for Field {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Mul for Field {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(((u128::from(self.0) * u128::from(rhs.0)) % u128::from(GOLDILOCKS_MODULUS)) as u64)
    }
}

impl std::ops::MulAssign for Field {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl std::ops::Neg for Field {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self == Self::ZERO {
            self
        } else {
            Self(GOLDILOCKS_MODULUS - self.0)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub column: usize,
    pub coefficient: Field,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Form {
    #[default]
    Empty,
    One(Entry),
    Many(Vec<Entry>),
}

impl Form {
    fn from_canonical_entries(mut entries: Vec<Entry>) -> Self {
        match entries.len() {
            0 => Self::Empty,
            1 => Self::One(entries.pop().expect("one form entry")),
            _ => Self::Many(entries),
        }
    }

    pub fn from_entries(mut entries: Vec<Entry>) -> Self {
        entries.sort_unstable_by_key(|entry| entry.column);
        let mut canonical: Vec<Entry> = Vec::with_capacity(entries.len());
        for entry in entries {
            if let Some(last) = canonical.last_mut() {
                if last.column == entry.column {
                    last.coefficient += entry.coefficient;
                    if last.coefficient == Field::ZERO {
                        canonical.pop();
                    }
                    continue;
                }
            }
            if entry.coefficient != Field::ZERO {
                canonical.push(entry);
            }
        }
        Self::from_canonical_entries(canonical)
    }

    pub fn singleton(column: usize, coefficient: Field) -> Self {
        if coefficient == Field::ZERO {
            Self::default()
        } else {
            Self::One(Entry { column, coefficient })
        }
    }

    pub fn entries(&self) -> &[Entry] {
        match self {
            Self::Empty => &[],
            Self::One(entry) => std::slice::from_ref(entry),
            Self::Many(entries) => entries,
        }
    }

    pub fn append(self, other: Self) -> Self {
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
        while let (Some(a), Some(b)) = (left.peek(), right.peek()) {
            match a.column.cmp(&b.column) {
                std::cmp::Ordering::Less => entries.push(left.next().expect("peeked left entry")),
                std::cmp::Ordering::Greater => entries.push(right.next().expect("peeked right entry")),
                std::cmp::Ordering::Equal => {
                    let a = left.next().expect("peeked left entry");
                    let b = right.next().expect("peeked right entry");
                    let coefficient = a.coefficient + b.coefficient;
                    if coefficient != Field::ZERO {
                        entries.push(Entry {
                            column: a.column,
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

    pub fn scaled(mut self, scalar: Field) -> Self {
        if scalar == Field::ZERO {
            return Self::default();
        }
        if scalar == Field::ONE {
            return self;
        }
        let entries = match &mut self {
            Self::Empty => &mut [],
            Self::One(entry) => std::slice::from_mut(entry),
            Self::Many(entries) => entries,
        };
        if scalar == -Field::ONE {
            for entry in entries {
                entry.coefficient = -entry.coefficient;
            }
        } else {
            for entry in entries {
                entry.coefficient *= scalar;
            }
        }
        self
    }

    pub fn validate(&self, logical_width: usize) -> Result<()> {
        let mut previous = None;
        for entry in self.entries() {
            if entry.column >= logical_width
                || entry.coefficient == Field::ZERO
                || previous.is_some_and(|column| column >= entry.column)
            {
                return Err("noncanonical independent sparse form".into());
            }
            previous = Some(entry.column);
        }
        Ok(())
    }

    fn into_entries(self) -> Vec<Entry> {
        match self {
            Self::Empty => Vec::new(),
            Self::One(entry) => vec![entry],
            Self::Many(entries) => entries,
        }
    }
}

pub fn empty_row() -> RowForms {
    std::array::from_fn(|_| Form::default())
}

pub fn array<'a>(value: &'a Value, label: &str) -> Result<&'a [Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label} is not an array"))
}

pub fn exact_array<'a>(value: &'a Value, length: usize, label: &str) -> Result<&'a [Value]> {
    let fields = array(value, label)?;
    if fields.len() != length {
        return Err(format!("{label} has length {}, expected {length}", fields.len()));
    }
    Ok(fields)
}

pub fn word(value: &Value, label: &str) -> Result<usize> {
    let raw = value
        .as_u64()
        .ok_or_else(|| format!("{label} is not a u64"))?;
    usize::try_from(raw).map_err(|_| format!("{label} exceeds usize"))
}

pub fn field(value: &Value, label: &str) -> Result<Field> {
    Field::checked(
        value
            .as_u64()
            .ok_or_else(|| format!("{label} is not a u64"))?,
        label,
    )
}

pub fn decode_list<T>(value: &Value, mut decode: impl FnMut(&Value) -> Result<T>, label: &str) -> Result<Vec<T>> {
    array(value, label)?.iter().map(&mut decode).collect()
}

pub fn checked_add(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| format!("{label} overflow"))
}

pub fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{label} overflow"))
}

pub fn decode_form(value: &Value, logical_width: usize) -> Result<Form> {
    let entries = array(value, "matrix sparse form")?
        .iter()
        .map(|entry| {
            let fields = exact_array(entry, 2, "matrix sparse entry")?;
            let column = word(&fields[0], "matrix sparse column")?;
            if column >= logical_width {
                return Err("matrix sparse column is out of range".into());
            }
            Ok(Entry {
                column,
                coefficient: field(&fields[1], "matrix sparse coefficient")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Form::from_entries(entries))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetainedKind {
    Bit,
    Centered,
    Field,
}

impl RetainedKind {
    fn width(self) -> usize {
        match self {
            Self::Bit | Self::Centered => 1,
            Self::Field => 41,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RetainedBlock {
    pub kind: RetainedKind,
    pub slot_count: usize,
    start: usize,
}

impl RetainedBlock {
    pub fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 3, "retained block")?;
        let kind = match word(&fields[0], "retained kind")? {
            0 => RetainedKind::Bit,
            1 => RetainedKind::Centered,
            2 => RetainedKind::Field,
            _ => return Err("unknown retained kind".into()),
        };
        Ok(Self {
            kind,
            slot_count: word(&fields[1], "retained slot count")?,
            start: word(&fields[2], "retained start")?,
        })
    }

    pub fn fits(&self, logical_width: usize) -> Result<bool> {
        Ok(checked_add(
            self.start,
            checked_mul(self.slot_count, self.kind.width(), "retained coordinate count")?,
            "retained end",
        )? <= logical_width)
    }

    pub fn validate(&self, logical_width: usize) -> Result<()> {
        if !self.fits(logical_width)? {
            return Err("retained block exceeds the logical column range".into());
        }
        Ok(())
    }

    pub fn form(&self, logical_width: usize, slot: usize) -> Result<Form> {
        if slot >= self.slot_count || !self.fits(logical_width)? {
            return Err("retained slot is out of range".into());
        }
        let width = self.kind.width();
        let first = checked_add(
            self.start,
            checked_mul(slot, width, "retained slot offset")?,
            "retained slot offset",
        )?;
        let mut entries = Vec::with_capacity(width);
        for (offset, &weight) in radix_weights()[..width].iter().enumerate() {
            entries.push(Entry {
                column: checked_add(first, offset, "retained coordinate")?,
                coefficient: weight,
            });
        }
        Ok(Form::from_canonical_entries(entries))
    }

    pub fn external_form(&self, logical_width: usize, slot_base: usize, lane: usize) -> Result<Form> {
        if lane >= 8 {
            return Err("retained external lane is out of range".into());
        }
        let state = (0..8)
            .map(|selected| self.form(logical_width, checked_add(slot_base, selected, "external slot")?))
            .collect::<Result<Vec<_>>>()?;
        Ok(external_layer(&state)?[lane].clone())
    }
}

fn radix_weights() -> &'static [Field; 41] {
    static WEIGHTS: OnceLock<[Field; 41]> = OnceLock::new();
    WEIGHTS.get_or_init(|| {
        let mut weights = [Field::ZERO; 41];
        let mut weight = Field::ONE;
        for coefficient in &mut weights {
            *coefficient = weight;
            weight *= Field(3);
        }
        weights
    })
}

pub fn external_layer(state: &[Form]) -> Result<Vec<Form>> {
    if state.len() != 8 {
        return Err("external layer requires eight forms".into());
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
                form = form.append(state[base + offset].clone().scaled(Field(coefficient)));
            }
            blocks.push(form);
        }
    }
    Ok((0..8)
        .map(|lane| {
            blocks[lane]
                .clone()
                .append(blocks[lane % 4].clone())
                .append(blocks[lane % 4 + 4].clone())
        })
        .collect())
}
