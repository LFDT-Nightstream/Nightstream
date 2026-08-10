//! Shared declarations for statically allocated WASM witness columns.
//!
//! Relation modules own their column declarations. This module only provides
//! the index assignment and metadata machinery used to compose those regions.

/// Declared intrinsic range for a witness column.
///
/// These declarations are meant to be enforced; otherwise the proof is not
/// sound. Enforcement can happen in the WASM CCS itself or through a lookup
/// argument. The selected approach may affect performance, not semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnWidth {
    /// Constrained to {0, 1}.
    Boolean,
    /// Constrained to [0, 256).
    Byte,
    /// Constrained to [0, 2^32).
    U32,
    /// No declared bound: the value is treated as a full field element.
    /// Use for columns whose intrinsic range has not been audited yet, or
    /// whose width depends on a row gate.
    Field,
}

/// Static metadata for one named witness column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmColumnSpec {
    pub index: usize,
    pub name: &'static str,
    pub role: &'static str,
    pub width: ColumnWidth,
}

/// Static metadata for one contiguous family in a subsystem-owned region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmColumnFamilySpec {
    pub region: &'static str,
    pub start: usize,
    pub len: usize,
    pub name: &'static str,
    pub role: &'static str,
    pub width: ColumnWidth,
}

impl WasmColumnFamilySpec {
    pub const fn end(&self) -> usize {
        self.start + self.len
    }
}

pub(crate) const fn f_prime_width(width: ColumnWidth) -> usize {
    match width {
        ColumnWidth::Boolean => 1,
        ColumnWidth::Byte => 8,
        ColumnWidth::U32 => 32,
        ColumnWidth::Field => 64,
    }
}

pub(crate) fn family_f_prime_widths(specs: &'static [WasmColumnFamilySpec]) -> impl Iterator<Item = usize> {
    specs
        .iter()
        .flat_map(|spec| core::iter::repeat_n(f_prime_width(spec.width), spec.len))
}

/// Define the named semantic prefix, starting at column zero.
macro_rules! define_columns {
    ($( ( $name:ident, $role:literal $(, $width:expr)? ) ),+ $(,)?) => {
        define_columns!(@assign 0usize; $($name),+);

        /// Macro-generated table of column metadata.
        pub const COLUMN_SPECS: &[$crate::column_registry::WasmColumnSpec] = &[
            $($crate::column_registry::WasmColumnSpec {
                index: $name,
                name: stringify!($name),
                role: $role,
                width: define_columns!(@maybe_width $($width)?),
            }),+
        ];
    };
    (@maybe_width $width:expr) => { $width };
    (@maybe_width) => { $crate::column_registry::ColumnWidth::Field };
    (@assign $idx:expr; $name:ident, $($rest:ident),+) => {
        pub const $name: usize = $idx;
        define_columns!(@assign $idx + 1usize; $($rest),+);
    };
    (@assign $idx:expr; $name:ident) => {
        pub const $name: usize = $idx;
        /// Number of macro-declared named columns. NOT the final witness
        /// width. Range constraints may be added, plus the F' transformation,
        /// lookup/mcc related constraints derived from the specs.
        pub const NAMED_COLUMN_COUNT: usize = $idx + 1usize;
    };
}

pub(crate) use define_columns;

/// Define a contiguous subsystem-owned region at an assigned absolute base.
macro_rules! define_column_region {
    (
        region: $region:literal,
        start: $start:expr,
        width: $width_vis:vis $width_name:ident,
        specs: $specs_vis:vis $specs_name:ident,
        columns: [
            $(($name:ident, $len:expr, $role:literal, $column_width:expr)),+ $(,)?
        ]
    ) => {
        define_column_region!(@assign $start; $(($name, $len)),+);

        $width_vis const $width_name: usize = 0usize $(+ $len)+;
        $specs_vis const $specs_name: &[$crate::column_registry::WasmColumnFamilySpec] = &[
            $($crate::column_registry::WasmColumnFamilySpec {
                region: $region,
                start: $name,
                len: $len,
                name: stringify!($name),
                role: $role,
                width: $column_width,
            }),+
        ];
    };
    (@assign $idx:expr; ($name:ident, $len:expr), $(($rest_name:ident, $rest_len:expr)),+) => {
        const $name: usize = $idx;
        define_column_region!(@assign $idx + $len; $(($rest_name, $rest_len)),+);
    };
    (@assign $idx:expr; ($name:ident, $len:expr)) => {
        const $name: usize = $idx;
    };
}

pub(crate) use define_column_region;
