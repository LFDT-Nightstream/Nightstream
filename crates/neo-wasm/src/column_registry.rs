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

/// Static metadata for one contiguous family of witness columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmColumnSpec {
    pub region: &'static str,
    pub start: usize,
    pub len: usize,
    pub name: &'static str,
    pub role: &'static str,
    pub width: ColumnWidth,
}

impl WasmColumnSpec {
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

pub(crate) fn expanded_f_prime_widths(specs: &'static [WasmColumnSpec]) -> impl Iterator<Item = usize> {
    specs
        .iter()
        .flat_map(|spec| core::iter::repeat_n(f_prime_width(spec.width), spec.len))
}

/// Define a contiguous subsystem-owned region at an assigned absolute base.
///
/// Each entry uses Rust's `[element; length]` array notation, with a declared
/// column width in the element position. The generated name is the absolute
/// index of the family's first column.
macro_rules! define_column_region {
    (
        region: $region:literal,
        start: $start:expr,
        width: $width_vis:vis $width_name:ident,
        specs: $specs_vis:vis $specs_name:ident,
        indices: $index_vis:vis,
        columns: [
            $($name:ident: [$column_width:ident; $len:expr] => $role:literal),+ $(,)?
        ]
    ) => {
        define_column_region!(@assign $index_vis, $start; $(($name, $len)),+);

        /// Number of witness columns allocated by this region.
        $width_vis const $width_name: usize = 0usize $(+ $len)+;
        /// Macro-generated metadata for the region's declared column families.
        $specs_vis const $specs_name: &[$crate::column_registry::WasmColumnSpec] = &[
            $($crate::column_registry::WasmColumnSpec {
                region: $region,
                start: $name,
                len: $len,
                name: stringify!($name),
                role: $role,
                width: $crate::column_registry::ColumnWidth::$column_width,
            }),+
        ];
    };
    (@assign $index_vis:vis, $idx:expr; ($name:ident, $len:expr), $(($rest_name:ident, $rest_len:expr)),+) => {
        $index_vis const $name: usize = $idx;
        define_column_region!(@assign $index_vis, $idx + $len; $(($rest_name, $rest_len)),+);
    };
    (@assign $index_vis:vis, $idx:expr; ($name:ident, $len:expr)) => {
        $index_vis const $name: usize = $idx;
    };
}

pub(crate) use define_column_region;
