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

    pub const fn contains(&self, column: usize) -> bool {
        self.start <= column && column < self.end()
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

pub(crate) const fn column_indices<const N: usize>(start: usize) -> [usize; N] {
    let mut indices = [0; N];
    let mut i = 0;
    while i < N {
        indices[i] = start + i;
        i += 1;
    }
    indices
}

/// Define a contiguous subsystem-owned region at an assigned absolute base.
///
/// Scalar entries name their column width directly. Families use Rust's
/// `[element; length]` array notation and generate arrays of absolute indices.
macro_rules! define_column_region {
    (
        region: $region:literal,
        start: $start:expr,
        width: $width_vis:vis $width_name:ident,
        specs: $specs_vis:vis $specs_name:ident,
        indices: $index_vis:vis,
        columns: [
            $($name:ident: $shape:tt => $role:literal),+ $(,)?
        ]
    ) => {
        define_column_region!(@assign $index_vis, $start; $(($name, $shape)),+);

        /// Number of witness columns allocated by this region.
        $width_vis const $width_name: usize =
            0usize $(+ define_column_region!(@len $shape))+;
        /// Macro-generated metadata for the region's declared column families.
        $specs_vis const $specs_name: &[$crate::column_registry::WasmColumnSpec] = &[
            $($crate::column_registry::WasmColumnSpec {
                region: $region,
                start: define_column_region!(@start $name, $shape),
                len: define_column_region!(@len $shape),
                name: stringify!($name),
                role: $role,
                width: define_column_region!(@width $shape),
            }),+
        ];
    };
    (@assign $index_vis:vis, $idx:expr; ($name:ident, $shape:tt), $(($rest_name:ident, $rest_shape:tt)),+) => {
        define_column_region!(@declare $index_vis, $idx; $name, $shape);
        define_column_region!(
            @assign $index_vis,
            $idx + define_column_region!(@len $shape);
            $(($rest_name, $rest_shape)),+
        );
    };
    (@assign $index_vis:vis, $idx:expr; ($name:ident, $shape:tt)) => {
        define_column_region!(@declare $index_vis, $idx; $name, $shape);
    };
    (@declare $index_vis:vis, $idx:expr; $name:ident, $column_width:ident) => {
        $index_vis const $name: usize = $idx;
    };
    (@declare $index_vis:vis, $idx:expr; $name:ident, [$column_width:ident; $len:expr]) => {
        $index_vis const $name: [usize; $len] =
            $crate::column_registry::column_indices::<$len>($idx);
    };
    (@start $name:ident, $column_width:ident) => { $name };
    (@start $name:ident, [$column_width:ident; $len:expr]) => { $name[0] };
    (@len $column_width:ident) => { 1usize };
    (@len [$column_width:ident; $len:expr]) => { $len };
    (@width $column_width:ident) => { $crate::column_registry::ColumnWidth::$column_width };
    (@width [$column_width:ident; $len:expr]) => { $crate::column_registry::ColumnWidth::$column_width };
}

pub(crate) use define_column_region;
