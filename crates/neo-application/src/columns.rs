//! Domain-neutral witness-column metadata and index ownership.

use std::collections::BTreeSet;

/// Declared intrinsic width of a witness column.
///
/// This declaration is compilation input, not an enforcement mechanism.
/// Relation construction must enforce it with constraints or a sound lookup
/// mechanism. Witness-layout and proof-shape planners may consume it as fact,
/// so an incorrect or unenforced declaration is a soundness error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnWidth {
    Boolean,
    Byte,
    U32,
    Field,
}

/// Metadata for one contiguous family of witness columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnSpec {
    pub region: &'static str,
    pub start: usize,
    pub len: usize,
    pub name: &'static str,
    pub role: &'static str,
    pub width: ColumnWidth,
}

impl ColumnSpec {
    pub const fn end(&self) -> usize {
        self.start + self.len
    }

    pub const fn contains(&self, column: usize) -> bool {
        self.start <= column && column < self.end()
    }
}

/// Define one contiguous region of statically allocated witness columns.
///
/// Scalar entries name their intrinsic width directly. Families use Rust's
/// `[width; length]` array notation and generate arrays of absolute indices.
#[macro_export]
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
        $crate::define_column_region!(@assign $index_vis, $start; $(($name, $shape)),+);

        /// Number of witness columns allocated by this region.
        $width_vis const $width_name: usize =
            0usize $(+ $crate::define_column_region!(@len $shape))+;
        /// Macro-generated metadata for the region's declared column families.
        $specs_vis const $specs_name: &[$crate::ColumnSpec] = &[
            $($crate::ColumnSpec {
                region: $region,
                start: $crate::define_column_region!(@start $name, $shape),
                len: $crate::define_column_region!(@len $shape),
                name: stringify!($name),
                role: $role,
                width: $crate::define_column_region!(@width $shape),
            }),+
        ];
    };
    (@assign $index_vis:vis, $idx:expr; ($name:ident, $shape:tt), $(($rest_name:ident, $rest_shape:tt)),+) => {
        $crate::define_column_region!(@declare $index_vis, $idx; $name, $shape);
        $crate::define_column_region!(
            @assign $index_vis,
            $idx + $crate::define_column_region!(@len $shape);
            $(($rest_name, $rest_shape)),+
        );
    };
    (@assign $index_vis:vis, $idx:expr; ($name:ident, $shape:tt)) => {
        $crate::define_column_region!(@declare $index_vis, $idx; $name, $shape);
    };
    (@declare $index_vis:vis, $idx:expr; $name:ident, $column_width:ident) => {
        $index_vis const $name: usize = $idx;
    };
    (@declare $index_vis:vis, $idx:expr; $name:ident, [$column_width:ident; $len:expr]) => {
        $index_vis const $name: [usize; $len] = {
            let mut indices = [0; $len];
            let mut i = 0;
            while i < $len {
                indices[i] = $idx + i;
                i += 1;
            }
            indices
        };
    };
    (@start $name:ident, $column_width:ident) => { $name };
    (@start $name:ident, [$column_width:ident; $len:expr]) => { $name[0] };
    (@len $column_width:ident) => { 1usize };
    (@len [$column_width:ident; $len:expr]) => { $len };
    (@width $column_width:ident) => { $crate::ColumnWidth::$column_width };
    (@width [$column_width:ident; $len:expr]) => { $crate::ColumnWidth::$column_width };
}

/// Complete, ordered ownership map for an application witness vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnRegistry {
    specs: Vec<ColumnSpec>,
    column_count: usize,
}

impl ColumnRegistry {
    pub fn new(specs: impl IntoIterator<Item = ColumnSpec>) -> Result<Self, ColumnRegistryError> {
        let specs: Vec<_> = specs.into_iter().collect();
        if specs.is_empty() {
            return Err(ColumnRegistryError::Empty);
        }

        let mut expected_start = 0usize;
        let mut names = BTreeSet::new();
        for spec in &specs {
            if spec.region.is_empty() {
                return Err(ColumnRegistryError::EmptyRegion { name: spec.name });
            }
            if spec.name.is_empty() {
                return Err(ColumnRegistryError::EmptyName { start: spec.start });
            }
            if spec.len == 0 {
                return Err(ColumnRegistryError::EmptyFamily { name: spec.name });
            }
            if !names.insert(spec.name) {
                return Err(ColumnRegistryError::DuplicateName { name: spec.name });
            }
            if spec.start != expected_start {
                return Err(ColumnRegistryError::NonContiguous {
                    name: spec.name,
                    expected_start,
                    actual_start: spec.start,
                });
            }
            expected_start = spec
                .start
                .checked_add(spec.len)
                .ok_or(ColumnRegistryError::IndexOverflow { name: spec.name })?;
        }

        Ok(Self {
            specs,
            column_count: expected_start,
        })
    }

    pub fn column_count(&self) -> usize {
        self.column_count
    }

    pub fn specs(&self) -> &[ColumnSpec] {
        &self.specs
    }

    pub fn spec_for_column(&self, column: usize) -> Option<&ColumnSpec> {
        if column >= self.column_count {
            return None;
        }

        // `new` guarantees a nonempty, contiguous registry starting at zero,
        // so every in-range column has exactly one preceding family start.
        let index = self.specs.partition_point(|spec| spec.start <= column) - 1;
        Some(&self.specs[index])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ColumnRegistryError {
    #[error("column registry requires at least one column family")]
    Empty,
    #[error("column family `{name}` has an empty region name")]
    EmptyRegion { name: &'static str },
    #[error("column family at index {start} has an empty name")]
    EmptyName { start: usize },
    #[error("column family `{name}` must contain at least one column")]
    EmptyFamily { name: &'static str },
    #[error("column family name `{name}` is declared more than once")]
    DuplicateName { name: &'static str },
    #[error("column family `{name}` starts at {actual_start}, expected {expected_start}")]
    NonContiguous {
        name: &'static str,
        expected_start: usize,
        actual_start: usize,
    },
    #[error("column family `{name}` overflows the column index space")]
    IndexOverflow { name: &'static str },
}
