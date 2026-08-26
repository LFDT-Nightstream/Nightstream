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

/// Declares one contiguous family of witness columns with a shared value width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnFamilySpec {
    pub region: &'static str,
    pub start: usize,
    pub len: usize,
    pub name: &'static str,
    pub role: &'static str,
    pub width: ColumnWidth,
}

impl ColumnFamilySpec {
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
        families: $families_vis:vis $families_name:ident,
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
        $families_vis const $families_name: &[$crate::ColumnFamilySpec] = &[
            $($crate::ColumnFamilySpec {
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
    families: Vec<ColumnFamilySpec>,
    column_count: usize,
}

impl ColumnRegistry {
    pub fn new(families: impl IntoIterator<Item = ColumnFamilySpec>) -> Result<Self, ColumnRegistryError> {
        let families: Vec<_> = families.into_iter().collect();
        if families.is_empty() {
            return Err(ColumnRegistryError::Empty);
        }

        let mut expected_start = 0usize;
        let mut names = BTreeSet::new();
        for family in &families {
            if family.region.is_empty() {
                return Err(ColumnRegistryError::EmptyRegion { name: family.name });
            }
            if family.name.is_empty() {
                return Err(ColumnRegistryError::EmptyName { start: family.start });
            }
            if family.len == 0 {
                return Err(ColumnRegistryError::EmptyFamily { name: family.name });
            }
            if !names.insert(family.name) {
                return Err(ColumnRegistryError::DuplicateName { name: family.name });
            }
            if family.start != expected_start {
                return Err(ColumnRegistryError::NonContiguous {
                    name: family.name,
                    expected_start,
                    actual_start: family.start,
                });
            }
            expected_start = family
                .start
                .checked_add(family.len)
                .ok_or(ColumnRegistryError::IndexOverflow { name: family.name })?;
        }

        Ok(Self {
            families,
            column_count: expected_start,
        })
    }

    pub fn column_count(&self) -> usize {
        self.column_count
    }

    pub fn families(&self) -> &[ColumnFamilySpec] {
        &self.families
    }

    pub fn family_for_column(&self, column: usize) -> Option<&ColumnFamilySpec> {
        if column >= self.column_count {
            return None;
        }

        // `new` guarantees a nonempty, contiguous registry starting at zero,
        // so every in-range column has exactly one preceding family start.
        let index = self
            .families
            .partition_point(|family| family.start <= column)
            - 1;
        Some(&self.families[index])
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
