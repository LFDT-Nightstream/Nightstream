//! Option B's two step circuits (Lean `Export/OptionB`): a light circuit and
//! a refresh circuit, each with its own relation.
mod relation;

pub(crate) use relation::Relation;

#[cfg(test)]
#[path = "../../../tests/option_b_folds/mod.rs"]
mod tests;
