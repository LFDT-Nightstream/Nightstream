//! Positional wire tuples of the Lean-emitted inner package.
//!
//! Owns only the serde shape. `validate` turns each tuple into the checked
//! package model.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::compact::{RawCompactRowInvocation, RawCompactRowTemplate};
use super::relation::RawCcsRelation;
use crate::witness::RawWitnessBatch;

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawPackage(
    pub(super) u64,
    pub(super) RawProfile,
    pub(super) RawPoseidonSchedule,
    pub(super) RawPhysicalLayout,
    pub(super) RawCcsRelation,
    pub(super) RawPermutationTemplate,
    pub(super) Vec<RawHashChain>,
    pub(super) Vec<RawPermutationInvocation>,
    pub(super) Vec<RawCompactRowTemplate>,
    pub(super) Vec<RawCompactRowInvocation>,
    pub(super) Vec<RawWitnessBatch>,
    pub(super) Vec<RawWitnessInstruction>,
    pub(super) Vec<RawSparseRow>,
    pub(super) Vec<Value>,
);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawProfile(
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawPoseidonSchedule(
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawSegment(pub(super) u64, pub(super) u64, pub(super) u64);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawPhysicalLayout(
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) Vec<RawSegment>,
    pub(super) Vec<RawSegment>,
);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawColumnRef(pub(super) u64, pub(super) u64);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawTemplateTerm(pub(super) RawColumnRef, pub(super) u64);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawTemplateCombination(pub(super) u64, pub(super) Vec<RawTemplateTerm>);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawTemplateRow(
    pub(super) u64,
    pub(super) RawTemplateCombination,
    pub(super) RawTemplateCombination,
    pub(super) RawTemplateCombination,
);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawPermutationTemplate(
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) Vec<RawTemplateRow>,
);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawHashChain(
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawPermutationInvocation(
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) Vec<RawSparseCombination>,
);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawWitnessInstruction(
    pub(super) u64,
    pub(super) u64,
    pub(super) RawSparseCombination,
    pub(super) RawSparseCombination,
);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawSparseTerm(pub(super) u64, pub(super) u64);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawSparseCombination(pub(super) u64, pub(super) Vec<RawSparseTerm>);

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawSparseRow(
    pub(super) u64,
    pub(super) RawSparseCombination,
    pub(super) RawSparseCombination,
    pub(super) RawSparseCombination,
);
