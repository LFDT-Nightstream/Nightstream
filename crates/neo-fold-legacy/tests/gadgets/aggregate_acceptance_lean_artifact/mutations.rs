//! Fail-closed mutation census for the aggregate-acceptance leaf artifact.
//!
//! Owns: one-at-a-time rejection checks for every generated active-row role,
//! coefficient, matrix binding, and sparse-polynomial coefficient/power.
//!
//! Does not own: singleton fixture geometry, production extraction, semantic
//! sufficiency, or the recursive outer image.
//!
//! Emits constraints: no.
//!
//! | Mutation family | Exhaustive surface | Expected branch |
//! |---|---|---|
//! | Shape | schema version and gate arity | shape drift |
//! | Rows | every matrix/coordinate role and coefficient | active-row drift |
//! | Bindings | every role/index plus length/order | binding drift |
//! | Polynomial | every coefficient/role/power plus length/order | polynomial drift |

use super::{validate_artifact, ArtifactAudit, ArtifactDrift, CoordinateRole, MatrixRole};

fn reject(production: &ArtifactAudit, candidate: ArtifactAudit, expected: ArtifactDrift) {
    assert_eq!(validate_artifact(&candidate, production), Err(expected));
}

fn alternate_coordinate(role: CoordinateRole) -> CoordinateRole {
    match role {
        CoordinateRole::One => CoordinateRole::ChunkBit(0),
        CoordinateRole::ChunkBit(_) => CoordinateRole::One,
        CoordinateRole::Accept => CoordinateRole::TreeOutput(0),
        CoordinateRole::TreeOutput(_) => CoordinateRole::Accept,
    }
}

fn alternate_matrix(role: MatrixRole) -> MatrixRole {
    match role {
        MatrixRole::Selector => MatrixRole::ProductLeft(0),
        MatrixRole::ProductLeft(index) => MatrixRole::ProductRight(index),
        MatrixRole::ProductRight(index) => MatrixRole::ProductLeft(index),
        MatrixRole::ProductOut => MatrixRole::QuadraticBitLeft,
        MatrixRole::QuadraticBitLeft => MatrixRole::QuadraticBitRight,
        MatrixRole::QuadraticBitRight => MatrixRole::ProductOut,
    }
}

fn reject_shape(production: &ArtifactAudit) {
    let mut candidate = production.clone();
    candidate.schema_version += 1;
    reject(production, candidate, ArtifactDrift::SchemaVersion);

    let mut candidate = production.clone();
    candidate.gate_arity += 1;
    reject(production, candidate, ArtifactDrift::GateArity);
}

fn reject_matrix_bindings(production: &ArtifactAudit) {
    for index in 0..production.matrix_bindings.len() {
        let mut candidate = production.clone();
        candidate.matrix_bindings[index].role = alternate_matrix(candidate.matrix_bindings[index].role);
        reject(production, candidate, ArtifactDrift::MatrixBindings);

        let mut candidate = production.clone();
        candidate.matrix_bindings[index].index += 1;
        reject(production, candidate, ArtifactDrift::MatrixBindings);
    }

    let mut candidate = production.clone();
    candidate.matrix_bindings.pop();
    reject(production, candidate, ArtifactDrift::MatrixBindings);
    let mut candidate = production.clone();
    candidate
        .matrix_bindings
        .push(candidate.matrix_bindings[0].clone());
    reject(production, candidate, ArtifactDrift::MatrixBindings);
    let mut candidate = production.clone();
    candidate.matrix_bindings.swap(0, 1);
    reject(production, candidate, ArtifactDrift::MatrixBindings);
}

fn reject_active_rows(production: &ArtifactAudit) {
    for row in 0..production.active_rows.len() {
        for combination in 0..production.active_rows[row].len() {
            let mut candidate = production.clone();
            candidate.active_rows[row][combination].role =
                alternate_matrix(candidate.active_rows[row][combination].role);
            reject(production, candidate, ArtifactDrift::ActiveRows);

            for term in 0..production.active_rows[row][combination].terms.len() {
                let mut candidate = production.clone();
                candidate.active_rows[row][combination].terms[term].role =
                    alternate_coordinate(candidate.active_rows[row][combination].terms[term].role);
                reject(production, candidate, ArtifactDrift::ActiveRows);

                let mut candidate = production.clone();
                candidate.active_rows[row][combination].terms[term].coefficient += 1;
                reject(production, candidate, ArtifactDrift::ActiveRows);
            }
        }
    }

    let mut candidate = production.clone();
    candidate.active_rows.pop();
    reject(production, candidate, ArtifactDrift::ActiveRows);
    let mut candidate = production.clone();
    candidate.active_rows.push(candidate.active_rows[0].clone());
    reject(production, candidate, ArtifactDrift::ActiveRows);
    let mut candidate = production.clone();
    candidate.active_rows.swap(0, 1);
    reject(production, candidate, ArtifactDrift::ActiveRows);
}

fn reject_polynomial(production: &ArtifactAudit) {
    for term in 0..production.polynomial_terms.len() {
        let mut candidate = production.clone();
        candidate.polynomial_terms[term].coefficient += 1;
        reject(production, candidate, ArtifactDrift::PolynomialTerms);

        for power in 0..production.polynomial_terms[term].powers.len() {
            let mut candidate = production.clone();
            candidate.polynomial_terms[term].powers[power].role =
                alternate_matrix(candidate.polynomial_terms[term].powers[power].role);
            reject(production, candidate, ArtifactDrift::PolynomialTerms);

            let mut candidate = production.clone();
            candidate.polynomial_terms[term].powers[power].power += 1;
            reject(production, candidate, ArtifactDrift::PolynomialTerms);
        }
    }

    let mut candidate = production.clone();
    candidate.polynomial_terms.pop();
    reject(production, candidate, ArtifactDrift::PolynomialTerms);
    let mut candidate = production.clone();
    candidate
        .polynomial_terms
        .push(candidate.polynomial_terms[0].clone());
    reject(production, candidate, ArtifactDrift::PolynomialTerms);
    let mut candidate = production.clone();
    candidate.polynomial_terms.swap(0, 1);
    reject(production, candidate, ArtifactDrift::PolynomialTerms);
}

pub(super) fn assert_all_fail_closed(production: &ArtifactAudit) {
    reject_shape(production);
    reject_matrix_bindings(production);
    reject_active_rows(production);
    reject_polynomial(production);
}
