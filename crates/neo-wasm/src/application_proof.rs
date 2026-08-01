//! Spartan and WHIR proof boundary for a Lean-owned WASM application manifest.
//!
//! This module validates one exact Lean manifest, lowers its active native CCS
//! rows to the same R1CS rows, and runs the direct Spartan backend. It does not
//! define WASM semantics or the recursive F-prime terminal relation.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use toy_spartan::{
    errors::SpartanError,
    provider::{goldi::F, GoldilocksWhirEngine},
    spartan::{RepeatedR1CSSNARK, SpartanProverKey, SpartanVerifierKey, R1CSSNARK},
    SparseMatrix, SplitR1CSShape,
};

use crate::WasmApplicationModule;

const SCHEMA_VERSION: u32 = 1;
const FORMAT_NAME: &str = "nightstream/wasm-application-proof";
const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;
const MATRIX_COUNT: usize = 4;
const POLYNOMIAL_DEGREE: usize = 3;

type Engine = GoldilocksWhirEngine;
type Snark = R1CSSNARK<Engine>;
type ProductionSnark = RepeatedR1CSSNARK<Engine>;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ColumnRole {
    One,
    ModuleByte,
    PrivateWitness,
    Output,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnBinding {
    index: usize,
    role: ColumnRole,
    role_index: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PolynomialTerm {
    sign: String,
    exponents: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct LinearTerm {
    column: usize,
    coefficient: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestRow {
    ordinal: usize,
    selector: usize,
    a: Vec<LinearTerm>,
    b: Vec<LinearTerm>,
    c: Vec<LinearTerm>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestCost {
    rows: usize,
    committed_columns: usize,
    lean_public_columns: usize,
    auxiliary_columns: usize,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestMetrics {
    r1cs_nonzero_coefficients: usize,
    native_ccs_nonzero_coefficients: usize,
    maximum_r1cs_row_density: usize,
    maximum_native_ccs_row_density: usize,
    poseidon2_calls: usize,
    maximum_live_witness_columns: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProofManifest {
    schema: u32,
    format: String,
    module_id: String,
    module_hex: String,
    goldilocks_modulus: u64,
    matrix_count: usize,
    polynomial_degree: usize,
    polynomial: Vec<PolynomialTerm>,
    columns: Vec<ColumnBinding>,
    rows: Vec<ManifestRow>,
    cost: ManifestCost,
    metrics: ManifestMetrics,
}

#[derive(Debug)]
struct RoleColumns {
    one: usize,
    module_bytes: Vec<usize>,
    private_witnesses: Vec<usize>,
    outputs: Vec<usize>,
}

/// Lean-derived physical and sparse relation measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasmApplicationProofStats {
    pub rows: usize,
    pub private_witness_columns: usize,
    pub public_values: usize,
    pub r1cs_nonzero_coefficients: usize,
    pub native_ccs_nonzero_coefficients: usize,
    pub maximum_r1cs_row_density: usize,
    pub maximum_native_ccs_row_density: usize,
    pub poseidon2_calls: usize,
    pub maximum_live_witness_columns: usize,
}

struct ValidatedRelation {
    shape: SplitR1CSShape<Engine>,
    private_witnesses: usize,
    outputs: usize,
    stats: WasmApplicationProofStats,
}

/// One lockstep Spartan proof whose PCS openings use WHIR.
#[derive(Serialize, Deserialize)]
pub struct WasmApplicationProof {
    inner: ProductionSnark,
}

impl WasmApplicationProof {
    /// Encode the proof for storage or transport.
    pub fn to_bytes(&self) -> Result<Vec<u8>, WasmApplicationProofError> {
        bincode::serialize(self).map_err(|error| WasmApplicationProofError::ProofEncoding(error.to_string()))
    }

    /// Decode an untrusted proof. Cryptographic validity is checked by
    /// `WasmApplicationProofSystem::verify`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WasmApplicationProofError> {
        bincode::deserialize(bytes).map_err(|error| WasmApplicationProofError::ProofEncoding(error.to_string()))
    }
}

/// Prepared proof system for one exact Lean-owned module and relation.
pub struct WasmApplicationProofSystem {
    module: WasmApplicationModule,
    private_witnesses: usize,
    outputs: usize,
    stats: WasmApplicationProofStats,
    prover_key: SpartanProverKey<Engine>,
    verifier_key: SpartanVerifierKey<Engine>,
}

impl WasmApplicationProofSystem {
    /// Validate the Lean manifest and prepare Spartan and WHIR keys.
    pub fn setup(module: WasmApplicationModule, manifest_json: &[u8]) -> Result<Self, WasmApplicationProofError> {
        let manifest: ProofManifest = serde_json::from_slice(manifest_json)?;
        let relation = validate_manifest(&module, &manifest)?;
        let (prover_key, verifier_key) = Snark::setup_direct(relation.shape)?;
        Ok(Self {
            module,
            private_witnesses: relation.private_witnesses,
            outputs: relation.outputs,
            stats: relation.stats,
            prover_key,
            verifier_key,
        })
    }

    pub fn module(&self) -> &WasmApplicationModule {
        &self.module
    }

    pub fn stats(&self) -> WasmApplicationProofStats {
        self.stats
    }

    /// Prove the declared private witness and public module outputs.
    pub fn prove(
        &self,
        private_witnesses: &[u64],
        outputs: &[u64],
    ) -> Result<WasmApplicationProof, WasmApplicationProofError> {
        let witness = canonical_values("private witness", private_witnesses, self.private_witnesses)?;
        let public = self.public_values(outputs)?;
        Ok(WasmApplicationProof {
            inner: ProductionSnark::prove_direct(&self.prover_key, &witness, &public, true)?,
        })
    }

    /// Verify the proof against verifier-owned module bytes and outputs.
    pub fn verify(
        &self,
        proof: &WasmApplicationProof,
        expected_outputs: &[u64],
    ) -> Result<(), WasmApplicationProofError> {
        let expected = self.public_values(expected_outputs)?;
        let actual = proof.inner.verify(&self.verifier_key)?;
        if actual != expected {
            return Err(WasmApplicationProofError::PublicValuesMismatch);
        }
        Ok(())
    }

    fn public_values(&self, outputs: &[u64]) -> Result<Vec<F>, WasmApplicationProofError> {
        let mut public = self
            .module
            .bytes()
            .iter()
            .map(|&byte| F::new(u64::from(byte)))
            .collect::<Vec<_>>();
        public.extend(canonical_values("output", outputs, self.outputs)?);
        Ok(public)
    }
}

fn validate_manifest(
    module: &WasmApplicationModule,
    manifest: &ProofManifest,
) -> Result<ValidatedRelation, WasmApplicationProofError> {
    require(manifest.schema == SCHEMA_VERSION, "unsupported proof manifest schema")?;
    require(manifest.format == FORMAT_NAME, "unsupported proof manifest format")?;
    require(manifest.module_id == module.module_id(), "module identifier mismatch")?;
    require(
        manifest.module_hex == encode_hex(module.bytes()),
        "module byte mismatch",
    )?;
    require(
        manifest.goldilocks_modulus == GOLDILOCKS_MODULUS,
        "field modulus mismatch",
    )?;
    require(
        manifest.matrix_count == MATRIX_COUNT,
        "native CCS matrix count mismatch",
    )?;
    require(
        manifest.polynomial_degree == POLYNOMIAL_DEGREE,
        "native CCS degree mismatch",
    )?;
    require(
        manifest.polynomial == selector_polynomial(),
        "native CCS selector polynomial mismatch",
    )?;

    let roles = validate_columns(module, manifest)?;
    validate_rows(module, manifest, &roles)?;
    validate_cost_and_metrics(module, manifest, &roles)?;

    let remap = backend_column_map(manifest.columns.len(), &roles);
    let backend_columns = manifest.columns.len();
    let a = sparse_matrix(&manifest.rows, backend_columns, &remap, |row| &row.a)?;
    let b = sparse_matrix(&manifest.rows, backend_columns, &remap, |row| &row.b)?;
    let c = sparse_matrix(&manifest.rows, backend_columns, &remap, |row| &row.c)?;
    let public_values = roles.module_bytes.len() + roles.outputs.len();
    let shape = SplitR1CSShape::new(
        2,
        manifest.rows.len(),
        0,
        0,
        roles.private_witnesses.len(),
        public_values,
        0,
        a,
        b,
        c,
    )?;

    Ok(ValidatedRelation {
        shape,
        private_witnesses: roles.private_witnesses.len(),
        outputs: roles.outputs.len(),
        stats: WasmApplicationProofStats {
            rows: manifest.cost.rows,
            private_witness_columns: roles.private_witnesses.len(),
            public_values,
            r1cs_nonzero_coefficients: manifest.metrics.r1cs_nonzero_coefficients,
            native_ccs_nonzero_coefficients: manifest.metrics.native_ccs_nonzero_coefficients,
            maximum_r1cs_row_density: manifest.metrics.maximum_r1cs_row_density,
            maximum_native_ccs_row_density: manifest.metrics.maximum_native_ccs_row_density,
            poseidon2_calls: manifest.metrics.poseidon2_calls,
            maximum_live_witness_columns: manifest.metrics.maximum_live_witness_columns,
        },
    })
}

fn validate_columns(
    module: &WasmApplicationModule,
    manifest: &ProofManifest,
) -> Result<RoleColumns, WasmApplicationProofError> {
    for (expected, binding) in manifest.columns.iter().enumerate() {
        require(binding.index == expected, "column indices are not canonical")?;
    }

    let one = canonical_role(&manifest.columns, ColumnRole::One)?;
    require(
        one.len() == 1 && one[0].0 == 0,
        "the one role must be unique at role index zero",
    )?;
    let module_bytes = canonical_role(&manifest.columns, ColumnRole::ModuleByte)?;
    let private_witnesses = canonical_role(&manifest.columns, ColumnRole::PrivateWitness)?;
    let outputs = canonical_role(&manifest.columns, ColumnRole::Output)?;
    require(
        module_bytes.len() == module.bytes().len(),
        "module-byte role count mismatch",
    )?;
    require(!outputs.is_empty(), "the proof relation has no public output")?;

    Ok(RoleColumns {
        one: one[0].1,
        module_bytes: module_bytes.into_iter().map(|pair| pair.1).collect(),
        private_witnesses: private_witnesses.into_iter().map(|pair| pair.1).collect(),
        outputs: outputs.into_iter().map(|pair| pair.1).collect(),
    })
}

fn canonical_role(
    columns: &[ColumnBinding],
    role: ColumnRole,
) -> Result<Vec<(usize, usize)>, WasmApplicationProofError> {
    let mut found = columns
        .iter()
        .filter(|binding| binding.role == role)
        .map(|binding| (binding.role_index, binding.index))
        .collect::<Vec<_>>();
    found.sort_unstable_by_key(|pair| pair.0);
    for (expected, (role_index, _)) in found.iter().enumerate() {
        require(*role_index == expected, "role indices are not contiguous")?;
    }
    Ok(found)
}

fn validate_rows(
    module: &WasmApplicationModule,
    manifest: &ProofManifest,
    roles: &RoleColumns,
) -> Result<(), WasmApplicationProofError> {
    for (ordinal, row) in manifest.rows.iter().enumerate() {
        require(row.ordinal == ordinal, "row ordinals are not canonical")?;
        require(
            row.selector == roles.one,
            "a selector is not the verifier-fixed one column",
        )?;
        validate_terms(&row.a, manifest.columns.len())?;
        validate_terms(&row.b, manifest.columns.len())?;
        validate_terms(&row.c, manifest.columns.len())?;
    }

    require(
        manifest.rows.len() >= module.bytes().len(),
        "module byte rows are missing",
    )?;
    for (index, &byte) in module.bytes().iter().enumerate() {
        let row = &manifest.rows[index];
        require(
            row.a == [term(roles.module_bytes[index], 1)],
            "module byte A row mismatch",
        )?;
        require(row.b == [term(roles.one, 1)], "module byte B row mismatch")?;
        let expected_c = if byte == 0 {
            Vec::new()
        } else {
            vec![term(roles.one, u64::from(byte))]
        };
        require(row.c == expected_c, "module byte C row mismatch")?;
    }
    Ok(())
}

fn validate_cost_and_metrics(
    module: &WasmApplicationModule,
    manifest: &ProofManifest,
    roles: &RoleColumns,
) -> Result<(), WasmApplicationProofError> {
    let cost = manifest.cost;
    require(cost.rows == manifest.rows.len(), "row cost does not equal emitted rows")?;
    require(
        cost.committed_columns == 0,
        "this manifest has no committed-column role",
    )?;
    require(
        cost.auxiliary_columns == roles.private_witnesses.len(),
        "auxiliary-column cost mismatch",
    )?;
    require(
        cost.lean_public_columns == 1 + module.bytes().len() + roles.outputs.len(),
        "public-column cost mismatch",
    )?;
    require(
        manifest.columns.len() == cost.committed_columns + cost.lean_public_columns + cost.auxiliary_columns,
        "column cost does not equal physical columns",
    )?;

    let supports = manifest
        .rows
        .iter()
        .map(|row| row.a.len() + row.b.len() + row.c.len())
        .collect::<Vec<_>>();
    let r1cs_nonzero = supports.iter().sum::<usize>();
    let max_r1cs = supports.iter().copied().max().unwrap_or(0);
    let metrics = manifest.metrics;
    require(
        metrics.r1cs_nonzero_coefficients == r1cs_nonzero,
        "R1CS nonzero count mismatch",
    )?;
    require(
        metrics.native_ccs_nonzero_coefficients == r1cs_nonzero + manifest.rows.len(),
        "native CCS nonzero count mismatch",
    )?;
    require(
        metrics.maximum_r1cs_row_density == max_r1cs,
        "R1CS row density mismatch",
    )?;
    require(
        metrics.maximum_native_ccs_row_density == max_r1cs + usize::from(!manifest.rows.is_empty()),
        "native CCS row density mismatch",
    )?;
    require(
        metrics.maximum_live_witness_columns <= roles.private_witnesses.len(),
        "live witness count exceeds private columns",
    )?;
    Ok(())
}

fn backend_column_map(column_count: usize, roles: &RoleColumns) -> Vec<usize> {
    let mut remap = vec![usize::MAX; column_count];
    for (index, &column) in roles.private_witnesses.iter().enumerate() {
        remap[column] = index;
    }
    remap[roles.one] = roles.private_witnesses.len();
    let public_base = roles.private_witnesses.len() + 1;
    for (index, &column) in roles.module_bytes.iter().enumerate() {
        remap[column] = public_base + index;
    }
    for (index, &column) in roles.outputs.iter().enumerate() {
        remap[column] = public_base + roles.module_bytes.len() + index;
    }
    remap
}

fn sparse_matrix(
    rows: &[ManifestRow],
    columns: usize,
    remap: &[usize],
    terms: impl Fn(&ManifestRow) -> &[LinearTerm],
) -> Result<SparseMatrix<F>, WasmApplicationProofError> {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(rows.len() + 1);
    indptr.push(0);
    for row in rows {
        let mut entries = terms(row)
            .iter()
            .map(|term| (remap[term.column], F::from_canonical_u64(term.coefficient)))
            .collect::<Vec<_>>();
        entries.sort_unstable_by_key(|entry| entry.0);
        for (column, coefficient) in entries {
            require(column != usize::MAX, "a row references an unbound column")?;
            indices.push(column);
            data.push(coefficient);
        }
        indptr.push(data.len());
    }
    Ok(SparseMatrix::from_csr(rows.len(), columns, data, indices, indptr)?)
}

fn validate_terms(terms: &[LinearTerm], column_count: usize) -> Result<(), WasmApplicationProofError> {
    let mut previous = None;
    for term in terms {
        require(term.column < column_count, "a term references an unknown column")?;
        require(
            term.coefficient > 0 && term.coefficient < GOLDILOCKS_MODULUS,
            "a coefficient is not canonical",
        )?;
        if let Some(previous) = previous {
            require(previous < term.column, "linear terms are not strictly ordered")?;
        }
        previous = Some(term.column);
    }
    Ok(())
}

fn canonical_values(label: &'static str, values: &[u64], expected: usize) -> Result<Vec<F>, WasmApplicationProofError> {
    if values.len() != expected {
        return Err(WasmApplicationProofError::ValueCount {
            label,
            expected,
            actual: values.len(),
        });
    }
    values
        .iter()
        .map(|&value| {
            if value >= GOLDILOCKS_MODULUS {
                Err(WasmApplicationProofError::NonCanonicalValue { label, value })
            } else {
                Ok(F::from_canonical_u64(value))
            }
        })
        .collect()
}

fn selector_polynomial() -> Vec<PolynomialTerm> {
    vec![
        PolynomialTerm {
            sign: "positive".to_string(),
            exponents: vec![1, 1, 0, 1],
        },
        PolynomialTerm {
            sign: "negative".to_string(),
            exponents: vec![0, 0, 1, 1],
        },
    ]
}

fn term(column: usize, coefficient: u64) -> LinearTerm {
    LinearTerm { column, coefficient }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn require(condition: bool, reason: &'static str) -> Result<(), WasmApplicationProofError> {
    if condition {
        Ok(())
    } else {
        Err(WasmApplicationProofError::InvalidManifest { reason })
    }
}

#[derive(Debug, Error)]
pub enum WasmApplicationProofError {
    #[error("failed to parse the Lean WASM proof manifest: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid Lean WASM proof manifest: {reason}")]
    InvalidManifest { reason: &'static str },
    #[error("{label} value count mismatch: expected {expected}, got {actual}")]
    ValueCount {
        label: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{label} value {value} is not canonical in Goldilocks")]
    NonCanonicalValue { label: &'static str, value: u64 },
    #[error("Spartan or WHIR rejected the application proof: {0}")]
    Spartan(#[from] SpartanError),
    #[error("failed to encode or decode the application proof: {0}")]
    ProofEncoding(String),
    #[error("the proof public values do not equal the verifier-owned module bytes and outputs")]
    PublicValuesMismatch,
}
