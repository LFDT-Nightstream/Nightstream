//! Validated boundary for a Lean-owned WASM application module.
//!
//! This module owns compact manifest parsing and checks parser-visible facts
//! against the exact WASM bytes. It does not own the VM relation, F-prime,
//! witness generation, or proof verification.

use serde::Deserialize;
use thiserror::Error;
use wasmparser::{ExternalKind, Parser, Payload};

use crate::{extract_wasm_program_artifacts, WasmBuildError, WasmProgramArtifacts};

const SCHEMA_VERSION: u32 = 1;
const FORMAT_NAME: &str = "nightstream/wasm-application-module";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModuleManifest {
    schema: u32,
    format: String,
    module_id: String,
    module_hex: String,
    entrypoint_hex: String,
    memory_minimum_pages: u32,
    memory_maximum_pages: u32,
    data_offset: u64,
    data_hex: String,
}

/// A WASM module whose bytes and parser-visible deployment facts agree with
/// one Lean-owned module manifest.
#[derive(Clone, Debug)]
pub struct WasmApplicationModule {
    module_id: String,
    bytes: Vec<u8>,
    entrypoint: String,
    artifacts: WasmProgramArtifacts,
}

impl WasmApplicationModule {
    /// Parse and validate one compact Lean module manifest.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, WasmApplicationManifestError> {
        let manifest: ModuleManifest = serde_json::from_slice(bytes)?;
        if manifest.schema != SCHEMA_VERSION {
            return Err(WasmApplicationManifestError::Schema {
                expected: SCHEMA_VERSION,
                actual: manifest.schema,
            });
        }
        if manifest.format != FORMAT_NAME {
            return Err(WasmApplicationManifestError::Format {
                expected: FORMAT_NAME,
                actual: manifest.format,
            });
        }
        if manifest.module_id.is_empty() {
            return Err(WasmApplicationManifestError::EmptyModuleId);
        }

        let module_bytes = decode_hex("module_hex", &manifest.module_hex)?;
        let entrypoint_bytes = decode_hex("entrypoint_hex", &manifest.entrypoint_hex)?;
        let entrypoint =
            String::from_utf8(entrypoint_bytes).map_err(|_| WasmApplicationManifestError::EntrypointNotUtf8)?;
        if entrypoint.is_empty() {
            return Err(WasmApplicationManifestError::MissingEntrypoint);
        }

        ensure_function_export(&module_bytes, &entrypoint)?;
        let artifacts = extract_wasm_program_artifacts(&module_bytes).map_err(WasmApplicationManifestError::Wasm)?;
        let tables = &artifacts.tables;
        if tables.has_imported_memory {
            return Err(WasmApplicationManifestError::ImportedMemory);
        }
        if tables.initial_memory_pages != Some(manifest.memory_minimum_pages) {
            return Err(WasmApplicationManifestError::InitialMemory {
                manifest: manifest.memory_minimum_pages,
                parsed: tables.initial_memory_pages,
            });
        }
        if tables.max_memory_pages != Some(manifest.memory_maximum_pages) {
            return Err(WasmApplicationManifestError::MaximumMemory {
                manifest: manifest.memory_maximum_pages,
                parsed: tables.max_memory_pages,
            });
        }

        let data = decode_hex("data_hex", &manifest.data_hex)?;
        let expected_data = data
            .iter()
            .enumerate()
            .map(|(index, &value)| (manifest.data_offset + index as u64, value))
            .collect::<Vec<_>>();
        if tables.linear_memory_init != expected_data {
            return Err(WasmApplicationManifestError::DataSegmentMismatch);
        }

        Ok(Self {
            module_id: manifest.module_id,
            bytes: module_bytes,
            entrypoint,
            artifacts,
        })
    }

    pub fn module_id(&self) -> &str {
        &self.module_id
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn entrypoint(&self) -> &str {
        &self.entrypoint
    }

    pub fn artifacts(&self) -> &WasmProgramArtifacts {
        &self.artifacts
    }
}

fn ensure_function_export(bytes: &[u8], entrypoint: &str) -> Result<(), WasmApplicationManifestError> {
    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload.map_err(|error| WasmApplicationManifestError::WasmParse(error.to_string()))?;
        if let Payload::ExportSection(reader) = payload {
            for export in reader {
                let export = export.map_err(|error| WasmApplicationManifestError::WasmParse(error.to_string()))?;
                if export.name == entrypoint && export.kind == ExternalKind::Func {
                    return Ok(());
                }
            }
        }
    }
    Err(WasmApplicationManifestError::MissingEntrypoint)
}

fn decode_hex(field: &'static str, value: &str) -> Result<Vec<u8>, WasmApplicationManifestError> {
    if value.len() % 2 != 0 {
        return Err(WasmApplicationManifestError::Hex { field });
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_nibble(pair[0]).ok_or(WasmApplicationManifestError::Hex { field })?;
            let low = hex_nibble(pair[1]).ok_or(WasmApplicationManifestError::Hex { field })?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[derive(Debug, Error)]
pub enum WasmApplicationManifestError {
    #[error("failed to parse the Lean WASM module manifest: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported Lean WASM module schema {actual}; expected {expected}")]
    Schema { expected: u32, actual: u32 },
    #[error("unsupported Lean WASM module format `{actual}`; expected `{expected}`")]
    Format {
        expected: &'static str,
        actual: String,
    },
    #[error("the Lean WASM module identifier is empty")]
    EmptyModuleId,
    #[error("manifest field `{field}` is not canonical lowercase hexadecimal")]
    Hex { field: &'static str },
    #[error("the manifest entrypoint is not UTF-8")]
    EntrypointNotUtf8,
    #[error("the exact WASM bytes do not export the declared function entrypoint")]
    MissingEntrypoint,
    #[error("failed to parse the exact WASM bytes: {0}")]
    WasmParse(String),
    #[error("failed to derive WASM proof artifacts: {0}")]
    Wasm(#[source] WasmBuildError),
    #[error("imported linear memory is not verifier-owned by this manifest")]
    ImportedMemory,
    #[error("initial memory mismatch: manifest={manifest}, parsed={parsed:?}")]
    InitialMemory { manifest: u32, parsed: Option<u32> },
    #[error("maximum memory mismatch: manifest={manifest}, parsed={parsed:?}")]
    MaximumMemory { manifest: u32, parsed: Option<u32> },
    #[error("the manifest data segment does not equal the data in the exact WASM bytes")]
    DataSegmentMismatch,
}
