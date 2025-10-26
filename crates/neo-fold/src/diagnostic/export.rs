//! Diagnostic export and import

use super::types::ConstraintDiagnostic;
use std::path::{Path, PathBuf};
use std::io::{Write, Read};

#[derive(Debug, Clone, Copy)]
pub enum DiagnosticFormat {
    /// Human-readable JSON
    Json,
    /// Compressed JSON (recommended)
    JsonGz,
    /// Binary CBOR (requires feature)
    #[cfg(feature = "prove-diagnostics-cbor")]
    Cbor,
}

impl DiagnosticFormat {
    /// Get format from environment variable
    pub fn from_env() -> Self {
        match std::env::var("NEO_DIAGNOSTIC_FORMAT").as_deref() {
            Ok("json") => DiagnosticFormat::Json,
            Ok("json.gz") => DiagnosticFormat::JsonGz,
            #[cfg(feature = "prove-diagnostics-cbor")]
            Ok("cbor") => DiagnosticFormat::Cbor,
            _ => DiagnosticFormat::JsonGz,  // Default: compressed
        }
    }
    
    fn extension(&self) -> &str {
        match self {
            DiagnosticFormat::Json => "json",
            DiagnosticFormat::JsonGz => "json.gz",
            #[cfg(feature = "prove-diagnostics-cbor")]
            DiagnosticFormat::Cbor => "cbor",
        }
    }
}

/// Export diagnostic to file
pub fn export_diagnostic(
    diagnostic: &ConstraintDiagnostic,
    format: DiagnosticFormat,
) -> Result<PathBuf, std::io::Error> {
    // NEO_DIAGNOSTICS_DIR controls where diagnostics are written (default: "diagnostics")
    // This is separate from NEO_DIAGNOSTICS (boolean flag to enable diagnostics)
    let dir = std::env::var("NEO_DIAGNOSTICS_DIR")
        .unwrap_or_else(|_| "diagnostics".to_string());
    
    std::fs::create_dir_all(&dir)?;
    
    // Check size limit
    let max_size: usize = std::env::var("NEO_DIAGNOSTIC_MAX_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10 * 1024 * 1024);  // 10MB default
    
    let filename = format!(
        "{}_step_{}_constraint_{}.{}",
        diagnostic.context.test,
        diagnostic.context.step_idx,
        diagnostic.structure.row_index,
        format.extension()
    );
    
    let path = Path::new(&dir).join(&filename);
    
    match format {
        DiagnosticFormat::Json => {
            let json = serde_json::to_string_pretty(&diagnostic)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            
            if json.len() > max_size {
                eprintln!("⚠️  Warning: Diagnostic size ({} bytes) exceeds limit ({} bytes)", 
                    json.len(), max_size);
                eprintln!("   Consider using json.gz format or adjusting NEO_DIAGNOSTIC_MAX_SIZE");
            }
            
            write_locked(&path, json.as_bytes())?;
        }
        
        DiagnosticFormat::JsonGz => {
            use flate2::write::GzEncoder;
            use flate2::Compression;
            
            let json = serde_json::to_vec(&diagnostic)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            
            // Allow configuring compression level
            let compression = match std::env::var("NEO_DIAGNOSTIC_COMPRESSION").ok().as_deref() {
                Some("fast") => Compression::fast(),
                Some("best") => Compression::best(),
                _ => Compression::default(),
            };
            
            let mut encoder = GzEncoder::new(Vec::new(), compression);
            encoder.write_all(&json)?;
            let compressed = encoder.finish()?;
            
            if compressed.len() > max_size {
                eprintln!("⚠️  Warning: Compressed diagnostic size ({} bytes) exceeds limit ({} bytes)", 
                    compressed.len(), max_size);
            }
            
            write_locked(&path, &compressed)?;
        }
        
        #[cfg(feature = "prove-diagnostics-cbor")]
        DiagnosticFormat::Cbor => {
            let cbor = serde_cbor::to_vec(&diagnostic)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            
            if cbor.len() > max_size {
                eprintln!("⚠️  Warning: CBOR diagnostic size ({} bytes) exceeds limit ({} bytes)", 
                    cbor.len(), max_size);
            }
            
            write_locked(&path, &cbor)?;
        }
    }
    
    eprintln!("📊 Constraint diagnostic captured: {}", path.display());
    
    Ok(path)
}

/// Write file with restricted permissions (0o600 on Unix)
/// This prevents accidental exposure of diagnostic data containing witness information
#[cfg(unix)]
fn write_locked(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(0o600)  // Owner read/write only
        .open(path)?;
    f.write_all(bytes)
}

#[cfg(not(unix))]
fn write_locked(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    // On non-Unix platforms, use standard write
    // TODO: Add Windows ACL restrictions if needed
    std::fs::write(path, bytes)
}

/// Load diagnostic from file (auto-detects format)
pub fn load_diagnostic(path: &Path) -> Result<ConstraintDiagnostic, std::io::Error> {
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    
    match extension {
        "json" => {
            let json = std::fs::read_to_string(path)?;
            serde_json::from_str(&json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }
        
        "gz" => {
            use flate2::read::GzDecoder;
            
            let file = std::fs::File::open(path)?;
            let mut decoder = GzDecoder::new(file);
            let mut json = String::new();
            decoder.read_to_string(&mut json)?;
            
            serde_json::from_str(&json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }
        
        #[cfg(feature = "prove-diagnostics-cbor")]
        "cbor" => {
            let cbor = std::fs::read(path)?;
            serde_cbor::from_slice(&cbor)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }
        
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Unknown diagnostic format: .{}", extension),
        ))
    }
}

