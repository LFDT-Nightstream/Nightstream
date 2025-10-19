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
    let dir = std::env::var("NEO_DIAGNOSTICS")
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
            
            std::fs::write(&path, json)?;
        }
        
        DiagnosticFormat::JsonGz => {
            use flate2::write::GzEncoder;
            use flate2::Compression;
            
            let json = serde_json::to_vec(&diagnostic)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            
            let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
            encoder.write_all(&json)?;
            let compressed = encoder.finish()?;
            
            if compressed.len() > max_size {
                eprintln!("⚠️  Warning: Compressed diagnostic size ({} bytes) exceeds limit ({} bytes)", 
                    compressed.len(), max_size);
            }
            
            std::fs::write(&path, compressed)?;
        }
        
        #[cfg(feature = "prove-diagnostics-cbor")]
        DiagnosticFormat::Cbor => {
            let cbor = serde_cbor::to_vec(&diagnostic)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            
            if cbor.len() > max_size {
                eprintln!("⚠️  Warning: CBOR diagnostic size ({} bytes) exceeds limit ({} bytes)", 
                    cbor.len(), max_size);
            }
            
            std::fs::write(&path, cbor)?;
        }
    }
    
    eprintln!("📊 Constraint diagnostic captured: {}", path.display());
    eprintln!("💡 Replay with: cargo run --bin neo-diag {}", path.display());
    
    Ok(path)
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

