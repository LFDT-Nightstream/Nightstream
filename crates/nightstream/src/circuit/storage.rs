//! Versioned package framing and atomic publication. Trust is chosen by the caller.

use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};

use nightstream_fprime::{load_compiled_application_package, PackageError};

use super::{ApplicationCircuit, CompiledCircuit, Error};

const MAGIC: &[u8; 8] = b"NSCIR\0\0\x01";
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct PendingFile {
    directory: PathBuf,
    path: PathBuf,
}

impl PendingFile {
    fn create(parent: &Path) -> std::io::Result<(Self, File)> {
        loop {
            let ordinal = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
            let directory = parent.join(format!(".nightstream-package-{}-{ordinal}", std::process::id()));
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            builder.mode(0o700);
            match builder.create(&directory) {
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
                Ok(()) => {}
            }
            let pending = Self {
                path: directory.join("package"),
                directory,
            };
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            let file = options.open(&pending.path)?;
            return Ok((pending, file));
        }
    }
}

impl Drop for PendingFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        let _ = fs::remove_dir(&self.directory);
    }
}

pub(super) fn write(path: &Path, circuit: &CompiledCircuit) -> Result<(), Error> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let (pending, file) = PendingFile::create(parent)?;
    let mut output = BufWriter::new(file);
    output.write_all(MAGIC)?;
    output.write_all(&circuit.application.prepared_output_forms()?)?;
    circuit.package.write_prepared(&mut output)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    // Same-directory hard linking publishes the complete file without replacing
    // an existing destination. The private temporary name is removed on drop.
    fs::hard_link(&pending.path, path)?;
    Ok(())
}

pub(super) fn read(path: &Path) -> Result<CompiledCircuit, Error> {
    let mut input = BufReader::new(File::open(path)?);
    let mut magic = [0; MAGIC.len()];
    input.read_exact(&mut magic)?;
    if magic != *MAGIC {
        return Err(PackageError::Invalid("compiled circuit file version").into());
    }
    let mut output_forms = [0; 4];
    input.read_exact(&mut output_forms)?;
    if output_forms.iter().any(|tag| !matches!(tag, 0 | 2)) {
        return Err(PackageError::Invalid("prepared output form").into());
    }
    let package = load_compiled_application_package(&mut input)?;
    let binding = package.production_verifier_binding()?;
    crate::lifecycle::validate_key_prefix(
        package.logical_column_count(),
        binding.verifier_context().commitment_key_words(),
    )?;
    let application = ApplicationCircuit::from_prepared(&package, output_forms)?;
    Ok(CompiledCircuit {
        application,
        package: Arc::new(package),
        binding,
    })
}
