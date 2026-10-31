//! Versioned package framing and atomic publication. Trust is chosen by the caller.
//! A file is the magic, the four output forms, the four output rows (each a
//! little-endian `u64`), then the prepared package. The magic also names a
//! memory relation, so a loaded memory circuit still configures only
//! `nebula::Verifier`.

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

use nightstream_fprime::load_prepared_package;

use super::{ApplicationCircuit, ApplicationError, CompiledCircuit, Error, Role};

const MAGIC: &[u8; 8] = b"NSCIR\0\0\x02";
const MEMORY_MAGIC: &[u8; 8] = b"NSMEM\0\0\x02";
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
    output.write_all(match circuit.role {
        Role::Stage1 => MAGIC,
        Role::Memory => MEMORY_MAGIC,
    })?;
    let application = circuit
        .application
        .as_ref()
        .ok_or(ApplicationError::NoApplication)?;
    output.write_all(&application.prepared_output_forms()?)?;
    for row in application.prepared_output_rows() {
        output.write_all(&(row as u64).to_le_bytes())?;
    }
    circuit.package.write_prepared(&mut output)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    // Same-directory hard linking publishes the complete file without replacing
    // an existing destination.
    fs::hard_link(&pending.path, path)?;
    Ok(())
}

pub(super) fn read(path: &Path) -> Result<CompiledCircuit, Error> {
    let mut input = BufReader::new(File::open(path)?);
    let mut magic = [0; MAGIC.len()];
    input.read_exact(&mut magic)?;
    let role = match &magic {
        MAGIC => Role::Stage1,
        MEMORY_MAGIC => Role::Memory,
        _ => return Err(Error::Circuit("compiled circuit file version")),
    };
    let mut output_forms = [0; 4];
    input.read_exact(&mut output_forms)?;
    if output_forms.iter().any(|tag| !matches!(tag, 0 | 2)) {
        return Err(ApplicationError::Prepared("prepared output form").into());
    }
    let mut output_rows = [0; 4];
    for row in &mut output_rows {
        let mut bytes = [0; 8];
        input.read_exact(&mut bytes)?;
        *row = usize::try_from(u64::from_le_bytes(bytes))
            .map_err(|_| ApplicationError::Prepared("prepared output rows"))?;
    }
    let package = load_prepared_package(&mut input)?;
    let binding = package.production_verifier_binding()?;
    crate::lifecycle::validate_key_prefix(
        package.logical_column_count(),
        binding.verifier_context().commitment_key_words(),
    )?;
    let application = ApplicationCircuit::from_prepared(&package, output_forms, output_rows)?;
    Ok(CompiledCircuit {
        role,
        application: Some(application),
        package: Arc::new(package),
        binding,
        matrix_window: Arc::default(),
    })
}
