//! Strictly loads one Lean-emitted package path and reports its identity and
//! validated shape. This tool does not build matrices or invoke a backend.

use std::{env, fs, io, path::PathBuf};

use nightstream_fprime::{load, PackageError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let path = PathBuf::from(
        arguments
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "usage: inspect-package <path>"))?,
    );
    if arguments.next().is_some() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "usage: inspect-package <path>").into());
    }

    let bytes = fs::read(path)?;
    let identity = match load(&bytes, [0; 4]) {
        Ok(package) => package.relation_identifier(),
        Err(PackageError::ExpectedIdentityMismatch { computed, .. }) => computed,
        Err(error) => return Err(error.into()),
    };
    let package = load(&bytes, identity)?;

    println!("identity={identity:?}");
    println!("rows={}", package.row_count());
    println!("private_columns={}", package.private_column_count());
    println!("private_inputs={}", package.private_input_count());
    println!("public_columns={}", package.public_column_count());
    println!("total_columns={}", package.total_column_count());
    println!("compact_templates={}", package.compact_template_count());
    println!("permutation_invocations={}", package.permutation_invocation_count());
    println!("compact_invocations={}", package.compact_invocation_count());
    println!("witness_instructions={}", package.witness_instruction_count());
    println!("assertion_rows={}", package.assertion_row_count());
    Ok(())
}
