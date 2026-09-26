//! Strictly loads one Lean-emitted package path and reports its identity and
//! validated shape. This tool does not build matrices or invoke a backend.

use std::{env, fs, io, path::PathBuf};

use nightstream_fprime::load_per_application_package;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let first = arguments.next().ok_or_else(usage)?;
    if first == "--sealed" {
        return inspect_sealed(&mut arguments);
    }

    Err(usage().into())
}

fn inspect_sealed(arguments: &mut impl Iterator<Item = std::ffi::OsString>) -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(arguments.next().ok_or_else(usage)?);
    let mut expected = [0u64; 4];
    for word in &mut expected {
        *word = arguments
            .next()
            .ok_or_else(usage)?
            .into_string()
            .map_err(|_| usage())?
            .parse()
            .map_err(|_| usage())?;
    }
    if arguments.next().is_some() {
        return Err(usage().into());
    }

    let bytes = fs::read(path)?;
    let package = load_per_application_package(&bytes, expected)?;
    let binding = package.production_verifier_binding()?;
    println!("structural_identity={:?}", package.structural_identifier());
    println!("package_identity={:?}", binding.package_identity());
    println!("verifier_context={:?}", binding.verifier_context().digest());
    println!(
        "verifier_context_descriptor={:?}",
        binding.verifier_context().descriptor_words()
    );
    println!("verification_key_digest={:?}", binding.verification_key_digest());
    println!(
        "authority_lengths={:?}",
        [
            binding.verifier_context().relation_words().len(),
            binding.verifier_context().application_word_count(),
            binding.verifier_context().nifs_key_words().len(),
            binding.verifier_context().commitment_key_words().len(),
        ]
    );
    println!("logical_rows={}", package.row_count());
    println!("logical_columns={}", package.logical_column_count());
    println!("application_rows={:?}", package.application().row_range());
    println!("next_preimage_rows={:?}", package.next_preimage_row_range());
    Ok(())
}

fn usage() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "usage: inspect-package --sealed <path> <id0> <id1> <id2> <id3>",
    )
}
