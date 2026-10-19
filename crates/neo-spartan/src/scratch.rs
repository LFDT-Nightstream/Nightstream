//! Unlinked scratch files in the system's temporary directory. A prover keeps
//! a large table there between the step that writes it and the step that
//! reads it back, so the table costs disk, not RAM, in between.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

/// Reads and writes go in blocks of this many words; it bounds only each
/// thread's buffer.
pub(crate) const BLOCK_WORDS: usize = 1 << 19;

/// A fresh file whose name is already removed: its space returns when it
/// closes.
pub(crate) fn file() -> io::Result<File> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "neo-spartan-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    std::fs::remove_file(&path)?;
    Ok(file)
}

/// Write `words` at word `offset`.
pub(crate) fn write(file: &File, offset: usize, words: &[u64]) -> io::Result<()> {
    words
        .par_chunks(BLOCK_WORDS)
        .enumerate()
        .try_for_each(|(block, chunk)| {
            let bytes: Vec<u8> = chunk.iter().flat_map(|word| word.to_le_bytes()).collect();
            file.write_all_at(&bytes, (8 * (offset + block * BLOCK_WORDS)) as u64)
        })
}

/// Fill `words` from word `offset`.
pub(crate) fn read(file: &File, offset: usize, words: &mut [u64]) -> io::Result<()> {
    words
        .par_chunks_mut(BLOCK_WORDS)
        .enumerate()
        .try_for_each(|(block, chunk)| {
            let mut bytes = vec![0u8; 8 * chunk.len()];
            file.read_exact_at(&mut bytes, (8 * (offset + block * BLOCK_WORDS)) as u64)?;
            for (word, bytes) in chunk.iter_mut().zip(bytes.chunks_exact(8)) {
                *word = u64::from_le_bytes(bytes.try_into().expect("eight bytes"));
            }
            Ok(())
        })
}
