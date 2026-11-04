//! Unlinked scratch files in the system's temporary directory. A prover keeps
//! a large table there between the step that writes it and the step that
//! reads it back, so the table costs disk, not RAM, in between. Words are
//! stored in native byte order: a file never outlives its process.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

/// Reads and writes go in blocks of this many words, in parallel.
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
        .try_for_each(|(block, chunk)| file.write_all_at(bytes(chunk), (8 * (offset + block * BLOCK_WORDS)) as u64))
}

/// Fill `words` from word `offset`.
pub(crate) fn read(file: &File, offset: usize, words: &mut [u64]) -> io::Result<()> {
    words
        .par_chunks_mut(BLOCK_WORDS)
        .enumerate()
        .try_for_each(|(block, chunk)| {
            file.read_exact_at(bytes_mut(chunk), (8 * (offset + block * BLOCK_WORDS)) as u64)
        })
}

fn bytes(words: &[u64]) -> &[u8] {
    // SAFETY: a `u64` is eight initialized bytes, and `u8` has alignment 1.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 8 * words.len()) }
}

fn bytes_mut(words: &mut [u64]) -> &mut [u8] {
    // SAFETY: as in `bytes`, and every byte pattern is a valid `u64`.
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), 8 * words.len()) }
}
