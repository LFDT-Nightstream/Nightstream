//! Private files and independent bounded cursors for application records.

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, FileExt, OpenOptionsExt};
#[cfg(not(unix))]
use std::sync::Mutex;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::PackageError;

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct Cleanup(Option<(PathBuf, PathBuf)>);

impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some((file, directory)) = &self.0 {
            let _ = fs::remove_file(file);
            let _ = fs::remove_dir(directory);
        }
    }
}

fn private_file() -> io::Result<(File, File, Cleanup)> {
    loop {
        let ordinal = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!("nightstream-application-{}-{ordinal}", std::process::id()));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        builder.mode(0o700);
        match builder.create(&directory) {
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
            Ok(()) => {}
        }
        let path = directory.join("records");
        #[allow(unused_mut)]
        let mut cleanup = Cleanup(Some((path.clone(), directory.clone())));
        let mut options = OpenOptions::new();
        options.create_new(true).read(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);
        let writer = options.open(&path)?;
        let reader = File::open(&path)?;
        #[cfg(unix)]
        {
            fs::remove_file(&path)?;
            fs::remove_dir(&directory)?;
            cleanup.0 = None;
        }
        // Other platforms retain private paths until the final handles close.
        // Their fallback does not promise cleanup after process termination.
        return Ok((writer, reader, cleanup));
    }
}

pub(super) struct Writer {
    output: BufWriter<File>,
    reader: File,
    cleanup: Cleanup,
    length: u64,
}

impl Writer {
    pub(super) fn new() -> Result<Self, PackageError> {
        let (writer, reader, cleanup) = private_file()?;
        Ok(Self {
            output: BufWriter::new(writer),
            reader,
            cleanup,
            length: 0,
        })
    }

    pub(super) fn position(&self) -> u64 {
        self.length
    }

    #[cfg(test)]
    pub(super) fn buffer_bytes(&self) -> usize {
        self.output.capacity()
    }

    #[cfg(test)]
    pub(super) fn truncate_for_test(&mut self) -> Result<(), PackageError> {
        self.output.flush()?;
        self.output.get_mut().set_len(self.length - 1)?;
        Ok(())
    }

    pub(super) fn word(&mut self, value: u64) -> Result<(), PackageError> {
        let next = self
            .length
            .checked_add(size_of::<u64>() as u64)
            .ok_or(PackageError::Invalid("application record offset overflow"))?;
        self.output.write_all(&value.to_le_bytes())?;
        self.length = next;
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Reader, PackageError> {
        self.output.flush()?;
        drop(self.output);
        if self.reader.metadata()?.len() != self.length {
            return Err(PackageError::Invalid("application record length differs"));
        }
        Ok(Reader {
            #[cfg(unix)]
            file: self.reader,
            #[cfg(not(unix))]
            file: Mutex::new(self.reader),
            _cleanup: self.cleanup,
            length: self.length,
        })
    }
}

impl Write for Writer {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let bytes = u64::try_from(buffer.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "private snapshot length overflow"))?;
        if self.length.checked_add(bytes).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "private snapshot length overflow",
            ));
        }
        let written = self.output.write(buffer)?;
        self.length += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

/// Crate-owned immutable bytes; serialization never retains the source tree.
pub(crate) struct PrivateSnapshot(Reader);

impl std::fmt::Debug for PrivateSnapshot {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output
            .debug_struct("PrivateSnapshot")
            .field("bytes", &self.len())
            .finish()
    }
}

impl PrivateSnapshot {
    pub(crate) fn create(write: impl FnOnce(&mut dyn Write) -> Result<(), PackageError>) -> Result<Self, PackageError> {
        let mut writer = Writer::new()?;
        write(&mut writer)?;
        Ok(Self(writer.finish()?))
    }

    pub(crate) fn len(&self) -> u64 {
        self.0.length
    }

    pub(crate) fn copy_to(&self, output: &mut (impl Write + ?Sized)) -> Result<(), PackageError> {
        let mut input = self.0.cursor(0, self.len())?;
        if io::copy(&mut input, output)? != self.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "private snapshot is truncated").into());
        }
        Ok(())
    }

    pub(crate) fn with_reader<T>(
        &self,
        read: impl FnOnce(&mut dyn Read) -> Result<T, PackageError>,
    ) -> Result<T, PackageError> {
        let mut input = self.0.cursor(0, self.len())?;
        read(&mut input)
    }
}

pub(super) struct Reader {
    #[cfg(unix)]
    file: File,
    #[cfg(not(unix))]
    file: Mutex<File>,
    _cleanup: Cleanup,
    length: u64,
}

impl Reader {
    pub(super) fn tail(&self, offset: u64) -> Result<BufReader<Cursor<'_>>, PackageError> {
        let bytes = self
            .length
            .checked_sub(offset)
            .ok_or(PackageError::Invalid("application record offset exceeds file"))?;
        self.cursor(offset, bytes)
    }

    #[cfg(test)]
    pub(super) fn rejects_writes(&self) -> bool {
        #[cfg(unix)]
        {
            self.file.set_len(0).is_err()
        }
        #[cfg(not(unix))]
        {
            self.file.lock().unwrap().set_len(0).is_err()
        }
    }

    pub(super) fn cursor(&self, offset: u64, bytes: u64) -> Result<BufReader<Cursor<'_>>, PackageError> {
        let end = offset
            .checked_add(bytes)
            .filter(|&end| end <= self.length)
            .ok_or(PackageError::Invalid("application record range exceeds file"))?;
        // Use the standard library's buffer policy. No row-sized buffer is kept.
        Ok(BufReader::new(Cursor {
            reader: self,
            position: offset,
            end,
        }))
    }

    pub(super) fn word(&self, offset: u64) -> Result<u64, PackageError> {
        read_word(&mut Cursor {
            reader: self,
            position: offset,
            end: offset
                .checked_add(8)
                .filter(|&end| end <= self.length)
                .ok_or(PackageError::Invalid("application index exceeds file"))?,
        })
    }
}

pub(super) struct Cursor<'a> {
    reader: &'a Reader,
    position: u64,
    end: u64,
}

impl Read for Cursor<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let len = buffer
            .len()
            .min(usize::try_from(self.end - self.position).unwrap_or(usize::MAX));
        #[cfg(unix)]
        let count = self
            .reader
            .file
            .read_at(&mut buffer[..len], self.position)?;
        #[cfg(not(unix))]
        let count = {
            let mut file = self
                .reader
                .file
                .lock()
                .map_err(|_| io::Error::other("application reader lock poisoned"))?;
            file.seek(SeekFrom::Start(self.position))?;
            file.read(&mut buffer[..len])?
        };
        self.position += count as u64;
        Ok(count)
    }
}

pub(super) fn read_word(input: &mut (impl Read + ?Sized)) -> Result<u64, PackageError> {
    let mut word = [0; size_of::<u64>()];
    input.read_exact(&mut word)?;
    Ok(u64::from_le_bytes(word))
}

/// A buffered continuation stack. Its resident block follows the standard
/// library I/O buffer policy; only deeper frames spill to the private file.
pub(super) struct RecipeStack {
    file: File,
    _reader: File,
    _cleanup: Cleanup,
    spilled: u64,
    frames: Vec<[u64; 3]>,
    block_frames: usize,
}

impl RecipeStack {
    pub(super) fn new() -> Result<Self, PackageError> {
        let (file, reader, cleanup) = private_file()?;
        let block_frames = (BufReader::new(&file).capacity() / size_of::<[u64; 3]>()).max(1);
        Ok(Self {
            file,
            _reader: reader,
            _cleanup: cleanup,
            spilled: 0,
            frames: Vec::with_capacity(block_frames),
            block_frames,
        })
    }

    pub(super) fn reset(&mut self) {
        self.spilled = 0;
        self.frames.clear();
    }

    pub(super) fn push(&mut self, frame: [u64; 3]) -> Result<(), PackageError> {
        if self.frames.len() == self.block_frames {
            let offset = self
                .spilled
                .checked_mul(size_of::<[u64; 3]>() as u64)
                .ok_or(PackageError::Invalid("application recipe stack overflow"))?;
            let next = self
                .spilled
                .checked_add(self.frames.len() as u64)
                .ok_or(PackageError::Invalid("application recipe stack overflow"))?;
            let bytes: Vec<_> = self
                .frames
                .iter()
                .flatten()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            self.file.seek(SeekFrom::Start(offset))?;
            self.file.write_all(&bytes)?;
            self.spilled = next;
            self.frames.clear();
        }
        self.frames.push(frame);
        Ok(())
    }

    pub(super) fn pop(&mut self) -> Result<Option<[u64; 3]>, PackageError> {
        if self.frames.is_empty() && self.spilled != 0 {
            let count = self.spilled.min(self.block_frames as u64) as usize;
            let start = self.spilled - count as u64;
            let offset = start
                .checked_mul(size_of::<[u64; 3]>() as u64)
                .ok_or(PackageError::Invalid("application recipe stack overflow"))?;
            self.file.seek(SeekFrom::Start(offset))?;
            let mut bytes = vec![0; count * size_of::<[u64; 3]>()];
            self.file.read_exact(&mut bytes)?;
            self.frames
                .extend(bytes.chunks_exact(size_of::<[u64; 3]>()).map(|frame| {
                    std::array::from_fn(|index| {
                        u64::from_le_bytes(
                            frame[index * 8..index * 8 + 8]
                                .try_into()
                                .expect("complete stack word"),
                        )
                    })
                }));
            self.spilled = start;
        }
        Ok(self.frames.pop())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/recipe_stack.rs"]
mod tests;
