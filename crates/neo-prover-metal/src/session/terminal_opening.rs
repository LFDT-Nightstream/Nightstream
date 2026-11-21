use super::*;
use neo_math::{field::Q, KExtensions, D, F, K};
use neo_reductions::superneo_eval::{OpeningMatrix, OpeningWitness};
use objc2_metal::MTLCommandEncoder;
use std::{marker::PhantomData, ptr::NonNull};

const TILE: usize = 16;
pub(super) struct Kernels {
    weights: Pipeline,
    heavy_weights: Pipeline,
    heavy_merge: Pipeline,
    opening: Pipeline,
    reduce: Pipeline,
    finish: Pipeline,
}

impl Kernels {
    pub(super) fn new(device: &Device, library: &ProtocolObject<dyn MTLLibrary>) -> Option<Self> {
        let get = |name, threads, simd: bool| {
            pipeline(device, library, name)
                .ok()
                .filter(|p| p.maxTotalThreadsPerThreadgroup() >= threads && (!simd || p.threadExecutionWidth() == 32))
        };
        Some(Self {
            weights: get("terminal_weights", 128, false)?,
            heavy_weights: get("terminal_heavy_weights", 32, true)?,
            heavy_merge: get("terminal_heavy_merge", 32, true)?,
            opening: get("terminal_opening", D, true)?,
            reduce: get("terminal_partial_openings", D, false)?,
            finish: get("terminal_finish_opening", D, false)?,
        })
    }
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum WitnessFormat {
    FieldWords = 0,
    SignMasks = 1,
    SignedBytes = 2,
}

struct Config {
    shape: [u32; 5],
    format: WitnessFormat,
    columns: [u32; 8],
    witness: u32,
    negative: u32,
    first: u32,
    tasks: u32,
    heavy: u32,
}
impl Config {
    fn words(&self) -> [u32; 19] {
        let [slots, blocks, stride, tile, workers] = self.shape;
        let [offsets, rows, tags, factors, eq, occupied, starts, widths] = self.columns;
        [
            slots,
            blocks,
            stride,
            tile,
            workers,
            self.format as u32,
            offsets,
            rows,
            tags,
            factors,
            eq,
            occupied,
            starts,
            widths,
            self.witness,
            self.negative,
            self.first,
            self.tasks,
            self.heavy,
        ]
    }
}

struct Span<'a> {
    head: Buffer,
    tail: Buffer,
    count: u32,
    _source: PhantomData<&'a [u8]>,
}

struct Workspace<'a> {
    session: &'a MetalSession,
    remaining: usize,
    page: usize,
}

unsafe extern "C" {
    fn getpagesize() -> std::ffi::c_int;
}

impl Workspace<'_> {
    fn reserve(&mut self, bytes: usize) -> Option<()> {
        if bytes == 0 || bytes > self.session.device.maxBufferLength() {
            return None;
        }
        let charge = bytes.checked_next_multiple_of(self.page)?;
        self.remaining = self.remaining.checked_sub(charge)?;
        Some(())
    }

    fn scratch(&mut self, bytes: usize, options: MTLResourceOptions) -> Option<Buffer> {
        self.reserve(bytes)?;
        let _guard = self.session.reserve_buffer(bytes).ok()?;
        let buffer = self
            .session
            .device
            .newBufferWithLength_options(bytes, options)?;
        self.session
            .activity
            .allocated_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Some(buffer)
    }

    fn copy<T: Copy>(&mut self, data: &[T]) -> Option<Buffer> {
        self.reserve(size_of_val(data))?;
        self.session.buffer_from_slice(data).ok()
    }

    fn view<'a, T: Copy>(&mut self, data: &'a [T], word: usize) -> Option<Span<'a>> {
        let bytes = size_of_val(data);
        let count = u32::try_from(bytes.checked_div(word)?).ok()?;
        let size = size_of::<T>();
        if bytes == 0 || !bytes.is_multiple_of(word) || !bytes.is_multiple_of(size) {
            return None;
        }
        let ptr = NonNull::from(data).cast::<std::ffi::c_void>();
        let prefix = bytes / self.page * self.page;
        if prefix != 0
            && self.page.is_multiple_of(word)
            && prefix.is_multiple_of(size)
            && ptr.as_ptr().addr().is_multiple_of(self.page)
            && prefix <= self.session.device.maxBufferLength()
        {
            let guard = self.session.reserve_buffer(prefix).ok()?;
            // SAFETY: The page-aligned prefix belongs to the input slice. Kernels only read it.
            // The caller retains every source through command completion and no deallocator releases it.
            let head = unsafe {
                self.session
                    .device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        ptr,
                        prefix,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
            };
            drop(guard);
            if let Some(head) = head {
                let tail = if prefix == bytes {
                    head.clone()
                } else {
                    self.copy(&data[prefix / size..])?
                };
                return Some(Span {
                    head,
                    tail,
                    count: (prefix / word) as u32,
                    _source: PhantomData,
                });
            }
        }
        let head = self.copy(data)?;
        Some(Span {
            tail: head.clone(),
            head,
            count,
            _source: PhantomData,
        })
    }
}

impl MetalSession {
    pub(crate) fn supports_tiled_openings(&self, blocks: usize) -> bool {
        self.terminal.is_some() && blocks >= 2048
    }

    pub(crate) fn tiled_opening(
        &self,
        matrices: &[OpeningMatrix<'_>],
        weights: &[K],
        witnesses: &[OpeningWitness<'_>],
        budget: usize,
        result: &mut [[K; D]],
    ) -> bool {
        objc2::rc::autoreleasepool(|_| {
            self.try_tiled_opening(matrices, weights, witnesses, budget, result)
                .is_some()
        })
    }

    fn try_tiled_opening(
        &self,
        matrices: &[OpeningMatrix<'_>],
        weights: &[K],
        witnesses: &[OpeningWitness<'_>],
        budget: usize,
        result: &mut [[K; D]],
    ) -> Option<()> {
        let k = self.terminal.as_ref()?;
        let first_matrix = matrices.first()?;
        let layout = first_matrix.layout;
        let blocks = first_matrix.tiles.first.len().checked_sub(1)?;
        if weights.is_empty()
            || witnesses.is_empty()
            || witnesses.len().checked_mul(matrices.len())? != result.len()
            || blocks == 0
            || blocks > (u32::MAX as usize) / D
            || matrices.iter().any(|m| {
                m.row_count != weights.len()
                    || m.tiles.first.len() != blocks + 1
                    || m.tags.len() > u32::MAX as usize - 256
                    || m.rows.len() > u32::MAX as usize
                    || m.slots.len() >= u32::MAX as usize
                    || !std::ptr::eq(m.layout.starts, layout.starts)
                    || !std::ptr::eq(m.layout.widths, layout.widths)
                    || m.layout.ratio != layout.ratio
            })
            || witnesses.iter().any(|w| match w {
                OpeningWitness::Signed { positive, negative } => positive.len() != blocks || negative.len() != blocks,
                OpeningWitness::Dense(rows) => rows.len() != blocks,
                OpeningWitness::Centered(rows) => rows.len() != blocks,
            })
        {
            return None;
        }
        // SAFETY: Darwin's getpagesize has no preconditions.
        let page = usize::try_from(unsafe { getpagesize() })
            .ok()
            .filter(|&n| n > 0)?;
        let mut ws = Workspace {
            session: self,
            remaining: budget,
            page,
        };
        let eq = ws.view(weights, 16)?;
        let starts = ws.view(layout.starts, 4)?;
        let widths = ws.view(layout.widths, 1)?;
        let powers = ws.copy(&first_matrix.tiles.powers)?;
        let mut inputs = Vec::with_capacity(witnesses.len());
        for witness in witnesses {
            let input = match witness {
                OpeningWitness::Signed { positive, negative } => (
                    WitnessFormat::SignMasks,
                    ws.view(positive, 8)?,
                    Some(ws.view(negative, 8)?),
                ),
                OpeningWitness::Dense(rows) => (WitnessFormat::FieldWords, ws.view(rows, 8)?, None),
                OpeningWitness::Centered(rows) => (WitnessFormat::SignedBytes, ws.view(rows, 1)?, None),
            };
            inputs.push(input);
        }
        let workers = blocks.div_ceil(TILE);
        let groups = workers.div_ceil(D);
        let private = MTLResourceOptions::StorageModePrivate;
        let max_slots = matrices.iter().map(|m| m.slots.len()).max()?;
        let max_tasks = matrices.iter().map(|m| m.tiles.tasks.len()).max()?;
        let forms = ws.scratch(max_slots.checked_mul(16)?, private)?;
        let partial = ws.scratch(workers.checked_mul(D * 16)?, private)?;
        let tier = ws.scratch(groups.checked_mul(D * 16)?, private)?;
        let pieces = if max_tasks == 0 {
            None
        } else {
            Some(ws.scratch(max_tasks.checked_mul(16)?, private)?)
        };
        let answers = ws.scratch(size_of_val(result), MTLResourceOptions::StorageModeShared)?;
        let command = self.command_buffer("nightstream.terminal_opening").ok()?;
        let pass = |p: &Pipeline, args: &[Arg<'_>], groups, threads| -> Option<()> {
            let encoder = command.computeCommandEncoder()?;
            self.launch(&encoder, p, args, Grid::Groups { groups, threads });
            encoder.endEncoding();
            Some(())
        };
        use Arg::{Buffer as B, BufferAt, Words};
        for (matrix_index, m) in matrices.iter().enumerate() {
            let slots = m.slots.len();
            let offsets = ws.view(m.offsets, 4)?;
            let rows = ws.view(m.rows, 1)?;
            let tags = ws.view(m.tags, 2)?;
            let factors = ws.view(m.factors, 8)?;
            let occupied = ws.view(m.slots, 4)?;
            let first = ws.view(&m.tiles.first, 4)?;
            let mut cfg = Config {
                shape: [
                    slots as u32,
                    blocks as u32,
                    m.row_bytes as u32,
                    TILE as u32,
                    workers as u32,
                ],
                format: WitnessFormat::FieldWords,
                columns: [
                    offsets.count,
                    rows.count,
                    tags.count,
                    factors.count,
                    eq.count,
                    occupied.count,
                    starts.count,
                    widths.count,
                ],
                witness: 0,
                negative: 0,
                first: first.count,
                tasks: 0,
                heavy: 0,
            };
            pass(
                &k.weights,
                &[
                    B(0, &offsets.head),
                    B(1, &rows.head),
                    B(2, &tags.head),
                    B(3, &factors.head),
                    B(4, &eq.head),
                    B(5, &forms),
                    Words(6, &cfg.words()),
                    B(7, &offsets.tail),
                    B(8, &rows.tail),
                    B(9, &tags.tail),
                    B(10, &factors.tail),
                    B(11, &eq.tail),
                ],
                slots.div_ceil(128),
                128,
            )?;
            if !m.tiles.tasks.is_empty() {
                let tasks = ws.view(&m.tiles.tasks, 8)?;
                cfg.tasks = tasks.count;
                let heavy = ws.view(&m.tiles.heavy, 16)?;
                cfg.heavy = heavy.count;
                let pieces = pieces.as_ref()?;
                pass(
                    &k.heavy_weights,
                    &[
                        B(0, &tasks.head),
                        B(7, &tasks.tail),
                        B(1, &rows.head),
                        B(2, &tags.head),
                        B(3, &factors.head),
                        B(4, &eq.head),
                        B(5, pieces),
                        Words(6, &cfg.words()),
                        B(8, &rows.tail),
                        B(9, &tags.tail),
                        B(10, &factors.tail),
                        B(11, &eq.tail),
                    ],
                    m.tiles.tasks.len(),
                    32,
                )?;
                pass(
                    &k.heavy_merge,
                    &[
                        B(0, &heavy.head),
                        B(1, pieces),
                        B(2, &forms),
                        B(3, &heavy.tail),
                        Words(4, &cfg.words()),
                    ],
                    m.tiles.heavy.len(),
                    32,
                )?;
            }
            for (witness_index, (format, source, negative)) in inputs.iter().enumerate() {
                cfg.format = *format;
                let neg = negative.as_ref().unwrap_or(source);
                let index = matrix_index * inputs.len() + witness_index;
                cfg.witness = source.count;
                cfg.negative = neg.count;
                pass(
                    &k.opening,
                    &[
                        B(0, &first.head),
                        B(15, &first.tail),
                        B(1, &occupied.head),
                        B(2, &forms),
                        B(3, &starts.head),
                        B(4, &widths.head),
                        B(5, &source.head),
                        B(6, &neg.head),
                        B(7, &powers),
                        B(8, &partial),
                        Words(9, &cfg.words()),
                        B(10, &occupied.tail),
                        B(11, &starts.tail),
                        B(12, &widths.tail),
                        B(13, &source.tail),
                        B(14, &neg.tail),
                    ],
                    workers,
                    D,
                )?;
                pass(
                    &k.reduce,
                    &[B(0, &partial), B(1, &tier), Words(2, &cfg.words())],
                    groups,
                    D,
                )?;
                pass(
                    &k.finish,
                    &[
                        B(0, &tier),
                        BufferAt(1, &answers, index * D * 16),
                        Words(2, &cfg.words()),
                        B(3, &powers),
                    ],
                    1,
                    D,
                )?;
            }
        }
        self.finish(&command).ok()?;
        let raw = host_view::<[[u64; 2]; D]>(&answers, result.len());
        if raw.iter().flatten().flatten().any(|&word| word >= Q) {
            return None;
        }
        for (ring, words) in result.iter_mut().zip(raw) {
            *ring = words.map(|pair| K::from_coeffs(pair.map(F::new)));
        }
        self.activity
            .downloaded_bytes
            .fetch_add(size_of_val(result) as u64, Ordering::Relaxed);
        Some(())
    }
}
