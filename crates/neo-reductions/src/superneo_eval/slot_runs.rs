#[derive(Clone, Copy)]
pub struct SlotRun {
    pub end: u32,
    pub slot: u32,
    pub width: u8,
}

#[derive(Clone)]
pub struct SlotRuns {
    runs: Vec<SlotRun>,
    pub widths: [bool; 256],
}

impl SlotRuns {
    pub fn new(starts: &[u32], widths: &[u8]) -> Option<Self> {
        let mut runs: Vec<SlotRun> = Vec::new();
        let mut seen = [false; 256];
        let mut end = 0;
        let mut old = 0;
        for (slot, (&start, &w)) in starts.iter().zip(widths).enumerate() {
            if w != 0 {
                if start != end {
                    return None;
                }
                end = start.checked_add(u32::from(w))?;
                if w != old {
                    if runs.len() > starts.len() / 8 + 1 {
                        return None;
                    }
                    seen[w as usize] = true;
                    runs.push(SlotRun {
                        end,
                        slot: u32::try_from(slot).ok()?,
                        width: w,
                    });
                } else {
                    runs.last_mut().unwrap().end = end;
                }
            }
            old = w;
        }
        runs.shrink_to_fit();
        Some(Self { runs, widths: seen })
    }
    pub fn slices(&self, first: usize, end: usize, mut emit: impl FnMut(usize, SlotRun)) {
        let start = self.runs.partition_point(|r| r.end as usize <= first);
        let mut col = start
            .checked_sub(1)
            .map_or(0, |i| self.runs[i].end as usize);
        for &run in &self.runs[start..] {
            if col >= end {
                break;
            }
            emit(col, run);
            col = run.end as usize;
        }
    }

    pub(super) fn last_column(&self) -> usize {
        self.runs.last().map_or(0, |r| r.end as usize)
    }
    pub(super) fn storage_bytes(&self) -> usize {
        self.runs.capacity() * size_of::<SlotRun>() + size_of::<Self>()
    }
    pub(super) fn slots_in(&self, first: usize, end: usize, slot_count: usize) -> std::ops::Range<usize> {
        let lookup = |index: usize, column: usize| {
            let start = index
                .checked_sub(1)
                .map_or(0, |i| self.runs[i].end as usize);
            self.runs[index].slot as usize + (column - start) / usize::from(self.runs[index].width)
        };
        let end = end.min(self.last_column());
        if first >= end {
            return slot_count..slot_count;
        }
        let lo = self.runs.partition_point(|x| (x.end as usize) <= first);
        let hi = self.runs.partition_point(|x| (x.end as usize) < end);
        lookup(lo, first)..lookup(hi, end - 1) + 1
    }
}
