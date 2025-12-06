#![cfg(feature = "fs-guard")]

use core::cell::RefCell;

#[derive(Clone, Debug)]
pub struct RecordedEvent(pub crate::debug::Event);

thread_local! {
    static GLOBAL: RefCell<Vec<crate::debug::Event>> = RefCell::new(Vec::new());
}

pub fn reset(tag: &str) {
    // Optional tag for dumps by the transcript
    std::env::set_var("NEO_TRANSCRIPT_TAG", tag);
    GLOBAL.with(|g| g.borrow_mut().clear());
}

pub fn record(evt: crate::debug::Event) {
    GLOBAL.with(|g| g.borrow_mut().push(evt));
}

pub fn take() -> Vec<crate::debug::Event> {
    GLOBAL.with(|g| core::mem::take(&mut *g.borrow_mut()))
}

/// Return the first mismatch (index, spec, actual)
pub fn first_mismatch<'a>(
    spec: &'a [crate::debug::Event],
    actual: &'a [crate::debug::Event],
) -> Option<(usize, &'a crate::debug::Event, &'a crate::debug::Event)> {
    let n = core::cmp::min(spec.len(), actual.len());
    for i in 0..n {
        let a = &spec[i];
        let b = &actual[i];
        if a.op != b.op || a.label != b.label || a.len != b.len {
            return Some((i, a, b));
        }
    }
    if spec.len() != actual.len() {
        let i = n;
        let a = spec.get(i).unwrap_or(&crate::debug::Event {
            op: "∅",
            label: b"",
            len: 0,
            st_prefix: [0; 4],
        });
        let b = actual.get(i).unwrap_or(&crate::debug::Event {
            op: "∅",
            label: b"",
            len: 0,
            st_prefix: [0; 4],
        });
        return Some((i, a, b));
    }
    None
}
