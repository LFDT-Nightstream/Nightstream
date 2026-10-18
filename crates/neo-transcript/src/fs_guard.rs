#![cfg(feature = "fs-guard")]

use core::cell::RefCell;

#[derive(Clone, Debug)]
pub struct RecordedEvent(pub crate::debug::Event);

thread_local! {
    static GLOBAL: RefCell<Vec<crate::debug::Event>> = RefCell::new(Vec::new());
}

pub fn reset(tag: &str) {
    std::env::set_var("NEO_TRANSCRIPT_TAG", tag);
    GLOBAL.with(|g| g.borrow_mut().clear());
}

pub fn record(evt: crate::debug::Event) {
    GLOBAL.with(|g| g.borrow_mut().push(evt));
}

pub fn take() -> Vec<crate::debug::Event> {
    GLOBAL.with(|g| core::mem::take(&mut *g.borrow_mut()))
}
