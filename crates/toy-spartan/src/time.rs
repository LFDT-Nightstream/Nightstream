#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
mod wasm {
  use js_sys::Date;
  use std::time::Duration;

  #[derive(Clone, Copy, Debug)]
  pub(crate) struct Instant(f64);

  impl Instant {
    pub(crate) fn now() -> Self {
      Self(Date::now())
    }

    pub(crate) fn elapsed(&self) -> Duration {
      let ms = Date::now() - self.0;
      if ms <= 0.0 {
        Duration::from_millis(0)
      } else {
        Duration::from_secs_f64(ms / 1_000.0)
      }
    }
  }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use wasm::Instant;
