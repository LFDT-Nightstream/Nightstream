#[inline]
pub(crate) fn parallelism_enabled() -> bool {
  cfg!(not(target_arch = "wasm32")) || cfg!(feature = "wasm-threads")
}

#[inline]
pub(crate) fn in_rayon_ctx() -> bool {
  parallelism_enabled() && rayon::current_thread_index().is_some()
}

#[inline]
pub(crate) fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
  A: FnOnce() -> RA + Send,
  B: FnOnce() -> RB + Send,
  RA: Send,
  RB: Send,
{
  if parallelism_enabled() {
    rayon::join(a, b)
  } else {
    (a(), b())
  }
}
