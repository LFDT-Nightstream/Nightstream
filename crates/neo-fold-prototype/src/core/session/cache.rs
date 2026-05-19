//! Owns optimized-structure cache selection for proof sessions.

use neo_ccs::CcsStructure;
use neo_math::F;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;

pub(super) fn maybe_build_optimized_cache(
    mode: &FoldingMode,
    s: &CcsStructure<F>,
    provided: Option<&OptimizedStructureCache>,
) -> Result<Option<OptimizedStructureCache>, PiCcsError> {
    if matches!(mode, FoldingMode::Optimized) && provided.is_none() {
        Ok(Some(OptimizedStructureCache::build(s)?))
    } else {
        Ok(None)
    }
}
