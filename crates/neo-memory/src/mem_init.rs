use neo_reductions::error::PiCcsError;
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// Public initial memory state for a Twist instance.
///
/// This is intentionally compact to support large memories without requiring a dense `Vec<F>`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MemInit<F> {
    /// All cells start at 0.
    Zero,
    /// A small set of non-zero initial cells (addr -> value).
    ///
    /// Canonical form: strictly increasing addresses, no duplicates, and all addresses < `k`.
    Sparse(Vec<(u64, F)>),
}

impl<F> Default for MemInit<F> {
    fn default() -> Self {
        Self::Zero
    }
}

impl<F: PrimeCharacteristicRing> MemInit<F> {
    pub fn validate(&self, k: usize) -> Result<(), PiCcsError> {
        match self {
            MemInit::Zero => Ok(()),
            MemInit::Sparse(pairs) => {
                let mut seen = BTreeSet::<u64>::new();
                let mut prev: Option<u64> = None;
                for (addr, _val) in pairs {
                    if let Some(prev) = prev {
                        if *addr <= prev {
                            return Err(PiCcsError::InvalidInput(
                                "MemInit::Sparse must be strictly increasing by address".into(),
                            ));
                        }
                    }
                    prev = Some(*addr);
                    let addr_usize = usize::try_from(*addr).map_err(|_| {
                        PiCcsError::InvalidInput(format!("MemInit::Sparse address doesn't fit usize: addr={addr}"))
                    })?;
                    if addr_usize >= k {
                        return Err(PiCcsError::InvalidInput(format!(
                            "MemInit::Sparse address out of range: addr={} >= k={}",
                            addr, k
                        )));
                    }
                    if !seen.insert(*addr) {
                        return Err(PiCcsError::InvalidInput(
                            "MemInit::Sparse must not contain duplicate addresses".into(),
                        ));
                    }
                }
                Ok(())
            }
        }
    }
}

pub fn mem_init_from_state_map<F: PrimeCharacteristicRing + Copy + PartialEq>(
    mem_id: u32,
    k: usize,
    state: &HashMap<u64, F>,
) -> Result<MemInit<F>, PiCcsError> {
    if state.is_empty() {
        return Ok(MemInit::Zero);
    }

    let mut pairs: Vec<(u64, F)> = state
        .iter()
        .filter_map(|(&addr, &val)| (val != F::ZERO).then_some((addr, val)))
        .collect();
    pairs.sort_by_key(|(addr, _)| *addr);

    if pairs.is_empty() {
        return Ok(MemInit::Zero);
    }

    if let Some((addr, _)) = pairs.last() {
        let addr_usize = usize::try_from(*addr).map_err(|_| {
            PiCcsError::InvalidInput(format!(
                "mem_id={mem_id}: MemInit address doesn't fit usize: addr={addr}"
            ))
        })?;
        if addr_usize >= k {
            return Err(PiCcsError::InvalidInput(format!(
                "mem_id={mem_id}: MemInit address out of range: addr={addr} >= k={k}"
            )));
        }
    }

    Ok(MemInit::Sparse(pairs))
}

pub fn mem_init_from_initial_mem<F: PrimeCharacteristicRing + Copy + PartialEq>(
    mem_id: u32,
    k: usize,
    initial_mem: &HashMap<(u32, u64), F>,
) -> Result<MemInit<F>, PiCcsError> {
    let mut state: HashMap<u64, F> = HashMap::new();

    for ((init_mem_id, addr), &val) in initial_mem.iter() {
        if *init_mem_id != mem_id || val == F::ZERO {
            continue;
        }

        let addr_usize = usize::try_from(*addr).map_err(|_| {
            PiCcsError::InvalidInput(format!(
                "mem_id={mem_id}: MemInit address doesn't fit usize: addr={addr}"
            ))
        })?;
        if addr_usize >= k {
            return Err(PiCcsError::InvalidInput(format!(
                "mem_id={mem_id}: MemInit address out of range: addr={addr} >= k={k}"
            )));
        }

        if state.insert(*addr, val).is_some() {
            return Err(PiCcsError::InvalidInput(format!(
                "mem_id={mem_id}: MemInit must not contain duplicate addresses: addr={addr}"
            )));
        }
    }

    mem_init_from_state_map(mem_id, k, &state)
}

/// Evaluate the multilinear extension of the initial memory table at `r_addr`.
///
/// For `Sparse`, this runs in `O(nnz * ell_addr)` time.
pub fn eval_init_at_r_addr<F, K>(init: &MemInit<F>, k: usize, r_addr: &[K]) -> Result<K, PiCcsError>
where
    F: PrimeCharacteristicRing + Copy,
    K: PrimeCharacteristicRing + From<F> + Copy,
{
    init.validate(k)?;

    match init {
        MemInit::Zero => Ok(K::ZERO),
        MemInit::Sparse(pairs) => {
            if r_addr.len() > 64 && !pairs.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "MemInit::Sparse only supports up to 64 address bits (got ell_addr={})",
                    r_addr.len()
                )));
            }

            let one = K::ONE;
            let mut acc = K::ZERO;

            for (addr, val_f) in pairs.iter() {
                let mut chi = one;
                for (bit_idx, &r) in r_addr.iter().enumerate() {
                    let bit = ((*addr >> bit_idx) & 1) as u8;
                    chi *= if bit == 1 { r } else { one - r };
                }
                acc += K::from(*val_f) * chi;
            }

            Ok(acc)
        }
    }
}
