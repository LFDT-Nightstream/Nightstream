#![cfg_attr(not(feature = "route-a-compiler-shadow"), allow(dead_code))]

use crate::memory_sidecar::claim_plan::TimeClaimMeta;
use crate::PiCcsError;
use std::collections::HashMap;

/// Route-A claim domain descriptor for canonical scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClaimDomain {
    TimeCycle,
}

impl ClaimDomain {
    #[inline]
    fn as_tag(self) -> u64 {
        match self {
            Self::TimeCycle => 1,
        }
    }
}

/// Primitive atom in the Route-A claim IR.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Atom {
    Label(&'static [u8]),
    DegreeBound(usize),
    DynamicFlag(bool),
    Domain(ClaimDomain),
}

/// Expression DAG id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Expr {
    Atom(Atom),
    Add(Vec<ExprId>),
    Mul(Vec<ExprId>),
    Neg(ExprId),
    Scale { c: i64, x: ExprId },
    Gate { pred: ExprId, body: ExprId },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ExprKey {
    Atom(Atom),
    Add(Vec<ExprId>),
    Mul(Vec<ExprId>),
    Neg(ExprId),
    Scale { c: i64, x: ExprId },
    Gate { pred: ExprId, body: ExprId },
}

#[derive(Clone, Debug)]
struct Node {
    expr: Expr,
    hash: u64,
}

/// Canonical claim descriptor emitted by the compiler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledClaim {
    pub label: &'static [u8],
    pub degree_bound: usize,
    pub is_dynamic: bool,
    pub domain: ClaimDomain,
    pub expr_hash: u64,
}

/// Deterministic compiled schedule and overall digest/hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledClaimSchedule {
    pub claims: Vec<CompiledClaim>,
    pub schedule_hash: u64,
    pub expr_nodes: usize,
}

#[derive(Default)]
struct ExprArena {
    nodes: Vec<Node>,
    intern: HashMap<ExprKey, ExprId>,
}

impl ExprArena {
    fn intern_or_push(&mut self, key: ExprKey, expr: Expr, hash: u64) -> ExprId {
        if let Some(id) = self.intern.get(&key).copied() {
            return id;
        }
        let id = ExprId(self.nodes.len());
        self.nodes.push(Node { expr, hash });
        self.intern.insert(key, id);
        id
    }

    #[inline]
    fn hash_of(&self, id: ExprId) -> u64 {
        self.nodes[id.0].hash
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn add_atom(&mut self, atom: Atom) -> ExprId {
        let hash = match &atom {
            Atom::Label(bytes) => hash_tagged(0xA001, &[hash_bytes(bytes)]),
            Atom::DegreeBound(v) => hash_tagged(0xA002, &[*v as u64]),
            Atom::DynamicFlag(v) => hash_tagged(0xA003, &[u64::from(*v)]),
            Atom::Domain(v) => hash_tagged(0xA004, &[v.as_tag()]),
        };
        self.intern_or_push(ExprKey::Atom(atom.clone()), Expr::Atom(atom), hash)
    }

    fn add_add(&mut self, children: Vec<ExprId>) -> ExprId {
        let mut flat: Vec<ExprId> = Vec::new();
        for child in children {
            match &self.nodes[child.0].expr {
                Expr::Add(inner) => flat.extend(inner.iter().copied()),
                _ => flat.push(child),
            }
        }
        if flat.len() == 1 {
            return flat[0];
        }
        flat.sort_by_key(|id| (self.hash_of(*id), id.0));
        let child_hashes: Vec<u64> = flat.iter().map(|id| self.hash_of(*id)).collect();
        let hash = hash_tagged(0xB001, &child_hashes);
        self.intern_or_push(ExprKey::Add(flat.clone()), Expr::Add(flat), hash)
    }

    fn add_mul(&mut self, children: Vec<ExprId>) -> ExprId {
        let mut flat: Vec<ExprId> = Vec::new();
        for child in children {
            match &self.nodes[child.0].expr {
                Expr::Mul(inner) => flat.extend(inner.iter().copied()),
                _ => flat.push(child),
            }
        }
        if flat.len() == 1 {
            return flat[0];
        }
        flat.sort_by_key(|id| (self.hash_of(*id), id.0));
        let child_hashes: Vec<u64> = flat.iter().map(|id| self.hash_of(*id)).collect();
        let hash = hash_tagged(0xB002, &child_hashes);
        self.intern_or_push(ExprKey::Mul(flat.clone()), Expr::Mul(flat), hash)
    }

    fn add_neg(&mut self, x: ExprId) -> ExprId {
        let hash = hash_tagged(0xB003, &[self.hash_of(x)]);
        self.intern_or_push(ExprKey::Neg(x), Expr::Neg(x), hash)
    }

    fn add_scale(&mut self, c: i64, x: ExprId) -> ExprId {
        let hash = hash_tagged(0xB004, &[c as u64, self.hash_of(x)]);
        self.intern_or_push(ExprKey::Scale { c, x }, Expr::Scale { c, x }, hash)
    }

    fn add_gate(&mut self, pred: ExprId, body: ExprId) -> ExprId {
        let hash = hash_tagged(0xB005, &[self.hash_of(pred), self.hash_of(body)]);
        self.intern_or_push(ExprKey::Gate { pred, body }, Expr::Gate { pred, body }, hash)
    }
}

/// Phase-1 Route-A claim compiler.
///
/// This compiler intentionally performs canonicalization and hashing only.
/// It does not alter claim semantics or schedule order.
pub struct RouteAClaimCompiler {
    arena: ExprArena,
}

impl Default for RouteAClaimCompiler {
    fn default() -> Self {
        Self {
            arena: ExprArena::default(),
        }
    }
}

impl RouteAClaimCompiler {
    fn compile_claim(
        &mut self,
        label: &'static [u8],
        degree_bound: usize,
        is_dynamic: bool,
        domain: ClaimDomain,
    ) -> CompiledClaim {
        // Phase-1 synthetic IR shape:
        // gate(dynamic, mul(label, degree_bound, domain))
        let a_label = self.arena.add_atom(Atom::Label(label));
        let a_degree = self.arena.add_atom(Atom::DegreeBound(degree_bound));
        let a_domain = self.arena.add_atom(Atom::Domain(domain));
        let a_dynamic = self.arena.add_atom(Atom::DynamicFlag(is_dynamic));

        let body = self.arena.add_mul(vec![a_label, a_degree, a_domain]);
        let gated = self.arena.add_gate(a_dynamic, body);
        // Exercise canonical ops in phase-1 to keep representation stable and tested.
        let gated_scaled = self.arena.add_scale(1, gated);
        let gated_zero = self.arena.add_scale(0, gated);
        let gated_zero_neg = self.arena.add_neg(gated_zero);
        let stabilized = self.arena.add_add(vec![gated_scaled, gated_zero_neg]);

        CompiledClaim {
            label,
            degree_bound,
            is_dynamic,
            domain,
            expr_hash: self.arena.hash_of(stabilized),
        }
    }

    #[allow(dead_code)]
    pub fn compile_schedule_from_metas(metas: &[TimeClaimMeta], domain: ClaimDomain) -> CompiledClaimSchedule {
        let enabled_mask = vec![true; metas.len()];
        Self::compile_schedule_from_metas_with_enable_mask(metas, &enabled_mask, domain)
            .expect("enabled mask length must match metas")
    }

    pub fn compile_schedule_from_metas_with_enable_mask(
        metas: &[TimeClaimMeta],
        enabled_mask: &[bool],
        domain: ClaimDomain,
    ) -> Result<CompiledClaimSchedule, PiCcsError> {
        if metas.len() != enabled_mask.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "route-a compiler enabled_mask length mismatch: metas={}, enabled_mask={}",
                metas.len(),
                enabled_mask.len()
            )));
        }
        let mut compiler = Self::default();
        let claims: Vec<CompiledClaim> = metas
            .iter()
            .zip(enabled_mask.iter().copied())
            .filter_map(|(meta, enabled)| {
                if !enabled {
                    return None;
                }
                Some(compiler.compile_claim(meta.label, meta.degree_bound, meta.is_dynamic, domain))
            })
            .collect();
        let schedule_hash = schedule_hash(&claims);
        Ok(CompiledClaimSchedule {
            claims,
            schedule_hash,
            expr_nodes: compiler.arena.node_count(),
        })
    }

    #[allow(dead_code)]
    pub fn compile_schedule_from_raw(
        labels: &[&'static [u8]],
        degree_bounds: &[usize],
        is_dynamic: &[bool],
        domain: ClaimDomain,
    ) -> Result<CompiledClaimSchedule, PiCcsError> {
        let enabled_mask = vec![true; labels.len()];
        Self::compile_schedule_from_raw_with_enable_mask(labels, degree_bounds, is_dynamic, &enabled_mask, domain)
    }

    pub fn compile_schedule_from_raw_with_enable_mask(
        labels: &[&'static [u8]],
        degree_bounds: &[usize],
        is_dynamic: &[bool],
        enabled_mask: &[bool],
        domain: ClaimDomain,
    ) -> Result<CompiledClaimSchedule, PiCcsError> {
        if labels.len() != enabled_mask.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "route-a compiler enabled_mask length mismatch: labels={}, enabled_mask={}",
                labels.len(),
                enabled_mask.len()
            )));
        }
        if labels.len() != degree_bounds.len() || labels.len() != is_dynamic.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "route-a compiler raw input length mismatch: labels={}, degree_bounds={}, is_dynamic={}",
                labels.len(),
                degree_bounds.len(),
                is_dynamic.len()
            )));
        }
        let mut compiler = Self::default();
        let claims: Vec<CompiledClaim> = labels
            .iter()
            .copied()
            .zip(degree_bounds.iter().copied())
            .zip(is_dynamic.iter().copied())
            .zip(enabled_mask.iter().copied())
            .filter_map(|(((label, degree_bound), is_dynamic), enabled)| {
                if !enabled {
                    return None;
                }
                Some(compiler.compile_claim(label, degree_bound, is_dynamic, domain))
            })
            .collect();
        let schedule_hash = schedule_hash(&claims);
        Ok(CompiledClaimSchedule {
            claims,
            schedule_hash,
            expr_nodes: compiler.arena.node_count(),
        })
    }
}

#[cfg(feature = "route-a-compiler-shadow")]
pub fn shadow_assert_compiled_schedule_matches_metas(
    labels: &[&'static [u8]],
    degree_bounds: &[usize],
    is_dynamic: &[bool],
    metas: &[TimeClaimMeta],
    context: &'static str,
) -> Result<(), PiCcsError> {
    if labels.len() != degree_bounds.len() || labels.len() != is_dynamic.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "{context}: route-a compiler shadow input length mismatch (labels={}, degree_bounds={}, is_dynamic={})",
            labels.len(),
            degree_bounds.len(),
            is_dynamic.len()
        )));
    }

    let mut raw_counts: HashMap<(&'static [u8], usize, bool), usize> = HashMap::new();
    for ((&label, &degree), &dyn_flag) in labels
        .iter()
        .zip(degree_bounds.iter())
        .zip(is_dynamic.iter())
    {
        *raw_counts.entry((label, degree, dyn_flag)).or_insert(0) += 1;
    }

    // Phase-2 dead-claim elimination (metadata-only):
    // compile only metas actually present in the assembled raw schedule.
    let enable_mask: Vec<bool> = metas
        .iter()
        .map(|meta| {
            let key = (meta.label, meta.degree_bound, meta.is_dynamic);
            if let Some(count) = raw_counts.get_mut(&key) {
                if *count > 0 {
                    *count -= 1;
                    return true;
                }
            }
            false
        })
        .collect();

    if raw_counts.values().any(|&count| count != 0) {
        return Err(PiCcsError::ProtocolError(format!(
            "{context}: route-a compiler shadow unmatched raw claims after metadata elimination"
        )));
    }

    let legacy = RouteAClaimCompiler::compile_schedule_from_raw_with_enable_mask(
        labels,
        degree_bounds,
        is_dynamic,
        &vec![true; labels.len()],
        ClaimDomain::TimeCycle,
    )?;
    let compiled =
        RouteAClaimCompiler::compile_schedule_from_metas_with_enable_mask(metas, &enable_mask, ClaimDomain::TimeCycle)?;

    if legacy.claims.len() != compiled.claims.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "{context}: route-a compiler shadow claim count mismatch (legacy={}, compiled={})",
            legacy.claims.len(),
            compiled.claims.len()
        )));
    }
    for (idx, (left, right)) in legacy.claims.iter().zip(compiled.claims.iter()).enumerate() {
        if left.label != right.label
            || left.degree_bound != right.degree_bound
            || left.is_dynamic != right.is_dynamic
            || left.domain != right.domain
            || left.expr_hash != right.expr_hash
        {
            return Err(PiCcsError::ProtocolError(format!(
                "{context}: route-a compiler shadow mismatch at idx={idx} label={:?}",
                left.label
            )));
        }
    }
    if legacy.schedule_hash != compiled.schedule_hash {
        return Err(PiCcsError::ProtocolError(format!(
            "{context}: route-a compiler shadow schedule hash mismatch (legacy={:#x}, compiled={:#x})",
            legacy.schedule_hash, compiled.schedule_hash
        )));
    }
    Ok(())
}

#[cfg(not(feature = "route-a-compiler-shadow"))]
#[inline]
pub fn shadow_assert_compiled_schedule_matches_metas(
    _labels: &[&'static [u8]],
    _degree_bounds: &[usize],
    _is_dynamic: &[bool],
    _metas: &[TimeClaimMeta],
    _context: &'static str,
) -> Result<(), PiCcsError> {
    Ok(())
}

#[inline]
fn schedule_hash(claims: &[CompiledClaim]) -> u64 {
    let mut values: Vec<u64> = Vec::with_capacity(claims.len() * 5);
    for claim in claims {
        values.push(hash_bytes(claim.label));
        values.push(claim.degree_bound as u64);
        values.push(u64::from(claim.is_dynamic));
        values.push(claim.domain.as_tag());
        values.push(claim.expr_hash);
    }
    hash_tagged(0xC001, &values)
}

#[inline]
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for byte in bytes {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[inline]
fn hash_tagged(tag: u64, values: &[u64]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    h ^= tag;
    h = h.wrapping_mul(0x100000001b3);
    for value in values {
        h ^= *value;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    const META_A: TimeClaimMeta = TimeClaimMeta {
        label: b"a/x",
        degree_bound: 3,
        is_dynamic: false,
    };
    const META_B: TimeClaimMeta = TimeClaimMeta {
        label: b"b/y",
        degree_bound: 7,
        is_dynamic: true,
    };

    #[test]
    fn route_a_compiler_is_deterministic_for_identical_schedule() {
        let metas = [META_A, META_B];
        let out1 = RouteAClaimCompiler::compile_schedule_from_metas(&metas, ClaimDomain::TimeCycle);
        let out2 = RouteAClaimCompiler::compile_schedule_from_metas(&metas, ClaimDomain::TimeCycle);
        assert_eq!(out1.claims, out2.claims);
        assert_eq!(out1.schedule_hash, out2.schedule_hash);
        assert_eq!(out1.expr_nodes, out2.expr_nodes);
    }

    #[test]
    fn route_a_compiler_schedule_hash_changes_when_order_changes() {
        let metas1 = [META_A, META_B];
        let metas2 = [META_B, META_A];
        let out1 = RouteAClaimCompiler::compile_schedule_from_metas(&metas1, ClaimDomain::TimeCycle);
        let out2 = RouteAClaimCompiler::compile_schedule_from_metas(&metas2, ClaimDomain::TimeCycle);
        assert_ne!(out1.schedule_hash, out2.schedule_hash);
    }

    #[test]
    fn route_a_compiler_raw_inputs_len_mismatch_is_rejected() {
        let err = RouteAClaimCompiler::compile_schedule_from_raw(&[b"a/x"], &[3, 4], &[false], ClaimDomain::TimeCycle)
            .expect_err("length mismatch must fail");
        assert!(format!("{err}").contains("length mismatch"), "unexpected error: {err}");
    }

    #[test]
    fn route_a_compiler_cse_reuses_identical_claim_nodes() {
        let metas = [META_A, META_A, META_A];
        let out = RouteAClaimCompiler::compile_schedule_from_metas(&metas, ClaimDomain::TimeCycle);
        assert_eq!(out.claims.len(), 3);
        assert!(
            out.expr_nodes <= 10,
            "expected CSE to reuse identical subexpressions, got expr_nodes={}",
            out.expr_nodes
        );
    }

    #[test]
    fn route_a_compiler_dead_claim_elimination_by_mask() {
        let metas = [META_A, META_B, META_A];
        let mask = [true, false, true];
        let out =
            RouteAClaimCompiler::compile_schedule_from_metas_with_enable_mask(&metas, &mask, ClaimDomain::TimeCycle)
                .expect("compile metas with mask");
        assert_eq!(out.claims.len(), 2);
        assert_eq!(out.claims[0].label, META_A.label);
        assert_eq!(out.claims[1].label, META_A.label);
    }

    #[test]
    fn route_a_compiler_dead_claim_mask_len_mismatch_is_rejected() {
        let metas = [META_A, META_B];
        let err =
            RouteAClaimCompiler::compile_schedule_from_metas_with_enable_mask(&metas, &[true], ClaimDomain::TimeCycle)
                .expect_err("mask length mismatch must fail");
        assert!(format!("{err}").contains("length mismatch"), "unexpected error: {err}");
    }
}
