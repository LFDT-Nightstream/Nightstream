# Goal: complete Phases 2–5 of the canonical encoding, autonomously

You are continuing formal work in `formal/nightstream-lean`. Work to completion
without asking for direction. Ask only under the narrow conditions in
**§7 Escalation**. Everything else is your call.

---

## 0. Resuming — do this first, every iteration

You may arrive with fresh, compacted, or continuous context. Never assume.
Establish position before writing anything:

1. `tail -3 assurance/evidence-ledger.jsonl` — the newest entry's
   `remaining_dependency` field is the authoritative work queue. It is ordered.
2. Read the modules under `Nightstream/Implementation/R1CS/Canonical/`. What
   exists there is what is done; the ledger describes it but the code is the
   authority.
3. `lake build tests.Axioms` — confirm green *before* you change anything, so a
   later failure is attributable to your work.
4. Take the first item in `remaining_dependency` that is not already closed in
   code. Do not re-derive closed items. Do not skip ahead to a more interesting
   one.

If the ledger and the code disagree, the code wins and the ledger entry was
wrong — supersede it explicitly (§8).

---

## 1. The objective, precisely

Derive, entirely within Lean and with no input from generated R1CS artifacts,
the exact constraint cost of the fixed-one F′ program, and prove that cost
corresponds to a semantically correct encoding.

Four phases, in order:

**Phase 2 — canonical Poseidon2 permutation (width-8 F′/`neo_ccs` only).**
Instantiate the concrete 8/22 round schedule and both linear matrices; prove
the support recurrence; prove `permutationProgram_exec_iff_spec` by round
induction; construct honest satisfying assignments; derive exact row, column,
and nonzero-coefficient counts.

**Phase 3 — sponge and hash recipes.** `hashPrior` and `hashNext` over the
Phase-2 permutation. Owns absorption, padding, rate, capacity, and domain
separation — none of which belong to Phase 2.

**Phase 4 — application codecs and terminal checks.** Complete the selected
profile and construct independent `runningCheck` and `freshCheck` recipes.

**Phase 5 — setup-owned recipes and assembly.** Accept complete proof-carrying
physical programs for the HyperNova application `step` and selected
`nifsVerify`, then assemble all eleven calls into the canonical Step and
Terminal programs. Derive their complete `Typed.Cost` tuples from receipt
folds. Selecting one deployment profile and comparing Rust remain later
refinement obligations.

---

## 2. What "complete" means, per obligation

A recipe or program is complete only when **all** of the following hold. A
subset is a subtotal, not a result.

- Exact row program, constructively defined — not a declared count.
- Row count derived from the construction, matching any declared footprint.
- Exact row ownership: every emitted row belongs to exactly one receipt.
- Exact column ownership: every allocated column belongs to exactly one owner;
  distinct owners never collide; shared reads are distinguished from
  allocations.
- Conservation: no row touches a column outside the allocation plus declared
  shared reads.
- Soundness: satisfaction implies the frozen semantic relation, or an exact
  named event.
- Honest completeness: an honest execution yields a satisfying assignment.
- Cost in the project's `Typed.Cost` — `(recurringRows, committedColumns,
  publicColumns, auxiliaryColumns)` — not a parallel cost type.
- Fail-closed axiom guard for every headline theorem, with measured reports.
- Per-property spec and evidence-ledger entry.

---

## 3. What does not count

Producing any of these is not progress. Recognise and reject them in your own
output:

- **A count without a construction.** "N rows" where N is declared, measured,
  or inherited rather than derived from an emitted row program.
- **A subtotal presented as a total.** If linear layers, terminal binding,
  wrapper rows, or output ports are excluded, the number is a subtotal and must
  be named as one.
- **A premise that moved rather than closed.** Replacing a caller-supplied
  callback with a caller-supplied record is not a discharge. If the new premise
  has no constructor invoked by a real consumer, nothing was closed.
- **A definitional theorem read as semantic.** `f x = []` proved by `rfl`
  because `f := fun _ => []` records a definition, not a fact about the world.
- **Exclusivity presented as exhaustiveness.** `A → ¬B` does not establish
  `A ∨ B`. If a third case exists, the classification must name it.
- **A reduction to another unproved statement of comparable strength.** If
  closing X requires Y, and Y is as hard and as unproved as X, the route is
  blocked. Say so; do not report progress.
- **A green build presented as a proof of the claim.** The gate proves the
  statements that exist. It says nothing about a statement you described but
  did not write.
- **A type too weak for its claim.** If the docstring asserts a property the
  representation cannot express, the defect is the type, not the wording.

---

## 4. Traps specific to this codebase

Each of these has already caused a defect here. Check for them before
reporting.

**4.1 Provenance.** Every number is either *derived* (the conclusion of a proof
from non-count inputs) or *measured* (imported from outside). Before using any
number, determine which. Known measured values that must never be presented as
derived: the 4,193,134-row artifact, the 600-row Poseidon2 artifact, the
47,020,034 emitted total, any `parameters.footprints` value taken from a Rust
configuration, and the production codec widths.

*Legitimate exception:* Poseidon2's round constants and matrices may be taken
from Rust, because the goal is to re-encode the selected permutation and it
must compute the same function. What may never be taken is the artifact's row
count or row layout.

**4.2 Scope of a search.** Do not assert absence from a scoped search. Six
false "does not exist" claims here came from grepping one directory, one
qualified name, or one filename pattern. Before writing "there is no X",
search by unqualified name, by concept, and across `crates/` as well as
`Nightstream/`.

**4.3 Scope of a statement.** A number can be exactly right about something
narrower than the sentence containing it. Before quoting a figure, state what
it quantifies over. "344 rows" was true of S-box rows and false of the
permutation.

**4.4 Cost composition.** Distinguish *allocated* from *referenced* columns.
Preallocated inputs are references; counting them as allocations double-counts
across any program with several instructions touching the same carrier.

**4.5 Paper authority.** SuperNeo and HyperNova specify an abstract hash and an
abstract random oracle. They do not select Poseidon2 or any of its parameters.
`d = 54`, `b = 2`, `k = 14`, `T = 216` are paper-derived (SuperNeo App. B.2);
width 8, 8/22 rounds, `x^7`, and the linear layers are production choices.
Ownership splits further: `neo-params` owns width/capacity/rate/digest/seed;
p3 and the Rust circuit own the degree, round split, and matrices.

**4.6 Two hashes, one permutation.** Construction 2's binding hash and the
Fiat–Shamir random oracle are distinct objects with distinct security
contracts. They may share arithmetic. Arithmetic correctness proves neither
binding nor RO soundness; both remain named events.

**4.7 Syntactic size versus mathematical support.** A concatenating
combination grows without bound while its support does not. An S-box output is
a fresh variable, so S-boxing *resets* support. Do not conclude a normal form
is impractical from unaggregated list length.

---

## 5. Search management

- Open several independent routes before committing to one.
- Keep incompatible routes alive until one is closed, not until one looks
  promising.
- Attack your own lemmas: for each, try to construct a counterexample before
  trying to prove it.
- **Mark a route blocked** if it only reduces the problem to another unproved
  statement of comparable strength. Record the blockage as a result.
- A kernel-checked obstruction is a valid outcome and should be recorded with
  the same discipline as a positive result — provided it reaches the exact
  interface in question, not a surrogate.
- When a route is blocked, do not weaken the target to make it pass. Report the
  blockage and take a different route.

---

## 6. Adversarial self-check, before every report

For each headline theorem you are about to claim:

1. Restate what it quantifies over, in one sentence, without the surrounding
   prose.
2. Name the property it would silently eliminate if it were wrong — for a
   derivation, which named event does it make unreachable?
3. For every number: derived or measured? If derived, from what non-count
   inputs?
4. For every negative witness: prove the fixture is otherwise accepted, then
   confirm the mutation isolates the intended branch.
5. For every premise you introduced: name the consumer that constructs it. If
   none exists, you moved the obligation.
6. Print raw `#print axioms` — not the normalizing macro — and confirm no
   unexpected `Lean.trustCompiler`.

If any check fails, fix it before reporting rather than reporting with a
caveat.

---

## 7. Escalation — the only reasons to stop and ask

Proceed without asking on everything except these:

- A **protocol decision** that changes a mapped paper definition, a frozen
  relation, or a verifier branch (spec §16 change control).
- A **production defect**: evidence that shipping Rust does not enforce a
  frozen relation.
- A **blocked route** where every alternative you identified is also blocked.
- A change that would **modify Rust, regenerate artifacts, or alter frozen
  semantics**.

Do not ask whether to proceed, whether a subtotal is acceptable, which of two
provable lemmas to prove first, or for permission to record an obstruction.

---

## 8. Recording discipline

- One per-property spec per obligation, in `specs/`, using the §10 contract
  format.
- One evidence-ledger entry per cycle. Include `remaining_dependency` and
  `non_goals` honestly; a promotion with an empty remaining list is a claim
  that nothing is left.
- `remaining_dependency` is the resume anchor for the next iteration (§0).
  Write it as an ordered queue of concrete next items, not a mood.
- Evidence states are those defined in spec §8. Do not invent tiers.
- When you correct a previous entry, supersede it explicitly and say what was
  wrong.

---

## 9. Validation

Every Lean command under the 25-minute cap; every `cargo test` under the
5-minute cap. Before each report:

```
lake build <focused targets>
lake build tests.Axioms
```

No `sorry`, `admit`, `axiom`, `postulate`, `unsafe`, `sorryAx`. No new
`native_decide` or `Lean.trustCompiler` on any headline path. Do not commit or
push.

---

## 10. Definition of done

Phases 2–5 are complete at the profile-indexed canonical boundary when:

- `CanonicalProgram` exists, compiles, and its cost is a receipt fold;
- `N_canonical` is stated as a full `Typed.Cost` tuple with row, column, and
  nonzero-coefficient counts;
- every recipe satisfies all of §2;
- every open item is a named property with a spec, not a comment;
- `tests.Axioms` is green and every headline theorem is guarded.

---

## 11. Ending the loop

If you are running under `/loop`, stop it — `ScheduleWakeup({stop: true})` —
on any of:

- **Done.** §10 is fully met.
- **Escalation.** A §7 condition fired. Stop and state the decision needed.
- **No progress.** Two consecutive iterations closed nothing new. Do not spin;
  report what is blocking.

Never leave an iteration mid-edit: each one must end with the build green and a
ledger entry written, because the next iteration may start with no memory of
this one.

Report at the end with: what closed, what is blocked and why, every remaining
assumption, and the exact `N_canonical` tuple with its scope stated.
