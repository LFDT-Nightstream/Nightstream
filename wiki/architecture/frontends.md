# Frontends

Frontends turn application data into the CCS instances that the folding
lifecycle accepts. The frontend owns the translation. The NIFS core does not
infer application semantics.

## Authority boundary

The recursive R1CS and Nebula frontends compile the authoritative F' relation.
Each recursive branch verifies the preceding NIFS step in-circuit. Their
terminal verifier can check the final running accumulator and latest instance.

The direct CCS frontend proves satisfaction of caller-supplied CCS instances
and NIFS continuity. It does not claim that each instance is an encoded F'
step. Multi-chunk direct CCS verification therefore keeps and replays the
audit trail.

## direct_ccs

`frontends/direct_ccs/` accepts a fixed R1CS shape `(A, B, C, m_in)` and an
assignment `z = [x | w]`. It checks `Az * Bz = Cz`, packs the witness, and
creates the Ajtai commitment.

## f_prime

`frontends/f_prime/` owns the application-neutral low-norm F' image,
recursive image plan, projection structure, accumulator handles, and
Poseidon2 traces. It does not compile an application by itself.

## r1cs_f_prime

`frontends/r1cs_f_prime/` lowers a verifier-owned R1CS relation into one
fixed low-norm relation with base and recursive branches.
`r1cs_f_prime::ivc::R1csIvc` owns the chain-facing flow. The emitted relation
is also the single source for its compilation audit.

## nebula

`frontends/nebula/` combines the fixed F' relation with Nebula's offline
memory checking. Its three branches cover the base step, the first recursive
step, and later recursive steps. The relation owns its fixed-point shape and
rejects shape cycles.

## bellpepper

`frontends/bellpepper.rs` converts a Bellpepper circuit into sparse R1CS. It
is an adapter only. Preprocessing and proving still use one of the frontends
above.
