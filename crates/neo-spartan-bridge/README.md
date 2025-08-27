# neo-spartan-bridge

Last‑mile compression: take the final **ME(b,L)** claim and produce a succinct proof using **Spartan2** with a **real FRI PCS** (no simulated FRI).

## Surface
- Adapter from `neo-ccs::MEInstance/Witness` to Spartan's linearized CCS.
- Prove/verify: run Spartan2 setup/prove/verify; return proof bytes.
- Binding: recompute public IO digest identically to `neo-fold`'s transcript expectations.

## Tests
- Smoke: tiny ME instance round‑trip.
- Tamper: flip c/X/r/y → verify must fail.
- Proof size/time metrics (optional).
