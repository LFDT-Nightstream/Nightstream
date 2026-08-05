# Neo and SuperNeo: Post-quantum folding with pay-per-bit costs over small fields

Wilson Nguyen  
Stanford University, New York University,  
and Microsoft Research

Srinath Setty  
Microsoft Research

**Abstract.** We construct the first folding scheme that simultaneously achieves six desirable properties: plausible post-quantum security, pay-per-bit commitment costs, field-native arithmetic (the sum-check and norm checks run purely over a small field), support for general (non-SIMD) constraint systems, small-field support (e.g., Goldilocks), and low recursion overheads. No existing scheme satisfies all six: group-based schemes (e.g., HyperNova) lack post-quantum security and are tied to large elliptic-curve fields; lattice-based schemes (e.g., LatticeFold) require expensive ring arithmetic, lose pay-per-bit costs, and impose SIMD constraints; and hash-based schemes (e.g., Arc) incur large verifier circuits.

We present two lattice-based folding schemes for CCS—an NP-complete relation generalizing R1CS, Plonkish, and AIR—called Neo and SuperNeo. Neo satisfies five of the six properties but requires SIMD constraint systems; SuperNeo removes this restriction and satisfies all six. Both run a single invocation of the sum-check protocol over a small field extension and achieve pay-per-bit costs via new folding-friendly instantiations of Ajtai commitments under the Module-SIS assumption. At the core of our constructions are two new norm-preserving embeddings of field vectors into ring vectors that respect an evaluation homomorphism required for folding. We also introduce *interactive reductions*, a framework that generalizes reductions of knowledge and enables modular security proofs for composed lattice-based protocols.
