// Canonical PaperExact coverage is in tests/paper_rectangular_parity.rs.
// This explicit target remains empty so old pre-SuperNeo fixtures cannot be
// mistaken for the active paper reference.
#![cfg(all(feature = "paper-exact", feature = "testing"))]
