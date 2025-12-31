#![allow(non_snake_case)]

use super::cpu_bus::append_bus_openings_to_me_instance;
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_memory::cpu::build_bus_layout_for_instances;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn chi_for_row_index(r: &[K], idx: usize) -> K {
    let mut acc = K::ONE;
    for (bit, &ri) in r.iter().enumerate() {
        let is_one = ((idx >> bit) & 1) == 1;
        acc *= if is_one { ri } else { K::ONE - ri };
    }
    acc
}

fn recompose_base_b_digits(params: &NeoParams, digits: &[K]) -> K {
    let bK = K::from(F::from_u64(params.b as u64));
    let mut pow = K::ONE;
    let mut acc = K::ZERO;
    for &v in digits.iter().take(D) {
        acc += pow * v;
        pow *= bK;
    }
    acc
}

#[test]
fn append_bus_openings_matches_manual_for_chunk_size_2() {
    let params = NeoParams::goldilocks_127();

    // Minimal shared-bus layout: one Shout instance with ell_addr=2.
    let m_in = 2usize;
    let chunk_size = 2usize;
    let m = 32usize;
    let bus = build_bus_layout_for_instances(m, m_in, chunk_size, [2usize], [])
        .expect("bus layout should build");
    assert!(bus.bus_cols > 0);

    // Witness decomposition Z (D×m).
    let mut Z = Mat::zero(D, m, F::ZERO);
    for rho in 0..D {
        for c in 0..m {
            Z[(rho, c)] = F::from_u64(((rho as u64) + 1) * ((c as u64) + 1));
        }
    }

    // Start with a core ME instance of length core_t=1.
    let y_pad = (params.d as usize).next_power_of_two();
    let core_t = 1usize;
    let mut me = neo_ccs::MeInstance::<Commitment, F, K> {
        c: Commitment::zeros(params.d as usize, 1),
        X: Mat::zero(D, m_in, F::ZERO),
        r: vec![
            K::from(F::from_u64(3)),
            K::from(F::from_u64(5)),
            K::from(F::from_u64(7)),
            K::from(F::from_u64(11)),
        ],
        y: vec![vec![K::ZERO; y_pad]],
        y_scalars: vec![K::ZERO],
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    append_bus_openings_to_me_instance(&params, &bus, core_t, &Z, &mut me).expect("append");

    assert_eq!(me.y.len(), core_t + bus.bus_cols);
    assert_eq!(me.y_scalars.len(), core_t + bus.bus_cols);

    // Manual check of each appended bus opening.
    let time_weights: Vec<K> = (0..bus.chunk_size)
        .map(|j| chi_for_row_index(&me.r, bus.time_index(j)))
        .collect();

    for col_id in 0..bus.bus_cols {
        let j_idx = core_t + col_id;
        let y_row = &me.y[j_idx];
        assert_eq!(y_row.len(), y_pad);

        // Check per-digit row values.
        for rho in 0..D {
            let mut acc = K::ZERO;
            for j in 0..bus.chunk_size {
                let w = time_weights[j];
                let z_idx = bus.bus_cell(col_id, j);
                acc += w * K::from(Z[(rho, z_idx)]);
            }
            assert_eq!(y_row[rho], acc, "rho={rho}, col_id={col_id}");
        }
        for rho in D..y_pad {
            assert_eq!(y_row[rho], K::ZERO);
        }

        // Check scalar recomposition.
        let expect_scalar = recompose_base_b_digits(&params, y_row);
        assert_eq!(me.y_scalars[j_idx], expect_scalar, "col_id={col_id}");
    }
}

