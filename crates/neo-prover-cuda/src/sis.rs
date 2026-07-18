//! Device execution of the protocol's fixed-seed SIS digest compressor.
//!
//! `neo-fold-clean` owns the domain/configuration and `neo-ajtai` owns the
//! seeded map. This module caches materialized maps, prepares balanced-
//! ternary messages on device, and returns the canonical Poseidon2 digest.

use cuda_core::DeviceBuffer;
use neo_ajtai::setup_par;
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{
    accumulator_digest_envelope_prefix, SisAccumulatorConfig, SIS_DIGEST_COMPRESSION_CONFIG,
};
use neo_math::{D, F};
use p3_field::PrimeField64;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::commit::DeviceAjtai;
use crate::device::{uninit_u64_device_buffer, upload_u64_device_buffer, Device};
use crate::kernels::ajtai::launch_plane_copy;
use crate::kernels::poseidon2::launch_hash_contiguous_cooperative;
use crate::kernels::sis::{
    launch_balanced_ternary_message, load_sis_kernels, SisKernelModule, BALANCED_TERNARY_DIGITS,
};
use crate::reduce::ccs::{CcsDeviceError, SumcheckKernels};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SisMapKey {
    seed: [u8; 32],
    kappa: usize,
    field_count: usize,
}

struct DeviceSisMap {
    key: SisMapKey,
    message_cols: usize,
    ajtai: DeviceAjtai,
    active_flag: DeviceBuffer<u64>,
}

#[derive(Default)]
pub(crate) struct DeviceSisCache {
    module: Option<SisKernelModule>,
    maps: Vec<DeviceSisMap>,
}

impl DeviceSisCache {
    pub(crate) fn module(&mut self, device: &Device) -> Result<&SisKernelModule, CcsDeviceError> {
        if self.module.is_none() {
            self.module = Some(load_sis_kernels(device.ctx()).map_err(CcsDeviceError::ModuleLoad)?);
        }
        Ok(self.module.as_ref().expect("loaded above"))
    }

    pub(crate) fn digest_host(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        config: SisAccumulatorConfig,
        fields: &[F],
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        if fields.is_empty() {
            return Err(CcsDeviceError::Shape("SIS digest requires fields"));
        }
        let words = fields
            .iter()
            .map(|field| field.as_canonical_u64())
            .collect::<Vec<_>>();
        let fields = upload_u64_device_buffer(device.stream(), &words)?;
        self.digest_device(device, kernels, config, &fields, words.len())
    }

    pub(crate) fn prepare_digest(
        &mut self,
        device: &Device,
        config: SisAccumulatorConfig,
        field_count: usize,
    ) -> Result<(), CcsDeviceError> {
        if field_count == 0 {
            return Err(CcsDeviceError::Shape("SIS digest requires fields"));
        }
        let _ = self.module(device)?;
        ensure_map(
            device,
            &mut self.maps,
            SisMapKey {
                seed: config.seed,
                kappa: config.kappa,
                field_count,
            },
        )?;
        ensure_map(
            device,
            &mut self.maps,
            SisMapKey {
                seed: SIS_DIGEST_COMPRESSION_CONFIG.seed,
                kappa: SIS_DIGEST_COMPRESSION_CONFIG.kappa,
                field_count: config.kappa * D,
            },
        )?;
        Ok(())
    }

    pub(crate) fn digest_device(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        config: SisAccumulatorConfig,
        fields: &DeviceBuffer<u64>,
        field_count: usize,
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        if field_count == 0 || fields.len() != field_count {
            return Err(CcsDeviceError::Shape("SIS device field shape mismatch"));
        }
        if self.module.is_none() {
            self.module = Some(load_sis_kernels(device.ctx()).map_err(CcsDeviceError::ModuleLoad)?);
        }
        let module = self.module.as_ref().expect("loaded above");

        let binding_key = SisMapKey {
            seed: config.seed,
            kappa: config.kappa,
            field_count,
        };
        let binding = commit_fields(device, module, &mut self.maps, binding_key, fields)?;

        let compression_key = SisMapKey {
            seed: SIS_DIGEST_COMPRESSION_CONFIG.seed,
            kappa: SIS_DIGEST_COMPRESSION_CONFIG.kappa,
            field_count: config.kappa * D,
        };
        let compression = commit_fields(device, module, &mut self.maps, compression_key, &binding)?;

        let prefix = accumulator_digest_envelope_prefix(config, field_count);
        let prefix_words = prefix
            .iter()
            .map(|field| field.as_canonical_u64())
            .collect::<Vec<_>>();
        let mut envelope_words = prefix_words.clone();
        envelope_words.resize(prefix_words.len() + compression.len(), 0);
        let mut envelope = upload_u64_device_buffer(device.stream(), &envelope_words)?;
        let copy_module = self
            .maps
            .iter()
            .find(|map| map.key == compression_key)
            .expect("compression map exists")
            .ajtai
            .module();
        launch_plane_copy(
            copy_module,
            device.stream(),
            &compression,
            prefix_words.len(),
            &mut envelope,
        )?;

        let mut digest = uninit_u64_device_buffer(device.stream(), 4)?;
        launch_hash_contiguous_cooperative(
            &kernels.poseidon,
            device.stream(),
            &envelope,
            envelope_words.len(),
            &mut digest,
            &kernels.poseidon_rc,
        )?;
        Ok(digest)
    }
}

fn commit_fields(
    device: &Device,
    module: &SisKernelModule,
    maps: &mut Vec<DeviceSisMap>,
    key: SisMapKey,
    fields: &DeviceBuffer<u64>,
) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
    let map_index = ensure_map(device, maps, key)?;
    let map = &mut maps[map_index];
    let mut message = DeviceBuffer::zeroed(device.stream(), map.message_cols * D)?;
    launch_balanced_ternary_message(
        module,
        device.stream(),
        fields,
        key.field_count,
        map.message_cols,
        &mut message,
    )?;
    map.ajtai
        .commit_signed_unit_device_columns(device, &message, &map.active_flag)
        .map_err(|_| CcsDeviceError::Shape("device SIS commitment failed"))
}

fn ensure_map(device: &Device, maps: &mut Vec<DeviceSisMap>, key: SisMapKey) -> Result<usize, CcsDeviceError> {
    let index = match maps.iter().position(|map| map.key == key) {
        Some(index) => index,
        None => {
            let message_cols = (key.field_count * BALANCED_TERNARY_DIGITS).div_ceil(D);
            let mut rng = ChaCha8Rng::from_seed(key.seed);
            let pp = setup_par(&mut rng, D, key.kappa, message_cols)
                .map_err(|_| CcsDeviceError::Shape("materialize seeded SIS PP failed"))?;
            let ajtai =
                DeviceAjtai::upload(device, &pp).map_err(|_| CcsDeviceError::Shape("upload seeded SIS PP failed"))?;
            maps.push(DeviceSisMap {
                key,
                message_cols,
                ajtai,
                active_flag: upload_u64_device_buffer(device.stream(), &[1])?,
            });
            maps.len() - 1
        }
    };
    Ok(index)
}
