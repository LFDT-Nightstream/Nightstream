//! CUDA graph capture and launch ownership.
//!
//! Owns the raw driver handles needed to capture already-enqueued CUDA work
//! into a replayable graph. It does not decide what protocol phase is worth
//! capturing; callers own graph shape and buffer lifetime.

use std::mem::MaybeUninit;

use cuda_core::{sys, CudaStream, DeviceBuffer, DriverError, IntoResult};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GraphAllocations {
    ptrs: Vec<u64>,
}

impl GraphAllocations {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push<T>(&mut self, buffer: &DeviceBuffer<T>) {
        self.ptrs.push(buffer.cu_deviceptr());
    }
}

pub struct CapturedGraph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
}

#[derive(Debug)]
pub enum CaptureError<E> {
    Body(E),
    Driver(DriverError),
}

impl CapturedGraph {
    pub fn capture(stream: &CudaStream, body: impl FnOnce() -> Result<(), DriverError>) -> Result<Self, DriverError> {
        Self::capture_checked(stream, body).map_err(|error| match error {
            CaptureError::Body(error) | CaptureError::Driver(error) => error,
        })
    }

    pub fn capture_checked<E>(
        stream: &CudaStream,
        body: impl FnOnce() -> Result<(), E>,
    ) -> Result<Self, CaptureError<E>> {
        stream
            .context()
            .bind_to_thread()
            .map_err(CaptureError::Driver)?;
        unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()
            .map_err(CaptureError::Driver)?;
        }

        let body_result = body();
        let mut graph = MaybeUninit::uninit();
        let end_result = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), graph.as_mut_ptr()).result() };

        match (body_result, end_result) {
            (Err(error), Ok(())) => {
                let graph = unsafe { graph.assume_init() };
                destroy_graph(graph);
                Err(CaptureError::Body(error))
            }
            (Err(error), Err(_)) => Err(CaptureError::Body(error)),
            (Ok(()), Err(error)) => Err(CaptureError::Driver(error)),
            (Ok(()), Ok(())) => {
                let graph = unsafe { graph.assume_init() };
                if graph.is_null() {
                    return Err(CaptureError::Driver(DriverError(
                        sys::cudaError_enum_CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
                    )));
                }
                let mut exec = MaybeUninit::uninit();
                let instantiate = unsafe { sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), graph, 0).result() };
                match instantiate {
                    Ok(()) => Ok(Self {
                        graph,
                        exec: unsafe { exec.assume_init() },
                    }),
                    Err(error) => {
                        destroy_graph(graph);
                        Err(CaptureError::Driver(error))
                    }
                }
            }
        }
    }

    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.context().bind_to_thread()?;
        unsafe { sys::cuGraphLaunch(self.exec, stream.cu_stream()).result() }
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuGraphExecDestroy(self.exec);
        }
        destroy_graph(self.graph);
    }
}

fn destroy_graph(graph: sys::CUgraph) {
    if !graph.is_null() {
        unsafe {
            let _ = sys::cuGraphDestroy(graph);
        }
    }
}
