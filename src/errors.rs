//! GPU error types

use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum GpuError {
    #[error("GPU not available")]
    GpuNotAvailable,

    #[error("No GPU devices found")]
    NoDevicesFound,

    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),

    #[error("GPU memory allocation failed: {0}")]
    MemoryAllocationFailed(String),

    #[error("GPU kernel execution failed: {0}")]
    KernelExecutionFailed(String),

    #[error("GPU memory transfer failed: {0}")]
    MemoryTransferFailed(String),

    #[error("GPU computation error: {0}")]
    ComputationError(String),

    #[error("GPU backend not supported: {0}")]
    BackendNotSupported(String),

    #[error("GPU device not found")]
    DeviceNotFound,

    #[error("GPU timeout")]
    Timeout,

    #[error("Invalid GPU configuration: {0}")]
    InvalidConfiguration(String),

    #[error("GPU resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Internal GPU error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, GpuError>;

impl From<std::io::Error> for GpuError {
    fn from(err: std::io::Error) -> Self {
        GpuError::Internal(err.to_string())
    }
}
