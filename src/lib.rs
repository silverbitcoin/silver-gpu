//! GPU Acceleration for SilverBitcoin Mining and Computation
//!
//! This module provides GPU acceleration for:
//! - SHA-512 mining (PoW)
//! - Cryptographic operations
//! - Parallel computation
//!
//! Supports multiple GPU backends:
//! - OpenCL (cross-platform)
//! - CUDA (NVIDIA)

pub mod errors;
pub mod gpu_miner;
pub mod gpu_context;
pub mod kernels;

pub use errors::{GpuError, Result};
pub use gpu_context::GpuContext;
pub use gpu_miner::GpuMiner;
pub use silver_pow::StratumPoolClient;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuInitError {
    #[error("No GPU devices found")]
    NoDevicesFound,

    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Unsupported GPU backend: {0}")]
    UnsupportedBackend(String),

    #[error("GPU error: {0}")]
    GpuError(String),
}

pub type GpuInitResult<T> = std::result::Result<T, GpuInitError>;

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// OpenCL backend (cross-platform)
    OpenCL,
    /// CUDA backend (NVIDIA only)
    Cuda,
    /// CPU fallback
    Cpu,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::OpenCL => write!(f, "OpenCL"),
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Cpu => write!(f, "CPU"),
        }
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub id: usize,
    pub name: String,
    pub backend: GpuBackend,
    pub compute_capability: String,
    pub memory_mb: u64,
    pub max_threads: u32,
}

impl std::fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU {} ({}): {} - {} MB, {} threads",
            self.id, self.backend, self.name, self.memory_mb, self.max_threads
        )
    }
}

/// GPU configuration
#[derive(Debug, Clone, Copy)]
pub struct GpuConfig {
    /// Preferred GPU backend
    pub backend: GpuBackend,
    /// Number of GPU threads per block
    pub threads_per_block: u32,
    /// Number of blocks
    pub num_blocks: u32,
    /// Enable GPU memory optimization
    pub optimize_memory: bool,
    /// Enable GPU computation caching
    pub enable_caching: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::OpenCL,
            threads_per_block: 256,
            num_blocks: 1024,
            optimize_memory: true,
            enable_caching: true,
        }
    }
}

impl GpuConfig {
    pub fn with_backend(mut self, backend: GpuBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads_per_block = threads;
        self
    }

    pub fn with_blocks(mut self, blocks: u32) -> Self {
        self.num_blocks = blocks;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_display() {
        assert_eq!(GpuBackend::OpenCL.to_string(), "OpenCL");
        assert_eq!(GpuBackend::Cuda.to_string(), "CUDA");
        assert_eq!(GpuBackend::Cpu.to_string(), "CPU");
    }

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.threads_per_block, 256);
        assert_eq!(config.num_blocks, 1024);
        assert!(config.optimize_memory);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_gpu_config_builder() {
        let config = GpuConfig::default()
            .with_backend(GpuBackend::Cuda)
            .with_threads(512)
            .with_blocks(2048);

        assert_eq!(config.backend, GpuBackend::Cuda);
        assert_eq!(config.threads_per_block, 512);
        assert_eq!(config.num_blocks, 2048);
    }
}
