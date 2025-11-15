//! # SilverBitcoin GPU Acceleration
//!
//! GPU acceleration layer supporting OpenCL, CUDA, and Metal.
//!
//! This crate provides:
//! - GPU abstraction layer
//! - Batch signature verification (100-1000x speedup)
//! - Parallel hash computation (10-100x speedup)
//! - GPU-accelerated transaction execution
//! - Automatic CPU/GPU load balancing

#![warn(missing_docs, rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod backend;
pub mod signature_verification;
pub mod hashing;
pub mod executor;
pub mod scheduler;

pub use backend::{GPUBackend, GPUDevice, GPUAccelerator};
pub use signature_verification::GPUSignatureVerifier;
pub use hashing::GPUHasher;
pub use executor::GPUExecutor;
pub use scheduler::HybridExecutor;
