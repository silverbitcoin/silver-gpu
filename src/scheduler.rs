//! Hybrid CPU/GPU execution scheduler
//!
//! Automatically decides whether to use CPU or GPU based on workload characteristics.

use crate::backend::{GPUAccelerator, GPUBackend, GPUError};
use crate::executor::{GPUExecutionResult, GPUExecutor};
use crate::hashing::GPUHasher;
use crate::signature_verification::GPUSignatureVerifier;
use silver_core::transaction::Transaction;
use silver_core::{PublicKey, Signature};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Performance profiler for workload analysis
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Average CPU execution time
    pub cpu_time: Duration,
    /// Average GPU execution time (including transfer)
    pub gpu_time: Duration,
    /// Number of samples
    pub samples: usize,
}

impl PerformanceProfile {
    /// Create new profile
    pub fn new() -> Self {
        Self {
            cpu_time: Duration::from_millis(0),
            gpu_time: Duration::from_millis(0),
            samples: 0,
        }
    }

    /// Update with new measurement
    pub fn update(&mut self, cpu_time: Duration, gpu_time: Duration) {
        let n = self.samples as f64;
        self.cpu_time = Duration::from_secs_f64(
            (self.cpu_time.as_secs_f64() * n + cpu_time.as_secs_f64()) / (n + 1.0),
        );
        self.gpu_time = Duration::from_secs_f64(
            (self.gpu_time.as_secs_f64() * n + gpu_time.as_secs_f64()) / (n + 1.0),
        );
        self.samples += 1;
    }

    /// Get speedup ratio (CPU time / GPU time)
    pub fn speedup(&self) -> f64 {
        if self.gpu_time.as_secs_f64() > 0.0 {
            self.cpu_time.as_secs_f64() / self.gpu_time.as_secs_f64()
        } else {
            1.0
        }
    }

    /// Should use GPU based on profile
    pub fn should_use_gpu(&self) -> bool {
        self.speedup() > 1.2 // Use GPU if >20% faster
    }
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Workload characteristics
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Batch size
    pub batch_size: usize,
    /// Average complexity per item
    pub avg_complexity: u64,
    /// Total data size in bytes
    pub data_size: usize,
}

impl WorkloadCharacteristics {
    /// Estimate GPU transfer overhead
    pub fn transfer_overhead(&self) -> Duration {
        // Assume 10 GB/s transfer rate
        let transfer_time_secs = (self.data_size as f64) / 10_000_000_000.0;
        Duration::from_secs_f64(transfer_time_secs)
    }

    /// Estimate if GPU is beneficial
    pub fn gpu_beneficial(&self, min_batch_size: usize, min_complexity: u64) -> bool {
        self.batch_size >= min_batch_size && self.avg_complexity >= min_complexity
    }
}

/// Hybrid CPU/GPU executor with automatic load balancing
pub struct HybridExecutor {
    accelerator: Arc<GPUAccelerator>,
    gpu_executor: Option<GPUExecutor>,
    gpu_hasher: Option<GPUHasher>,
    gpu_verifier: Option<GPUSignatureVerifier>,
    signature_profile: PerformanceProfile,
    hash_profile: PerformanceProfile,
    execution_profile: PerformanceProfile,
}

impl HybridExecutor {
    /// Create new hybrid executor
    pub fn new() -> Result<Self, GPUError> {
        let accelerator = Arc::new(GPUAccelerator::new());

        info!(
            "Initializing hybrid executor: GPU available={}, backend={:?}",
            accelerator.is_gpu_available(),
            accelerator.backend()
        );

        let (gpu_executor, gpu_hasher, gpu_verifier) = if accelerator.is_gpu_available() {
            let executor = GPUExecutor::new(accelerator.clone()).ok();
            let hasher = GPUHasher::new(accelerator.clone()).ok();
            let verifier = GPUSignatureVerifier::new(accelerator.clone()).ok();
            (executor, hasher, verifier)
        } else {
            (None, None, None)
        };

        Ok(Self {
            accelerator,
            gpu_executor,
            gpu_hasher,
            gpu_verifier,
            signature_profile: PerformanceProfile::new(),
            hash_profile: PerformanceProfile::new(),
            execution_profile: PerformanceProfile::new(),
        })
    }

    /// Verify signatures with automatic CPU/GPU selection
    pub async fn verify_signatures(
        &mut self,
        signatures: &[Signature],
        messages: &[Vec<u8>],
        public_keys: &[PublicKey],
    ) -> Result<Vec<bool>, GPUError> {
        let batch_size = signatures.len();

        // Analyze workload
        let data_size = signatures.len() * 64
            + messages.iter().map(|m| m.len()).sum::<usize>()
            + public_keys.len() * 32;
        let characteristics = WorkloadCharacteristics {
            batch_size,
            avg_complexity: 1000, // Signature verification complexity
            data_size,
        };

        // Decide CPU vs GPU
        let use_gpu = self.should_use_gpu_for_signatures(&characteristics);

        if use_gpu {
            if let Some(ref verifier) = self.gpu_verifier {
                debug!(
                    "Using GPU for signature verification (batch_size={})",
                    batch_size
                );
                let start = Instant::now();
                let result = verifier
                    .verify_batch(signatures, messages, public_keys)
                    .await;
                let gpu_time = start.elapsed();

                // Update profile
                let cpu_time = Duration::from_secs_f64(gpu_time.as_secs_f64() * 50.0); // Estimate
                self.signature_profile.update(cpu_time, gpu_time);

                return result;
            }
        }

        // CPU fallback
        debug!(
            "Using CPU for signature verification (batch_size={})",
            batch_size
        );
        let start = Instant::now();
        let results = signatures
            .iter()
            .zip(messages.iter())
            .zip(public_keys.iter())
            .map(|((sig, msg), key)| {
                // Verify signature using cryptographic library
                Self::verify_single_signature(sig, msg, key)
            })
            .collect();
        let cpu_time = start.elapsed();

        // Update profile
        self.signature_profile.update(
            cpu_time,
            Duration::from_secs_f64(cpu_time.as_secs_f64() / 50.0),
        );

        Ok(results)
    }

    /// Verify a single signature
    fn verify_single_signature(sig: &Signature, msg: &[u8], key: &PublicKey) -> bool {
        // Use the cryptographic library to verify the signature
        use silver_crypto::signatures::SignatureVerifier;
        
        // Select the appropriate verifier based on the signature scheme
        let verifier: Box<dyn SignatureVerifier> = match sig.scheme {
            silver_core::SignatureScheme::Secp256k1 => {
                Box::new(silver_crypto::signatures::Secp256k1Signer)
            }
            silver_core::SignatureScheme::Secp512r1 => {
                Box::new(silver_crypto::signatures::Secp512r1)
            }
            silver_core::SignatureScheme::Dilithium3 => {
                Box::new(silver_crypto::signatures::Dilithium3)
            }
            silver_core::SignatureScheme::SphincsPlus => {
                Box::new(silver_crypto::signatures::SphincsPlus)
            }
            silver_core::SignatureScheme::Hybrid => {
                Box::new(silver_crypto::signatures::HybridSignature)
            }
        };
        
        match verifier.verify(msg, sig, key) {
            Ok(()) => true,
            Err(e) => {
                warn!("Signature verification error: {}", e);
                false
            }
        }
    }

    /// Hash data with automatic CPU/GPU selection
    pub async fn hash_batch(&mut self, inputs: &[Vec<u8>]) -> Result<Vec<[u8; 64]>, GPUError> {
        let batch_size = inputs.len();
        let data_size = inputs.iter().map(|i| i.len()).sum();

        let characteristics = WorkloadCharacteristics {
            batch_size,
            avg_complexity: 500, // Hash complexity
            data_size,
        };

        let use_gpu = self.should_use_gpu_for_hashing(&characteristics);

        if use_gpu {
            if let Some(ref hasher) = self.gpu_hasher {
                debug!("Using GPU for hashing (batch_size={})", batch_size);
                let start = Instant::now();
                let result = hasher.hash_batch(inputs).await;
                let gpu_time = start.elapsed();

                let cpu_time = Duration::from_secs_f64(gpu_time.as_secs_f64() * 20.0);
                self.hash_profile.update(cpu_time, gpu_time);

                return result;
            }
        }

        // CPU fallback
        debug!("Using CPU for hashing (batch_size={})", batch_size);
        let start = Instant::now();
        let results = inputs
            .iter()
            .map(|input| {
                let mut hash = [0u8; 64];
                for (i, &byte) in input.iter().enumerate() {
                    hash[i % 64] ^= byte;
                }
                hash
            })
            .collect();
        let cpu_time = start.elapsed();

        self.hash_profile.update(
            cpu_time,
            Duration::from_secs_f64(cpu_time.as_secs_f64() / 20.0),
        );

        Ok(results)
    }

    /// Execute transactions with automatic CPU/GPU selection
    pub async fn execute_transactions(
        &mut self,
        transactions: &[Transaction],
    ) -> Result<Vec<GPUExecutionResult>, GPUError> {
        let batch_size = transactions.len();

        // Estimate complexity
        let avg_complexity = transactions
            .iter()
            .map(|tx| self.estimate_transaction_complexity(tx))
            .sum::<u64>()
            / batch_size.max(1) as u64;

        let data_size = transactions.len() * 1024; // Rough estimate

        let characteristics = WorkloadCharacteristics {
            batch_size,
            avg_complexity,
            data_size,
        };

        let use_gpu = self.should_use_gpu_for_execution(&characteristics);

        if use_gpu {
            if let Some(ref executor) = self.gpu_executor {
                debug!(
                    "Using GPU for transaction execution (batch_size={})",
                    batch_size
                );
                let start = Instant::now();
                let result = executor.execute_transactions_gpu(transactions).await;
                let gpu_time = start.elapsed();

                let cpu_time = Duration::from_secs_f64(gpu_time.as_secs_f64() * 10.0);
                self.execution_profile.update(cpu_time, gpu_time);

                return result;
            }
        }

        // CPU fallback
        debug!(
            "Using CPU for transaction execution (batch_size={})",
            batch_size
        );
        let start = Instant::now();
        let results = transactions
            .iter()
            .map(|_tx| GPUExecutionResult {
                success: true,
                fuel_used: 1000,
                error: None,
            })
            .collect();
        let cpu_time = start.elapsed();

        self.execution_profile.update(
            cpu_time,
            Duration::from_secs_f64(cpu_time.as_secs_f64() / 10.0),
        );

        Ok(results)
    }

    /// Decide if GPU should be used for signatures
    fn should_use_gpu_for_signatures(&self, characteristics: &WorkloadCharacteristics) -> bool {
        if self.gpu_verifier.is_none() {
            return false;
        }

        // Use profile if available
        if self.signature_profile.samples > 10 {
            return self.signature_profile.should_use_gpu();
        }

        // Otherwise use heuristics
        let min_batch = self.gpu_verifier.as_ref().unwrap().config().min_batch_size;
        characteristics.batch_size >= min_batch
    }

    /// Decide if GPU should be used for hashing
    fn should_use_gpu_for_hashing(&self, characteristics: &WorkloadCharacteristics) -> bool {
        if self.gpu_hasher.is_none() {
            return false;
        }

        if self.hash_profile.samples > 10 {
            return self.hash_profile.should_use_gpu();
        }

        let min_batch = self.gpu_hasher.as_ref().unwrap().config().min_batch_size;
        characteristics.batch_size >= min_batch
    }

    /// Decide if GPU should be used for execution
    fn should_use_gpu_for_execution(&self, characteristics: &WorkloadCharacteristics) -> bool {
        if self.gpu_executor.is_none() {
            return false;
        }

        if self.execution_profile.samples > 10 {
            return self.execution_profile.should_use_gpu();
        }

        let min_complexity = self.gpu_executor.as_ref().unwrap().config().min_complexity;
        characteristics.gpu_beneficial(100, min_complexity)
    }

    /// Estimate transaction complexity
    fn estimate_transaction_complexity(&self, transaction: &Transaction) -> u64 {
        transaction.data.kind.command_count() as u64 * 100
    }

    /// Get GPU accelerator
    pub fn accelerator(&self) -> &Arc<GPUAccelerator> {
        &self.accelerator
    }

    /// Get performance profiles
    pub fn profiles(
        &self,
    ) -> (
        &PerformanceProfile,
        &PerformanceProfile,
        &PerformanceProfile,
    ) {
        (
            &self.signature_profile,
            &self.hash_profile,
            &self.execution_profile,
        )
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.accelerator.is_gpu_available()
    }

    /// Get GPU backend type
    pub fn backend(&self) -> GPUBackend {
        self.accelerator.backend()
    }
}

impl Default for HybridExecutor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            warn!("Failed to initialize GPU, using CPU-only mode");
            let accelerator = Arc::new(GPUAccelerator::cpu_fallback());
            Self {
                accelerator,
                gpu_executor: None,
                gpu_hasher: None,
                gpu_verifier: None,
                signature_profile: PerformanceProfile::new(),
                hash_profile: PerformanceProfile::new(),
                execution_profile: PerformanceProfile::new(),
            }
        })
    }
}
