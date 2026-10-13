//! Error type shared across the crate.

use std::path::PathBuf;

/// Everything that can go wrong while loading or running ARDY.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Filesystem access failed.
    #[error("{path}: {source}")]
    Io {
        /// The path the failed operation touched.
        path: PathBuf,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// A safetensors container is malformed or has unexpected dtypes.
    #[error("safetensors: {0}")]
    Safetensors(String),

    /// An `.npy` file is malformed or has an unsupported dtype.
    #[error("npy: {0}")]
    Npy(String),

    /// A checkpoint bundle is incomplete or inconsistent with the contract.
    #[error("bundle: {0}")]
    Bundle(String),

    /// A checkpoint field disagrees with the frozen motion contract.
    #[error("contract: {0}")]
    Contract(String),
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
