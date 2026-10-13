//! Integration tests against the real converted checkpoint bundle.
//!
//! Skipped with a message when the bundle is absent, mirroring ardy-mlx's
//! test conventions. Default discovery: sibling `../ardy-mlx/checkpoints`.

use std::path::{Path, PathBuf};

use ardy::io::{FeatureLayout, preset};

fn bundle_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ARDY_BUNDLE") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
        return None;
    }
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ardy-mlx/checkpoints/ARDY-Core-RP-20FPS-Horizon40");
    p.is_dir().then_some(p)
}

#[test]
fn contract_constants_and_presets() {
    let core = preset("ARDY-Core-RP-20FPS-Horizon40").expect("core preset");
    assert_eq!(
        core,
        FeatureLayout {
            nbjoints: 27,
            fps: 20,
            horizon: 40,
            skeleton: "CoreSkeleton27",
        }
    );
    assert!(preset("not-a-preset").is_none());
}

#[test]
fn opens_bundle_and_matches_contract() {
    let Some(dir) = bundle_dir() else {
        eprintln!("skip: no bundle (set ARDY_BUNDLE)");
        return;
    };
    let bundle = ardy::io::bundle::Bundle::open(&dir).expect("bundle opens");

    // Converted safetensors tensor counts (weight-manifest.json).
    assert_eq!(bundle.tokenizer().len(), 204);
    assert_eq!(bundle.denoiser().len(), 224);

    // Config sanity against the contract constants.
    let cfg = bundle.config();
    assert_eq!(cfg.num_base_steps, 10);
    assert_eq!(cfg.cfg_type, "separated");
    assert_eq!(cfg.gen_horizon_len, 40);
    assert_eq!(cfg.autoencoder.latent_dim, 512);
    assert_eq!(cfg.autoencoder.num_layers, 8);
    assert_eq!(cfg.autoencoder.num_heads, 4);
    assert!(cfg.autoencoder.causal_encoder && cfg.autoencoder.causal_decoder);
    assert_eq!(cfg.denoiser.latent_dim, 1024);
    assert_eq!(cfg.denoiser.num_layers, 8);
    assert_eq!(cfg.denoiser.num_heads, 8);
    assert!(!cfg.denoiser.use_text_mask);
    assert_eq!(cfg.denoiser.llm_shape, vec![1, 4096]);

    // FeatureLayout derived from the config's motion_rep + horizon.
    let layout = bundle.layout();
    assert_eq!(layout.nbjoints, 27);
    assert_eq!(layout.fps, 20);
    assert_eq!(layout.motion_rep_dim(), 330);
    assert_eq!(layout.dim_token(), 148);
    assert_eq!(layout.default_history_frames(), 160);

    // Spot-check known weight tensors (all F32, PyTorch [out, in] layout).
    let proj = bundle
        .denoiser()
        .get("denoiser.backbone.body_model.embed_text.weight")
        .expect("embed_text weight");
    assert_eq!(proj.shape, vec![1024, 4096]);
    let prefix = bundle
        .denoiser()
        .get("denoiser.backbone.body_model.learned_prefix_embedding.embedding.weight")
        .expect("learned prefix");
    assert_eq!(prefix.shape, vec![3, 1024]);
}

#[test]
fn stats_have_contract_widths() {
    let Some(dir) = bundle_dir() else {
        eprintln!("skip: no bundle (set ARDY_BUNDLE)");
        return;
    };
    let bundle = ardy::io::bundle::Bundle::open(&dir).expect("bundle opens");

    // motion stats are 334-wide on disk and sliced to 330 by the model code.
    let motion = bundle.stats("motion").expect("motion stats");
    assert_eq!(motion.mean.shape, vec![334]);
    assert_eq!(motion.std.shape, vec![334]);

    for kind in ["pre_quantization", "post_quantization"] {
        let s = bundle.stats(kind).unwrap_or_else(|e| panic!("{kind}: {e}"));
        assert_eq!(s.mean.shape, vec![128], "{kind} mean width");
        assert_eq!(s.std.shape, vec![128], "{kind} std width");
        assert!(s.std.data.iter().all(|v| *v > 0.0), "{kind} std positive");
    }
}
