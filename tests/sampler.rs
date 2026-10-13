//! Stage-sampler replay against the golden torch-reference dumps
//! (`ardy-mlx/artifacts/golden-run-4s/raw`).
//!
//! The golden run generated 80 frames = two autoregressive windows with a
//! deterministic `x_T` per window (`randn_{0,1}`), so the whole sampling chain
//! is replayable bit-for-bit at float32 roundoff:
//!
//! * window 0: 10 DDIM steps from `randn_0` (`denoise_step{0..9}_x_out`),
//! * inter-window recenter / requantize / heading refresh
//!   (`recenter0_*` = `window1_history_in`),
//! * window 1: 10 more steps (`denoise_step{10..19}_x_out`),
//! * the final translate (`final_hybrid_translated`).
//!
//! `final_motion_decoded_autoregressive` needs the M5 decoder, not the sampler.

use ardy::io::{DEFAULT_CFG_WEIGHT, bundle::Bundle};
use ardy::model::denoiser::{AutoLatentTwostageDenoiser, split_hybrid_free};
use ardy::model::diffusion::Diffusion;
use ardy::model::fsq::FsqVaeTransformer;
use ardy::model::motion_rep::{MotionRep, Stats};
use ardy::model::nn::WeightStore;
use ardy::model::sampler::{
    convert_frame_mask_to_token_mask, generate, get_three_mask_from_len, recenter_step,
};
use mlx_rs::Array;
use mlx_rs::ops;

mod common;
use common::*;

/// The golden mask/token-mask dumps rebuilt through the Rust helpers, per
/// window. Also pins the `latent_utils.py:207` trap: window 0's frame-level
/// future mask covers 40 frames while its token-level future mask is all
/// false.
#[test]
fn golden_replay_mask_helpers() {
    let Some(gdir) = golden_dir() else {
        eprintln!("skip: no golden run (set ARDY_GOLDEN)");
        return;
    };

    for k in 0..2 {
        let masks = npz(&gdir, &format!("masks_{k}.npz"));
        let tmasks = npz(&gdir, &format!("token_masks_{k}.npz"));
        let (h, g, f) = get_three_mask_from_len(
            &arr(npz_member(&masks, "history_len")),
            &arr(npz_member(&masks, "generation_len")),
            &arr(npz_member(&masks, "future_len")),
            80,
        )
        .unwrap();
        assert_close(
            &flat(&h),
            &flat(&arr(npz_member(&masks, "history"))),
            0.0,
            &format!("window{k} history mask"),
        );
        assert_close(
            &flat(&g),
            &flat(&arr(npz_member(&masks, "generation"))),
            0.0,
            &format!("window{k} generation mask"),
        );
        assert_close(
            &flat(&f),
            &flat(&arr(npz_member(&masks, "future"))),
            0.0,
            &format!("window{k} future mask"),
        );

        let (th, tg, tf) = convert_frame_mask_to_token_mask(&h, &g, &f, None, 4).unwrap();
        // flat() reads f32; the golden dumps stored every mask (bool included)
        // as f32, so cast both sides before comparing
        let as_f32 = |b: &Array| b.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        assert_close(
            &flat(&as_f32(&th)),
            &flat(&as_f32(&bool_arr(npz_member(&tmasks, "history")))),
            0.0,
            &format!("window{k} history token mask"),
        );
        assert_close(
            &flat(&as_f32(&tg)),
            &flat(&as_f32(&bool_arr(npz_member(&tmasks, "generation")))),
            0.0,
            &format!("window{k} generation token mask"),
        );
        assert_close(
            &flat(&as_f32(&tf)),
            &flat(&as_f32(&bool_arr(npz_member(&tmasks, "future")))),
            0.0,
            &format!("window{k} future token mask"),
        );
    }

    // the trap, asserted on the rebuilt window-0 masks: 40 future frames, 0
    // future tokens
    let masks = npz(&gdir, "masks_0.npz");
    let tmasks = npz(&gdir, "token_masks_0.npz");
    let future_frames: f32 = flat(&arr(npz_member(&masks, "future"))).iter().sum();
    // count straight off the dump's f32 payload (flat() only reads float arrays)
    let future_tokens: f32 = npz_member(&tmasks, "future")
        .data
        .iter()
        .filter(|&&x| x > 0.5)
        .count() as f32;
    assert_eq!(future_frames, 40.0);
    assert_eq!(future_tokens, 0.0);
}

/// The full two-window autoregressive generate: one run, all 20 per-step
/// post-DDIM tensors checked, then the endpoint triple (offsets, accumulated
/// translation, translated hybrid).
#[test]
fn golden_replay_generate_two_windows() {
    let Some(bdir) = bundle_dir() else {
        eprintln!("skip: no bundle (set ARDY_BUNDLE)");
        return;
    };
    let Some(gdir) = golden_dir() else {
        eprintln!("skip: no golden run (set ARDY_GOLDEN)");
        return;
    };

    let bundle = Bundle::open(&bdir).unwrap();
    let mr = MotionRep::from_bundle(&bundle).unwrap();
    let diffusion = Diffusion::new(10).unwrap();
    let mut dstore = WeightStore::new(bundle.denoiser().clone());
    let denoiser = AutoLatentTwostageDenoiser::new(&mut dstore, &mr, "denoiser").unwrap();
    let mut tstore = WeightStore::new(bundle.tokenizer().clone());
    let mut fsq = FsqVaeTransformer::new(&mut tstore, &mr, "pose_net").unwrap();
    let post = bundle.stats("post_quantization").unwrap();
    let pre = bundle.stats("pre_quantization").unwrap();
    fsq.load_stats(
        Stats::from_npy(&post.mean, &post.std).unwrap(),
        Stats::from_npy(&pre.mean, &pre.std).unwrap(),
    );

    let text_feat = arr(&npy(&gdir, "text_feat.npy"));
    let heading0 = arr(&npy(&gdir, "window0_first_heading_in.npy"));
    let (w_text, w_cstr) = DEFAULT_CFG_WEIGHT;

    let out = generate(
        &denoiser,
        &fsq,
        &mr,
        &diffusion,
        &text_feat,
        80,
        10,
        (w_text as f32, w_cstr as f32),
        Some(heading0),
        40,
        Some(160),
        &|w| Ok(arr(&npy(&gdir, &format!("randn_{w}.npy")))),
        Some(&mut |w: usize, i: usize, x: &Array| {
            // identity subsample: call k of window w runs loop index 9-i
            let k = w * 10 + (9 - i);
            assert_close(
                &flat(x),
                &flat(&arr(&npy(&gdir, &format!("denoise_step{k}_x_out.npy")))),
                // measured: worst 8.8e-5 (trig roundoff compounds mildly over
                // a window's 10 DDIM steps); 10x headroom for MLX versions
                5e-4,
                &format!("denoising step {k}"),
            );
        }),
    )
    .unwrap();

    assert_eq!(out.offsets, vec![0, 40, 80], "offsets");

    // accumulated translation == the two recenter centers summed
    let want_transl = ops::add(
        arr(&npy(&gdir, "recenter0_center_pos.npy")),
        arr(&npy(&gdir, "recenter1_center_pos.npy")),
    )
    .unwrap();
    assert_close(
        &flat(&out.global_transl),
        &flat(&want_transl),
        1e-4,
        "global_transl",
    );

    assert_close(
        &flat(&out.hybrid_translated),
        &flat(&arr(&npy(&gdir, "final_hybrid_translated.npy"))),
        // measured: 3.5e-6
        1e-4,
        "final hybrid translated",
    );
}

/// The inter-window boundary replayed in isolation from the dumps:
/// `recenter_step` on the recorded window outputs must reproduce the recorded
/// recentered histories, the centers, and the refreshed headings.
#[test]
fn golden_replay_recenter_boundary() {
    let Some(bdir) = bundle_dir() else {
        eprintln!("skip: no bundle (set ARDY_BUNDLE)");
        return;
    };
    let Some(gdir) = golden_dir() else {
        eprintln!("skip: no golden run (set ARDY_GOLDEN)");
        return;
    };

    let bundle = Bundle::open(&bdir).unwrap();
    let mr = MotionRep::from_bundle(&bundle).unwrap();
    let mut tstore = WeightStore::new(bundle.tokenizer().clone());
    let mut fsq = FsqVaeTransformer::new(&mut tstore, &mr, "pose_net").unwrap();
    let post = bundle.stats("post_quantization").unwrap();
    let pre = bundle.stats("pre_quantization").unwrap();
    fsq.load_stats(
        Stats::from_npy(&post.mean, &post.std).unwrap(),
        Stats::from_npy(&pre.mean, &pre.std).unwrap(),
    );

    // window 0 -> 1: history is 10 tokens, recenter center 0 + 40 - 1 = 39
    let hist0 = arr(&npy(&gdir, "window0_history_out.npy"));
    let rec0 = recenter_step(&hist0, &mr, &fsq, 4, 20, 0, 40, Some(160), true).unwrap();
    assert_close(
        &flat(&rec0.center_pos),
        &flat(&arr(&npy(&gdir, "recenter0_center_pos.npy"))),
        // measured: exact 0.0 (pure slicing)
        1e-6,
        "recenter0 center_pos",
    );
    assert_close(
        &flat(&rec0.history),
        &flat(&arr(&npy(&gdir, "recenter0_history_recentered.npy"))),
        // measured: 2.4e-7
        1e-5,
        "recenter0 history",
    );
    // the next window's history is exactly this tensor
    assert_close(
        &flat(&rec0.history),
        &flat(&arr(&npy(&gdir, "window1_history_in.npy"))),
        1e-5,
        "boundary: window1 history_in",
    );
    // refreshed heading is `heading[:, cropped_start]` -> [B]; crop 160 of an
    // 80-frame sequence never truncates, so cropped_start = 0
    assert_close(
        &flat(&rec0.heading.unwrap()),
        &flat(&arr(&npy(&gdir, "window1_first_heading_in.npy"))),
        1e-5,
        "boundary: window1 first_heading",
    );

    // window 1 end: history is 20 tokens, center 40 + 40 - 1 = 79
    let hist1 = arr(&npy(&gdir, "window1_history_out.npy"));
    let rec1 = recenter_step(&hist1, &mr, &fsq, 4, 20, 40, 40, Some(160), true).unwrap();
    assert_close(
        &flat(&rec1.center_pos),
        &flat(&arr(&npy(&gdir, "recenter1_center_pos.npy"))),
        1e-6,
        "recenter1 center_pos",
    );
    assert_close(
        &flat(&rec1.history),
        &flat(&arr(&npy(&gdir, "recenter1_history_recentered.npy"))),
        1e-5,
        "recenter1 history",
    );
    // the refreshed heading for window 1 has no dedicated npy dump; check the
    // shape contract ([B], a scalar frame index per batch row)
    assert_eq!(
        rec1.heading.unwrap().shape(),
        &[1],
        "recenter1 heading shape"
    );

    // and the final translate of the last recentered history is the endpoint
    let (root_motion, latent) =
        split_hybrid_free(&rec1.history, 4, 20, mr.root_block_dim()).unwrap();
    let global_transl = ops::add(
        arr(&npy(&gdir, "recenter0_center_pos.npy")),
        arr(&npy(&gdir, "recenter1_center_pos.npy")),
    )
    .unwrap();
    let new_root = mr
        .translate_normalized_root_motion(&root_motion, &global_transl)
        .unwrap();
    let final_hybrid =
        AutoLatentTwostageDenoiser::join_hybrid(&new_root.reshape(&[1, -1, 20]).unwrap(), &latent)
            .unwrap();
    assert_close(
        &flat(&final_hybrid),
        &flat(&arr(&npy(&gdir, "final_hybrid_translated.npy"))),
        // measured: 3.5e-6
        1e-4,
        "final hybrid translated (boundary path)",
    );
}
