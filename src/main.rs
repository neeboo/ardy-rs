//! `ardy-generate`: end-to-end motion generation from a precomputed text
//! embedding.
//!
//! Replays the golden run's contract (`scripts/generate.py --no-postprocess`
//! upstream): sample the autoregressive diffusion chain, decode the hybrid
//! sequence to explicit motion features, and invert those to world-space
//! joints — writing one `.npy` per output tensor plus an `offsets.json`
//! sidecar, with names matching the golden run's `.npz` members so
//! `--golden` compares 1:1.
//!
//! ```text
//! cargo run --release -- \
//!     --bundle ../ardy-mlx/checkpoints/ARDY-Core-RP-20FPS-Horizon40 \
//!     --text-embedding ../ardy-mlx/artifacts/golden-run-4s/raw/text_feat.npy \
//!     --seed 0 --out /tmp/ardy-out \
//!     --golden ../ardy-mlx/artifacts/golden-run-4s/out/golden_core_h40_4s_seed0.npz
//! ```
//!
//! Noise: per-window `x_T` tensors come from `--noise a.npy b.npy …` when
//! given (the golden run's `randn_*.npy`, MLX cannot reproduce CUDA RNG
//! state), otherwise from `mlx_rs::random` under `--seed`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ardy::io::DEFAULT_CFG_WEIGHT;
use ardy::io::bundle::Bundle;
use ardy::io::npy::{NpyArray, read_npy_f32, read_npz_f32, write_npy_f32};
use ardy::model::decode::{inverse, motion_from_hybrid};
use ardy::model::denoiser::AutoLatentTwostageDenoiser;
use ardy::model::diffusion::Diffusion;
use ardy::model::fsq::FsqVaeTransformer;
use ardy::model::motion_rep::{MotionRep, Stats};
use ardy::model::nn::WeightStore;
use ardy::model::sampler::generate;
use ardy::model::skeleton::Skeleton;
use ardy::{Error, Result};
use mlx_rs::random;
use mlx_rs::{Array, ops};

/// Core checkpoint contract (`conftest.py` / the golden run record).
const FRAMES_PER_TOKEN: usize = 4;
const NFRAME_ROOT_DIM: usize = 20;
const GEN_HORIZON_LEN: usize = 40;
const DEFAULT_CROP_HISTORY: usize = 160;

struct Args {
    bundle: PathBuf,
    text_embedding: PathBuf,
    frames: usize,
    num_steps: usize,
    cfg: (f32, f32),
    history_frames: usize,
    seed: u64,
    noise: Vec<PathBuf>,
    out: PathBuf,
    golden: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        bundle: PathBuf::new(),
        text_embedding: PathBuf::new(),
        frames: 80,
        num_steps: 10,
        cfg: (DEFAULT_CFG_WEIGHT.0 as f32, DEFAULT_CFG_WEIGHT.1 as f32),
        history_frames: DEFAULT_CROP_HISTORY,
        seed: 0,
        noise: Vec::new(),
        out: PathBuf::from("out"),
        golden: None,
    };

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    let value = |i: &mut usize, argv: &[String], what: &str| -> Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| Error::Contract(format!("--{what} needs a value (see --help)")))
    };
    while i < argv.len() {
        let flag = argv[i].clone();
        match flag.as_str() {
            "--bundle" => args.bundle = value(&mut i, &argv, "bundle")?.into(),
            "--text-embedding" => {
                args.text_embedding = value(&mut i, &argv, "text-embedding")?.into()
            }
            "--frames" => {
                args.frames = value(&mut i, &argv, "frames")?
                    .parse()
                    .map_err(|_| Error::Contract("--frames must be an integer".to_string()))?
            }
            "--num-steps" => {
                args.num_steps = value(&mut i, &argv, "num-steps")?
                    .parse()
                    .map_err(|_| Error::Contract("--num-steps must be an integer".to_string()))?
            }
            "--cfg" => {
                let v = value(&mut i, &argv, "cfg")?;
                let parts: Vec<&str> = v.split(',').collect();
                args.cfg = match parts.as_slice() {
                    [t] => (t.parse().map_err(|_| bad_cfg(&v))?, args.cfg.1),
                    [t, c] => (
                        t.parse().map_err(|_| bad_cfg(&v))?,
                        c.parse().map_err(|_| bad_cfg(&v))?,
                    ),
                    _ => return Err(bad_cfg(&v)),
                };
            }
            "--history-frames" => {
                args.history_frames =
                    value(&mut i, &argv, "history-frames")?
                        .parse()
                        .map_err(|_| {
                            Error::Contract("--history-frames must be an integer".to_string())
                        })?
            }
            "--seed" => {
                args.seed = value(&mut i, &argv, "seed")?
                    .parse()
                    .map_err(|_| Error::Contract("--seed must be an integer".to_string()))?
            }
            "--noise" => {
                // one or more .npy paths, up to the next --flag
                while i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                    i += 1;
                    args.noise.push(argv[i].clone().into());
                }
                if args.noise.is_empty() {
                    return Err(Error::Contract(
                        "--noise needs at least one .npy path (see --help)".to_string(),
                    ));
                }
            }
            "--out" => args.out = value(&mut i, &argv, "out")?.into(),
            "--golden" => args.golden = Some(value(&mut i, &argv, "golden")?.into()),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(Error::Contract(format!(
                    "unknown argument {other:?} (see --help)"
                )));
            }
        }
        i += 1;
    }
    if args.bundle.as_os_str().is_empty() || args.text_embedding.as_os_str().is_empty() {
        return Err(Error::Contract(
            "--bundle and --text-embedding are required (see --help)".to_string(),
        ));
    }
    Ok(args)
}

fn bad_cfg(v: &str) -> Error {
    Error::Contract(format!(
        "--cfg must be `TEXT[,CONSTRAINT]` floats, got {v:?}"
    ))
}

fn print_help() {
    println!(
        "ardy-generate — ARDY motion generation (MLX, Rust)

USAGE:
  ardy-generate --bundle DIR --text-embedding FILE.npy [flags]

REQUIRED:
  --bundle DIR            checkpoint bundle (ARDY-Core-RP-20FPS-Horizon40)
  --text-embedding FILE   precomputed text embedding (.npy)

OPTIONS:
  --frames N              output frames (default 80; multiple of 4)
  --num-steps N           DDIM steps per window (default 10)
  --cfg T[,C]             classifier-free guidance weights (default 2,2)
  --history-frames N      autoregressive history crop (default 160)
  --seed N                mlx-rs RNG seed for the noise fallback (default 0)
  --noise F.npy [F.npy …] per-window x_T tensors; replaces --seed.
                          Must be the last flag on the command line.
  --out DIR               output directory (default ./out)
  --golden FILE.npz       compare outputs against a golden archive
  -h, --help"
    );
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

struct Output {
    name: &'static str,
    array: Array,
}

fn run(args: &Args) -> Result<()> {
    let bundle = Bundle::open(&args.bundle)?;
    let mr = MotionRep::from_bundle(&bundle)?;
    let diffusion = Diffusion::new(args.num_steps)?;
    let mut dstore = WeightStore::new(bundle.denoiser().clone());
    let denoiser = AutoLatentTwostageDenoiser::new(&mut dstore, &mr, "denoiser")?;
    let mut tstore = WeightStore::new(bundle.tokenizer().clone());
    let mut fsq = FsqVaeTransformer::new(&mut tstore, &mr, "pose_net")?;
    let post = bundle.stats("post_quantization")?;
    let pre = bundle.stats("pre_quantization")?;
    fsq.load_stats(
        Stats::from_npy(&post.mean, &post.std)?,
        Stats::from_npy(&pre.mean, &pre.std)?,
    );

    let text = read_npy_f32(&args.text_embedding)?;
    let dims: Vec<i32> = text.shape.iter().map(|&d| d as i32).collect();
    let text_feat = Array::from_slice(&text.data, &dims);
    let bs = text.shape[0] as i32;

    let gen_tokens = (GEN_HORIZON_LEN / FRAMES_PER_TOKEN) as i32;
    let hybrid_width = NFRAME_ROOT_DIM + fsq.latent_dim();
    let noise_shape = [bs, gen_tokens, hybrid_width as i32];

    let noise: Box<dyn Fn(usize) -> Result<Array>> = if args.noise.is_empty() {
        random::seed(args.seed)?;
        let key = random::key(args.seed)?;
        eprintln!("noise: mlx-rs RNG, seed {}", args.seed);
        Box::new(move |_| random::normal::<f32>(&noise_shape, None, None, &key).map_err(Into::into))
    } else {
        let files = args.noise.clone();
        eprintln!("noise: {} file(s) from --noise", files.len());
        Box::new(move |w: usize| {
            let path = files.get(w).ok_or_else(|| {
                Error::Contract(format!(
                    "--noise has {} file(s) but window {w} was needed",
                    files.len()
                ))
            })?;
            let n = read_npy_f32(path)?;
            let dims: Vec<i32> = n.shape.iter().map(|&d| d as i32).collect();
            Ok(Array::from_slice(&n.data, &dims))
        })
    };

    eprintln!(
        "sampling: {} frames, {} window(s), {} step(s), cfg {:?}",
        args.frames,
        args.frames.div_ceil(GEN_HORIZON_LEN),
        args.num_steps,
        args.cfg
    );
    let out = generate(
        &denoiser,
        &fsq,
        &mr,
        &diffusion,
        &text_feat,
        args.frames,
        args.num_steps,
        args.cfg,
        None, // first_heading_angle: the CLI starts standing, heading 0
        GEN_HORIZON_LEN,
        Some(args.history_frames),
        &*noise,
        None::<&mut dyn FnMut(usize, usize, &Array)>,
    )?;

    let features = motion_from_hybrid(
        &out.hybrid_translated,
        &fsq,
        &mr,
        FRAMES_PER_TOKEN,
        NFRAME_ROOT_DIM,
        GEN_HORIZON_LEN,
        Some(args.history_frames),
        args.frames,
    )?;
    let decoded = inverse(&mr, Skeleton::core(), &features, true)?;

    let f32_of = |a: &Array| a.as_dtype(mlx_rs::Dtype::Float32).map_err(Error::from);
    let outputs = vec![
        Output {
            name: "motion_normalized",
            array: features,
        },
        Output {
            name: "local_rot_mats",
            array: decoded.local_rot_mats.clone(),
        },
        Output {
            name: "global_rot_mats",
            array: decoded.global_rot_mats.clone(),
        },
        Output {
            name: "posed_joints",
            array: decoded.posed_joints.clone(),
        },
        Output {
            name: "root_positions",
            array: decoded.root_positions.clone(),
        },
        Output {
            name: "smooth_root_pos",
            array: decoded.root_positions.clone(),
        },
        Output {
            name: "foot_contacts",
            array: f32_of(&decoded.foot_contacts)?,
        },
        Output {
            name: "global_root_heading",
            array: decoded.global_root_heading.clone(),
        },
    ];

    std::fs::create_dir_all(&args.out).map_err(|source| Error::Io {
        path: args.out.clone(),
        source,
    })?;
    for o in &outputs {
        let flat = o.array.reshape(&[-1])?;
        flat.eval().map_err(Error::from)?;
        let shape: Vec<usize> = o.array.shape().iter().map(|&d| d as usize).collect();
        write_npy_f32(
            &args.out.join(format!("{}.npy", o.name)),
            flat.as_slice::<f32>(),
            &shape,
        )?;
    }

    // the sampling sidecar: cumulative token->frame offsets and the
    // accumulated recentering translation
    let transl = ops::reshape(&out.global_transl, &[-1])?;
    transl.eval().map_err(Error::from)?;
    let sidecar = serde_json::json!({
        "frames": args.frames,
        "num_steps": args.num_steps,
        "cfg": args.cfg,
        "history_frames": args.history_frames,
        "seed": args.seed,
        "offsets": out.offsets,
        "global_transl": transl.as_slice::<f32>(),
    });
    std::fs::write(
        args.out.join("offsets.json"),
        serde_json::to_string_pretty(&sidecar)
            .map_err(|e| Error::Npy(format!("offsets.json: {e}")))?,
    )
    .map_err(|source| Error::Io {
        path: args.out.join("offsets.json"),
        source,
    })?;

    eprintln!("wrote {} tensors to {}", outputs.len(), args.out.display());
    if let Some(golden) = &args.golden {
        compare_golden(&outputs, golden)?;
    }
    Ok(())
}

fn compare_golden(outputs: &[Output], golden: &Path) -> Result<()> {
    let members = read_npz_f32(golden)?;
    let as_arr = |n: &NpyArray| {
        let dims: Vec<i32> = n.shape.iter().map(|&d| d as i32).collect();
        Array::from_slice(&n.data, &dims)
    };
    println!("{:<22} {:>12}", "tensor", "max abs diff");
    for o in outputs {
        let key = format!("{o}.npy", o = o.name);
        let Some(n) = members
            .iter()
            .find(|(name, _)| *name == key || name.strip_suffix(".npy") == Some(o.name))
            .map(|(_, a)| a)
        else {
            println!("{:<22} {:>12}", o.name, "(no golden member)");
            continue;
        };
        let golden = as_arr(n);
        let flat_gold = golden.reshape(&[-1])?;
        let flat_mine = o.array.reshape(&[-1])?;
        if flat_gold.shape() != flat_mine.shape() {
            println!(
                "{:<22} {:>12}",
                o.name,
                format!("shape {:?} vs {:?}", flat_mine.shape(), flat_gold.shape())
            );
            continue;
        }
        let diff = ops::max(&ops::abs(ops::subtract(&flat_mine, &flat_gold)?)?, None)?;
        let d = diff.reshape(&[-1])?;
        d.eval().map_err(Error::from)?;
        let v = d.as_slice::<f32>()[0];
        println!("{:<22} {:>12.3e}", o.name, v);
    }
    Ok(())
}
