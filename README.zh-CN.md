English: [README.md](README.md)

# ardy-rs

**ARDY**(NVIDIA,SIGGRAPH 2026 —— 基于混合动作表征的自回归扩散模型)的
Rust 实现,运行于 Apple Silicon。移植自
[`ardy-mlx`](https://github.com/neeboo/ardy-mlx)(其对齐参考实现
`nv-tlabs/ardy` 至 float32 舍入级,约 2e-05),通过
[`mlx-rs`](https://crates.io/crates/mlx-rs) 跑在
[MLX](https://github.com/ml-explore/mlx) 上。

**状态:对 `ARDY-Core-RP-20FPS-Horizon40` 检查点已完成**
(`cskel27` 核心骨架,`has_double_ankle_joints = false`):

| 里程碑 | 范围 | 状态 |
|---|---|---|
| M0 | 检查点 bundle I/O:契约常量、safetensors 权重、stats `.npy` | 完成 |
| M1 | nn 算子层(linear、layernorm、gelu/silu、PE、mask、attention) | 完成 |
| M2 | Stage A tokenizer(FSQ VAE transformer)—— golden 逐值复现 | 完成 |
| M3 | Stage B denoiser(twostage,3 路 CFG batch)—— 对齐至 1.4e-6 | 完成 |
| M4 | 采样:cosine schedule、DDIM `eta=0`、自回归滑窗 | 完成 |
| M5 | motion_rep 解码 + 骨架 FK + CLI,端到端对齐 golden(5.2e-6) | 完成 |
| M6 | MotionCorrection 移植(轨迹校正 + IK 后处理),与真实 C++ 差分 ≤ 1e-5 | 完成 |
| M7 | 文档、署名 | 完成 |

与 ardy-mlx v1 相同的范围外项:8B 文本编码器(只吃预计算 embedding)、
CUDA/TensorRT、实时流式生成。

## 命令行

```sh
cargo run --release -- \
    --bundle ../ardy-mlx/checkpoints/ARDY-Core-RP-20FPS-Horizon40 \
    --text-embedding ../ardy-mlx/artifacts/golden-run-4s/raw/text_feat.npy \
    --seed 0 --frames 80 \
    --out /tmp/ardy-out \
    --postprocess
```

每个输出张量写一个 `.npy`(`motion_normalized`、`local_rot_mats`、
`global_rot_mats`、`posed_joints`、`root_positions`、`smooth_root_pos`、
`foot_contacts`、`global_root_heading`),外加 `offsets.json` 边车文件。

主要参数(全部见 `--help`):

- `--noise a.npy b.npy …` —— 每个窗口的 `x_T` 张量,替代 `--seed` 的
  RNG(golden run 用的是 CUDA 噪声,MLX 无法复现其随机状态)。
- `--golden FILE.npz` —— 与 golden 归档 1:1 对比。
- `--postprocess` —— 对解码后的关节跑 MotionCorrection 的 IK/轨迹校正
  后处理(上游默认参数,无约束)。默认关闭;golden 契约不带它跑。

## 测试

```sh
cargo test
```

golden 门控的测试套件从本地 ardy-mlx 检出回放中间张量与最终片段
(默认发现路径 `../ardy-mlx`,可用 `ARDY_BUNDLE` 覆盖);没有 bundle
时自动跳过。

MotionCorrection 移植额外做了**对真实上游 C++ 的差分验证**:
`tests/postprocess.rs` 用合成的 LCG 记录(不含 NVIDIA 数据)同时喂给
Rust 移植版和由未改动上游源码编译出的探针二进制。构建探针见
`scripts/mc_probe/README.md`;没有探针时这些测试空转通过。

## 一个已验证的上游怪癖

差分探针证实,上游 `TrajectoryCorrector::computeDiffMats` 的速度加权是
**死代码**:`multVelWeights` 把加权三元组算进局部向量,却从未调用
`setFromTriplets` 写回,因此速度差分矩阵 `V` 始终未加权。Rust 移植版
镜像实测行为,并在 `DiffMats` 的文档注释中记录——包括自由帧在内,
约束帧与 C++ 完全一致(≤ 1e-5)。

## 检查点

转换后的 MLX 权重(`checkpoints/ARDY-Core-RP-20FPS-Horizon40/mlx/`)
**不入库**,golden 输出同样不入库(NVIDIA 条款,见 `NOTICE.md`)。
测试指向本地 ardy-mlx 检出(默认发现同级目录
`../ardy-mlx/checkpoints/...`,可用 `ARDY_BUNDLE` 覆盖)。

## 许可

Apache-2.0。署名与检查点条款记录见 `NOTICE.md`。
