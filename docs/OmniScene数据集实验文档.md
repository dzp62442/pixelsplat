# OmniScene 数据集实验说明（规划稿）

## 配置概览
- **基础入口**：沿用 `config/main.yaml` 的 Hydra 结构，命令格式与 README 中的 `python3 -m src.main +experiment=re10k ...` 完全一致。针对 OmniScene，我们将新增 `+experiment=omniscene_112x200` 与 `+experiment=omniscene_224x400` 两套实验文件，通过 `python3 -m src.main +experiment=omniscene_112x200`（或 224x400 版本）启动训练/测试。
- **数据集配置文件**：新增 `config/dataset/omniscene.yaml`，结构仿照 `re10k`，但设置 `name: omniscene`、`defaults.view_sampler: all`，并将 `roots` 指向 `datasets/omniscene`。基础分辨率默认为 `224x400`，并保持黑色背景/非环形相机；是否改为 `112x200` 交由实验文件覆盖。
- **实验配置文件**：`config/experiment/omniscene_112x200.yaml` 与 `config/experiment/omniscene_224x400.yaml` 主要做以下覆盖：
  - 将 `dataset` 指向 OmniScene，并分别设置 `image_shape=[112,200]` 与 `[224,400]`。
  - `data_loader.train/val/test.batch_size = 1`，确保三阶段 batch size 一致。
  - `trainer.max_steps = 100_001`，`trainer.val_check_interval = 0.01`。Lightning 以“epoch”为 chunk 计算，因此 0.01 表示每累计约 1% 的数据就做一次验证。
  - `wandb.name/tags` 标注实验分辨率，方便与 re10k 区分。
  - `loss`、`model.encoder/decoder`、`optimizer.lr` 等参数维持与 re10k 一致（即仍使用 epipolar encoder + CUDA splatting decoder + mse/lpips 配置），确保对比实验仅改变数据。
- **训练/验证/测试节奏**：
  - 训练 batch size 同上均为 1。
  - 验证频率由 `trainer.val_check_interval=0.01` 控制。
  - README 中未提供 `eval_model_every_n_val` 的等价字段；本项目的 `train` 配置不包含该开关，因此阶段性测试可通过显式运行 `mode=test` 的指令完成（无法直接在训练循环内自动触发）。
  - 测试命令也沿用 README 样式：`python3 -m src.main +experiment=omniscene_112x200 mode=test dataset/view_sampler=evaluation ...`；evaluation sampler 的参数将视需要新增（默认维持 `all`，如需固定评测列表，可以仿照 re10k 的 assets）。

## 数据加载流程
1. **注册入口**：与 DepthSplat 相同，需在 `src/dataset/__init__.py` 中注册 `DatasetOmniScene`，以便 `DataModule` 通过 `cfg.dataset.name` 自动实例化。
2. **数据集实现**：
   - 我们将新建 `src/dataset/dataset_omniscene.py` 和配套工具函数（如 `utils_omniscene.py`），整体逻辑参考 depthsplat：根据阶段读取 `bins_*_3.2m.json`，再从 `bin_infos_3.2m/{token}.pkl` 中抽取各摄像头帧。
   - **Train/Val/Test 切分**：stage="train" 读取 `bins_train_3.2m.json` 全量；stage="val" 采用“前 30000 个 bin，每隔 3000 取 1 个，再取前 10 个”策略；stage="test" 默认保留 mini-test（`[0::14][:2048]`），若未来需要完整测试集，可在本文件中放宽限制。
   - **输入/输出视图**：与 depthsplat 一样，默认 6 个 key-frame 作为 context，输出帧包含每个摄像头 index `[1,2]` 的 sweeps，再拼回输入帧以便监督。由于 pixelsplat 依赖 view sampler，我们将 `defaults.view_sampler: all` 并在数据集内部直接构建 context/target 字段，不再调用 sampler。
   - **图像/掩码处理**：复用 depthsplat 的 `load_conditions` 思路，把 JPEG 缩放到目标分辨率并按宽/高归一化 intrinsics；输出帧带入动态掩码，输入帧保持全 1。为了兼容 `EncoderEpipolar` 的 patch shim，还需要在 shim 中检测 `masks` 字段并同步裁剪（类似 depthsplat 中对 mask 的扩展）。
   - **Near/Far**：配置文件中给定 `near=0.5`、`far=100.0`（如 depthsplat 112x200/224x400 实验），读取后通过 `repeat` 扩展到各视角。若需要根据实际 baseline 缩放，可额外引入 `baseline_scale_bounds` 开关。
   - **返回格式**：输出与 re10k 相同的 dict（`context`/`target`/`scene`）。唯一的新增字段是 `target["masks"]`，供后续损失（可选）屏蔽动态区域；该字段是可选项，不影响现有无掩码代码。
3. **差异与复用**：
   - DepthSplat 的数据集继承自常规 `Dataset`，像素 Splat 当前的 `DatasetRE10k` 则是 `IterableDataset` 并通过预处理 chunk 提供固定形状图像。OmniScene 使用 JSON/PKL 索引和多相机结构，两边的数据排布完全不同，无法直接把 depthsplat 的 loader 拷贝过来；但逻辑流程（解析 bin、加载 JPEG、生成 context/target）可以复用，只需把路径/依赖改为 pixelsplat 的工具链。
   - DepthSplat 还需要 `utils_omniscene.get_ray_directions` 等深度计算；pixelSplat 不直接用这些函数，但图像 resize/内参归一化/掩码读取可以照搬。新的工具文件会只保留必要部分，避免冗余依赖。

## 主程序调用方式
- **Lightning 入口**：`src/main.py` 中的 Trainer/ModelWrapper/DataModule 构成与 depthsplat 相同的 pipeline。完成配置与数据集注册后，`python3 -m src.main +experiment=omniscene_224x400` 会自动加载新数据集并调用现有 epipolar encoder + splatting decoder 进行训练。
- **与 depthsplat 的差异**：
  1. pixelSplat 的 `ModelWrapper` 当前没有动态掩码分支，也不会在 loss 中读取 `batch["target"]["masks"]`。若需要屏蔽动态区域，需要在 loss 计算或数据 shim 中显式使用掩码；本规划阶段先聚焦于数据/配置接入，等实现阶段再决定是否增添 mask 逻辑。
  2. DepthSplat 的 `train.eval_model_every_n_val`/`train.use_dynamic_mask` 等字段在 pixelSplat 中不存在，因此“每 10 次验证触发一次测试”需要通过手动运行测试脚本来模拟；配置文档里会注明这一点，确保与 depthsplat 的节奏要求对齐。
  3. pixelSplat 的 `view_sampler` 主要面向随机采样左右 context 的通用数据（re10k 等）。OmniScene 期望一次性给出全部环视输入，因此我们会把 sampler 设为 `all` 并在数据集内部控制输入/输出视图数量，这一实现与 depthsplat 的“固定 6 个输入 + 若干输出”方案一致。
- **调用示例**：
  - 训练（112x200）：`python3 -m src.main +experiment=omniscene_112x200`
  - 训练（224x400）：`python3 -m src.main +experiment=omniscene_224x400`
  - 测试：`python3 -m src.main +experiment=omniscene_224x400 mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/omniscene_eval_index.json checkpointing.load=checkpoints/omniscene.ckpt`
  这些命令与 README 中的 re10k/acid 指令保持同一风格，便于统一脚本调用。

## 后续实现要点
- 增补 `docs/` 中的此文档后，将按文档内容在 pixelsplat 中实现配置、数据集、工具函数以及（必要时）mask-aware 的数据 shim，代码风格保持与 depthsplat 相同。
- 若后续需要评测轨迹或 figure 生成，可在 `assets/` 下仿照 re10k 添加 evaluation index，以便 `dataset/view_sampler=evaluation` 复用；此部分将在实现阶段视需要补充。
