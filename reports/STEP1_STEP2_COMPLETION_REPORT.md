# NeuralToneTransform 第 1/2 步完成报告

## 1. 报告信息

- 项目路径: `<repo-root>`
- 报告日期: `2026-05-28`
- 阶段范围:
  - 第 1 步: 环境搭建与可用性验证
  - 第 2 步: 官方 NAM baseline 跑通与产物验收
- 阶段结论:
  - 第 1 步: 完成
  - 第 2 步: 完成

## 2. 环境信息

- OS: Windows PowerShell
- Python 虚拟环境: `.venv`
- PyTorch: `2.11.0+cpu`
- GPU 状态: `torch.cuda.is_available() = False`（本次 baseline 在 CPU 完成）
- 主要依赖:
  - `torch`, `torchaudio`, `librosa`, `matplotlib`, `tensorboard`
  - `soundfile`, `scipy`, `numpy`, `pandas`
  - `neural-amp-modeler==0.12.2`

## 3. 数据源与官方链接

### 3.1 官方文档

- NAM 用户入口: `https://www.neuralampmodeler.com/users`
- NAM 安装文档: `https://neural-amp-modeler.readthedocs.io/en/latest/installation.html`
- NAM Full Trainer 教程: `https://neural-amp-modeler.readthedocs.io/en/stable/tutorials/full.html`

### 3.2 Baseline 音频数据

- 输入音频下载链接:
  - `https://drive.google.com/uc?export=download&id=1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G`
- 目标音频下载链接:
  - `https://drive.google.com/uc?export=download&id=1NrpQLBbCDHyu0RPsne4YcjIpi5-rEP6w`

## 4. 执行步骤与命令记录

### 4.1 第 1 步: 环境搭建

执行命令:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
```

执行结果:

- `.venv` 创建并可复用
- 依赖安装完成
- 通过 `scripts/check_env.py` 验证:
  - import 全通过
  - WAV 读写成功
  - 波形图生成成功
  - CUDA 可用性检查完成

### 4.2 第 2 步: 官方 baseline

执行命令:

```powershell
.\scripts\run_step2_nam_baseline.ps1
```

补充验收命令:

```powershell
.\scripts\finalize_step2_without_rerun.ps1
```

执行结果:

- baseline 数据下载成功
- baseline 配置生成成功
- `nam-full` 训练流程正常执行并到达 `max_epochs=10`
- 成功定位并归档模型文件到标准路径

## 5. 产物清单

### 5.1 脚本与文档

- `scripts/setup_step1.ps1`
- `scripts/check_env.py`
- `scripts/prepare_nam_baseline.py`
- `scripts/run_step2_nam_baseline.ps1`
- `scripts/finalize_step2_without_rerun.ps1`
- `scripts/verify_reproducibility.ps1`
- `requirements.txt`
- `README.md`
- `reports/step2_baseline_notes.md`

### 5.2 数据与配置

- `data/raw/smoke_test_tone.wav`
- `data/raw/baseline_input.wav`
- `data/raw/baseline_output.wav`
- `configs/nam_baseline/data.json`
- `configs/nam_baseline/model.json`
- `configs/nam_baseline/learning.json`

### 5.3 训练产物

- 源模型:
  - `outputs/nam_baseline/2026-05-28-01-43-27/model.nam`
- 标准模型:
  - `outputs/nam_baseline/model.nam`
- 验收信息:
  - size(bytes): `297454`
  - last_write: `2026-05-28 02:01:32`

### 5.4 环境验证可视化

- `reports/waveform_smoke.png`

### 5.5 运行日志

- `logs/step1_env_check.log`
- `logs/step2_prepare_nam_baseline.log`
- `logs/step2_nam_training.log`
- `logs/step2_finalize_model.log`

## 6. Reproducibility Upgrade

1. Added `requirements.txt`.
2. Removed hard-coded local absolute paths from NAM baseline config generation.
3. Added logs for environment check, baseline preparation, NAM training, and model finalization.
4. Added `README.md` reproduction instructions.
5. Added `scripts/verify_reproducibility.ps1`.

## 7. 验收结论

第 1/2 步验收通过，满足以下条件:

- Step1:
  - 依赖安装完成
  - 环境检查通过
- Step2:
  - 官方 baseline 训练流程完成
  - `.nam` 模型文件已生成并归档到标准路径

当前可进入下一阶段（建议顺序）:

1. 数据采集与采样点级对齐（`align.py`）
2. 切片与数据集划分（`dataset.py` + metadata）
3. 最小可用自研模型与损失（`model.py`, `loss.py`）
