# 如何通过 MiniMind 理解大模型的训练、测试与评测

> 本文档是一份循序渐进的学习指南，帮助你通过阅读和运行 MiniMind 项目，系统地理解大语言模型（LLM）从零到一的完整生命周期。

---

## 目录

1. [项目简介](#1-项目简介)
2. [环境准备](#2-环境准备)
3. [项目结构速览](#3-项目结构速览)
4. [学习路径：从零理解大模型](#4-学习路径从零理解大模型)
   - [第一步：理解 Tokenizer（分词器）](#第一步理解-tokenizer分词器)
   - [第二步：理解模型架构](#第二步理解模型架构)
   - [第三步：预训练（Pretrain）](#第三步预训练pretrain)
   - [第四步：监督微调（SFT）](#第四步监督微调sft)
   - [第五步：LoRA 参数高效微调](#第五步lora-参数高效微调)
   - [第六步：偏好对齐（DPO）](#第六步偏好对齐dpo)
   - [第七步：强化学习训练（RLAIF）](#第七步强化学习训练rlaif)
   - [第八步：模型蒸馏](#第八步模型蒸馏)
5. [模型测试与推理](#5-模型测试与推理)
6. [模型评测（Benchmark）](#6-模型评测benchmark)
7. [进阶：可视化训练过程](#7-进阶可视化训练过程)
8. [关键概念速查表](#8-关键概念速查表)
9. [推荐学习顺序](#9-推荐学习顺序)

---

## 1. 项目简介

MiniMind 是一个从零开始、完全使用 PyTorch 原生代码实现的极小型中文语言模型项目。
最小版本仅有 **25.8M 参数**，可在普通个人 GPU 上于 2 小时内完成训练。

项目覆盖了 LLM 的完整训练流程：

| 阶段 | 关键文件 | 说明 |
|------|----------|------|
| 分词器训练 | `trainer/train_tokenizer.py` | 训练自定义 BPE Tokenizer |
| 预训练 | `trainer/train_pretrain.py` | 在大规模语料上学习语言规律 |
| 监督微调 | `trainer/train_full_sft.py` | 学习指令跟随能力 |
| LoRA 微调 | `trainer/train_lora.py` | 参数高效微调 |
| 偏好对齐 | `trainer/train_dpo.py` | 利用人类偏好数据对齐模型 |
| 强化学习 | `trainer/train_ppo.py` / `train_grpo.py` / `train_spo.py` | RLAIF 强化训练 |
| 推理模型 | `trainer/train_reason.py` | 训练带有思维链的推理模型 |
| 模型蒸馏 | `trainer/train_distillation.py` | 大模型知识蒸馏到小模型 |
| 推理测试 | `eval_llm.py` | 模型对话测试 |
| 基准评测 | README 中描述的评测方法 | C-Eval、C-MMLU 等标准榜单 |

---

## 2. 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/jingyaogong/minimind.git
cd minimind

# 2. 安装依赖（建议使用 Python 3.9+，CUDA 11.8+）
pip install -r requirements.txt

# 3. 下载数据集（放置到 dataset/ 目录）
# 详见 dataset/dataset.md 中的说明
```

**推荐硬件配置：**
- 最低：单张 NVIDIA GPU，显存 ≥ 6GB（训练 25.8M 模型）
- 推荐：NVIDIA 3090（24GB 显存）可在 2 小时内完成完整预训练

---

## 3. 项目结构速览

```
minimind/
├── model/
│   ├── model_minimind.py      # 核心：模型架构（Transformer + MoE）
│   ├── model_lora.py          # LoRA 适配器实现
│   └── tokenizer.json         # 自定义分词器文件
│
├── trainer/
│   ├── train_tokenizer.py     # 分词器训练
│   ├── train_pretrain.py      # 预训练
│   ├── train_full_sft.py      # 全参数监督微调
│   ├── train_lora.py          # LoRA 微调
│   ├── train_dpo.py           # DPO 偏好对齐
│   ├── train_ppo.py           # PPO 强化学习
│   ├── train_grpo.py          # GRPO 强化学习
│   ├── train_spo.py           # SPO 强化学习
│   ├── train_reason.py        # 推理模型训练
│   ├── train_distillation.py  # 模型蒸馏
│   └── trainer_utils.py       # 公共工具函数
│
├── dataset/
│   ├── lm_dataset.py          # 各阶段数据集加载逻辑
│   └── dataset.md             # 数据集下载说明
│
├── scripts/
│   ├── web_demo.py            # Streamlit 聊天前端
│   ├── serve_openai_api.py    # OpenAI API 兼容服务端
│   └── chat_openai_api.py     # API 客户端示例
│
└── eval_llm.py                # 推理与对话测试入口
```

---

## 4. 学习路径：从零理解大模型

### 第一步：理解 Tokenizer（分词器）

**概念：** Tokenizer 是大模型的"词典"，负责将文字转换为模型能理解的整数 ID 序列。

**关键文件：** `trainer/train_tokenizer.py`

**学习要点：**
- 了解 BPE（字节对编码）算法原理
- 理解 `vocab_size`（词表大小）对模型容量的影响
- MiniMind 使用自定义的中文 Tokenizer，词表大小为 6400

```bash
# 训练自定义分词器
python trainer/train_tokenizer.py
```

**阅读建议：** 先阅读 `model/tokenizer.json`，理解词表结构，再阅读训练代码。

---

### 第二步：理解模型架构

**概念：** MiniMind 基于 Transformer 架构，实现了现代 LLM 的所有核心组件。

**关键文件：** `model/model_minimind.py`

**学习要点：**

| 组件 | 作用 |
|------|------|
| `RMSNorm` | 替代 LayerNorm 的归一化方式，计算更高效 |
| `RotaryEmbedding` (RoPE) | 相对位置编码，支持长文本外推 |
| `CausalSelfAttention` | 因果自注意力机制（含 GQA/MQA 分组查询注意力） |
| `FeedForward` (SwiGLU) | 前馈网络，使用 SwiGLU 激活函数 |
| `MoEGate` + `MOEFeedForward` | 混合专家（MoE）架构，提升参数利用率 |
| `MiniMindBlock` | 单层 Transformer Block |
| `MiniMindForCausalLM` | 完整的因果语言模型 |

**模型尺寸对照：**

| 版本 | `hidden_size` | `num_hidden_layers` | 参数量 |
|------|--------------|---------------------|--------|
| Small | 512 | 8 | ~26M |
| MoE | 640 | 8 | ~145M |
| Base | 768 | 16 | ~104M |

**阅读建议：** 重点理解 `MiniMindBlock` 的 `forward()` 函数，这是一次 token 经过单层的完整计算过程。

---

### 第三步：预训练（Pretrain）

**概念：** 在海量无标注文本上，通过"预测下一个 token"的任务让模型学习语言规律。这是 LLM 获得语言能力的基础。

**关键文件：** `trainer/train_pretrain.py`、`dataset/lm_dataset.py`（`PretrainDataset` 类）

**核心代码片段（`train_epoch`）：**
```python
# 前向传播 + 计算损失
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss  # aux_loss 用于 MoE 负载均衡

# 梯度累积 + 混合精度训练
scaler.scale(loss).backward()
scaler.step(optimizer)
```

**关键超参数：**

| 参数 | 说明 |
|------|------|
| `--epochs` | 训练轮数 |
| `--batch_size` | 批次大小 |
| `--learning_rate` | 学习率（支持 cosine 调度） |
| `--accumulation_steps` | 梯度累积步数（模拟更大 batch） |
| `--grad_clip` | 梯度裁剪，防止梯度爆炸 |

```bash
# 启动预训练（单卡）
python trainer/train_pretrain.py

# 多卡训练（DDP）
torchrun --nproc_per_node 2 trainer/train_pretrain.py
```

**学习要点：**
- 理解交叉熵损失（cross-entropy loss）在语言模型中的含义
- 理解混合精度训练（`torch.cuda.amp`）如何节省显存
- 理解学习率调度（cosine decay with warmup）

---

### 第四步：监督微调（SFT）

**概念：** 在预训练模型基础上，用高质量的"指令-回答"对进行微调，让模型学会按照人类意图回答问题。

**关键文件：** `trainer/train_full_sft.py`、`dataset/lm_dataset.py`（`SFTDataset` 类）

**与预训练的关键区别：**
- 数据格式变为结构化的对话格式（`<s>user` + 换行 + `问题</s><s>assistant` + 换行 + `回答</s>`）
- Loss 只计算在 **assistant 回复** 部分（即只对模型需要生成的内容计算损失）
- 学习率更小（避免破坏预训练学到的知识）

```bash
# 启动 SFT 训练
python trainer/train_full_sft.py
```

**学习要点：**
- 对比 `PretrainDataset` 和 `SFTDataset` 的数据处理逻辑，理解 label mask 的作用
- 理解为什么 SFT 能让模型从"续写文本"变成"回答问题"

---

### 第五步：LoRA 参数高效微调

**概念：** 不修改原始模型权重，而是在注意力层添加低秩矩阵（A × B），只训练这两个小矩阵，大幅减少可训练参数数量。

**关键文件：** `model/model_lora.py`、`trainer/train_lora.py`

**LoRA 核心原理：**
```
原始权重 W 不变
新输出 = W·x + (B·A)·x  ← 只训练 A 和 B（参数量极少）
```

**学习要点：**
- 阅读 `model/model_lora.py` 中的 `LoRALinear` 类，理解秩分解如何实现
- 对比全参数 SFT 和 LoRA 微调在显存占用和效果上的差异

```bash
# 启动 LoRA 微调（以 identity 风格为例）
python trainer/train_lora.py --lora_name lora_identity
```

---

### 第六步：偏好对齐（DPO）

**概念：** 给模型一组"好回答 vs. 差回答"的对比数据，通过 DPO（Direct Preference Optimization）算法直接优化模型使其偏向更好的回答，无需单独训练奖励模型。

**关键文件：** `trainer/train_dpo.py`

**DPO 损失函数（代码中 `dpo_loss` 函数）：**
```python
# 计算策略模型和参考模型对 chosen/rejected 回答的对数概率
# 优化方向：使 chosen 概率相对于参考模型升高，rejected 概率降低
```

**学习要点：**
- 理解 DPO 相比 PPO 的简化：省去了独立的奖励模型训练
- 理解参考模型（reference model）在对齐训练中的作用（防止模型退化）

```bash
python trainer/train_dpo.py
```

---

### 第七步：强化学习训练（RLAIF）

**概念：** 利用 AI 反馈（RLAIF：Reinforcement Learning from AI Feedback）通过强化学习进一步提升模型能力，包括 PPO、GRPO、SPO 三种算法。

**关键文件：**
- `trainer/train_ppo.py`（近端策略优化，经典 RLHF 方法）
- `trainer/train_grpo.py`（Group Relative Policy Optimization，DeepSeek-R1 采用）
- `trainer/train_spo.py`（Simplified Policy Optimization）
- `trainer/train_reason.py`（训练带思维链的推理模型）

**GRPO 核心思想：**
- 对同一个问题采样多个回答，计算组内相对奖励
- 奖励来源：格式奖励（`<think>...</think><answer>...</answer>`）+ 外部奖励模型

**学习要点：**
- 理解强化学习在 LLM 训练中的作用：探索更好的回答策略
- 理解 KL 散度惩罚项：防止模型过度偏离 SFT 基础
- 阅读 `train_grpo.py` 中的 `calculate_rewards` 函数，理解奖励设计

```bash
# 训练推理模型（带思维链）
python trainer/train_reason.py

# GRPO 强化学习
python trainer/train_grpo.py
```

---

### 第八步：模型蒸馏

**概念：** 用大模型（Teacher）的软输出（概率分布）来指导小模型（Student）的训练，将大模型的知识压缩到小模型中。

**关键文件：** `trainer/train_distillation.py`

**蒸馏损失 = KL 散度（学生与教师的输出分布之间）**

**学习要点：**
- 理解"软标签"（soft labels）vs. "硬标签"（hard labels）的区别
- 理解温度参数（temperature）在蒸馏中如何平滑概率分布

```bash
python trainer/train_distillation.py
```

---

## 5. 模型测试与推理

训练完成后，通过以下方式测试模型：

### 命令行对话测试

```bash
# 使用 SFT 模型进行对话（默认）
python eval_llm.py

# 使用预训练模型（仅续写）
python eval_llm.py --weight pretrain

# 使用 LoRA 权重
python eval_llm.py --lora_weight lora_identity

# 使用 Hugging Face 格式的模型
python eval_llm.py --load_from /path/to/hf_model
```

**关键参数说明：**

| 参数 | 说明 |
|------|------|
| `--weight` | 权重类型（pretrain/full_sft/rlhf/reason/grpo） |
| `--hidden_size` | 模型隐藏层维度（决定加载哪个尺寸的模型） |
| `--temperature` | 生成温度（越高越随机，范围 0~1） |
| `--top_p` | nucleus 采样（累计概率阈值） |
| `--max_new_tokens` | 最大生成长度 |
| `--historys` | 携带多少轮历史对话 |

### Web 界面测试

```bash
# 启动 Streamlit 聊天界面
streamlit run scripts/web_demo.py
```

### API 服务测试

```bash
# 启动 OpenAI 兼容的 API 服务
python scripts/serve_openai_api.py

# 客户端调用
python scripts/chat_openai_api.py
```

---

## 6. 模型评测（Benchmark）

项目支持在标准中文基准测评集上评估模型能力：

### 支持的评测集

| 评测集 | 类型 | 说明 |
|--------|------|------|
| **C-Eval** | 多选题 | 覆盖中文各学科知识，52 个科目 |
| **C-MMLU** | 多选题 | 中文大规模多任务语言理解 |
| **OpenBookQA** | 多选题 | 开放式科学问题理解 |

### 评测方法

评测通过让模型在多选题上计算各选项的对数概率，选取概率最大的选项，与标准答案比较计算准确率。

**参考 README 中的评测配置步骤：**

```bash
# 下载评测数据集
# 参考 README 中的数据集下载链接

# 运行评测（详见 README 中的评测部分）
```

### 理解评测结果

- 参数量越大的模型通常得分越高，但 MiniMind 的目标是**以极少参数实现合理性能**
- 预训练模型 → SFT 模型 → RLHF 模型，评分应逐步提升
- 可对比不同训练阶段模型的得分，直观感受各训练阶段的价值

---

## 7. 进阶：可视化训练过程

项目支持 **Weights & Biases (wandb)** 和 **SwanLab** 两种可视化工具：

```bash
# 使用 wandb 监控训练
pip install wandb
python trainer/train_pretrain.py --use_wandb 1

# 使用 SwanLab 监控训练
pip install swanlab
python trainer/train_pretrain.py --use_swanlab 1
```

**可以观察的指标：**
- `loss`：总损失（应随训练下降）
- `logits_loss`：语言模型预测损失
- `aux_loss`：MoE 负载均衡辅助损失
- `learning_rate`：学习率变化曲线

---

## 8. 关键概念速查表

| 概念 | 简单解释 | 项目中对应位置 |
|------|----------|----------------|
| **Tokenizer** | 文字→数字的映射工具 | `model/tokenizer.json` |
| **Transformer** | LLM 的核心神经网络结构 | `model/model_minimind.py` |
| **Attention** | 让模型关注相关 token 的机制 | `CausalSelfAttention` 类 |
| **RoPE** | 旋转位置编码，支持长文本 | `RotaryEmbedding` 类 |
| **MoE** | 混合专家，提升参数效率 | `MOEFeedForward` 类 |
| **Pretrain** | 无监督语言模型预训练 | `train_pretrain.py` |
| **SFT** | 指令跟随微调 | `train_full_sft.py` |
| **LoRA** | 参数高效微调方法 | `model_lora.py` |
| **DPO** | 直接偏好优化（对齐方法） | `train_dpo.py` |
| **GRPO** | 基于组相对策略的强化学习 | `train_grpo.py` |
| **蒸馏** | 大模型知识迁移到小模型 | `train_distillation.py` |
| **Perplexity** | 困惑度，衡量语言模型质量 | 损失的指数：`exp(loss)` |

---

## 9. 推荐学习顺序

根据你的背景，选择适合的学习路径：

### 🔰 入门路径（无 LLM 基础）

```
1. 阅读 README.md 了解项目全貌
2. model/model_minimind.py → 理解 Transformer 结构
3. dataset/lm_dataset.py → 理解数据格式
4. trainer/train_pretrain.py → 理解预训练流程
5. eval_llm.py → 运行模型推理感受效果
```

### 🚀 进阶路径（有深度学习基础）

```
1. 完整训练一遍：Pretrain → SFT → eval_llm.py 验证
2. 对比 SFT 和 LoRA 的效果差异
3. 阅读 DPO/GRPO 代码，理解对齐训练
4. 在评测集上测试不同阶段的模型得分
```

### 🎯 研究路径（希望深入理解算法）

```
1. 精读 model_minimind.py（尤其是 MoE 实现）
2. 精读 train_grpo.py（从零实现的 GRPO 算法）
3. 精读 train_distillation.py（知识蒸馏）
4. 修改超参数、数据集，观察训练效果变化
5. 尝试在下游任务上 fine-tune 并评测
```

---

> 💡 **提示：** 所有核心算法代码均从零使用 PyTorch 原生实现，不依赖第三方高层封装。这意味着每一行代码都值得仔细阅读——你看到的就是模型真正在做的事情。
>
> 如有疑问，欢迎在项目 [Issues](https://github.com/jingyaogong/minimind/issues) 中交流探讨。
