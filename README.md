一、 实验环境与核心目标
- 硬件配置：单卡 RTX 4090 (24GB 显存) + 高配 CPU (15核以上) + 充足内存。
- 基础框架：LLaMA-Factory。
- 核心模型：Qwen2.5-VL-7B-Instruct (多模态视觉语言模型)。
- 实验目标：在 25,000 张高分辨率物理材料图片（带有瑕疵或特征标注）上进行有监督微调 (SFT)，训练一个专属的物理材料分析大模型。

---
二、 数据集准备与环境联通
多模态微调的第一步是让框架准确找到图片。我们排除了“找不到文件”的报错，确立了正确的文件映射逻辑。
1. 数据存放路径：所有图片统一放置在 /root/autodl-tmp/VLM-Physics-Finetuning-Data/material_dataset/ 目录下。
2. JSON 映射配置 (dataset_info.json)： 在 LLaMA-Factory 的 dataset_info.json 中，注册我们的自定义数据集：
  - 指定 dataset: material_physics。
  - 设定根目录 "image_folder": "/root/autodl-tmp/VLM-Physics-Finetuning-Data"。
  - 这样，当 JSON 文件中出现相对路径 material_dataset/mat_img_xxxx.jpg 时，系统能完美拼接出绝对路径。

---
三、 依赖库的“炼狱”与版本锁定
在尝试优化显存时，我们遇到了极其经典的依赖库冲突与版本兼容问题。这是深度学习工程中最容易卡关的地方，最终我们得出了以下完美环境配方。
1. 核心加速库安装
为了让 4090 跑得动多模态模型，必须安装底层加速插件。我们在终端执行了现场编译安装：
- 命令：pip install bitsandbytes flash-attn --no-build-isolation
- 作用：bitsandbytes 用于后续可能的优化器状态压缩；flash-attn (FlashAttention-2) 则是大幅降低注意力矩阵显存占用的“救命神药”。编译过程耗时约 15 分钟，必须耐心等待。
2. Transformers 版本锁定
由于 Hugging Face 官方删除了新版模型底层的 .visual 属性，导致 LLaMA-Factory 在尝试量化或打补丁时频繁报错。
- 降级到 4.49.0：解决了视觉模块报错，但缺失了 LLaMA-Factory 强依赖的 transformers.video_utils。
- 升级到 5.3.0：触发了 LLaMA-Factory 的安全锁（它最高只支持 5.2.0）。
- 最终黄金版本：我们在终端执行 pip install "transformers==5.2.0"，完美平衡了新特性与框架兼容性。

---
四、 24GB 显存的极限压榨战
本实验最大的挑战是如何将一个原本需要 15GB 显存加载、推理极度消耗显存的 70 亿参数多模态模型，硬塞进 24GB 的 4090 中进行训练。我们分阶段实施了以下“显存保卫战”：
1. 摒弃有 Bug 的 4-bit 量化
原本计划通过 quantization_bit: 4 将模型压缩，但由于框架与模型在 4-bit 模式下的视觉模块兼容问题（反复报错 AttributeError），我们果断彻底删除了 4-bit 量化，决定在全量状态下硬刚。
2. 核心参数的“微创手术”
在不修改 cutoff_len: 2048（为了保住耗时数十分钟才算好的 Tokenizer 缓存）的前提下，我们修改了 YAML 文件：
- 降低 Batch Size：per_device_train_batch_size: 1（单次只喂 1 张图），配合 gradient_accumulation_steps: 16 维持总体训练效果。
- 削减 LoRA 矩阵：将 lora_rank 降至 4，只微调核心层 (q_proj, v_proj)。
- 启用 BF16：将 fp16 改为 bf16。4090 原生支持 Bfloat16，这直接砍掉了传统 FP16 训练时在后台偷占显存的 FP32 缩放器。
- 删除 Eval 模块：删除了 ### eval 和 val_size 相关的配置（后续为了找回缓存又补回了 val_size: 0.05），避免验证集计算图长期驻留显存。
- 引入分页优化器：使用 optim: paged_adamw_8bit。它不仅把优化器体积压缩到了 8-bit，还能在显卡即将 OOM 时，动态向 CPU 内存借用空间。
3. 解决显存碎片化
在还差 200MB 就爆显存的最后关头，我们没有修改代码，而是使用了 PyTorch 的底层环境变量。
- 对策：在启动命令前加上防碎片指令。它允许显存像橡皮筋一样动态伸缩，将零散的显存碎片拼凑起来使用。

---
五、 最终的训练配置与执行
结合上述所有经验，我们最终定稿的 train_qwen_physics.yaml 核心配置如下：
YAML
### modelmodel_name_or_path: /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct### methodstage: sftdo_train: truefinetuning_type: loralora_target: q_proj,v_projlora_rank: 4lora_alpha: 4### datasetdataset: material_physicstemplate: qwen2_vlcutoff_len: 2048max_samples: 25000overwrite_cache: falsepreprocessing_num_workers: 8### outputoutput_dir: /root/autodl-tmp/output/qwen_physics_v1logging_steps: 10save_steps: 100plot_loss: trueoverwrite_output_dir: true### trainper_device_train_batch_size: 1gradient_accumulation_steps: 16learning_rate: 5.0e-5num_train_epochs: 3.0lr_scheduler_type: cosinebf16: trueflash_attn: fa2optim: paged_adamw_8bit### evalval_size: 0.05
最终点火命令：
Bash
PYTORCH_ALLOC_CONF=expandable_segments:True llamafactory-cli train train_qwen_physics.yaml

---
六、 实验结果与推理部署
1. 训练耗时与结果： 启动后，系统完美读取了预处理缓存（0 秒过 Tokenizer）。经过约 12 个小时的不间断计算，4455 个 Steps 全部跑完。最终 train_loss 降至极其优秀的 0.1762，模型权重成功保存在了目标目录下。
2. 成本控制策略： 由于训练时间较长，我们采用了 AutoDL 的自动关机功能（按 GPU 利用率 < 10% 持续 10 分钟触发），在训练结束后自动切断了计费。
3. 模型推理测试： 针对微调后的模型测试，文档明确指出了**不推荐使用无显卡模式（纯 CPU）**进行对话，因为多模态推理的矩阵计算量极大，CPU 会导致响应极慢甚至内存溢出。
  - 建议方案：租赁带有 RTX 3080/3090/4070Ti 等显存 ≥ 16GB 的高性价比显卡服务器。
  - 启动命令：在带有显卡的终端中运行 llamafactory-cli webchat train_qwen_physics.yaml，即可通过 WebUI 直接给它发送物理图片进行专属能力的检验。
---
七、 模型权重与数据开源 (Hugging Face)
本项目遵循 MLOps 工业级规范，采用“代码与模型资产分离”的存储策略。基础大模型（Qwen2.5-VL-7B）使用官方开源版本，而本次实验生成的专属 LoRA 权重与物理数据集已永久托管于 Hugging Face：

🧠 物理模型权重 (LoRA Adapter): https://huggingface.co/yaoyuanlf/qwen2.5-vl-physics-lora

📊 物理材料瑕疵数据集: https://huggingface.co/datasets/yaoyuanlf/physics-vlm-dataset

(注：部署时，只需拉取 GitHub 的运行代码与配置，框架将自动从 Hugging Face 合并底座模型与 LoRA 权重，实现一键推理。)
