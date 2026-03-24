import os
import json
from datasets import load_dataset
from PIL import Image

# 1. 设置你想保存数据的文件夹名字
SAVE_DIR = "material_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. 基于 OmniScience 数据集扩展的材料学关键词（注意大小写）
TARGET_SUBJECT = "physics"
KEYWORDS = [
    # === 显微镜技术 ===
    "SEM", "sem", "Sem",
    "scanning electron microscope", "Scanning Electron Microscope", "scanning electron microscopy",
    "TEM", "tem", "Tem",
    "transmission electron microscope", "Transmission Electron Microscope", "transmission electron microscopy",
    "electron microscope", "Electron Microscope", "electron microscopy", "Electron Microscopy",
    "optical microscope", "Optical Microscope", "optical microscopy", "Optical Microscopy",
    "AFM", "afm", "atomic force microscope",
    "STM", "stm", "scanning tunneling microscope",
    "micrograph", "Micrograph", "microscopy", "Microscopy",

    # === 微观结构 ===
    "microstructure", "Microstructure", "micro-structure", "micro structure",
    "morphology", "Morphology", "surface morphology",
    "grain", "Grain", "grain structure", "grain boundary", "grain size",
    "crystal", "Crystal", "crystal structure", "crystalline", "crystallinity",
    "lattice", "Lattice", "lattice structure",
    "phase", "Phase", "phase structure", "multiphase",
    "texture", "Texture",

    # === 金相学 ===
    "metallographic", "Metallographic", "metallography", "Metallography",
    "polished", "etched", "etching",
    "cross-section", "cross section", "Cross-section",

    # === 材料类型 ===
    "alloy", "Alloy", "steel", "Steel",
    "metal", "Metal", "metallic",
    "polymer", "Polymer", "polymeric",
    "ceramic", "Ceramic",
    "composite", "Composite",
    "nanoparticle", "Nanoparticle", "nanostructure", "Nanostructure",
    "thin film", "Thin Film", "coating", "Coating",

    # === 缺陷和特征 ===
    "defect", "Defect", "dislocation", "Dislocation",
    "crack", "Crack", "fracture", "Fracture", "fracture surface",
    "void", "Void", "pore", "porosity",
    "inclusion", "Inclusion", "precipitate", "Precipitate",
    "interface", "Interface",

    # === 表征和分析 ===
    "characterization", "Characterization",
    "imaging", "Imaging", "image analysis",
    "analysis", "Analysis"
]

# 3. 准备存放给 LLaMA-Factory 用的 JSON 列表
llama_factory_data = []
found_count = 0

print("🚀 开始连接 Hugging Face，准备流式淘金...")

# ⚠️ 注意：请把下面的 "dataset_name" 替换成 OmniScience 真实的 Hugging Face 仓库名
# 比如 "xxx/OmniScience" (因为我不知道你们用的是哪个具体的细分版本)
dataset = load_dataset("UniParser/OmniScience", split="train", streaming=True)

for idx, row in enumerate(dataset):
    if idx > 0 and idx % 5000 == 0:
        print(f"⏳ 疯狂扫描中... 目前已排查了 {idx} 条数据，已找到 {found_count} 张材料图。继续挖...")

    # 假设数据集里有 'subject' 和 'caption'（或者 'text'）这两个字段
    subject = row.get("subject", "").lower()
    caption = row.get("caption", "")  # 保留原始大小写，因为关键词已包含大小写变体

    # 4. 核心过滤逻辑：必须是 physics，且 caption 里包含我们想要的关键词
    if TARGET_SUBJECT in subject and any(kw in caption for kw in KEYWORDS):
        try:
            # 5. 保存图片
            image = row["image"] # 假设图片字段叫 'image'
            img_filename = f"mat_img_{found_count}.jpg"
            img_path = os.path.join(SAVE_DIR, img_filename)
            
            # 如果图片是 RGB 格式，直接保存
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path)
            
            # 6. 组装成 LLaMA-Factory 需要的格式
            data_item = {
                "messages": [
                    {"role": "user", "content": "<image>Please describe this material microstructure in detail."},
                    {"role": "assistant", "content": row.get("caption", "")}
                ],
                "images": [img_path]
            }
            llama_factory_data.append(data_item)
            
            found_count += 1
            print(f"✅ 找到第 {found_count} 张材料学图片！")
            
                
        except Exception as e:
            print(f"⚠️ 第 {idx} 行图片处理报错，已跳过: {e}")

# 7. 把整理好的文本列表保存为 JSON 文件
with open("material_train.json", "w", encoding="utf-8") as f:
    json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)

print(f"\n🎉 淘金结束！成功提取 {found_count} 条数据。")
print("图片保存在 material_dataset 文件夹下，训练配置文件为 material_train.json")

