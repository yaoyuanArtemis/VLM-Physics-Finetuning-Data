import os
import json

# 配置路径
SAVE_DIR = "material_dataset"
OUTPUT_JSON = "material_train.json"

llama_factory_data = []

# 遍历已经下载好的图片
img_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')], 
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))

print(f"检测到 {len(img_files)} 张图片，开始修复账本...")

for img_filename in img_files:
    img_path = os.path.join(SAVE_DIR, img_filename)
    
    # 构造条目（注意：这里由于丢失了原始caption，我们先填入一个占位符，
    # 或者如果你能忍受重新跑一小会儿，可以改脚本。
    # 但最快的方式是：既然我们要微调，其实可以先用一个通用的描述或者空着）
    data_item = {
        "messages": [
            {"role": "user", "content": "<image>Please describe this material microstructure in detail."},
            {"role": "assistant", "content": "This is a specialized material microstructure image."} # 占位符
        ],
        "images": [img_path]
    }
    llama_factory_data.append(data_item)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)

print(f"✅ 修复完成！{OUTPUT_JSON} 已生成。")
