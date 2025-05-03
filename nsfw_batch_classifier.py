import os
import json
import torch
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from torchvision.io import read_image
from transformers import AutoModelForImageClassification
from torch.amp import autocast
from pathlib import Path
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==== 日志文件路径（固定在脚本同目录） ====
BASE_DIR         = Path(__file__).parent
success_log_path = BASE_DIR / "success_log.txt"
failed_log_path  = BASE_DIR / "failed_log.txt"

# ==== 读取已处理过的缓存（只做一次 I/O，统一大小写 & 规范分隔符） ====
processed_cache = set()
if success_log_path.exists():
    for line in success_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.strip()
        if not p:
            continue
        norm = os.path.normcase(os.path.normpath(p))
        processed_cache.add(norm)

# ==== 初始化设备，仅打印一次 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n🚀 当前使用设备：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ==== 模型加载与 warm-up ====
model_name = "Falconsai/nsfw_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_name).eval().to(device)

image_size = getattr(model.config, "image_size", 224)
id2label   = getattr(model.config, "id2label", {0: "safe", 1: "nsfw"})

with torch.no_grad():
    dummy = torch.rand(1, 3, image_size, image_size).to(device)
    _ = model(pixel_values=dummy)

# ==== 参数配置 ====
valid_exts       = (".jpg", ".jpeg", ".png", ".webp", ".gif")
batch_size       = 32
max_workers_load = 4
max_workers_scan = 8

# ==== GPU 端预处理流水线 ====
transform_gpu = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std =[0.5, 0.5, 0.5]),
])

write_pool = ThreadPoolExecutor(max_workers=4)

# ==== 扫描 .info 文件夹，取第一张符合条件的图片路径 ====
def find_first_image(folder_path: str) -> str | None:
    for entry in os.scandir(folder_path):
        if not entry.is_file():
            continue
        name = entry.name.lower()
        if name.endswith(valid_exts) and "thumbnail" not in name:
            return entry.path
    return None

def scan_info_folders(root_dir: str) -> list[str]:
    # 阶段1：列出所有以 .info 结尾的子目录
    info_dirs = [
        entry.path for entry in os.scandir(root_dir)
        if entry.is_dir() and entry.name.endswith(".info")
    ]
    # 阶段2：并行查找每个目录的第一张图
    results = []
    with ThreadPoolExecutor(max_workers=max_workers_scan) as executor:
        for img_path in tqdm(
                executor.map(find_first_image, info_dirs),
                total=len(info_dirs),
                desc="扫描 .info 文件夹",
                unit="dir",
        ):
            if img_path:
                results.append(img_path)
    return results

# ==== 加载并预处理单张图片为 GPU Tensor ====
def load_image_tensor(image_path: Path):
    try:
        tensor = read_image(str(image_path))  # [C,H,W]
        c = tensor.shape[0]
        if c == 4:
            tensor = tensor[:3]
        elif c == 1:
            tensor = tensor.expand(3, -1, -1)
        elif c != 3:
            raise ValueError(f"Unsupported channel size: {c}")
        return transform_gpu(tensor).to(device)
    except Exception as e:
        print(f"[跳过] 读取失败：{image_path} -> {e}")
        failed_log_path.open("a", encoding="utf-8").write(f"{image_path}\n")
        return None

# ==== 批量推理 + 写日志 / 打标签 / 输出图片 ====
def process_batch(images, paths, mode, out_root=None):
    batch = torch.stack(images)
    torch.cuda.synchronize()
    with torch.no_grad(), autocast(device_type="cuda"):
        outputs = model(pixel_values=batch)
        probs   = torch.softmax(outputs.logits, dim=1)
        preds   = torch.argmax(probs, dim=1)
    torch.cuda.synchronize()

    with success_log_path.open("a", encoding="utf-8") as f_succ:
        for path_str, pred in zip(paths, preds):
            label = id2label[pred.item()]
            # 写入日志时也只写原始路径
            f_succ.write(f"{path_str}\n")
            if mode == "1":
                apply_eagle_tag(Path(path_str), label)
            else:
                save_to_output_folder(Path(path_str), label, out_root)

# ==== Eagle 模式：修改 metadata.json 中的 tags ====
metadata_cache = {}
def apply_eagle_tag(image_path: Path, label: str):
    meta_path = image_path.parent / "metadata.json"
    if not meta_path.exists():
        return
    try:
        if meta_path in metadata_cache:
            data = metadata_cache[meta_path]
        else:
            data = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
            metadata_cache[meta_path] = data

        tags = data.get("tags", [])
        if "色情" in tags or "非色情" in tags:
            return

        tags = [t for t in tags if t not in ("色情", "非色情")]
        tags.append("色情" if label == "nsfw" else "非色情")
        data["tags"] = tags
        meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[失败] 更新 metadata: {meta_path} -> {e}")
        failed_log_path.open("a", encoding="utf-8").write(f"{meta_path}\n")

# ==== 普通模式：异步复制文件到 output 文件夹 ====
def async_copy(src: Path, dst: str):
    try:
        shutil.copy(str(src), dst)
    except Exception as e:
        print(f"[复制失败] {src} -> {e}")
        failed_log_path.open("a", encoding="utf-8").write(f"{src}\n")

def save_to_output_folder(image_path: Path, label: str, out_root: str):
    folder = os.path.join(out_root, "色情" if label == "nsfw" else "非色情")
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, image_path.name)
    write_pool.submit(async_copy, image_path, dst)

# ==== 用户交互：选择模式 & 目录 ====
def choose_mode():
    print("\n输入模式 (1=Eagle, 2=普通):", end="")
    while True:
        c = input().strip()
        if c in ("1", "2"):
            return c
        print("无效输入，请输入 1 或 2:", end="")

def choose_folder():
    root = tk.Tk()
    root.withdraw()
    d = filedialog.askdirectory(title="请选择要处理的图片根目录")
    if not d:
        print("未选择文件夹，程序退出。")
        exit()
    return d

# ==== Eagle 模式主流程：先过滤再进度条 ====
def run_eagle_mode(root_dir: str):
    # 1) 扫描所有 .info 首张图
    info_images = scan_info_folders(root_dir)
    print(f"共发现 {len(info_images)} 张待处理图片")

    # 2) 在内存中一次性过滤已处理路径（无额外 I/O）
    filtered = [
        p for p in info_images
        if os.path.normcase(os.path.normpath(p)) not in processed_cache
    ]
    print(f"排除已处理后剩余 {len(filtered)} 张待处理图片")

    # 3) 批量推理
    batch_imgs, batch_paths = [], []
    for img_path_str in tqdm(filtered, desc="处理进度", unit="img"):
        p = Path(img_path_str)
        tensor = load_image_tensor(p)
        if tensor is None:
            continue
        batch_imgs.append(tensor)
        batch_paths.append(str(p))
        if len(batch_imgs) >= batch_size:
            process_batch(batch_imgs, batch_paths, mode="1")
            batch_imgs, batch_paths = [], []
    if batch_imgs:
        process_batch(batch_imgs, batch_paths, mode="1")

# ==== 普通模式主流程 ====
def run_normal_mode(root_dir: str):
    out_root = os.path.join(root_dir, "output")
    os.makedirs(out_root, exist_ok=True)

    # 收集所有符合条件的文件
    all_files = [
        Path(root_dir) / f for f in os.listdir(root_dir)
        if Path(root_dir, f).is_file() and f.lower().endswith(valid_exts)
    ]
    # 并行加载预处理
    with ThreadPoolExecutor(max_workers=max_workers_load) as ex:
        results = list(ex.map(load_image_tensor, all_files))

    # 过滤已处理
    valid = [
        (img, p) for img, p in zip(results, all_files)
        if img is not None and os.path.normcase(os.path.normpath(str(p))) not in processed_cache
    ]

    # 分 batch 推理
    for i in tqdm(range(0, len(valid), batch_size), desc="处理进度", unit="batch"):
        batch = valid[i : i + batch_size]
        imgs, paths = zip(*batch)
        process_batch(imgs, paths, mode="2", out_root=out_root)

# ==== 程序入口 ====
if __name__ == "__main__":
    mode      = choose_mode()
    input_dir = choose_folder()
    if mode == "1":
        run_eagle_mode(input_dir)
    else:
        run_normal_mode(input_dir)
    print("\n🎉 全部完成！")
