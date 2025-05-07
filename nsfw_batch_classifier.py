from typing import List, Optional, Dict
import os
import json
import torch
import signal
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from transformers import AutoModelForImageClassification
from torch.amp import autocast
import shutil
import warnings
from PIL import Image
import math
import time

warnings.filterwarnings("ignore", category=UserWarning)

STOP = False
def _signal_handler(sig, frame):
    global STOP
    print("\n⚠️ 收到终止信号，正在安全退出……")
    STOP = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

BASE_DIR = Path(__file__).parent
SUCCESS_LOG_PATH = BASE_DIR / "success_log.txt"
FAILED_LOG_PATH = BASE_DIR / "failed_log.txt"

processed_cache = set()
if SUCCESS_LOG_PATH.exists():
    for line in SUCCESS_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.strip()
        if p:
            processed_cache.add(os.path.normcase(os.path.normpath(p)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 当前使用设备：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

model_name = "Falconsai/nsfw_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_name).eval().to(device)
image_size = getattr(model.config, "image_size", 224)
id2label = getattr(model.config, "id2label", {0: "safe", 1: "nsfw"})
with torch.no_grad():
    dummy = torch.rand(1, 3, image_size, image_size).to(device)
    _ = model(pixel_values=dummy)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
BATCH_SIZE = 32
MAX_WORKERS_LOAD = max(1, os.cpu_count() - 2)
MAX_WORKERS_SCAN = min(16, os.cpu_count() or 4)

transform_gpu = T.Compose([
    T.Resize([image_size, image_size]),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

write_pool = ThreadPoolExecutor(max_workers=4)

def count_failed_log() -> int:
    if not FAILED_LOG_PATH.exists():
        return 0
    return sum(1 for _ in FAILED_LOG_PATH.open(encoding="utf-8"))

def find_first_image(folder_path: str) -> Optional[str]:
    if STOP:
        return None
    for entry in os.scandir(folder_path):
        if entry.is_file():
            name = entry.name.lower()
            if name.endswith(VALID_EXTS) and "thumbnail" not in name:
                return entry.path
    return None

def scan_info_folders(root_dir: str) -> List[str]:
    info_dirs = [e.path for e in os.scandir(root_dir) if e.is_dir() and e.name.endswith(".info")]
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCAN) as pool:
        for img_path in tqdm(pool.map(find_first_image, info_dirs),
                             total=len(info_dirs),
                             desc="扫描 .info 文件夹",
                             unit="dir"):
            if STOP:
                break
            if img_path:
                results.append(img_path)
    return results

def load_image_tensor(image_path: Path):
    if STOP:
        return None
    try:
        suffix = image_path.suffix.lower()
        if suffix == ".gif":
            with Image.open(image_path) as gif:
                img = gif.convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
        tensor = TF.pil_to_tensor(img).float().div(255.0)
        return transform_gpu(tensor).to(device)
    except Exception as e:
        print(f"[跳过] 读取失败：{image_path} -> {e}")
        with FAILED_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{image_path}\n")
        return None

def process_batch(images: List[torch.Tensor], paths: List[str], mode: str, out_root: Optional[str] = None):
    if STOP or not images:
        return
    batch = torch.stack(images, dim=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.cuda.is_available():
        with torch.no_grad(), autocast(device_type="cuda"):
            outputs = model(pixel_values=batch)
    else:
        with torch.no_grad():
            outputs = model(pixel_values=batch)
    preds = torch.argmax(torch.softmax(outputs.logits, dim=1), dim=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with SUCCESS_LOG_PATH.open("a", encoding="utf-8") as f_succ:
        for path_str, pred in zip(paths, preds):
            if STOP:
                break
            label = id2label[pred.item()]
            norm = os.path.normcase(os.path.normpath(path_str))
            f_succ.write(norm + "\n")
            if mode == "1":
                apply_eagle_tag(Path(path_str), label)
            else:
                save_to_output_folder(Path(path_str), label, out_root)

metadata_cache: Dict[str, dict] = {}
def apply_eagle_tag(image_path: Path, label: str):
    if STOP:
        return
    meta = image_path.parent / "metadata.json"
    if not meta.exists():
        return
    try:
        meta_key = str(meta.resolve())
        if meta_key in metadata_cache:
            data = metadata_cache[meta_key]
        else:
            data = json.loads(meta.read_text(encoding="utf-8", errors="ignore"))
            metadata_cache[meta_key] = data
        tags = data.get("tags", [])
        if "色情" in tags or "非色情" in tags:
            return
        tags = [t for t in tags if t not in ("色情", "非色情")]
        tags.append("色情" if label == "nsfw" else "非色情")
        data["tags"] = tags
        meta.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.utime(str(meta), times=(time.time(), time.time()))
    except Exception as e:
        print(f"[失败] 更新 metadata: {meta} -> {e}")
        with FAILED_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{meta}\n")

def async_copy(src: Path, dst: str):
    if STOP:
        return
    try:
        shutil.copy(str(src), dst)
    except Exception as e:
        print(f"[复制失败] {src} -> {e}")
        with FAILED_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{src}\n")

def save_to_output_folder(image_path: Path, label: str, out_root: str):
    if STOP:
        return
    folder = os.path.join(out_root, "色情" if label == "nsfw" else "非色情")
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, image_path.name)
    write_pool.submit(async_copy, image_path, dst)

def choose_mode() -> str:
    print("输入模式 (1=Eagle, 2=普通, 3=测试, 4=压缩metadata):", end="")
    while not STOP:
        c = input().strip()
        if c in ("1", "2", "3", "4"):
            return c
        print("无效，请输入 1、2、3 或 4:", end="")

def choose_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    d = filedialog.askdirectory(title="请选择要处理的图片根目录")
    root.destroy()
    if not d:
        print("未选文件夹，退出。")
        raise KeyboardInterrupt
    return d

def batch_generator(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def run_eagle_mode(root_dir: str):
    info_imgs = scan_info_folders(root_dir)
    print(f"共发现 {len(info_imgs)} 张待处理图片")
    if STOP:
        return
    filtered = [p for p in info_imgs if os.path.normcase(os.path.normpath(p)) not in processed_cache]
    print(f"排除已处理后剩余 {len(filtered)} 张")
    total_images = len(filtered)
    success_count = 0
    with tqdm(total=total_images, desc="处理进度", unit="张") as pbar:
        for batch_paths in batch_generator(filtered, BATCH_SIZE):
            if STOP:
                break
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_LOAD) as loader:
                tensors = list(loader.map(load_image_tensor, map(Path, batch_paths)))
            imgs, paths = [], []
            for t, p in zip(tensors, batch_paths):
                if STOP:
                    break
                if t is not None:
                    imgs.append(t)
                    paths.append(p)
            process_batch(imgs, paths, mode="1")
            pbar.update(len(paths))
            success_count += len(paths)
    print(f"✅ 成功处理图片数：{success_count}")
    print(f"❌ 加载失败图片数（见日志）：{count_failed_log()}")

def run_normal_mode(root_dir: str):
    out_root = os.path.join(root_dir, "output")
    os.makedirs(out_root, exist_ok=True)
    all_files = [str(Path(root_dir) / f) for f in os.listdir(root_dir)
                 if f.lower().endswith(VALID_EXTS) and Path(root_dir, f).is_file()]
    filtered = [p for p in all_files if os.path.normcase(os.path.normpath(p)) not in processed_cache]
    print(f"普通模式共 {len(filtered)} 张")
    total_images = len(filtered)
    success_count = 0
    with tqdm(total=total_images, desc="处理进度", unit="张") as pbar:
        for batch_paths in batch_generator(filtered, BATCH_SIZE):
            if STOP:
                break
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_LOAD) as loader:
                tensors = list(loader.map(load_image_tensor, map(Path, batch_paths)))
            imgs, paths = [], []
            for t, p in zip(tensors, batch_paths):
                if STOP:
                    break
                if t is not None:
                    imgs.append(t)
                    paths.append(p)
            process_batch(imgs, paths, mode="2", out_root=out_root)
            pbar.update(len(paths))
            success_count += len(paths)
    print(f"✅ 成功处理图片数：{success_count}")
    print(f"❌ 加载失败图片数（见日志）：{count_failed_log()}")

def run_test_mode(root_dir: str):
    print(f"进入测试模式，目录：{root_dir}")
    img_path = None
    for file in os.listdir(root_dir):
        if file.lower().endswith(VALID_EXTS) and "thumbnail" not in file.lower():
            img_path = str(Path(root_dir) / file)
            break
    if not img_path:
        print("❌ 未找到有效测试图片。")
        return
    print(f"✅ 发现测试图片：{img_path}")
    tensor = load_image_tensor(Path(img_path))
    if tensor is None:
        print("❌ 图片加载失败。")
        return
    process_batch([tensor], [img_path], mode="1")
    print("✅ 测试完成。")

def run_compress_mode(root_dir: str):
    print(f"📦 启动压缩模式，遍历 {root_dir} 中所有 metadata.json")
    json_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "metadata.json":
                json_paths.append(os.path.join(dirpath, filename))

    print(f"🔍 共找到 {len(json_paths)} 个 metadata.json")

    success_count = 0
    for path in tqdm(json_paths, desc="压缩中", unit="文件"):
        if STOP:
            break
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            success_count += 1
        except Exception as e:
            print(f"\n❌ 压缩失败: {path} -> {e}")

    print(f"\n🎉 完成压缩：共 {success_count} 个 metadata.json 文件")


if __name__ == "__main__":
    try:
        mode = choose_mode()
        input_dir = choose_folder()
        if mode == "1":
            run_eagle_mode(input_dir)
        elif mode == "2":
            run_normal_mode(input_dir)
        elif mode == "3":
            run_test_mode(input_dir)
        elif mode == "4":
            run_compress_mode(input_dir)
    except KeyboardInterrupt:
        print("\n✅ 已安全终止，所有已完成的操作均已写入日志。")
    finally:
        write_pool.shutdown(wait=False)
        print("🎉 程序结束。")
