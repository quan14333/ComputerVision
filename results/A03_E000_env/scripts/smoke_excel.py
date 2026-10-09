
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCEL = ROOT / "third_party" / "ExCEL"

# ExCEL dùng đường dẫn tương đối cho attributes_text.
os.chdir(str(EXCEL))
sys.path.insert(0, str(EXCEL))

import torch
import torchvision
import numpy as np
import mmcv
import pydensecrf.densecrf
import clip
from model.model_excel import ExCEL_model

print("Python:", sys.version)
print("Python executable:", sys.executable)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", np.__version__)
print("MMCV:", mmcv.__version__)
print("CLIP source:", clip.__file__)
print("CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert Path(clip.__file__).resolve().parent == EXCEL / "clip", (
    "Import nhầm CLIP ngoài repository."
)
assert torch.cuda.is_available(), "PyTorch chưa nhận GPU."

torch.cuda.set_device(0)
print("GPU:", torch.cuda.get_device_name(0))

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

attr_json = (
    EXCEL / "attributes_text" /
    "descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json"
)
assert attr_json.exists(), "Thiếu file mô tả thuộc tính VOC."

print("IMPORT_PASS")

# Không cần lưu graph khi khởi tạo text embeddings.
with torch.no_grad():
    model = ExCEL_model(
        clip_model="ExCEL_ViT-B/16",
        embedding_dim=256,
        in_channels=768,
        dataset_name="pascal_voc",
        num_classes=21,
        num_atrr_clusters=112,
        json_file=str(attr_json),
        img_size=320,
        mode="train_aug",
        device="cuda",
    ).cuda().eval()

    image = torch.randn(1, 3, 320, 320, device="cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    outputs = model(image)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

assert isinstance(outputs, (tuple, list)) and len(outputs) == 5

def inspect_output(value, name):
    if isinstance(value, torch.Tensor):
        print(name, "shape=", tuple(value.shape), "dtype=", value.dtype)
        assert torch.isfinite(value).all().item(), name + " có NaN/Inf"
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            inspect_output(item, name + "[" + str(i) + "]")
    else:
        raise TypeError("Output không mong đợi: " + name)

for index, output in enumerate(outputs):
    inspect_output(output, "output_" + str(index))

seg = outputs[0]
assert tuple(seg.shape) == (1, 21, 20, 20), (
    "Shape segmentation không đúng: " + str(tuple(seg.shape))
)

print("Forward seconds:", elapsed)
print("Peak allocated MiB:", torch.cuda.max_memory_allocated() / 2**20)
print("SMOKE_FORWARD_PASS")
