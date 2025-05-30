import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import math
import random
import traceback
import csv
import time

# ==============================================================================
#                           CONFIGURATION SETTINGS
# ==============================================================================
# --- Data ---
DATA_DIR = "./datasets/BD25"
ANNOTATION_FILENAME = "annotations.xml"
IMAGE_EXT_LIST = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
# Drone images for qualitative assessment (expected in DATA_DIR/test/images/)
DRONE_IMAGE_FILENAMES = [
    f"drone{i}.JPG" for i in range(1, 7)
]  # drone1.JPG to drone6.JPG

# --- Target Image Size & Padding ---
# Final size the model will see after resize AND padding
TARGET_H = 768
TARGET_W = 1024
# Value for padding (0-255). 128 (grey) is common.
PADDING_VALUE = 128

# --- Experiment Control ---
APPROACH = "combined"  # Options: 'f1_only', 'combined'
SEED = 42
PERFORM_TRAINING = True

# --- Fixed Validation Set Counts ---
VAL_F1_COUNT = 5
VAL_F2_COUNT = 5

# --- Training Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 2
LR = 5e-5
AUG_FLIP = True
AUG_COLOR_JITTER_STRENGTH = 0.2

# --- Model & Density Map ---
OUTPUT_STRIDE = 8
DENSITY_SIGMA = 4.0  # Sigma applied to density map of size TARGET_H/S x TARGET_W/S
POINT_LABEL = "head"

# --- System & Setup ---
NUM_WORKERS = 1
USE_GPU = True
OUTPUT_DIR = "./output"
VIS_DIR = os.path.join(OUTPUT_DIR, f'vis_preds_{APPROACH}')
VAL_VIS = os.path.join(VIS_DIR, "val")
TEST_VIS = os.path.join(VIS_DIR, "test")
DRONE_VIS = os.path.join(VIS_DIR, "drone_qualitative")  # For drone images

# --- Visualization ---
MAX_VIS_IMAGES = 999
# ==============================================================================

# --- Basic Validation ---
if TARGET_H % OUTPUT_STRIDE != 0 or TARGET_W % OUTPUT_STRIDE != 0:
    print(
        f"ERROR: TARGET_H ({TARGET_H}) and TARGET_W ({TARGET_W}) must be divisible by OUTPUT_STRIDE ({OUTPUT_STRIDE})"
    )
    exit()

# --- Setup ---
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
    print("Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available or USE_GPU=False. Using CPU.")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_VIS, exist_ok=True)
os.makedirs(TEST_VIS, exist_ok=True)
os.makedirs(DRONE_VIS, exist_ok=True)
print(f"Output will be saved to: {os.path.abspath(OUTPUT_DIR)}")
main_pid = os.getpid()


# --- Utils ---
def get_annotated_images_from_xml(xml_path, point_label="head"):
    annotated_image_names = set()
    if not os.path.exists(xml_path):
        print(f"Warning: Annotation file not found: {xml_path}")
        return annotated_image_names
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for image_elem in root.findall(".//image"):
            img_name = image_elem.get("name")
            if not img_name:
                continue
            has_label = False
            for point_elem in image_elem.findall("./points"):
                if point_elem.get("label") == point_label:
                    has_label = True
                    break
            if has_label:
                annotated_image_names.add(img_name)
        return annotated_image_names
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return annotated_image_names
    except Exception as e:
        print(f"Unexpected error reading XML {xml_path}: {e}")
        return annotated_image_names


def load_points(xml_file, image_name):
    points = []
    if not os.path.exists(xml_file):
        return np.array(points, dtype=np.float32).reshape(0, 2)
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for img in root.findall(".//image"):
            if img.attrib.get("name", "").lower() == image_name.lower():
                for pt in img.findall("./points"):
                    if pt.attrib.get("label") == POINT_LABEL:
                        x, y = map(float, pt.attrib["points"].split(","))
                        points.append((x, y))
                break
    except Exception as e:
        print(f"Error parsing {xml_file} for {image_name}: {e}")
    points_np = np.array(points, dtype=np.float32)
    if points_np.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if points_np.ndim == 1:
        points_np = points_np.reshape(1, 2)
    return points_np


def create_density_map(points, target_shape_hw, sigma, output_stride):
    # (Using the efficient single-filter method, points are already scaled+offset for target_shape_hw)
    h, w = target_shape_hw
    target_h, target_w = h // output_stride, w // output_stride
    point_map = np.zeros((target_h, target_w), dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return point_map
    for x, y in points:
        target_x, target_y = int(x / output_stride), int(y / output_stride)
        if 0 <= target_y < target_h and 0 <= target_x < target_w:
            point_map[target_y, target_x] = 1.0
    density_map = gaussian_filter(
        point_map, sigma=sigma
    )  # Filter the map of deltas once
    current_sum = density_map.sum()
    if current_sum > 1e-6:
        density_map *= gt_count / current_sum
    return density_map


def visualize_prediction(img_tensor, gt_map, pred_map, out_path, is_qualitative=False):
    try:
        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
        gt = gt_map.squeeze().cpu().numpy()
        pred = pred_map.squeeze().detach().cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        fig_title = (
            "Qualitative Prediction" if is_qualitative else "Prediction Visualization"
        )
        gt_title = "GT (N/A)" if is_qualitative else f"GT ({gt.sum():.1f})"

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_np)
        axes[0].set_title("Image")
        axes[0].axis("off")
        im1 = axes[1].imshow(gt, cmap="jet" if not is_qualitative else "gray", vmin=0)
        axes[1].set_title(gt_title)
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(pred, cmap="jet", vmin=0)
        axes[2].set_title(f"Pred ({pred.sum():.1f})")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.suptitle(fig_title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        print(f"Error during visualization for {out_path}: {e}")


# --- Dataset ---
class CrowdDataset(Dataset):
    def __init__(
        self,
        image_paths,
        data_dir,
        annotation_filename,
        transform=None,
        is_train=False,
        is_qualitative=False,
    ):
        self.image_paths = image_paths
        self.data_dir = data_dir
        self.annotation_filename = annotation_filename
        # Transform here is for ToTensor, Normalize, ColorJitter. Resize is handled internally.
        self.transform = transform
        self.is_train = is_train
        self.is_qualitative = is_qualitative  # Flag for unannotated images
        self.target_h = TARGET_H
        self.target_w = TARGET_W
        self.padding_value = PADDING_VALUE
        self.density_sigma = DENSITY_SIGMA
        self.output_stride = OUTPUT_STRIDE
        self.point_label = POINT_LABEL
        self.aug_flip = AUG_FLIP

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_filename = os.path.basename(img_path)
        try:
            img = Image.open(img_path).convert("RGB")
            original_w, original_h = img.size
            points = []
            gt_count = 0

            if not self.is_qualitative:
                norm_img_path = img_path.replace("\\", "/")
                floor_dir = (
                    "train"
                    if f"/{os.path.basename(self.data_dir)}/train/images/"
                    in norm_img_path
                    else "test"
                )
                if floor_dir not in ["train", "test"]:
                    floor_dir = "train" if "/train/images/" in norm_img_path else "test"
                if floor_dir not in ["train", "test"]:
                    raise ValueError(f"Cannot determine floor: {img_path}")
                ann_path = os.path.join(
                    self.data_dir, floor_dir, self.annotation_filename
                )
                points = load_points(ann_path, img_filename)
                gt_count = len(points)

            # --- Augmentation: Apply Flip FIRST (on original image and points) ---
            apply_flip = self.is_train and self.aug_flip and random.random() < 0.5
            if apply_flip:
                img = TF.hflip(img)
                if gt_count > 0:
                    points[:, 0] = original_w - 1 - points[:, 0]

            # --- Resize maintaining aspect ratio to fit within TARGET_W, TARGET_H ---
            scale = min(self.target_w / original_w, self.target_h / original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            # Ensure new dimensions are at least 1 pixel
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            # --- Scale points according to this resize factor ---
            if gt_count > 0:
                scaled_points = points * scale
            else:
                scaled_points = np.empty((0, 2), dtype=np.float32)

            # --- Create Padded Image (PIL) to TARGET_W, TARGET_H ---
            padded_img_pil = Image.new(
                "RGB",
                (self.target_w, self.target_h),
                (self.padding_value, self.padding_value, self.padding_value),
            )
            pad_left = (self.target_w - new_w) // 2
            pad_top = (self.target_h - new_h) // 2
            padded_img_pil.paste(img_resized, (pad_left, pad_top))

            # --- Offset scaled points according to padding ---
            if gt_count > 0:
                padded_points = scaled_points + np.array(
                    [pad_left, pad_top], dtype=np.float32
                )
            else:
                padded_points = np.empty((0, 2), dtype=np.float32)

            # --- Apply main transforms (ColorJitter, ToTensor, Normalize) to padded image ---
            if self.transform:
                img_tensor = self.transform(padded_img_pil)
            else:
                img_tensor = transforms.ToTensor()(padded_img_pil)

            # --- Create density map using FINAL points and FINAL target size ---
            # For qualitative images, points will be empty -> density_map will be zeros
            density_map = create_density_map(
                padded_points,
                (self.target_h, self.target_w),
                self.density_sigma,
                self.output_stride,
            )
            density_map_tensor = torch.from_numpy(density_map).unsqueeze(0).float()

            return img_tensor, density_map_tensor, float(gt_count), img_filename
        except Exception as e:
            print(f"CRITICAL ERROR in Dataset for {img_path}: {e}")
            traceback.print_exc()
            dummy_img = torch.zeros((3, self.target_h, self.target_w))
            dummy_density = torch.zeros(
                (
                    1,
                    self.target_h // self.output_stride,
                    self.target_w // self.output_stride,
                )
            )
            return dummy_img, dummy_density, 0.0, "error_img"


# --- Model ---
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(512, 1, 1)
        nn.init.normal_(self.output_layer.weight, std=1e-5)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return nn.functional.relu(x)


# --- Training ---
def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, total_mae, num_samples = 0, 0, 0
    pbar = tqdm(
        enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1} Train", leave=False
    )
    for batch_idx, batch_data in pbar:
        if batch_data is None:
            continue
        try:
            imgs, maps, counts, _ = batch_data
        except ValueError:
            imgs, maps, counts = batch_data
        if imgs.shape[1:] != torch.Size([3, TARGET_H, TARGET_W]):
            continue  # Check target size
        imgs, maps = imgs.to(device), maps.to(device)
        gt_counts = counts.to(device).float().unsqueeze(1)
        preds = model(imgs)
        loss = criterion(preds, maps)
        if torch.isnan(loss):
            print("ERROR: NaN loss!")
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        with torch.no_grad():
            pred_counts = preds.view(preds.size(0), -1).sum(dim=1, keepdim=True)
            total_mae += torch.abs(pred_counts - gt_counts).sum().item()
            num_samples += imgs.size(0)
        pbar.set_postfix(
            loss=loss.item(),
            mae_batch=torch.abs(pred_counts - gt_counts).mean().item(),
            refresh=True,
        )
    if num_samples == 0:
        return 0.0, 0.0
    return total_loss / num_samples, total_mae / num_samples


# --- Evaluation ---
def evaluate(
    model, loader, criterion, device, epoch, save_dir=None, is_qualitative_eval=False
):
    model.eval()
    total_loss, total_mae, total_mse, num_samples = 0, 0, 0, 0
    results = []
    vis_count = 0
    desc_prefix = (
        "Qualitative Eval"
        if is_qualitative_eval
        else f"Epoch {epoch+1 if isinstance(epoch, int) else epoch} Eval"
    )
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc=desc_prefix, leave=False)
        for i, batch_data in pbar:
            if batch_data is None:
                continue
            try:
                imgs, maps, counts, names = batch_data
            except ValueError:
                imgs, maps, counts = batch_data
                names = [f"batch{i}_img{j}" for j in range(imgs.size(0))]
            if imgs.shape[1:] != torch.Size([3, TARGET_H, TARGET_W]):
                continue  # Check target size
            imgs, maps = imgs.to(device), maps.to(device)
            gt_counts = counts.to(device).float().unsqueeze(1)
            preds = model(imgs)
            if not is_qualitative_eval:
                loss = criterion(preds, maps)
                if torch.isnan(loss):
                    print(f"WARN: NaN loss in eval batch {i}")
                    continue
                total_loss += loss.item() * imgs.size(0)
                pred_counts = preds.view(preds.size(0), -1).sum(dim=1, keepdim=True)
                total_mae += torch.abs(pred_counts - gt_counts).sum().item()
                total_mse += ((pred_counts - gt_counts) ** 2).sum().item()
                for j in range(imgs.size(0)):
                    results.append(
                        (names[j], gt_counts[j].item(), pred_counts[j].item())
                    )
            num_samples += imgs.size(0)
            if not is_qualitative_eval:
                pbar.set_postfix(loss=loss.item(), refresh=True)
            # Save visualization based on MAX_VIS_IMAGES for quant, save all for qual
            if save_dir and vis_count < (
                len(loader.dataset) if is_qualitative_eval else MAX_VIS_IMAGES
            ):
                out_file = os.path.join(
                    save_dir,
                    f"{'drone_' if is_qualitative_eval else ''}{'epoch_final' if not isinstance(epoch,int) else 'epoch'+str(epoch+1)}_{names[0]}",
                )
                visualize_prediction(
                    imgs[0],
                    maps[0],
                    preds[0],
                    out_file,
                    is_qualitative=is_qualitative_eval,
                )
                vis_count += 1
    if is_qualitative_eval:
        return  # No metrics for qualitative
    if num_samples == 0:
        return 0.0, 0.0, 0.0, []
    avg_loss = total_loss / num_samples
    avg_mae = total_mae / num_samples
    avg_mse = math.sqrt(total_mse / num_samples)
    return avg_loss, avg_mae, avg_mse, results


# --- Main ---
if __name__ == "__main__":
    print("\n--- Crowd Counting Pipeline (Padding, Drone Eval, Custom Splits) ---")
    print(f"Approach: {APPROACH}, Training: {PERFORM_TRAINING}")

    # --- Define Transforms ---
    eval_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    eval_transform = transforms.Compose(eval_transform_list)

    train_transform_list = []
    if AUG_COLOR_JITTER_STRENGTH > 0:
        print(f"Applying Color Jitter with strength: {AUG_COLOR_JITTER_STRENGTH}")
        train_transform_list.append(
            transforms.ColorJitter(
                brightness=AUG_COLOR_JITTER_STRENGTH,
                contrast=AUG_COLOR_JITTER_STRENGTH,
                saturation=AUG_COLOR_JITTER_STRENGTH,
                hue=AUG_COLOR_JITTER_STRENGTH / 2.0,
            )
        )
    # Flip handled in Dataset
    train_transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_transform = transforms.Compose(train_transform_list)
    print(f"Applying Random Horizontal Flip (in Dataset): {AUG_FLIP}")

    # --- Prepare Data Lists & Filter ---
    print("Scanning images & annotations...")
    f1_dir = os.path.join(DATA_DIR, "train")
    f2_dir = os.path.join(DATA_DIR, "test")
    f1_img_dir = os.path.join(f1_dir, "images")
    f1_ann_path = os.path.join(f1_dir, ANNOTATION_FILENAME)
    f2_img_dir = os.path.join(f2_dir, "images")
    f2_ann_path = os.path.join(f2_dir, ANNOTATION_FILENAME)

    f1_ann_names = get_annotated_images_from_xml(f1_ann_path, POINT_LABEL)
    f2_ann_names = get_annotated_images_from_xml(f2_ann_path, POINT_LABEL)

    f1_all_paths = []
    f2_all_paths = []
    drone_paths = []
    for ext in IMAGE_EXT_LIST:
        for cf in [str.lower, str.upper]:
            f1_all_paths.extend(glob.glob(os.path.join(f1_img_dir, f"*{cf(ext)}")))
            f2_all_paths.extend(glob.glob(os.path.join(f2_img_dir, f"*{cf(ext)}")))
    f1_all_paths = sorted(list(set(f1_all_paths)))
    f2_all_paths = sorted(list(set(f2_all_paths)))

    f1_imgs_ann = [p for p in f1_all_paths if os.path.basename(p) in f1_ann_names]
    f2_imgs_ann = [p for p in f2_all_paths if os.path.basename(p) in f2_ann_names]

    # Get drone image paths (expected in test/images)
    for fname in DRONE_IMAGE_FILENAMES:
        drone_img_path = os.path.join(
            f2_img_dir, fname
        )  # Drone images are in test/images
        if os.path.exists(drone_img_path):
            drone_paths.append(drone_img_path)
        else:
            print(f"WARN: Drone image {drone_img_path} not found.")
    print(
        f"F1 Annotated: {len(f1_imgs_ann)}, F2 Annotated: {len(f2_imgs_ann)}, Drone Images: {len(drone_paths)}"
    )

    # --- Splitting Data based on APPROACH and fixed Val counts ---
    train_img_paths, val_img_paths, quant_test_img_paths = [], [], []
    if len(f1_imgs_ann) < VAL_F1_COUNT:
        exit(
            f"Not enough F1 images ({len(f1_imgs_ann)}) for validation ({VAL_F1_COUNT})"
        )
    np.random.shuffle(f1_imgs_ann)
    val_f1_paths = f1_imgs_ann[:VAL_F1_COUNT]
    train_f1_paths = f1_imgs_ann[VAL_F1_COUNT:]

    if APPROACH == "f1_only":
        print(
            "\nApproach 'f1_only': Train/Val on F1, Quant Test on F2, Qual Test on Drone"
        )
        train_img_paths = train_f1_paths
        val_img_paths = val_f1_paths
        quant_test_img_paths = (
            f2_imgs_ann  # All annotated F2 images for quantitative test
        )
        print(
            f"Train:{len(train_img_paths)} (F1), Val:{len(val_img_paths)} (F1), Quant Test:{len(quant_test_img_paths)} (F2)"
        )
    elif APPROACH == "combined":
        print(
            "\nApproach 'combined': Train on F1+F2 (minus val), Val on F1+F2, Qual Test on Drone"
        )
        if len(f2_imgs_ann) < VAL_F2_COUNT:
            exit(
                f"Not enough F2 images ({len(f2_imgs_ann)}) for validation ({VAL_F2_COUNT})"
            )
        np.random.shuffle(f2_imgs_ann)
        val_f2_paths = f2_imgs_ann[:VAL_F2_COUNT]
        train_f2_paths = f2_imgs_ann[VAL_F2_COUNT:]
        train_img_paths = train_f1_paths + train_f2_paths
        val_img_paths = val_f1_paths + val_f2_paths
        quant_test_img_paths = (
            val_img_paths  # Quantitative test on the combined validation set
        )
        print(
            f"Train:{len(train_img_paths)} (F1+F2), Val/Quant Test:{len(val_img_paths)} (F1+F2)"
        )
    else:
        exit(f"Invalid APPROACH: {APPROACH}")

    # --- Create Datasets ---
    train_dataset = (
        CrowdDataset(
            train_img_paths,
            DATA_DIR,
            ANNOTATION_FILENAME,
            transform=train_transform,
            is_train=True,
        )
        if train_img_paths
        else None
    )
    val_dataset = (
        CrowdDataset(
            val_img_paths,
            DATA_DIR,
            ANNOTATION_FILENAME,
            transform=eval_transform,
            is_train=False,
        )
        if val_img_paths
        else None
    )
    quant_test_dataset = (
        CrowdDataset(
            quant_test_img_paths,
            DATA_DIR,
            ANNOTATION_FILENAME,
            transform=eval_transform,
            is_train=False,
        )
        if quant_test_img_paths
        else None
    )
    drone_dataset = (
        CrowdDataset(
            drone_paths,
            DATA_DIR,
            ANNOTATION_FILENAME,
            transform=eval_transform,
            is_train=False,
            is_qualitative=True,
        )
        if drone_paths
        else None
    )

    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        if train_dataset
        else None
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        if val_dataset
        else None
    )
    quant_test_loader = (
        DataLoader(
            quant_test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        if quant_test_dataset
        else None
    )
    drone_loader = (
        DataLoader(
            drone_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        if drone_dataset
        else None
    )
    print("DataLoaders created.")
    if not train_loader:
        exit("Error: Training loader not created (empty dataset).")
    if not val_loader:
        print("Warn: No validation loader.")
    if not quant_test_loader:
        print(f"Warn: No quantitative test loader.")
    if not drone_loader:
        print(f"Warn: No drone qualitative test loader.")

    # --- Model, Criterion, Optimizer ---
    print("Initializing Model...")
    model = CSRNet().to(device)
    criterion = nn.MSELoss(reduction="mean").to(device)
    print("Model Initialized.")
    best_model_path = os.path.join(OUTPUT_DIR, f"best_model_{APPROACH}.pth")
    best_val_mae = float("inf")

    # --- Training OR Loading ---
    if PERFORM_TRAINING:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        print("\n--- Starting Training ---")
        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
            start_t = time.time()
            train_loss, train_mae = train(
                model, train_loader, optimizer, criterion, device, epoch
            )
            curr_val_mae = float("inf")
            if val_loader:
                val_loss, val_mae, val_mse, _ = evaluate(
                    model, val_loader, criterion, device, epoch, save_dir=VAL_VIS
                )
                print(
                    f" Epoch Summary: Train L={train_loss:.4f},MAE={train_mae:.2f} | Val L={val_loss:.4f},MAE={val_mae:.2f},MSE={val_mse:.2f}"
                )
                curr_val_mae = val_mae
            else:
                print(
                    f" Epoch Summary (No Val): Train L={train_loss:.4f},MAE={train_mae:.2f}"
                )
            if val_loader and curr_val_mae < best_val_mae:
                best_val_mae = curr_val_mae
                torch.save(model.state_dict(), best_model_path)
                print(f" *** New best Val MAE: {best_val_mae:.2f}. Saved model ***")
            elif not val_loader and epoch == EPOCHS - 1:
                torch.save(model.state_dict(), best_model_path)
                print(f" Saved final model (no val) to {best_model_path}")
            print(f"  Epoch Duration: {time.time()-start_t:.1f}s")
        print("\n--- Training Finished ---")
    else:
        print("\n--- Skipping Training ---")
        print("Attempting to load model...")
        if os.path.exists(best_model_path):
            try:
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                print(f"Loaded model: {best_model_path}")
                best_val_mae = float("nan")
            except Exception as e:
                print(f"Error loading model: {e}. Cannot proceed.")
                exit()
        else:
            print(f"Error: Model file not found: {best_model_path}. Cannot eval.")
            exit()

    # --- Final Evaluation ---
    print(
        f"\n--- Final Evaluation Phase (Using {'Best Trained' if PERFORM_TRAINING and val_loader else 'Loaded/Final Epoch'} Model) ---"
    )
    final_val_mae, final_val_mse = float("nan"), float("nan")
    val_results_list = []
    if val_loader:
        print(f"\nFinal Eval Val Set ({len(val_dataset)}):")
        final_val_loss, final_val_mae, final_val_mse, val_results_list = evaluate(
            model, val_loader, criterion, device, epoch="final-val", save_dir=VAL_VIS
        )
        print(f" Val MAE:{final_val_mae:.2f}, MSE:{final_val_mse:.2f}")
    else:
        print("\nSkip final val eval.")

    final_quant_test_mae, final_quant_test_mse = float("nan"), float("nan")
    quant_test_results_list = []
    if quant_test_loader:
        print(f"\nFinal Quantitative Test ({len(quant_test_dataset)}):")
        (
            final_quant_test_loss,
            final_quant_test_mae,
            final_quant_test_mse,
            quant_test_results_list,
        ) = evaluate(
            model,
            quant_test_loader,
            criterion,
            device,
            epoch="final-quant-test",
            save_dir=TEST_VIS,
        )
        print(
            f" Quant Test MAE:{final_quant_test_mae:.2f}, MSE:{final_quant_test_mse:.2f}"
        )
    else:
        print("\nSkip quantitative test eval.")

    # --- Qualitative Drone Image Evaluation ---
    if drone_loader:
        print(
            f"\n--- Qualitative Drone Image Evaluation ({len(drone_dataset)} images) ---"
        )
        evaluate(
            model,
            drone_loader,
            criterion,
            device,
            epoch="final-drone",
            save_dir=DRONE_VIS,
            is_qualitative_eval=True,
        )
        print("Drone image visualizations saved.")
    else:
        print("\nSkipping drone image qualitative evaluation.")

    # --- Reporting Summary ---
    print("\n--- Final Report Summary ---")
    print(f"Approach: {APPROACH}")
    print(f"Best Validation MAE (during training, if run): {best_val_mae:.2f}")
    print(
        f"Final Quantitative Test MAE: {final_quant_test_mae:.2f} | MSE: {final_quant_test_mse:.2f}"
    )

    # --- Save Quantitative Test Results to CSV ---
    print("\n--- Saving Detailed Quantitative Test Results ---")
    results_to_save = quant_test_results_list
    csv_filename = None
    report_set_name = f"Quant_Test_Set({APPROACH})"
    if results_to_save is not None and len(results_to_save) > 0:
        csv_filename = os.path.join(OUTPUT_DIR, f"quant_test_results_{APPROACH}.csv")
        print(f"Saving results for {report_set_name} to: {csv_filename}")
        try:
            results_to_save.sort(key=lambda x: os.path.basename(x[0]))
            with open(csv_filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["Image Filename", "GT Count", "Pred Count", "Abs Error"]
                )
                for img_path, gt_count, pred_count in results_to_save:
                    abs_error = abs(pred_count - gt_count)
                    writer.writerow(
                        [
                            os.path.basename(img_path),
                            f"{gt_count:.0f}",
                            f"{pred_count:.2f}",
                            f"{abs_error:.2f}",
                        ]
                    )
            print(f"CSV saved successfully to {csv_filename}")
        except Exception as e:
            print(f"An unexpected error occurred while saving CSV: {e}")
    else:
        print(f"No quantitative test results to save for '{APPROACH}'.")

    print("\n--- Script Finished ---")