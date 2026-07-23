from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
Num_epochs = 20
class GlassSegDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=402):
        self.rgb_dir   = Path(data_root) / split / "images"
        self.radar_dir = Path(data_root) / split / "radar_mask"
        self.mask_dir  = Path(data_root) / split / "gt_masks"

        self.img_size = img_size

        self.files = sorted(list(self.rgb_dir.glob("*.png")) +
                            list(self.rgb_dir.glob("*.jpg")))

        if len(self.files) == 0:
            raise ValueError(f"No images in {self.rgb_dir}")

        resize_tf = [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0
            )

        ]

        if split == "train":
            self.transform = A.Compose(
                resize_tf + [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=10, p=0.3),
                    ToTensorV2()
                ]
            )
        else:
            self.transform = A.Compose(
                resize_tf + [
                    ToTensorV2()
                ]
            )


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rgb_path = self.files[idx]
        radar_path = self.radar_dir / rgb_path.name
        mask_path  = self.mask_dir  / rgb_path.name

        # --- Load ---
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        radar = cv2.imread(str(radar_path), cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(str(mask_path),  cv2.IMREAD_GRAYSCALE)

        # --- FORCE SAME SIZE ---
        H, W = radar.shape[:2]

        if rgb.shape[:2] != (H,W):
            rgb = cv2.resize(rgb, (W,H), interpolation=cv2.INTER_NEAREST)

        if mask.shape[:2] != (H,W):
            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)

        # --- Augment ---
        transformed = self.transform(image=rgb, masks=[radar, mask])

        rgb   = transformed["image"].float()/255.0
        radar = transformed["masks"][0].unsqueeze(0).float()/255.0
        mask  = transformed["masks"][1].unsqueeze(0).float()/255.0

        return rgb, radar, mask, rgb_path.stem


def get_loaders(data_root, batch_size=8, img_size=256, num_workers=2):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_root: Root directory of prepared dataset
        batch_size: Batch size for training
        img_size: Image size (default 256)
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = GlassSegDataset(data_root, split='train', img_size=img_size)
    val_ds = GlassSegDataset(data_root, split='val', img_size=img_size)
    test_ds = GlassSegDataset(data_root, split='test', img_size=img_size)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':        
    print("Testing GlassSegDataset")
    print("="*50)

    data_root = '/home/astik/glass_seg_model_suhani/dataset/anotation_target2/GlassSegDataset_STRATIFIED_LIGHTING'

    try:
        train_loader, val_loader, test_loader = get_loaders(
            data_root, 
            batch_size=4, 
            img_size=402
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Get one batch
        rgb, radar, masks, names = next(iter(train_loader))
        print(f"RGB shape: {rgb.shape}")
        print(f"Radar shape: {radar.shape}")
        print(f"Masks shape: {masks.shape}")
        print(f"Names: {names}")
        print("="*50)
        print("Dataloader test passed")
        
    except Exception as e:
        print(f"Error: {e}")
