import os
import argparse
import yaml
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
from typing import List, Union, Tuple
from terratorch.tasks import SemanticSegmentationTask
from terratorch.models import EncoderDecoderFactory
from terratorch.datamodules import Sen1Floods11NonGeoDataModule
from terratorch.models.backbones.prithvi_vit import PrithviViT
from terratorch.models.decoders.upernet_decoder import UperNetDecoder
from terratorch.models.necks import SelectIndices, ReshapeTokensToImage
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.cli_tools import LightningInferenceModel
import re
import datetime
from einops import rearrange
from PIL import Image

# Constants
NO_DATA_FLOAT = -9999.0

NO_DATA = -9999
NO_DATA_FLOAT = -9999.0
OFFSET = 0
PERCENTILE = 99


def process_channel_group(orig_img, channels):
    """
    Args:
        orig_img: torch.Tensor representing original image (reference) with shape = (bands, H, W).
        channels: list of indices representing RGB channels.

    Returns:
        torch.Tensor with shape (num_channels, height, width) for original image
    """
    orig_img = orig_img[channels, ...]
    valid_mask = torch.ones_like(orig_img, dtype=torch.bool)
    valid_mask[orig_img == NO_DATA_FLOAT] = False

    # Rescale (enhancing contrast)
    max_value = max(3000, np.percentile(orig_img[valid_mask], PERCENTILE))
    min_value = OFFSET

    orig_img = torch.clamp((orig_img - min_value) / (max_value - min_value), 0, 1)

    # No data as zeros
    orig_img[~valid_mask] = 0

    return orig_img


def read_geotiff(file_path: str) -> Tuple[np.ndarray, dict, Tuple[float, float]]:
    """Read a GeoTIFF file.

    Args:
        file_path: path to input file

    Returns:
        img: image data
        meta: metadata
        coords: coordinates (longitude, latitude)
    """
    print(f"\nReading file: {file_path}")
    with rasterio.open(file_path) as src:
        img = src.read()
        meta = src.meta
        coords = src.xy(0, 0)  # Get coordinates of top-left corner

    print("Meta information:")
    print(f"- Number of bands: {meta['count']}")
    print(f"- Image shape: {img.shape}")
    print(f"- Data type: {meta['dtype']}")
    print(f"- Driver: {meta['driver']}")
    print(f"- CRS: {meta['crs']}")
    print(f"- Transform: {meta['transform']}")
    print(f"Coordinates: {coords}")

    return img, meta, coords


def save_geotiff(image, output_path: str, meta: dict):
    """Save multi-band image in Geotiff file.

    Args:
        image: np.ndarray with shape (bands, height, width)
        output_path: path where to save the image
        meta: dict with meta info.
    """
    with rasterio.open(output_path, "w", **meta) as dest:
        for i in range(image.shape[0]):
            dest.write(image[i, :, :], i + 1)

    return


def _convert_np_uint8(float_image: torch.Tensor):
    image = float_image.numpy() * 255.0
    image = image.astype(dtype=np.uint8)

    return image


def load_example(
    file_path: str,
    input_indices: List[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an example from a file.

    Args:
        file_path: path to input file
        input_indices: indices of bands to use (0-based)

    Returns:
        imgs: input images
        temporal_coords: temporal coordinates
        location_coords: location coordinates
    """
    # Read image
    img, meta, coords = read_geotiff(file_path)
    print(f"Reading {file_path}")
    print(f"Number of bands: {meta['count']}")
    print(f"Image shape: {img.shape}")

    # Select bands if specified
    if input_indices is not None:
        img = img[input_indices]
        print(f"Selected bands shape: {img.shape}")

    # Normalize image
    img = img.astype(np.float32)
    img = (img - img.mean(axis=(1, 2), keepdims=True)) / (
        img.std(axis=(1, 2), keepdims=True) + 1e-6
    )

    # Add batch dimension
    imgs = np.expand_dims(img, axis=0)
    print(f"Final image shape: {imgs.shape}")

    # Create temporal coordinates (using current year and Julian day)
    # Shape: (batch_size, num_frames, 2) - [year, julian_day]
    current_year = 2024
    julian_day = 1  # 1月1日
    temporal_coords = np.array([[[current_year, julian_day]]], dtype=np.float32)

    # Create location coordinates (using image center)
    # Shape: (batch_size, 2)
    location_coords = np.array([coords], dtype=np.float32)

    return imgs, temporal_coords, location_coords


def pad_to_size(imgs: np.ndarray, img_size: int) -> np.ndarray:
    """Pad images to multiple of img_size.

    Args:
        imgs: input images with shape (batch_size, channels, height, width)
        img_size: target size

    Returns:
        padded images
    """
    # Get original size
    original_h, original_w = imgs.shape[2:]

    # Calculate padding
    h1 = (original_h // img_size + 1) * img_size
    w1 = (original_w // img_size + 1) * img_size

    # Pad
    padded_imgs = np.pad(
        imgs,
        ((0, 0), (0, 0), (0, h1 - original_h), (0, w1 - original_w)),
        mode="constant",
        constant_values=0,
    )

    return padded_imgs


def run_model(
    imgs: np.ndarray,
    temporal_coords: np.ndarray,
    location_coords: np.ndarray,
    model: SemanticSegmentationTask,
    datamodule: Sen1Floods11NonGeoDataModule,
    img_size: int = 256,
) -> np.ndarray:
    """Run model on input images.

    Args:
        imgs: input images
        temporal_coords: temporal coordinates
        location_coords: location coordinates
        model: model to run
        datamodule: datamodule for preprocessing
        img_size: size of input images

    Returns:
        predictions
    """
    # Pad images to img_size
    print(f"Input data shape before padding: {imgs.shape}")
    imgs = pad_to_size(imgs, img_size)
    print(f"Input data shape after padding: {imgs.shape}")

    # Convert to torch tensors
    imgs = torch.from_numpy(imgs).float()
    print(f"Input data shape after torch conversion: {imgs.shape}")

    # Convert temporal_coords to tensor if provided
    if temporal_coords is not None:
        temporal_coords = torch.from_numpy(temporal_coords).float()
        print(f"Temporal coords shape: {temporal_coords.shape}")

    # Convert location_coords to tensor if provided
    if location_coords is not None:
        location_coords = torch.from_numpy(location_coords).float()
        print(f"Location coords shape: {location_coords.shape}")

    # Move to device
    device = next(model.parameters()).device
    imgs = imgs.to(device)
    if temporal_coords is not None:
        temporal_coords = temporal_coords.to(device)
    if location_coords is not None:
        location_coords = location_coords.to(device)

    # Run model
    with torch.no_grad():
        outputs = model.model(imgs, temporal_coords=temporal_coords, location_coords=location_coords)
        # デバッグ: outputsの構造を確認
        print(f"Output type: {type(outputs)}")
        print(f"Output attributes: {dir(outputs)}")
        if hasattr(outputs, '__dict__'):
            print(f"Output dict: {outputs.__dict__}")
        
        # ModelOutputの場合は属性からテンソルを取得
        if hasattr(outputs, 'output'):
            pred = outputs.output
            print("Using outputs.output")
        elif hasattr(outputs, 'logits'):
            pred = outputs.logits
            print("Using outputs.logits")
        elif hasattr(outputs, 'pred'):
            pred = outputs.pred
            print("Using outputs.pred")
        elif hasattr(outputs, 'prediction'):
            pred = outputs.prediction
            print("Using outputs.prediction")
        elif hasattr(outputs, 'last_hidden_state'):
            pred = outputs.last_hidden_state
            print("Using outputs.last_hidden_state")
        elif isinstance(outputs, dict):
            pred = outputs['pred']
            print("Using outputs['pred']")
        else:
            pred = outputs
            print("Using outputs directly")

    # Convert to numpy
    pred = pred.cpu().numpy()

    return pred


def save_prediction(pred: np.ndarray, output_file: str, rgb_outputs: bool = False, input_image: np.ndarray = None) -> None:
    """Save prediction to file.

    Args:
        pred: prediction array
        output_file: path to output file
        rgb_outputs: whether to save RGB outputs
        input_image: original input image for RGB visualization
    """
    # Remove padding and get original size
    pred = pred[0]  # Remove batch dimension
    
    # Get original image size (before padding)
    if input_image is not None:
        original_height, original_width = input_image.shape[2], input_image.shape[3]
        # Crop prediction to original size (remove padding)
        pred = pred[:, :original_height, :original_width]
    
    # Apply softmax to get probabilities
    pred_probs = np.exp(pred) / np.sum(np.exp(pred), axis=0, keepdims=True)
    
    # Get class predictions (argmax)
    pred_classes = np.argmax(pred_probs, axis=0)
    
    # Save class predictions
    class_output_file = output_file.replace('.tif', '_classes.tif')
    
    # Convert to uint8 for saving
    pred_classes_uint8 = (pred_classes * 255).astype(np.uint8)
    
    # Save as PNG for now (simpler than TIFF with georeferencing)
    png_output_file = output_file.replace('.tif', '_prediction.png')
    Image.fromarray(pred_classes_uint8).save(png_output_file)
    print(f"Prediction saved to: {png_output_file}")
    
    # Save input RGB image (bands 3, 2, 1 = RED, GREEN, BLUE)
    if input_image is not None:
        # Get original image without batch dimension
        original_img = input_image[0]  # Remove batch dimension
        
        # Extract bands 3, 2, 1 (RED, GREEN, BLUE)
        # Our selected bands are [1, 2, 3, 8, 11, 12] from 13 bands
        # So band indices in our 6-band array are:
        # Band 1 (BLUE) = index 0 in our array
        # Band 2 (GREEN) = index 1 in our array  
        # Band 3 (RED) = index 2 in our array
        rgb_bands = original_img[[2, 1, 0], :, :]  # RED, GREEN, BLUE
        
        # Normalize to 0-255 range
        rgb_image = np.zeros((rgb_bands.shape[1], rgb_bands.shape[2], 3), dtype=np.uint8)
        for i in range(3):
            band = rgb_bands[i]
            # Normalize using percentile stretch
            p2, p98 = np.percentile(band, (2, 98))
            band_norm = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
            rgb_image[:, :, i] = band_norm.astype(np.uint8)
        
        rgb_input_file = output_file.replace('.tif', '_input_rgb.png')
        Image.fromarray(rgb_image).save(rgb_input_file)
        print(f"Input RGB image saved to: {rgb_input_file}")
    
    if rgb_outputs:
        # Save RGB visualization (overlay flood areas on input RGB)
        if input_image is not None:
            # Get original image without batch dimension
            original_img = input_image[0]  # Remove batch dimension
            
            # Create base RGB image (same as input_rgb)
            rgb_bands = original_img[[2, 1, 0], :, :]  # RED, GREEN, BLUE
            
            # Normalize to 0-255 range
            base_image = np.zeros((rgb_bands.shape[1], rgb_bands.shape[2], 3), dtype=np.uint8)
            for i in range(3):
                band = rgb_bands[i]
                # Normalize using percentile stretch
                p2, p98 = np.percentile(band, (2, 98))
                band_norm = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                base_image[:, :, i] = band_norm.astype(np.uint8)
            
            print(f"Base image shape: {base_image.shape}")
            print(f"Prediction classes shape: {pred_classes.shape}")
            print(f"Unique prediction values: {np.unique(pred_classes)}")
            
            # Create flood overlay (red areas for flood)
            flood_mask = pred_classes == 1  # Assuming class 1 is flood
            flood_count = np.sum(flood_mask)
            print(f"Flood pixels detected: {flood_count}")
            
            # Create overlay image
            overlay_image = base_image.copy()
            overlay_image[flood_mask, 0] = 255  # Set red channel to 255 for flood areas
            overlay_image[flood_mask, 1] = 0    # Set green channel to 0 for flood areas
            overlay_image[flood_mask, 2] = 0    # Set blue channel to 0 for flood areas
            
            # Blend images: 30% base image + 70% overlay where there's flood
            alpha = 0.3  # 30% base image, 70% overlay for flood areas
            blended_image = base_image.copy().astype(np.float32)
            
            # Only blend where there are flood pixels
            if flood_count > 0:
                blended_image[flood_mask] = (
                    base_image[flood_mask].astype(np.float32) * alpha + 
                    overlay_image[flood_mask].astype(np.float32) * (1 - alpha)
                )
            
            blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
            
            rgb_file = output_file.replace('.tif', '_rgb.png')
            Image.fromarray(blended_image).save(rgb_file)
            print(f"RGB visualization (overlay) saved to: {rgb_file}")
        else:
            # Fallback: simple flood visualization
            rgb_file = output_file.replace('.tif', '_rgb.png')
            rgb_image = np.zeros((pred_classes.shape[0], pred_classes.shape[1], 3), dtype=np.uint8)
            rgb_image[:, :, 0] = pred_classes * 255  # Red channel for flood areas
            Image.fromarray(rgb_image).save(rgb_file)
            print(f"RGB visualization saved to: {rgb_file}")


def main(
    data_file: str,
    config: str,
    checkpoint: str,
    output_dir: str,
    rgb_outputs: bool = False,
) -> None:
    """Run inference on a single file.

    Args:
        data_file: path to input file
        config: path to config file
        checkpoint: path to checkpoint file
        output_dir: path to output directory
        rgb_outputs: whether to save RGB outputs
    """
    # Load config
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    # Create model
    model = SemanticSegmentationTask(
        model_args={
            "backbone_pretrained": True,
            "backbone": "prithvi_eo_v2_300_tl",
            "decoder": "UperNetDecoder",
            "decoder_channels": 256,
            "decoder_scale_modules": True,
            "num_classes": 2,
            "rescale": True,
            "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "head_dropout": 0.1,
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"},
            ],
        },
        model_factory="EncoderDecoderFactory",
        loss="ce",
        ignore_index=-1,
        lr=0.001,
        freeze_backbone=False,
        freeze_decoder=False,
        plot_on_val=10,
    )

    # Load checkpoint and fix keys
    checkpoint_dict = torch.load(checkpoint, map_location=torch.device('cpu'))["state_dict"]
    new_state_dict = {}
    for k, v in checkpoint_dict.items():
        if k.startswith("model.encoder._timm_module."):
            new_key = k.replace("model.encoder._timm_module.", "model.encoder.")
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # Load datamodule
    datamodule = Sen1Floods11NonGeoDataModule(config)

    # Load data
    imgs, temporal_coords, location_coords = load_example(
        data_file,
        input_indices=[1, 2, 3, 8, 11, 12],  # Sentinel-2の6バンド
    )

    # Run model
    pred = run_model(
        imgs,
        temporal_coords,
        location_coords,
        model,
        datamodule,
    )

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(data_file))
    save_prediction(pred, output_file, rgb_outputs=rgb_outputs, input_image=imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE run inference", add_help=False)

    parser.add_argument(
        "--data_file",
        type=str,
        default="examples/India_900498_S2Hand.tif",
        help="Path to the file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to yaml file containing model parameters.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
        help="Path to a checkpoint file to load from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the directory where to save outputs.",
    )
    parser.add_argument(
        "--rgb_outputs",
        action="store_true",
        help="If present, output files will only contain RGB channels. "
        "Otherwise, all bands will be saved.",
    )
    args = parser.parse_args()

    main(**vars(args)) 