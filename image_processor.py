import numpy as np
import rasterio
from PIL import Image
import torch
from skimage.transform import resize
import cv2

class ImageProcessor:
    def __init__(self):
        # Sentinel-2バンドマッピング
        self.band_mapping = {
            'B2': 0,   # Blue
            'B3': 1,   # Green  
            'B4': 2,   # Red
            'B8A': 3,  # Narrow NIR
            'B11': 4,  # SWIR1
            'B12': 5   # SWIR2
        }
        self.target_size = (512, 512)
    
    def process_sentinel2_image(self, uploaded_file):
        """Sentinel-2画像を処理"""
        try:
            # アップロードされたファイルを一時保存
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Rasterioで画像読み込み
            with rasterio.open(temp_path) as src:
                # 全バンドを読み込み
                image_data = src.read()
                
                # バンド数確認
                if image_data.shape[0] < 6:
                    raise ValueError(f"不十分なバンド数: {image_data.shape[0]} < 6")
                
                # 必要な6バンドを選択
                selected_bands = image_data[:6]  # 最初の6バンドを使用
                
                # データ型変換 (uint16 -> int16)
                if selected_bands.dtype == np.uint16:
                    selected_bands = selected_bands.astype(np.int16)
                
                # サイズ調整
                processed_bands = []
                for band in selected_bands:
                    resized_band = resize(band, self.target_size, preserve_range=True)
                    processed_bands.append(resized_band)
                
                processed_image = np.stack(processed_bands, axis=0)
                
                # 正規化
                processed_image = self.normalize_image(processed_image)
                
                return processed_image, temp_path
                
        except Exception as e:
            raise Exception(f"画像処理エラー: {e}")
    
    def normalize_image(self, image):
        """画像を正規化"""
        # Prithviモデル用の正規化
        # 値域を1000-3000に調整
        image = np.clip(image, 1000, 3000)
        image = (image - 1000) / 2000.0  # 0-1に正規化
        return image
    
    def create_rgb_image(self, image_data):
        """RGB画像を作成（可視化用）"""
        # バンド3(Red), 2(Green), 1(Blue)を使用
        rgb = np.stack([
            image_data[2],  # Red
            image_data[1],  # Green  
            image_data[0]   # Blue
        ], axis=-1)
        
        # 0-255に正規化
        rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
        
        return rgb
    
    def create_prediction_overlay(self, rgb_image, prediction_mask):
        """予測マスクをRGB画像にオーバーレイ"""
        overlay = rgb_image.copy()
        
        # 洪水領域を赤色でオーバーレイ
        flood_mask = prediction_mask == 1
        overlay[flood_mask] = [255, 0, 0]  # 赤色
        
        # 透明度を適用
        alpha = 0.6
        result = cv2.addWeighted(rgb_image, 1-alpha, overlay, alpha, 0)
        
        return result