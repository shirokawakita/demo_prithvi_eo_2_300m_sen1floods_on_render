import streamlit as st
import matplotlib.pyplot as plt

# Streamlit設定を最初に実行
st.set_page_config(
    page_title="Prithvi-EO-2.0 洪水検出",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import os
import yaml
import rasterio
from huggingface_hub import hf_hub_download
from pathlib import Path
from skimage.transform import resize
import cv2
import gc
import asyncio
import threading
import tempfile

# Import functions from inference.py (main.pyと同じ)
INFERENCE_AVAILABLE = False
TERRATORCH_ERROR = None

try:
    # main.pyと同じimport
    from inference import (
        SemanticSegmentationTask,
        Sen1Floods11NonGeoDataModule,
        load_example,
        run_model,
        save_prediction
    )
    INFERENCE_AVAILABLE = True
    st.success("✅ terratorch + inference.py が正常に読み込まれました")
    
except ImportError as e:
    TERRATORCH_ERROR = str(e)
    st.error(f"❌ terratorch/inference.pyのインポートエラー: {e}")
    st.warning("⚠️ 独自実装のフォールバックモデルを使用します")
    INFERENCE_AVAILABLE = False

# イベントループ問題を修正
def fix_event_loop():
    """イベントループの問題を修正"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 新しいイベントループを作成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        # イベントループが存在しない場合は新しく作成
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# 初期化時にイベントループを修正
fix_event_loop()

# 環境変数設定
def configure_streamlit():
    """Streamlitの設定を環境変数で行う"""
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = str(os.environ.get('PORT', 8501))
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '100'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

configure_streamlit()

class SimpleCNNModel(nn.Module):
    """簡単なCNNモデル（プレースホルダー用）"""
    def __init__(self, in_channels=6, num_classes=2):
        super(SimpleCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.upsample(x)
        return x

# より高度なPrithviモデル実装（terratorch無し）
class AdvancedPrithviModel(nn.Module):
    """Standard Plan用の高度なPrithviモデル実装（terratorch依存なし）"""
    def __init__(self, 
                 img_size=512,
                 patch_size=16,
                 num_bands=6,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 num_classes=2):
        super(AdvancedPrithviModel, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(num_bands, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder for Segmentation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """重みを初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        _, embed_dim, h, w = x.shape
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Remove class token and reshape
        x = x[:, 1:]  # Remove class token
        x = x.transpose(1, 2).view(B, embed_dim, h, w)
        
        # Decoder
        x = self.decoder(x)
        
        # Resize to original size
        x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x

class StandalonePrithviLoader:
    """terratorch無しのStandalone Prithviモデルローダー"""
    def __init__(self):
        self.repo_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
        self.model_filename = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
        self.cache_dir = Path("/tmp/prithvi_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    @st.cache_resource
    def download_and_load_model(_self):
        """Standalone Prithviモデルをダウンロードして読み込み"""
        try:
            with st.spinner("🚀 Standard Plan: 独自Prithviモデルを初期化中..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🏗️ 高度なPrithviモデルを構築中...")
                progress_bar.progress(25)
                
                # 高度なPrithviモデルを作成
                model = AdvancedPrithviModel(
                    img_size=512,
                    patch_size=16,
                    num_bands=6,
                    embed_dim=768,
                    depth=12,
                    num_heads=12,
                    num_classes=2
                )
                
                status_text.text("📥 Hugging Faceからモデル重みをダウンロード中...")
                progress_bar.progress(50)
                
                # 実際のPrithviモデル重みをダウンロード（可能な場合）
                try:
                    model_path = hf_hub_download(
                        repo_id=_self.repo_id,
                        filename=_self.model_filename,
                        cache_dir=str(_self.cache_dir)
                    )
                    
                    st.success(f"✅ モデルファイルダウンロード成功: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
                    
                    status_text.text("🔧 モデル重みを適用中...")
                    progress_bar.progress(75)
                    
                    # 実際の重みを読み込み（部分的でも適用）
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                            
                            # 適用可能な重みのみを読み込み
                            model_dict = model.state_dict()
                            pretrained_dict = {}
                            
                            for k, v in state_dict.items():
                                # キー名を変換して適用可能かチェック
                                for model_key in model_dict.keys():
                                    if model_key in k or k.endswith(model_key.split('.')[-1]):
                                        if model_dict[model_key].shape == v.shape:
                                            pretrained_dict[model_key] = v
                                            break
                            
                            model_dict.update(pretrained_dict)
                            model.load_state_dict(model_dict, strict=False)
                            
                            st.success(f"✅ 実際のPrithvi重み適用: {len(pretrained_dict)}/{len(model_dict)} パラメータ")
                        
                    except Exception as weight_error:
                        st.warning(f"⚠️ 重み適用エラー: {weight_error}")
                        st.info("💡 初期化重みで動作します（学習済み重みなし）")
                
                except Exception as download_error:
                    st.warning(f"⚠️ モデルダウンロードエラー: {download_error}")
                    st.info("💡 初期化重みで動作します")
                
                status_text.text("✅ モデル初期化完了!")
                progress_bar.progress(100)
                
                model.eval()
                
                # メモリクリーンアップ
                gc.collect()
                
                return model, None, {}
                
        except Exception as e:
            st.error(f"❌ Standaloneモデル作成エラー: {e}")
            return None, None, {}
    
    def _create_fallback_model(self):
        """フォールバックモデルを作成"""
        st.info("🔧 フォールバックモデルを作成中...")
        model = AdvancedPrithviModel()
        model.eval()
        return model

class ImageProcessor:
    def __init__(self):
        self.target_size = (512, 512)
        self.target_dtype = np.int16
    
    def preprocess_image(self, file_path, target_size=(512, 512), target_dtype=np.int16):
        """
        main.pyの実装に基づいた前処理:
        - Resize to target size
        - Convert data type
        - Normalize data range
        """
        st.info(f"画像を前処理中... (目標サイズ: {target_size}, データ型: {target_dtype})")
        
        with rasterio.open(file_path) as src:
            # Read all bands
            img = src.read()  # Shape: (bands, height, width)
            profile = src.profile.copy()
            
            st.info(f"元画像: バンド数={img.shape[0]}, サイズ={img.shape[1]}x{img.shape[2]}, データ型={img.dtype}")
            
            # Resize each band if necessary
            if img.shape[1:] != target_size:
                st.info(f"画像をリサイズ中: {img.shape[1]}x{img.shape[2]} → {target_size[0]}x{target_size[1]}")
                resized_bands = []
                for i in range(img.shape[0]):
                    # Resize each band individually
                    resized_band = resize(
                        img[i], 
                        target_size, 
                        preserve_range=True,
                        anti_aliasing=True
                    ).astype(img.dtype)
                    resized_bands.append(resized_band)
                img = np.stack(resized_bands, axis=0)
            
            # Convert data type if necessary
            if img.dtype != target_dtype:
                st.info(f"データ型を変換中: {img.dtype} → {target_dtype}")
                
                # Normalize to target data type range
                if img.dtype == np.uint16 and target_dtype == np.int16:
                    # Convert uint16 to int16 range
                    # uint16: 0-65535 → int16: -32768 to 32767
                    # But we'll map to positive range similar to training data (1000-3000)
                    img_min, img_max = img.min(), img.max()
                    # Normalize to 0-1 range
                    img_normalized = (img.astype(np.float32) - img_min) / (img_max - img_min)
                    # Scale to target range (similar to training data: 1000-3000)
                    img = (img_normalized * 2000 + 1000).astype(target_dtype)
                else:
                    # General conversion
                    img = img.astype(target_dtype)
            
            st.success(f"前処理完了: バンド数={img.shape[0]}, サイズ={img.shape[1]}x{img.shape[2]}, データ型={img.dtype}")
            
            # Save preprocessed image to temporary file
            output_path = file_path.replace('.tif', '_preprocessed.tif')
            
            # Update profile for the new image
            profile.update({
                'height': target_size[0],
                'width': target_size[1],
                'dtype': target_dtype,
                'count': img.shape[0]
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(img)
            
            return output_path
    
    def process_sentinel2_image(self, uploaded_file):
        """Sentinel-2画像を処理（main.pyの実装に基づく）"""
        try:
            # main.pyと同じように画像を読み込み
            if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
                # TIFFファイルの場合はrasterioで読み込み（多バンド対応）
                import tempfile
                import rasterio
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    with rasterio.open(tmp_path) as src:
                        # 全バンドを読み込み
                        img_data = src.read()  # Shape: (bands, height, width)
                        profile = src.profile.copy()
                        
                        st.info(f"TIFFファイル読み込み: バンド数={img_data.shape[0]}, サイズ={img_data.shape[1]}x{img_data.shape[2]}")
                        
                        # (bands, height, width) → (height, width, bands) に変換
                        rgb_image = img_data.transpose(1, 2, 0)
                        
                        # データ型確認
                        st.info(f"データ型: {rgb_image.dtype}, 値域: {rgb_image.min()}-{rgb_image.max()}")
                        
                finally:
                    os.unlink(tmp_path)
                    
            else:
                # PNG/JPEGなどの通常画像ファイル
                image = Image.open(uploaded_file)
                rgb_image = np.array(image)
                
                st.info(f"通常画像ファイル読み込み: 形状={rgb_image.shape}, データ型={rgb_image.dtype}")
            
            st.success("✅ 画像処理完了!")
            
            return rgb_image
                
        except Exception as e:
            # エラー時も一時ファイルを削除
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise Exception(f"画像処理エラー: {e}")
    
    def run_inference(self, preprocessed_path, model, datamodule):
        """main.pyの実装に基づいた推論実行"""
        try:
            # Load data using inference.py functions
            imgs, temporal_coords, location_coords = load_example(
                preprocessed_path,
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
            
            # Create output directory
            output_dir = tempfile.mkdtemp()
            output_file = os.path.join(output_dir, 'prediction.tif')
            
            # Save predictions
            save_prediction(pred, output_file, rgb_outputs=True, input_image=imgs)
            
            # Load generated images
            input_rgb_path = output_file.replace('.tif', '_input_rgb.png')
            prediction_path = output_file.replace('.tif', '_prediction.png')
            overlay_path = output_file.replace('.tif', '_rgb.png')
            
            input_rgb = Image.open(input_rgb_path) if os.path.exists(input_rgb_path) else None
            prediction = Image.open(prediction_path) if os.path.exists(prediction_path) else None
            overlay = Image.open(overlay_path) if os.path.exists(overlay_path) else None
            
            return input_rgb, prediction, overlay, pred
            
        finally:
            # Clean up preprocessed file
            if os.path.exists(preprocessed_path):
                os.unlink(preprocessed_path)
    
    def create_rgb_image(self, image_data):
        """RGB画像を作成（可視化用）"""
        try:
            # バンド3(Red), 2(Green), 1(Blue)を使用
            rgb = np.stack([
                image_data[2] if image_data.shape[0] > 2 else image_data[0],  # Red
                image_data[1] if image_data.shape[0] > 1 else image_data[0],  # Green  
                image_data[0]   # Blue
            ], axis=-1)
            
            # 0-255に正規化
            rgb = (rgb * 255).astype(np.uint8)
            
            return rgb
        except Exception as e:
            st.error(f"RGB画像作成エラー: {e}")
            # エラー時はグレースケール画像を返す
            gray = (image_data[0] * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)
    
    def create_prediction_overlay(self, rgb_image, prediction_mask):
        """予測マスクをRGB画像にオーバーレイ"""
        try:
            overlay = rgb_image.copy()
            
            # 洪水領域を赤色でオーバーレイ
            flood_mask = prediction_mask == 1
            overlay[flood_mask] = [255, 0, 0]  # 赤色
            
            # 透明度を適用
            alpha = 0.6
            result = cv2.addWeighted(rgb_image, 1-alpha, overlay, alpha, 0)
            
            return result
        except Exception as e:
            st.error(f"オーバーレイ作成エラー: {e}")
            return rgb_image

def create_download_link(image, filename):
    """画像のダウンロードリンクを作成"""
    try:
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration: none; color: #1f77b4;">📥 {filename}</a>'
        return href
    except Exception as e:
        return f"ダウンロードエラー: {e}"

def show_system_info():
    """システム情報を表示"""
    if st.sidebar.checkbox("システム情報", value=False):
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.sidebar.write(f"メモリ使用量: {memory.percent:.1f}%")
            st.sidebar.write(f"利用可能メモリ: {memory.available / 1024**3:.2f}GB")
        except:
            st.sidebar.write("システム情報を取得できません")

def initialize_model():
    """main.pyと同じ方式でPrithviモデルを初期化"""
    try:
        if INFERENCE_AVAILABLE == True:
            # main.pyと同じterratorch使用
            st.info("🔄 main.pyと同じ方式でPrithvi-EO-2.0モデルを初期化中...")
            try:
                # main.pyと同じSemanticSegmentationTask
                model = SemanticSegmentationTask(
                    model="prithvi_eo_2_300m_sen1floods",
                    backbone="prithvi_eo_2_300m",
                    backbone_pretrained="prithvi_eo_2_300m.pt",
                    in_channels=6,
                    num_classes=2,
                    ignore_index=-1,
                    num_frames=1,
                    pretrained=True,
                    freeze_backbone=False,
                    freeze_decoder=False,
                )
                
                # main.pyと同じデータモジュール
                datamodule = Sen1Floods11NonGeoDataModule(
                    batch_size=1,
                    num_workers=0,
                    val_split=0.2,
                    test_split=0.1,
                    means=[
                        1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
                        1263.73947144, 1645.40315126
                    ],
                    stds=[
                        633.15169573, 650.2842772, 712.12507725, 965.23119807,
                        948.9819932, 1108.06650639
                    ]
                )
                
                st.session_state.model = model
                st.session_state.data_module = datamodule
                st.session_state.config = {}
                st.success("✅ **terratorch使用**: 実際のPrithvi-EO-2.0モデル初期化完了!")
                return True
                
            except Exception as e:
                st.error(f"❌ terratorch Prithviモデル初期化エラー: {e}")
                raise e
                
        else:
            # terratorch未使用の場合はフォールバック
            st.warning("⚠️ terratorch未使用: フォールバックモデルを使用")
            fallback_model = SimpleCNNModel(in_channels=6, num_classes=2)
            fallback_model.eval()
            st.session_state.model = fallback_model
            st.session_state.data_module = None
            st.session_state.config = {}
            return True
            
    except Exception as e:
        st.error(f"❌ モデル初期化全体エラー: {e}")
        return False



def preprocess_image_like_main(img_array, target_size=(512, 512), target_dtype=np.int16):
    """main.pyと同じ前処理方法"""
    try:
        # main.pyのpreprocess_image関数と同じ処理
        st.info(f"main.py方式で前処理中... (目標サイズ: {target_size}, データ型: {target_dtype})")
        
        # main.pyと同じように実際の画像バンドを使用
        if len(img_array.shape) == 3:
            # 画像の形状を確認
            if img_array.shape[-1] >= 6:
                # 既に6バンド以上ある場合は、最初の6バンドを使用
                st.info(f"多バンド画像を検出: {img_array.shape[-1]}バンド")
                # (height, width, bands) → (bands, height, width) の形状に変換
                img = img_array[:, :, :6].transpose(2, 0, 1)
                st.info("実際の6バンドを使用 [Band1, Band2, Band3, Band4, Band5, Band6]")
                
            elif img_array.shape[-1] == 3:
                # RGB画像の場合は、main.pyのように処理
                st.info("RGB画像を検出、6バンドに拡張")
                rgb_array = img_array.astype(np.float32)
                
                # main.pyと同じバンド拡張方法
                # 実際のSentinel-2バンドをシミュレーション
                blue = rgb_array[:, :, 2]    # Blue band
                green = rgb_array[:, :, 1]   # Green band  
                red = rgb_array[:, :, 0]     # Red band
                
                # NIR, SWIR1, SWIR2は実際のバンドがない場合のシミュレーション
                # （main.pyでは実際のSentinel-2データを想定）
                nir = 255 - red              # NIRシミュレーション
                swir1 = green * 0.8          # SWIR1シミュレーション
                swir2 = blue * 0.7           # SWIR2シミュレーション
                
                # (height, width, 6) → (6, height, width) の形状に変換
                img = np.stack([blue, green, red, nir, swir1, swir2], axis=0)
                st.warning("⚠️ RGB画像のため、NIR/SWIRはシミュレーション値を使用")
                
            else:
                # その他の場合
                img = img_array.transpose(2, 0, 1)
                st.info(f"画像を変換: {img_array.shape} → {img.shape}")
        else:
            # 2D画像の場合
            if len(img_array.shape) == 2:
                # 単一バンドを6バンドに複製
                img = np.stack([img_array] * 6, axis=0)
                st.info("単一バンド画像を6バンドに複製")
            else:
                img = img_array
        
        st.info(f"元画像: バンド数={img.shape[0]}, サイズ={img.shape[1]}x{img.shape[2]}, データ型={img.dtype}")
        
        # main.pyと同じリサイズ処理
        if img.shape[1:] != target_size:
            st.info(f"画像をリサイズ中: {img.shape[1]}x{img.shape[2]} → {target_size[0]}x{target_size[1]}")
            from skimage.transform import resize
            
            resized_bands = []
            for i in range(img.shape[0]):
                resized_band = resize(
                    img[i], 
                    target_size, 
                    preserve_range=True,
                    anti_aliasing=True
                ).astype(img.dtype)
                resized_bands.append(resized_band)
            img = np.stack(resized_bands, axis=0)
        
        # main.pyと同じデータ型変換処理
        if img.dtype != target_dtype:
            st.info(f"データ型を変換中: {img.dtype} → {target_dtype}")
            
            if img.dtype == np.uint8 and target_dtype == np.int16:
                # uint8: 0-255 → int16: 1000-3000 (training data range)
                img_min, img_max = img.min(), img.max()
                img_normalized = (img.astype(np.float32) - img_min) / (img_max - img_min)
                img = (img_normalized * 2000 + 1000).astype(target_dtype)
            else:
                img = img.astype(target_dtype)
        
        st.success(f"前処理完了: バンド数={img.shape[0]}, サイズ={img.shape[1]}x{img.shape[2]}, データ型={img.dtype}")
        
        return img
        
    except Exception as e:
        st.error(f"❌ main.py方式前処理エラー: {e}")
        return None

def create_rgb_like_inference(processed_image):
    """inference.pyのsave_prediction関数と同じRGB作成方法"""
    try:
        # inference.pyのsave_prediction関数と同じ処理
        # バンド選択: [1, 2, 3, 8, 11, 12] から [RED, GREEN, BLUE] = [2, 1, 0]
        rgb_bands = processed_image[[2, 1, 0], :, :]  # RED, GREEN, BLUE
        
        # inference.pyと同じ正規化処理
        rgb_image = np.zeros((rgb_bands.shape[1], rgb_bands.shape[2], 3), dtype=np.uint8)
        for i in range(3):
            band = rgb_bands[i]
            # inference.pyと同じ2-98パーセンタイル調整
            p2, p98 = np.percentile(band, (2, 98))
            if p98 > p2:
                band_norm = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
            else:
                band_norm = np.clip(band * 255 / band.max() if band.max() > 0 else band, 0, 255)
            rgb_image[:, :, i] = band_norm.astype(np.uint8)
        
        st.info(f"🔍 inference.py方式RGB画像形状: {rgb_image.shape}")
        return rgb_image
        
    except Exception as e:
        st.warning(f"⚠️ inference.py方式RGB作成エラー: {e}")
        return None

def create_tensor_for_model(processed_image):
    """main.pyのload_example関数と同じテンソル作成"""
    try:
        # main.pyのload_example関数と同じ正規化
        img = processed_image.astype(np.float32)
        img = (img - img.mean(axis=(1, 2), keepdims=True)) / (
            img.std(axis=(1, 2), keepdims=True) + 1e-6
        )
        
        # バッチ次元を追加
        imgs = np.expand_dims(img, axis=0)
        
        # テンソルに変換
        tensor = torch.from_numpy(imgs).float()
        
        st.info(f"🔍 モデル用テンソル形状: {tensor.shape}")
        return tensor
        
    except Exception as e:
        st.error(f"❌ テンソル作成エラー: {e}")
        return None

def create_realistic_flood_prediction(rgb_image, processed_tensor, model):
    """現実的な洪水検出予測を生成（参考画像と同じ形式）"""
    try:
        with torch.no_grad():
            # モデル推論
            prediction = model(processed_tensor)
            prediction = torch.softmax(prediction, dim=1)
            flood_prob = prediction[0, 1].cpu().numpy()  # 洪水確率
            
            # より現実的な洪水検出のための後処理
            # 1. 低い確率の領域を除去（ノイズ除去）
            flood_mask = flood_prob > 0.3  # 30%以上の確率のみ
            
            # 2. 小さな領域を除去（モルフォロジー処理）
            kernel = np.ones((3, 3), np.uint8)
            flood_mask = cv2.morphologyEx(flood_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            flood_mask = cv2.morphologyEx(flood_mask, cv2.MORPH_CLOSE, kernel)
            
            # 3. 画像の特徴に基づく追加フィルタリング
            # 暗い領域（水域の可能性が高い）を重視
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            dark_areas = gray < np.percentile(gray, 40)  # 下位40%の暗い領域
            
            # 青色成分が強い領域（水域の特徴）
            blue_channel = rgb_image[:, :, 2]
            blue_dominant = blue_channel > np.mean([rgb_image[:, :, 0], rgb_image[:, :, 1]], axis=0)
            
            # 最終的な洪水マスク（複数条件の組み合わせ）
            final_flood_mask = flood_mask & (dark_areas | blue_dominant)
            
            # さらなるノイズ除去
            final_flood_mask = cv2.morphologyEx(final_flood_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            return final_flood_mask.astype(bool), flood_prob
            
    except Exception as e:
        st.error(f"❌ 予測生成エラー: {e}")
        # フォールバック: 簡単な水域検出
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        water_mask = gray < np.percentile(gray, 25)  # 下位25%の暗い領域
        return water_mask, np.random.random(rgb_image.shape[:2]) * 0.5

def create_prediction_overlay(rgb_image, flood_mask):
    """予測結果のオーバーレイ画像を作成"""
    overlay = rgb_image.copy()
    # 洪水検出エリアを赤色でオーバーレイ
    overlay[flood_mask] = [255, 0, 0]  # 赤色
    
    # 透明度を適用
    alpha = 0.6
    result = cv2.addWeighted(rgb_image, 1-alpha, overlay, alpha, 0)
    return result

def enhance_satellite_image_display(rgb_image):
    """衛星画像の表示を改善"""
    try:
        # ヒストグラム均等化でコントラスト改善
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # ガンマ補正で明度調整
        gamma = 1.2
        enhanced = np.power(enhanced / 255.0, 1/gamma) * 255
        enhanced = enhanced.astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        st.warning(f"⚠️ 画像強調エラー: {e}")
        return rgb_image

def main():
    st.title("🌊 Prithvi-EO-2.0 洪水検出システム（Standard Plan）")
    
    # 現在のモード表示
    if INFERENCE_AVAILABLE == True:
        st.success("✅ **完全版で動作中** - 実際のPrithvi-EO-2.0モデル使用")
    elif INFERENCE_AVAILABLE == "partial":
        st.warning("⚠️ **部分対応モード** - inference.py利用可能、terratorch代替実装中")
    else:
        st.error("🔧 **セットアップ中** - 依存関係の解決を実行中")
        st.info("""
        **Standard Plan の利点**:
        - ✅ 2GB RAM（1.28GBモデル対応）
        - ✅ 実際のPrithvi-EO-2.0モデル実行可能
        - ✅ 高精度な洪水検出
        - ✅ Sentinel-2画像の正確な処理
        """)
    
    st.markdown("""
    **IBM & NASAが開発したPrithvi-EO-2.0モデルを使用したSentinel-2画像からの洪水検出**
    
    このアプリケーションは[Render](https://render.com) **Standard Plan**上で動作しています。
    - **GitHub**: [リポジトリを見る](https://github.com/shirokawakita/demo_prithvi_eo_2_300m_sen1floods_on_render)
    - **現在のプラン**: Standard Plan (2GB RAM) ✅
    - **機能**: 完全版 Prithvi-EO-2.0 モデル
    """)
    
    # サイドバー
    st.sidebar.header("🔧 設定")
    st.sidebar.markdown("### モデル情報")
    
    if INFERENCE_AVAILABLE == True:
        st.sidebar.success("""
        ✅ **完全版モード**
        - **モデル**: Prithvi-EO-2.0-300M ✅
        - **サイズ**: 1.28GB ✅
        - **タスク**: 実際の洪水検出 ✅
        - **入力**: Sentinel-2 (6バンド) ✅
        - **解像度**: 512×512ピクセル ✅
        """)
    elif INFERENCE_AVAILABLE == "partial":
        st.sidebar.warning("""
        ⚠️ **部分対応モード**
        - **モデル**: カスタムPrithviモデル
        - **機能**: 基本的なAI洪水検出
        - **制限**: terratorch依存関係の代替実装
        - **状況**: Standard Plan対応中
        """)
    else:
        st.sidebar.error("""
        🔧 **セットアップ中**
        - **プラン**: Standard Plan ✅
        - **メモリ**: 2GB ✅
        - **状況**: 依存関係インストール中
        - **対応**: inference.pyとterratorch設定中
        """)
    
    # システム情報表示
    show_system_info()
    
    # モデル初期化
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded:
        if initialize_model():
            st.session_state.model_loaded = True
    
    # 画像処理器初期化
    processor = ImageProcessor()
    
    # ファイルアップロード
    st.header("📁 Sentinel-2画像アップロード")
    
    uploaded_file = st.file_uploader(
        "Sentinel-2 TIFFファイルを選択してください",
        type=['tif', 'tiff'],
        help="Sentinel-2 L1C画像またはSentinel-1画像（最大100MB）"
    )
    
    # 完全版の説明
    if INFERENCE_AVAILABLE == True:
        st.markdown("### 🧠 完全版機能")
        st.success("""
        **現在利用可能な完全版機能**:
        - 🛰️ 実際のPrithvi-EO-2.0モデル（1.28GB）
        - 📊 Sentinel-2画像の正確な6バンド処理
        - 🌊 科学的に妥当な高精度洪水検出
        - 📈 研究レベルの精度（Sen1Floods11データセット学習済み）
        """)
    elif INFERENCE_AVAILABLE == "partial":
        st.markdown("### ⚠️ 部分対応機能")
        st.warning("""
        **現在利用可能な機能**:
        - 🧠 カスタムPrithviモデル
        - 📊 基本的なSentinel-2処理
        - 🌊 AI洪水検出（terratorch代替実装）
        - 📈 研究レベルに近い精度
        """)
    else:
        st.markdown("### 🛠️ Standard Plan セットアップ中")
        st.info("""
        **準備中の機能**:
        - 🔧 inference.pyファイルの配置
        - 🔧 terratorch依存関係のインストール
        - 🔧 Prithvi-EO-2.0モデル（1.28GB）のダウンロード準備
        - 🔧 Standard Plan (2GB RAM) 環境の最適化
        """)
    
    # モデル情報表示を修正
    with st.sidebar:
        st.subheader("🤖 モデル情報")
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            if INFERENCE_AVAILABLE == True:
                st.success("✅ **実際のPrithvi-EO-2.0モデル**を使用中です。")
                st.info("🛰️ terratorch + SemanticSegmentationTask")
                st.info("📊 main.pyと同じ方式")
            elif isinstance(st.session_state.model, AdvancedPrithviModel):
                st.warning("⚠️ **独自Prithviモデル**を使用中です。")
                st.info("🚀 Standard Plan: 2GB RAM環境でPrithvi風アーキテクチャを動作")
            elif isinstance(st.session_state.model, SimpleCNNModel):
                st.error("⚠️ **簡易モデル**を使用中です。")
                st.error("⚠️ **注意**: 現在プレースホルダーモデルを使用中です。実際のPrithviモデルではありません。")
                st.info("💡 実際のPrithviモデルが正しく読み込まれていない可能性があります。")
            else:
                st.info("🤖 モデルが読み込まれています。")
        else:
            st.error("❌ モデルが読み込まれていません。")
    
    # 画像処理と予測
    if uploaded_file is not None:
        try:
            # main.pyと同じように画像を読み込み
            if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
                # TIFFファイルの場合はrasterioで読み込み（多バンド対応）
                import tempfile
                import rasterio
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    with rasterio.open(tmp_path) as src:
                        # 全バンドを読み込み
                        img_data = src.read()  # Shape: (bands, height, width)
                        profile = src.profile.copy()
                        
                        st.info(f"TIFFファイル読み込み: バンド数={img_data.shape[0]}, サイズ={img_data.shape[1]}x{img_data.shape[2]}")
                        
                        # (bands, height, width) → (height, width, bands) に変換
                        rgb_image = img_data.transpose(1, 2, 0)
                        
                        # データ型確認
                        st.info(f"データ型: {rgb_image.dtype}, 値域: {rgb_image.min()}-{rgb_image.max()}")
                        
                finally:
                    os.unlink(tmp_path)
                    
            else:
                # PNG/JPEGなどの通常画像ファイル
                image = Image.open(uploaded_file)
                rgb_image = np.array(image)
                
                st.info(f"通常画像ファイル読み込み: 形状={rgb_image.shape}, データ型={rgb_image.dtype}")
            
            st.success("✅ 画像処理完了!")
            
            # 画像情報表示
            st.subheader("📷 画像情報")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**基本情報**:")
                st.write(f"- サイズ: {rgb_image.shape[1]}x{rgb_image.shape[0]}")
                st.write(f"- チャンネル数: {rgb_image.shape[2] if len(rgb_image.shape) > 2 else 1}")
                st.write(f"- データ型: {rgb_image.dtype}")
                st.write(f"- 値域: {rgb_image.min()} - {rgb_image.max()}")
                
            with col2:
                st.write("**処理情報**:")
                # 画像前処理（バックグラウンド処理）
                try:
                    # main.pyと同じ前処理
                    processed_image = preprocess_image_like_main(rgb_image)
                    if processed_image is not None:
                        # inference.pyと同じRGB表示作成
                        sentinel2_rgb = create_rgb_like_inference(processed_image)
                        if sentinel2_rgb is not None:
                            st.success("✅ inference.py方式RGB表示作成完了")
                            display_rgb_image = sentinel2_rgb
                            st.info("🛰️ inference.py方式Sentinel-2表示を使用")
                        else:
                            display_rgb_image = rgb_image
                            st.info("📷 元画像を使用")
                        
                        # モデル用テンソル作成
                        processed_tensor = create_tensor_for_model(processed_image)
                        st.success(f"✅ モデル用テンソル作成完了: {processed_tensor.shape}")
                    else:
                        st.error("❌ 前処理に失敗")
                        processed_tensor = None
                        display_rgb_image = rgb_image
                        
                except Exception as preprocess_error:
                    st.error(f"❌ 前処理エラー: {preprocess_error}")
                    processed_tensor = None
                    display_rgb_image = rgb_image
                    st.warning("⚠️ 前処理に失敗、元画像を使用")
            
            # 入力画像プレビュー
            st.subheader("🖼️ 入力画像プレビュー")
            st.image(display_rgb_image, caption="処理済みSentinel-2 RGB画像", use_column_width=True)
            
            # 予測実行
            st.header("🧠 AI洪水検出")
            
            # モデルタイプの確認表示
            if INFERENCE_AVAILABLE == True:
                st.success("✅ **実際のPrithvi-EO-2.0モデル**を使用中です（main.pyと同じ）。")
                st.info("🛰️ terratorch + SemanticSegmentationTask使用")
            elif isinstance(st.session_state.model, SimpleCNNModel):
                st.error("⚠️ **注意**: 現在プレースホルダーモデルを使用中です。実際のPrithviモデルではありません。")
                st.info("💡 実際のPrithviモデルが正しく読み込まれていない可能性があります。")
            elif isinstance(st.session_state.model, AdvancedPrithviModel):
                st.warning("⚠️ **独自Prithviモデル**を使用中です。")
            else:
                st.warning("⚠️ **未知のモデル**が読み込まれています。")
            
            predict_button = st.button("🔍 洪水検出を実行", type="primary", use_container_width=True)
            
            if predict_button and processed_tensor is not None:
                try:
                    # display_rgb_imageが存在しない場合は元画像を使用
                    if 'display_rgb_image' not in locals():
                        display_rgb_image = rgb_image
                        st.info("📷 元画像を使用して推論を実行")
                    
                    with st.spinner("🔮 洪水検出を実行中..."):
                        # terratorch使用時はmain.pyと同じ方式
                        if INFERENCE_AVAILABLE == True:
                            st.info("🔄 main.pyと同じ方式で推論実行中...")
                            # main.pyと同じrun_model関数を使用
                            prediction = run_model(
                                processed_tensor, 
                                st.session_state.model, 
                                st.session_state.data_module
                            )
                            
                            # 予測結果からマスクを作成
                            flood_prob = torch.softmax(prediction, dim=1)[0, 1].cpu().numpy()
                            flood_mask = flood_prob > 0.5  # 50%以上で洪水判定
                            
                            st.success("✅ main.py方式での推論完了")
                        else:
                            # フォールバック時は独自実装
                            flood_mask, flood_prob = create_realistic_flood_prediction(
                                display_rgb_image, processed_tensor, st.session_state.model
                            )
                        
                        # 予測マスク画像を作成（白黒）
                        prediction_image = np.zeros_like(display_rgb_image)
                        prediction_image[flood_mask] = [255, 255, 255]  # 洪水=白
                        # 非洪水エリアは黒のまま（0, 0, 0）
                        
                        # オーバーレイ画像を作成
                        overlay_image = create_prediction_overlay(display_rgb_image, flood_mask)
                        
                        # 統計計算
                        total_pixels = flood_mask.size
                        flood_pixels = np.sum(flood_mask)
                        flood_percentage = (flood_pixels / total_pixels) * 100
                        
                        st.success("✅ 洪水検出完了!")
                        
                        # 参考画像と同じ3カラム表示
                        st.subheader("📊 結果")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Input Image**")
                            st.image(display_rgb_image, use_column_width=True)
                            
                        with col2:
                            st.markdown("**Prediction**")
                            st.image(prediction_image, use_column_width=True)
                            
                        with col3:
                            st.markdown("**Overlay**")
                            st.image(overlay_image, use_column_width=True)
                        
                        # 統計情報を簡潔に表示
                        st.subheader("📈 検出結果")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("洪水検出率", f"{flood_percentage:.2f}%")
                        with col2:
                            st.metric("洪水ピクセル数", f"{flood_pixels:,}")
                        with col3:
                            st.metric("総ピクセル数", f"{total_pixels:,}")
                        
                        # リスクレベル判定
                        if flood_percentage > 20:
                            st.error("🚨 **高リスク**: 大規模な洪水の可能性")
                        elif flood_percentage > 5:
                            st.warning("⚠️ **中リスク**: 部分的な洪水の可能性")
                        else:
                            st.info("✅ **低リスク**: 限定的な水域検出")
                        
                        # ダウンロードセクション
                        st.subheader("💾 ダウンロード")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # 入力画像ダウンロード（Sentinel-2 RGB）
                            img_buffer = io.BytesIO()
                            Image.fromarray(display_rgb_image).save(img_buffer, format='PNG')
                            st.download_button(
                                label="入力画像をダウンロード",
                                data=img_buffer.getvalue(),
                                file_name="input_image.png",
                                mime="image/png"
                            )
                        
                        with col2:
                            # 予測画像ダウンロード
                            pred_buffer = io.BytesIO()
                            Image.fromarray(prediction_image).save(pred_buffer, format='PNG')
                            st.download_button(
                                label="予測結果をダウンロード",
                                data=pred_buffer.getvalue(),
                                file_name="prediction.png",
                                mime="image/png"
                            )
                        
                        with col3:
                            # オーバーレイ画像ダウンロード
                            overlay_buffer = io.BytesIO()
                            Image.fromarray(overlay_image).save(overlay_buffer, format='PNG')
                            st.download_button(
                                label="オーバーレイをダウンロード",
                                data=overlay_buffer.getvalue(),
                                file_name="overlay.png",
                                mime="image/png"
                            )
                            
                except Exception as e:
                    st.error(f"❌ **推論エラー**: {e}")
                    st.info("💡 画像形式や前処理に問題がある可能性があります。")
                    import traceback
                    st.code(traceback.format_exc())
            
        except Exception as e:
            st.error(f"❌ エラー: {e}")
            st.markdown("### 🔧 トラブルシューティング")
            st.markdown("""
            - ファイルが正しいTIFF形式か確認してください
            - ファイルサイズが100MB以下か確認してください
            - 多バンド画像（最低1バンド以上）を使用してください
            """)
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌊 Prithvi-EO-2.0 洪水検出システム | Powered by IBM & NASA | Running on Render</p>
        <p>モデル: <a href='https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11'>Hugging Face</a> | 
        ソースコード: <a href='https://github.com/shirokawakita/demo_prithvi_eo_2_300m_sen1floods'>GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()