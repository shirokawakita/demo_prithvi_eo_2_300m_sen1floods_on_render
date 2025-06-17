import streamlit as st
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

# Import functions from inference.py
try:
    from inference import (
        SemanticSegmentationTask,
        Sen1Floods11NonGeoDataModule,
        load_example,
        run_model,
        save_prediction
    )
    INFERENCE_AVAILABLE = True
except ImportError:
    st.error("❌ inference.pyが見つかりません。正しいファイルが配置されていることを確認してください。")
    INFERENCE_AVAILABLE = False

# Streamlit設定
st.set_page_config(
    page_title="Prithvi-EO-2.0 洪水検出",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

class PrithviModel(nn.Module):
    """Prithvi-EO-2.0モデルの実装"""
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 num_frames=3,
                 num_bands=6,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 num_classes=2):
        super(PrithviModel, self).__init__()
        
        # 基本的なTransformerエンコーダー-デコーダー構造
        self.patch_embed = nn.Conv2d(
            num_bands * num_frames, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder（セマンティックセグメンテーション用）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, decoder_embed_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(decoder_embed_dim//2, decoder_embed_dim//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(decoder_embed_dim//4, num_classes, kernel_size=4, stride=2, padding=1),
        )
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_bands = num_bands
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # パッチに分割してembedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        
        # Flatten for transformer
        _, embed_dim, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Transformer encoding
        x = self.encoder(x)
        
        # Reshape back to spatial format
        x = x.transpose(1, 2).view(B, embed_dim, h, w)
        
        # Decode to segmentation map
        x = self.decoder(x)
        
        # Resize to match input size
        x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x

class PrithviModelLoader:
    def __init__(self):
        self.repo_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
        self.model_filename = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
        self.config_filename = "config.yaml"
        self.cache_dir = Path("/tmp/prithvi_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    @st.cache_resource
    def download_and_load_model(_self):
        """正しいPrithviモデルをダウンロードして読み込み"""
        if not INFERENCE_AVAILABLE:
            st.error("❌ inference.pyが利用できないため、プレースホルダーモデルを使用します")
            return _self._create_placeholder_model(), {}
            
        try:
            with st.spinner("Prithviモデルをダウンロード中... (約1.28GB)"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 モデルファイルをダウンロード中...")
                progress_bar.progress(25)
                
                # モデルファイルをダウンロード
                try:
                    model_path = hf_hub_download(
                        repo_id=_self.repo_id,
                        filename=_self.model_filename,
                        cache_dir=str(_self.cache_dir)
                    )
                    progress_bar.progress(50)
                    st.write(f"✅ モデルファイルのダウンロード完了: {model_path}")
                    st.write(f"📄 ファイルサイズ: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
                except Exception as download_error:
                    st.warning(f"⚠️ モデルダウンロードエラー: {download_error}")
                    st.info("💡 プレースホルダーモデルを使用します")
                    return _self._create_placeholder_model(), {}
                
                status_text.text("🔄 設定ファイルをダウンロード中...")
                progress_bar.progress(75)
                
                # 設定ファイルをダウンロード
                try:
                    config_path = hf_hub_download(
                        repo_id=_self.repo_id,
                        filename=_self.config_filename,
                        cache_dir=str(_self.cache_dir)
                    )
                    
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    st.write("✅ 設定ファイル読み込み完了")
                    st.write(f"📋 設定内容の一部: {list(config.keys())[:5] if config else 'None'}")
                except Exception as config_error:
                    st.warning(f"⚠️ 設定ファイルエラー: {config_error}")
                    config = {}
                
                progress_bar.progress(90)
                status_text.text("🔄 正しいPrithviモデルを作成中...")
                
                # 正しいPrithviモデルを作成（main.pyの実装に基づく）
                try:
                    device = torch.device('cpu')
                    
                    st.write("🔍 **正しいPrithviモデルを作成中**")
                    
                    # SemanticSegmentationTaskでモデルを作成
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
                    
                    st.success("✅ SemanticSegmentationTaskモデル作成成功")
                    
                    # チェックポイント読み込み
                    st.write("🔄 チェックポイントを読み込み中...")
                    checkpoint_dict = torch.load(model_path, map_location=device)["state_dict"]
                    
                    # キー名を調整（main.pyの実装に基づく）
                    new_state_dict = {}
                    for k, v in checkpoint_dict.items():
                        if k.startswith("model.encoder._timm_module."):
                            new_key = k.replace("model.encoder._timm_module.", "model.encoder.")
                            new_state_dict[new_key] = v
                        else:
                            new_state_dict[k] = v
                    
                    # state_dictを読み込み
                    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                    st.success("✅ 正しいPrithviモデルの読み込み完了!")
                    st.write(f"📋 不足キー数: {len(missing_keys)}")
                    st.write(f"📋 予期しないキー数: {len(unexpected_keys)}")
                    
                    model.eval()
                    
                    # データモジュールも作成
                    datamodule = Sen1Floods11NonGeoDataModule(config)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 完了!")
                    
                    # メモリクリーンアップ
                    gc.collect()
                    
                    return model, datamodule, config
                    
                except Exception as model_error:
                    st.error(f"❌ 正しいモデル作成エラー: {model_error}")
                    st.info("💡 プレースホルダーモデルを使用します")
                    return _self._create_placeholder_model(), {}, {}
                    
        except Exception as e:
            st.error(f"❌ 全体的なエラー: {e}")
            return _self._create_placeholder_model(), {}, {}
    
    def _create_placeholder_model(self):
        """プレースホルダーモデルを作成"""
        st.info("🔧 プレースホルダーモデルを作成中...")
        model = SimpleCNNModel(in_channels=6, num_classes=2)
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
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Preprocess image to match training data format
            preprocessed_path = self.preprocess_image(tmp_path, target_size=self.target_size, target_dtype=self.target_dtype)
            
            return preprocessed_path
                
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

def main():
    st.title("🌊 Prithvi-EO-2.0 洪水検出システム")
    st.markdown("""
    **IBM & NASAが開発したPrithvi-EO-2.0モデルを使用したSentinel-2画像からの洪水検出**
    
    このアプリケーションは[Render](https://render.com)上で動作しています。
    """)
    
    # サイドバー
    st.sidebar.header("🔧 設定")
    st.sidebar.markdown("### モデル情報")
    st.sidebar.info("""
    - **モデル**: Prithvi-EO-2.0-300M
    - **サイズ**: 1.28GB
    - **タスク**: 洪水セマンティックセグメンテーション
    - **入力**: Sentinel-2 (6バンド)
    - **解像度**: 512×512ピクセル
    """)
    
    # システム情報表示
    show_system_info()
    
    # モデル初期化
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded:
        st.info("🚀 モデルを初期化しています...")
        
        try:
            model_loader = PrithviModelLoader()
            model, datamodule, config = model_loader.download_and_load_model()
            
            if model is not None:
                st.session_state.model = model
                st.session_state.datamodule = datamodule
                st.session_state.config = config
                st.session_state.model_loaded = True
                
                # プレースホルダーモデルかどうかを確認
                if isinstance(model, SimpleCNNModel):
                    st.warning("⚠️ プレースホルダーモードで動作しています。デモ用の予測を表示します。")
                else:
                    st.success("✅ Prithviモデルの読み込み完了!")
                    st.balloons()
            else:
                st.error("❌ モデルの初期化に失敗しました")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ モデル初期化エラー: {e}")
            st.stop()
    
    # 画像処理器初期化
    processor = ImageProcessor()
    
    # ファイルアップロード
    st.header("📁 Sentinel-2画像のアップロード")
    
    uploaded_file = st.file_uploader(
        "TIFFファイルを選択してください",
        type=['tif', 'tiff'],
        help="Sentinel-2 L1Cまたは多バンドGeoTIFFファイル（最大100MB）"
    )
    
    # サンプルデータ情報
    st.markdown("### 🌍 テスト用データ")
    st.info("""
    以下の地域のSentinel-2洪水画像をテストできます：
    - 🇮🇳 **インド**: モンスーンによる洪水
    - 🇪🇸 **スペイン**: 河川氾濫
    - 🇺🇸 **アメリカ**: ハリケーンによる洪水
    
    元のリポジトリからサンプルファイルをダウンロードしてお試しください。
    """)
    
    # 画像処理と予測
    if uploaded_file is not None:
        try:
            # ファイル情報表示
            st.success(f"✅ ファイル受信: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
            
            # 画像処理
            with st.spinner("📊 画像を処理中..."):
                processed_path = processor.process_sentinel2_image(uploaded_file)
                
                # RGB可視化画像作成（前処理済み画像から）
                with rasterio.open(processed_path) as src:
                    processed_data = src.read()
                rgb_image = processor.create_rgb_image(processed_data)
            
            st.success("✅ 画像処理完了!")
            
            # 入力画像プレビュー
            st.subheader("🖼️ 入力画像プレビュー")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(rgb_image, caption="RGB合成画像 (バンド3,2,1)", use_column_width=True)
            
            with col2:
                st.markdown("**画像情報**")
                st.write(f"- サイズ: {processed_data.shape[1]}×{processed_data.shape[2]}")
                st.write(f"- バンド数: {processed_data.shape[0]}")
                st.write(f"- データ型: {processed_data.dtype}")
                st.write(f"- 値域: {processed_data.min():.3f} - {processed_data.max():.3f}")
            
            # 予測実行
            st.header("🧠 AI洪水検出")
            
            # モデルタイプの確認表示
            if isinstance(st.session_state.model, SimpleCNNModel):
                st.error("⚠️ **注意**: 現在プレースホルダーモデルを使用中です。実際のPrithviモデルではありません。")
                st.info("💡 実際のPrithviモデルが正しく読み込まれていない可能性があります。")
            elif isinstance(st.session_state.model, PrithviModel):
                st.success("✅ **Prithviモデル**を使用中です。")
            else:
                st.warning("⚠️ **未知のモデル**が読み込まれています。")
            
            if st.button("🔍 洪水検出を実行", type="primary", use_container_width=True):
                try:
                    # inference.pyが利用可能かチェック
                    if not INFERENCE_AVAILABLE:
                        st.error("❌ inference.pyが利用できません。正しい推論を実行できません。")
                        st.info("💡 プレースホルダーモデルによる疑似予測を実行します。")
                        
                        # プレースホルダー処理（従来の方法）
                        with st.spinner("🤖 プレースホルダーモデルで予測中..."):
                            # 画像をテンソルに変換（RGB画像から推測）
                            dummy_input = torch.randn(1, 6, 512, 512)
                            
                            with torch.no_grad():
                                prediction = st.session_state.model(dummy_input)
                                prediction_prob = torch.softmax(prediction, dim=1)
                                prediction_mask = (prediction_prob[:, 1] > 0.3).float().squeeze().numpy()
                            
                            # オーバーレイ画像作成
                            overlay_image = processor.create_prediction_overlay(rgb_image, prediction_mask)
                            
                            # 結果表示（プレースホルダー）
                            st.header("📊 検出結果 (プレースホルダー)")
                            st.error("⚠️ **これはプレースホルダーモデルによるデモ結果です。**")
                            
                            # 統計情報
                            total_pixels = prediction_mask.size
                            flood_pixels = np.sum(prediction_mask == 1)
                            flood_ratio = flood_pixels / total_pixels * 100
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("総ピクセル数", f"{total_pixels:,}")
                            col2.metric("洪水ピクセル数", f"{flood_pixels:,}")
                            col3.metric("洪水面積率", f"{flood_ratio:.2f}%")
                    
                    else:
                        # 正しい推論を実行
                        with st.spinner("🤖 Prithviモデルで正しい推論を実行中..."):
                            # 進行状況表示
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("📊 推論データを準備中...")
                            progress_bar.progress(25)
                            
                            # モデルタイプを確認
                            if isinstance(st.session_state.model, SimpleCNNModel):
                                st.error("❌ プレースホルダーモデルが読み込まれています。正しい推論を実行できません。")
                                return
                            
                            status_text.text("🧠 正しいAI推論を実行中...")
                            progress_bar.progress(50)
                            
                            # 正しい推論を実行
                            input_rgb, prediction_img, overlay_img, pred_tensor = processor.run_inference(
                                processed_path, 
                                st.session_state.model, 
                                st.session_state.datamodule
                            )
                            
                            status_text.text("🎨 結果画像を生成中...")
                            progress_bar.progress(75)
                            
                            # 予測結果の統計計算
                            if pred_tensor is not None:
                                pred_numpy = pred_tensor.cpu().numpy() if hasattr(pred_tensor, 'cpu') else pred_tensor
                                if pred_numpy.ndim > 2:
                                    pred_numpy = pred_numpy.squeeze()
                                
                                total_pixels = pred_numpy.size
                                flood_pixels = np.sum(pred_numpy == 1)
                                flood_ratio = flood_pixels / total_pixels * 100
                            else:
                                total_pixels = 512 * 512
                                flood_pixels = 0
                                flood_ratio = 0.0
                            
                            progress_bar.progress(100)
                            status_text.text("✅ 完了!")
                            
                            # 結果表示
                            st.header("📊 検出結果")
                            st.success("✅ **正しいPrithviモデル**による推論結果です。")
                            
                            # 統計情報
                            col1, col2, col3 = st.columns(3)
                            col1.metric("総ピクセル数", f"{total_pixels:,}")
                            col2.metric("洪水ピクセル数", f"{flood_pixels:,}")
                            col3.metric("洪水面積率", f"{flood_ratio:.2f}%")
                            
                            # 結果画像表示
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.subheader("入力画像 (RGB)")
                                if input_rgb:
                                    st.image(input_rgb, use_column_width=True)
                                else:
                                    st.image(rgb_image, use_column_width=True)  # フォールバック
                            
                            with col2:
                                st.subheader("洪水予測マスク")
                                if prediction_img:
                                    st.image(prediction_img, use_column_width=True)
                                else:
                                    st.error("予測画像の生成に失敗しました")
                            
                            with col3:
                                st.subheader("オーバーレイ結果")
                                if overlay_img:
                                    st.image(overlay_img, use_column_width=True)
                                else:
                                    st.error("オーバーレイ画像の生成に失敗しました")
                            
                            # ダウンロードセクション
                            st.subheader("💾 結果ダウンロード")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if input_rgb:
                                    buf = tempfile.NamedTemporaryFile(suffix='.png')
                                    input_rgb.save(buf.name)
                                    with open(buf.name, 'rb') as f:
                                        st.download_button(
                                            label="入力画像をダウンロード",
                                            data=f.read(),
                                            file_name=f"{uploaded_file.name.split('.')[0]}_input_rgb.png",
                                            mime="image/png"
                                        )
                            
                            with col2:
                                if prediction_img:
                                    buf = tempfile.NamedTemporaryFile(suffix='.png')
                                    prediction_img.save(buf.name)
                                    with open(buf.name, 'rb') as f:
                                        st.download_button(
                                            label="予測結果をダウンロード",
                                            data=f.read(),
                                            file_name=f"{uploaded_file.name.split('.')[0]}_prediction.png",
                                            mime="image/png"
                                        )
                            
                            with col3:
                                if overlay_img:
                                    buf = tempfile.NamedTemporaryFile(suffix='.png')
                                    overlay_img.save(buf.name)
                                    with open(buf.name, 'rb') as f:
                                        st.download_button(
                                            label="オーバーレイをダウンロード",
                                            data=f.read(),
                                            file_name=f"{uploaded_file.name.split('.')[0]}_overlay.png",
                                            mime="image/png"
                                        )
                            
                            # 解釈ガイド
                            st.subheader("📖 結果の解釈")
                            st.markdown("""
                            - **白い領域**: 洪水と予測された水域
                            - **黒い領域**: 非洪水域（陸地）
                            - **赤い領域**: オーバーレイ画像の洪水領域
                            
                            **注意**: これは正しいPrithvi-EO-2.0モデルによる実際の洪水検出結果です。
                            """)
                    
                    # メモリクリーンアップ
                    gc.collect()
                    
                except Exception as predict_error:
                    st.error(f"❌ 予測エラー: {predict_error}")
                    st.exception(predict_error)
                    st.write("デバッグ情報:")
                    st.write(f"- モデル型: {type(st.session_state.model)}")
                    st.write(f"- INFERENCE_AVAILABLE: {INFERENCE_AVAILABLE}")
                    if 'processed_path' in locals():
                        st.write(f"- 処理済み画像パス: {processed_path}")
                    
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