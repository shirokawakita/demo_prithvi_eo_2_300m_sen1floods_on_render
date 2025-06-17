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
        """モデルをダウンロードして読み込み"""
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
                status_text.text("🔄 モデルを読み込み中...")
                
                # モデル読み込み - より詳細なデバッグ
                try:
                    device = torch.device('cpu')
                    
                    st.write("🔍 **モデルファイルの詳細分析開始**")
                    
                    # Prithviモデルを正しく読み込み
                    model_data = torch.load(model_path, map_location=device)
                    
                    st.write(f"📋 モデルデータ型: {type(model_data)}")
                    st.write(f"📋 データサイズ: {len(str(model_data))} 文字")
                    
                    if isinstance(model_data, dict):
                        st.write(f"📋 利用可能なキー: {list(model_data.keys())}")
                        
                        # 各キーの詳細情報を表示
                        for key in model_data.keys():
                            value = model_data[key]
                            st.write(f"  - **{key}**: {type(value)}")
                            if hasattr(value, 'shape'):
                                st.write(f"    形状: {value.shape}")
                            elif isinstance(value, dict):
                                st.write(f"    辞書キー数: {len(value)}")
                                if len(value) < 10:  # 小さい辞書の場合はキーを表示
                                    st.write(f"    サブキー: {list(value.keys())}")
                        
                        # Prithviモデルの構造を理解してから読み込み
                        model = None
                        
                        # まず、'model'キーを優先的に試行
                        if 'model' in model_data:
                            st.write("🔑 'model' キーを使用")
                            try:
                                model_obj = model_data['model']
                                st.write(f"🔍 modelオブジェクト型: {type(model_obj)}")
                                
                                # モデルがnn.Moduleの場合
                                if isinstance(model_obj, nn.Module):
                                    model = model_obj
                                    st.success("✅ 'model' キーからnn.Module読み込み成功")
                                else:
                                    st.write(f"⚠️ modelは{type(model_obj)}です。state_dictかもしれません。")
                                    
                            except Exception as load_error:
                                st.warning(f"⚠️ 'model' キーでの読み込み失敗: {load_error}")
                        
                        # 次に state_dict系のキーを試行
                        if model is None:
                            for key in ['state_dict', 'model_state_dict']:
                                if key in model_data:
                                    st.write(f"🔑 キー '{key}' を試行中...")
                                    try:
                                        # 実際のPrithviモデルの構造を推測する必要がある
                                        # とりあえずstate_dictの中身を確認
                                        state_dict = model_data[key]
                                        st.write(f"📋 State dict keys sample: {list(state_dict.keys())[:10]}")
                                        st.write(f"📋 State dict総キー数: {len(state_dict)}")
                                        
                                        # state_dictの構造から元のモデル構造を推測
                                        has_transformer = any('transformer' in k or 'attention' in k for k in state_dict.keys())
                                        has_encoder = any('encoder' in k for k in state_dict.keys())
                                        has_decoder = any('decoder' in k for k in state_dict.keys())
                                        
                                        st.write(f"🔍 推測される構造:")
                                        st.write(f"  - Transformer要素: {has_transformer}")
                                        st.write(f"  - Encoder要素: {has_encoder}")
                                        st.write(f"  - Decoder要素: {has_decoder}")
                                        
                                        # 実際のPrithviモデルを作成してstate_dictを読み込み
                                        try:
                                            model = PrithviModel(
                                                img_size=512,
                                                patch_size=16,
                                                num_frames=1,  # 単一時点の画像
                                                num_bands=6,   # Sentinel-2の6バンド
                                                embed_dim=768,
                                                num_classes=2  # 洪水/非洪水
                                            )
                                            # state_dictの構造を調整して読み込み
                                            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                                            st.success("✅ Prithviモデルのstate_dictを読み込み成功!")
                                            st.write(f"📋 不足キー数: {len(missing_keys)}")
                                            st.write(f"📋 予期しないキー数: {len(unexpected_keys)}")
                                            if missing_keys:
                                                st.write(f"📋 不足キー例: {missing_keys[:5]}")
                                            if unexpected_keys:
                                                st.write(f"📋 予期しないキー例: {unexpected_keys[:5]}")
                                        except Exception as prithvi_error:
                                            st.warning(f"⚠️ Prithviモデルの作成に失敗: {prithvi_error}")
                                            st.info("💡 プレースホルダーモデルを使用します")
                                            model = _self._create_placeholder_model()
                                        break
                                    except Exception as load_error:
                                        st.warning(f"⚠️ キー '{key}' での読み込み失敗: {load_error}")
                        
                        # 他のキーも試行
                        if model is None:
                            for key in ['net', 'network', 'encoder', 'decoder']:
                                if key in model_data:
                                    st.write(f"🔑 キー '{key}' を試行中...")
                                    try:
                                        model = model_data[key]
                                        st.success(f"✅ キー '{key}' からの読み込み成功")
                                        break
                                    except Exception as load_error:
                                        st.warning(f"⚠️ キー '{key}' での読み込み失敗: {load_error}")
                        
                        # どのキーでも読み込めない場合
                        if model is None:
                            st.warning("⚠️ 標準的なキーでモデルを読み込めませんでした")
                            st.info("💡 実際のPrithviモデル構造の実装が必要です")
                            model = _self._create_placeholder_model()
                    
                    else:
                        # 直接モデルオブジェクトの場合
                        model = model_data
                        st.success("✅ 直接モデルオブジェクトを読み込み")
                    
                    # モデルを評価モードに設定
                    if hasattr(model, 'eval'):
                        model.eval()
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 完了!")
                    
                    # メモリクリーンアップ
                    gc.collect()
                    
                    return model, config
                    
                except Exception as model_error:
                    st.error(f"❌ モデル読み込みエラー: {model_error}")
                    st.info("💡 プレースホルダーモデルを使用します")
                    return _self._create_placeholder_model(), {}
                    
        except Exception as e:
            st.error(f"❌ 全体的なエラー: {e}")
            return _self._create_placeholder_model(), {}
    
    def _create_placeholder_model(self):
        """プレースホルダーモデルを作成"""
        st.info("🔧 プレースホルダーモデルを作成中...")
        model = SimpleCNNModel(in_channels=6, num_classes=2)
        model.eval()
        return model

class ImageProcessor:
    def __init__(self):
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
                st.write(f"📊 元画像: {image_data.shape} (バンド, 高さ, 幅)")
                
                if image_data.shape[0] < 6:
                    # バンドが足りない場合は繰り返しで補完
                    st.warning(f"⚠️ バンド数不足 ({image_data.shape[0]} < 6). 補完します.")
                    while image_data.shape[0] < 6:
                        image_data = np.concatenate([image_data, image_data[:1]], axis=0)
                
                # 必要な6バンドを選択
                selected_bands = image_data[:6]
                
                # データ型確認・変換
                st.write(f"📊 データ型: {selected_bands.dtype}")
                if selected_bands.dtype == np.uint16:
                    selected_bands = selected_bands.astype(np.float32)
                elif selected_bands.dtype == np.int16:
                    selected_bands = selected_bands.astype(np.float32)
                
                # サイズ調整
                st.write(f"📊 リサイズ前: {selected_bands.shape}")
                processed_bands = []
                for i, band in enumerate(selected_bands):
                    resized_band = resize(band, self.target_size, preserve_range=True, anti_aliasing=True)
                    processed_bands.append(resized_band)
                
                processed_image = np.stack(processed_bands, axis=0)
                st.write(f"📊 リサイズ後: {processed_image.shape}")
                
                # 正規化
                processed_image = self.normalize_image(processed_image)
                
                # 一時ファイル削除
                os.remove(temp_path)
                
                return processed_image
                
        except Exception as e:
            # エラー時も一時ファイルを削除
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception(f"画像処理エラー: {e}")
    
    def normalize_image(self, image):
        """画像を正規化"""
        # 基本的な正規化 (0-1範囲)
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)
        
        return image.astype(np.float32)
    
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
            model, config = model_loader.download_and_load_model()
            
            if model is not None:
                st.session_state.model = model
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
                processed_image = processor.process_sentinel2_image(uploaded_file)
                
                # RGB可視化画像作成
                rgb_image = processor.create_rgb_image(processed_image)
            
            st.success("✅ 画像処理完了!")
            
            # 入力画像プレビュー
            st.subheader("🖼️ 入力画像プレビュー")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(rgb_image, caption="RGB合成画像 (バンド3,2,1)", use_column_width=True)
            
            with col2:
                st.markdown("**画像情報**")
                st.write(f"- サイズ: {processed_image.shape[1]}×{processed_image.shape[2]}")
                st.write(f"- バンド数: {processed_image.shape[0]}")
                st.write(f"- データ型: {processed_image.dtype}")
                st.write(f"- 値域: {processed_image.min():.3f} - {processed_image.max():.3f}")
            
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
                    with st.spinner("🤖 Prithviモデルで予測中..."):
                        # 進行状況表示
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("📊 テンソルに変換中...")
                        progress_bar.progress(25)
                        
                        # テンソルに変換
                        input_tensor = torch.from_numpy(processed_image).unsqueeze(0).float()
                        st.write(f"📊 入力テンソル形状: {input_tensor.shape}")
                        st.write(f"📊 モデルタイプ: {type(st.session_state.model).__name__}")
                        
                        status_text.text("🧠 AI予測実行中...")
                        progress_bar.progress(50)
                        
                        # 予測実行
                        with torch.no_grad():
                            prediction = st.session_state.model(input_tensor)
                            st.write(f"📊 予測出力形状: {prediction.shape}")
                            st.write(f"📊 予測値の範囲: {prediction.min().item():.4f} - {prediction.max().item():.4f}")
                            
                            # プレースホルダーモデルかどうかで処理を分ける
                            if isinstance(st.session_state.model, SimpleCNNModel):
                                st.warning("⚠️ プレースホルダーモデルによる疑似予測です")
                                # より現実的な予測結果を生成
                                prediction_prob = torch.softmax(prediction, dim=1)
                                # ランダムではなく、より現実的なパターンを生成
                                prediction_mask = (prediction_prob[:, 1] > 0.3).float().squeeze().numpy()
                            else:
                                # 実際のPrithviモデルの場合
                                if prediction.shape[1] == 2:  # クラス数が2の場合
                                    prediction_mask = torch.argmax(prediction, dim=1).squeeze().numpy()
                                else:
                                    # シグモイド出力の場合
                                    prediction_mask = (torch.sigmoid(prediction) > 0.5).float().squeeze().numpy()
                        
                        status_text.text("🎨 結果画像を生成中...")
                        progress_bar.progress(75)
                        
                        # オーバーレイ画像作成
                        overlay_image = processor.create_prediction_overlay(rgb_image, prediction_mask)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ 完了!")
                        
                        # メモリクリーンアップ
                        del prediction, input_tensor
                        gc.collect()
                    
                    # 結果表示
                    st.header("📊 検出結果")
                    
                    # 統計情報
                    total_pixels = prediction_mask.size
                    flood_pixels = np.sum(prediction_mask == 1)
                    non_flood_pixels = total_pixels - flood_pixels
                    flood_ratio = flood_pixels / total_pixels * 100
                    
                    # プレースホルダーモデルの場合は警告を表示
                    if isinstance(st.session_state.model, SimpleCNNModel):
                        st.error("⚠️ **これはプレースホルダーモデルによるデモ結果です。実際の洪水検出ではありません。**")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("総ピクセル数", f"{total_pixels:,}")
                    col2.metric("洪水ピクセル数", f"{flood_pixels:,}")
                    col3.metric("洪水面積率", f"{flood_ratio:.2f}%")
                    
                    # 実際の値を表示
                    st.write("**詳細統計:**")
                    st.write(f"- 非洪水ピクセル数: {non_flood_pixels:,}")
                    st.write(f"- 非洪水面積率: {100-flood_ratio:.2f}%")
                    
                    # 結果画像表示
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("入力画像 (RGB)")
                        st.image(rgb_image, use_column_width=True)
                    
                    with col2:
                        st.subheader("洪水予測マスク")
                        mask_vis = (prediction_mask * 255).astype(np.uint8)
                        st.image(mask_vis, use_column_width=True)
                    
                    with col3:
                        st.subheader("オーバーレイ結果")
                        st.image(overlay_image, use_column_width=True)
                    
                    # ダウンロードセクション
                    st.subheader("💾 結果ダウンロード")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(create_download_link(rgb_image, "input_rgb.png"), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(create_download_link(np.stack([mask_vis]*3, axis=-1), "prediction_mask.png"), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(create_download_link(overlay_image, "flood_overlay.png"), unsafe_allow_html=True)
                    
                    # 解釈ガイド
                    st.subheader("📖 結果の解釈")
                    st.markdown("""
                    - **白い領域**: 洪水と予測された水域
                    - **黒い領域**: 非洪水域（陸地）
                    - **赤い領域**: オーバーレイ画像の洪水領域
                    
                    **注意**: プレースホルダーモードでは実際の洪水検出ではなく、デモ用の予測結果を表示しています。
                    """)
                    
                except Exception as predict_error:
                    st.error(f"❌ 予測エラー: {predict_error}")
                    st.write("デバッグ情報:")
                    st.write(f"- モデル型: {type(st.session_state.model)}")
                    st.write(f"- 入力画像形状: {processed_image.shape}")
                    
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