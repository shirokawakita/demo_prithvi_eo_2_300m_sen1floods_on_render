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
                except Exception as config_error:
                    st.warning(f"⚠️ 設定ファイルエラー: {config_error}")
                    config = {}
                
                progress_bar.progress(90)
                status_text.text("🔄 モデルを読み込み中...")
                
                # モデル読み込み
                try:
                    device = torch.device('cpu')
                    
                    # Torchモデルを読み込み
                    model_data = torch.load(model_path, map_location=device)
                    
                    st.write(f"📋 モデルデータ型: {type(model_data)}")
                    
                    if isinstance(model_data, dict):
                        st.write(f"📋 利用可能なキー: {list(model_data.keys())}")
                        
                        # 一般的なキーパターンを試行
                        model = None
                        for key in ['model', 'state_dict', 'model_state_dict', 'net', 'network']:
                            if key in model_data:
                                st.write(f"🔑 キー '{key}' を使用")
                                try:
                                    if key == 'state_dict' or key == 'model_state_dict':
                                        # state_dictの場合は新しいモデルを作成
                                        model = SimpleCNNModel()
                                        # 部分的にstate_dictを読み込み（サイズが合わない部分は無視）
                                        model.load_state_dict(model_data[key], strict=False)
                                    else:
                                        model = model_data[key]
                                    break
                                except Exception as load_error:
                                    st.warning(f"⚠️ キー '{key}' でのロードに失敗: {load_error}")
                                    continue
                        
                        # どのキーでも読み込めない場合
                        if model is None:
                            st.warning("⚠️ 標準的なキーでモデルを読み込めませんでした")
                            model = _self._create_placeholder_model()
                    
                    else:
                        # 直接モデルオブジェクトの場合
                        model = model_data
                    
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
                        
                        status_text.text("🧠 AI予測実行中...")
                        progress_bar.progress(50)
                        
                        # 予測実行
                        with torch.no_grad():
                            prediction = st.session_state.model(input_tensor)
                            st.write(f"📊 予測出力形状: {prediction.shape}")
                            prediction_mask = torch.argmax(prediction, dim=1).squeeze().numpy()
                        
                        status_text.text("🎨 結果画像を生成中...")
                        progress_bar.progress(75)
                        
                        # オーバーレイ画像作成
                        overlay_image = processor.create_prediction_overlay(rgb_image, prediction_mask)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ 完了!")
                        
                        # メモリクリーンアップ
                        del prediction, input_tensor
                        gc.collect()
                    
                    # プレースホルダーモデル使用時の警告
                    if isinstance(st.session_state.model, SimpleCNNModel):
                        st.warning("⚠️ プレースホルダーモデルによるデモ予測結果です。")
                    
                    # 結果表示
                    st.header("📊 検出結果")
                    
                    # 統計情報
                    total_pixels = prediction_mask.size
                    flood_pixels = np.sum(prediction_mask == 1)
                    flood_ratio = flood_pixels / total_pixels * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("総ピクセル数", f"{total_pixels:,}")
                    col2.metric("洪水ピクセル数", f"{flood_pixels:,}")
                    col3.metric("洪水面積率", f"{flood_ratio:.2f}%")
                    
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