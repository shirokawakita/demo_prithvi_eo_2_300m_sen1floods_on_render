import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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

# Streamlit設定
st.set_page_config(
    page_title="Prithvi-EO-2.0 洪水検出",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 環境変数設定（config.tomlの代替）
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

class PrithviModelLoader:
    def __init__(self):
        self.repo_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
        self.model_filename = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
        self.config_filename = "config.yaml"
        self.cache_dir = Path("/tmp/prithvi_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    @st.cache_resource
    def download_model(_self):
        """Hugging Face HubからPrithviモデルをダウンロード"""
        try:
            with st.spinner("Prithviモデルをダウンロード中... (約1.28GB)"):
                # プログレスバーを表示
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("モデルファイルをダウンロード中...")
                progress_bar.progress(25)
                
                # モデルファイルをダウンロード
                model_path = hf_hub_download(
                    repo_id=_self.repo_id,
                    filename=_self.model_filename,
                    cache_dir=str(_self.cache_dir)
                )
                
                progress_bar.progress(75)
                status_text.text("設定ファイルをダウンロード中...")
                
                # 設定ファイルをダウンロード
                config_path = hf_hub_download(
                    repo_id=_self.repo_id,
                    filename=_self.config_filename,
                    cache_dir=str(_self.cache_dir)
                )
                
                progress_bar.progress(100)
                status_text.text("ダウンロード完了!")
                
                return model_path, config_path
                
        except Exception as e:
            st.error(f"モデルのダウンロードに失敗しました: {e}")
            return None, None
    
    @st.cache_resource
    def load_model(_self):
        """モデルを読み込み"""
        model_path, config_path = _self.download_model()
        
        if model_path is None or config_path is None:
            return None, None
        
        try:
            with st.spinner("モデルを読み込み中..."):
                # 設定ファイル読み込み
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # モデル読み込み
                device = torch.device('cpu')  # Renderではcpu使用
                model = torch.load(model_path, map_location=device)
                model.eval()
                
                # メモリクリーンアップ
                gc.collect()
                
                return model, config
                
        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {e}")
            return None, None

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
                if image_data.shape[0] < 6:
                    raise ValueError(f"不十分なバンド数: {image_data.shape[0]} < 6")
                
                # 必要な6バンドを選択
                selected_bands = image_data[:6]
                
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
                
                # 一時ファイル削除
                os.remove(temp_path)
                
                return processed_image
                
        except Exception as e:
            # エラー時も一時ファイルを削除
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception(f"画像処理エラー: {e}")
    
    def normalize_image(self, image):
        """画像を正規化"""
        # Prithviモデル用の正規化
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

def create_download_link(image, filename):
    """画像のダウンロードリンクを作成"""
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration: none; color: #1f77b4;">📥 {filename}</a>'
    return href

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
    - **精度**: mIoU 88.68%
    """)
    
    # システム情報表示
    show_system_info()
    
    # 警告メッセージ
    st.sidebar.markdown("### ⚠️ 重要事項")
    st.sidebar.warning("""
    - 初回起動時は20-30分かかります
    - 処理時間: 約30-60秒/画像
    - 最大アップロード: 100MB
    """)
    
    # モデル初期化
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded:
        st.info("🚀 モデルを初期化しています...")
        model_loader = PrithviModelLoader()
        model, config = model_loader.load_model()
        
        if model is not None:
            st.session_state.model = model
            st.session_state.config = config
            st.session_state.model_loaded = True
            st.success("✅ モデルの読み込み完了!")
            st.balloons()
        else:
            st.error("❌ モデルの読み込みに失敗しました")
            st.stop()
    
    # 画像処理器初期化
    processor = ImageProcessor()
    
    # ファイルアップロード
    st.header("📁 Sentinel-2画像のアップロード")
    
    uploaded_file = st.file_uploader(
        "TIFFファイルを選択してください",
        type=['tif', 'tiff'],
        help="Sentinel-2 L1Cまたは6バンド対応のGeoTIFFファイル（最大100MB）"
    )
    
    # サンプルデータ情報
    st.markdown("### 🌍 サンプルデータについて")
    st.info("""
    サンプルデータは以下の地域の洪水画像です：
    - 🇮🇳 **インド**: モンスーンによる洪水
    - 🇪🇸 **スペイン**: 河川氾濫
    - 🇺🇸 **アメリカ**: ハリケーンによる洪水
    
    実際のサンプルファイルは元のリポジトリからダウンロードしてご利用ください。
    """)
    
    # 画像処理と予測
    if uploaded_file is not None:
        try:
            # ファイル情報表示
            st.success(f"✅ ファイル受信: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
            
            # 画像処理
            with st.spinner("画像を処理中..."):
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
            
            # 予測実行
            st.header("🧠 AI洪水検出")
            
            if st.button("🔍 洪水検出を実行", type="primary", use_container_width=True):
                with st.spinner("Prithviモデルで予測中..."):
                    # 進行状況表示
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("テンソルに変換中...")
                    progress_bar.progress(25)
                    
                    # テンソルに変換
                    input_tensor = torch.from_numpy(processed_image).unsqueeze(0).float()
                    
                    status_text.text("AI予測実行中...")
                    progress_bar.progress(50)
                    
                    # 予測実行
                    with torch.no_grad():
                        prediction = st.session_state.model(input_tensor)
                        prediction_mask = torch.argmax(prediction, dim=1).squeeze().numpy()
                    
                    status_text.text("結果画像を生成中...")
                    progress_bar.progress(75)
                    
                    # オーバーレイ画像作成
                    overlay_image = processor.create_prediction_overlay(rgb_image, prediction_mask)
                    
                    progress_bar.progress(100)
                    status_text.text("完了!")
                    
                    # メモリクリーンアップ
                    del prediction, input_tensor
                    gc.collect()
                
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
                    # 予測マスクを可視化用に変換
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
                
                **注意**: 雲や影の影響で誤検出が生じる場合があります。
                """)
                
        except Exception as e:
            st.error(f"❌ エラー: {e}")
            st.markdown("### 🔧 トラブルシューティング")
            st.markdown("""
            - ファイルが正しいTIFF形式か確認してください
            - ファイルサイズが100MB以下か確認してください
            - Sentinel-2データで6バンド以上含まれているか確認してください
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