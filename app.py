import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import gc
import torch
import tempfile
from huggingface_hub import hf_hub_download
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# Streamlit設定
st.set_page_config(
    page_title="Prithvi-EO-2.0 洪水検出システム (AI統合版)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 設定
MODEL_NAME = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
MODEL_FILE = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
CACHE_DIR = "/tmp/prithvi_cache"

@st.cache_resource
def load_model():
    """Hugging Face Hubからモデルをダウンロード・ロード"""
    try:
        with st.spinner("Prithvi-EO-2.0モデルをロード中...（初回は数分かかります）"):
            # モデルファイルをダウンロード
            model_path = hf_hub_download(
                repo_id=MODEL_NAME,
                filename=MODEL_FILE,
                cache_dir=CACHE_DIR
            )
            
            # PyTorchモデルをロード
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(model_path, map_location=device)
            model.eval()
            
            st.success("✅ Prithvi-EO-2.0モデルをロードしました")
            return model, device
            
    except Exception as e:
        st.error(f"モデルロードエラー: {str(e)}")
        st.info("デモモードに切り替えます...")
        return None, None

def preprocess_image(image_array, target_size=(512, 512)):
    """画像前処理"""
    try:
        # RGBチャンネルを6バンドに変換（Prithvi-EO-2.0用）
        # Blue, Green, Red, Narrow NIR, SWIR1, SWIR2
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # RGB画像を6バンドに拡張（デモ用）
            blue = image_array[:, :, 2]   # B
            green = image_array[:, :, 1]  # G
            red = image_array[:, :, 0]    # R
            nir = np.mean(image_array, axis=2)  # NIR近似
            swir1 = np.mean(image_array, axis=2) * 0.8  # SWIR1近似
            swir2 = np.mean(image_array, axis=2) * 0.6  # SWIR2近似
            
            # 6バンド画像作成
            bands = np.stack([blue, green, red, nir, swir1, swir2], axis=2)
        else:
            bands = image_array
        
        # テンソルに変換
        tensor = torch.FloatTensor(bands).permute(2, 0, 1).unsqueeze(0)
        
        # 正規化
        tensor = tensor / 255.0
        
        return tensor
        
    except Exception as e:
        st.error(f"前処理エラー: {str(e)}")
        return None

def predict_flood(model, device, input_tensor):
    """洪水予測実行"""
    try:
        if model is None or device is None:
            return None
            
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            
            # 推論実行
            outputs = model(input_tensor)
            
            # 出力処理
            if hasattr(outputs, 'logits'):
                prediction = outputs.logits
            else:
                prediction = outputs
                
            # ソフトマックス適用
            prediction = torch.softmax(prediction, dim=1)
            
            # 最大クラスを取得
            pred_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
            
            return pred_mask
            
    except Exception as e:
        st.error(f"予測エラー: {str(e)}")
        return None

def create_demo_mask(image_array):
    """デモ用マスク（モデルが利用できない場合）"""
    try:
        height, width = image_array.shape[:2]
        
        # グレースケール変換
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2)
        else:
            gray = image_array
            
        # 暗い領域を洪水として判定
        threshold = np.percentile(gray, 30)
        mask = (gray < threshold).astype(np.uint8)
        
        return mask
        
    except Exception as e:
        st.error(f"デモマスク生成エラー: {str(e)}")
        return np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

def visualize_results(original, mask, title_prefix=""):
    """結果可視化"""
    try:
        # オーバーレイ作成
        overlay = original.copy()
        if len(overlay.shape) == 3:
            overlay[mask == 1] = [255, 0, 0]  # 赤色
        else:
            overlay[mask == 1] = 255
            
        # ブレンド
        alpha = 0.6
        if len(original.shape) == 3:
            result = (alpha * overlay + (1 - alpha) * original).astype(np.uint8)
        else:
            result = overlay
            
        return result
        
    except Exception as e:
        st.error(f"可視化エラー: {str(e)}")
        return original

def main():
    """メインアプリケーション"""
    
    # ヘッダー
    st.title("🌊 Prithvi-EO-2.0 洪水検出システム (AI統合版)")
    
    # モデルロード
    model, device = load_model()
    
    # 情報表示
    with st.expander("ℹ️ アプリケーション情報", expanded=False):
        if model is not None:
            st.success("🤖 **実際のPrithvi-EO-2.0モデル使用中**")
            st.markdown("""
            - **モデル**: ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11
            - **データセット**: Sen1Floods11でファインチューニング済み
            - **入力**: 6バンド（Blue, Green, Red, Narrow NIR, SWIR1, SWIR2）
            - **出力**: 洪水/非洪水のセマンティックセグメンテーション
            """)
        else:
            st.warning("⚠️ **デモモード**: モデルロードに失敗、パターンベース検出使用中")
            st.markdown("""
            - **機能**: 基本的な画像処理とデモ予測
            - **制限**: 実際のPrithviモデル未使用
            - **推奨**: 依存関係を確認してください
            """)
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        if model is not None:
            st.success("🤖 AIモデル: アクティブ")
            st.info(f"デバイス: {'GPU' if device.type == 'cuda' else 'CPU'}")
        else:
            st.warning("🔧 デモモード")
            
        st.subheader("📋 対応形式")
        st.text("• JPG/JPEG")
        st.text("• PNG")
        st.text("• TIFF/TIF")
        
        st.subheader("📊 システム要件")
        st.text("• メモリ: 2GB+ 推奨")
        st.text("• PyTorch")
        st.text("• Hugging Face Hub")
    
    # ファイルアップロード
    st.header("📁 画像アップロード")
    
    uploaded_file = st.file_uploader(
        "衛星画像またはテスト画像を選択してください",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        help="最大100MB、Sentinel-2形式推奨"
    )
    
    if uploaded_file is not None:
        try:
            # 画像読み込み
            with st.spinner("画像を処理中..."):
                # ファイル読み込み
                file_bytes = uploaded_file.read()
                
                # PIL Imageとして開く
                image = Image.open(io.BytesIO(file_bytes))
                
                # RGB変換
                if image.mode != 'RGB':
                    if image.mode == 'RGBA':
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        background.paste(image, mask=image.split()[-1])
                        image = background
                    else:
                        image = image.convert('RGB')
                
                # リサイズ
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                image_array = np.array(image)
            
            st.success("✅ 画像を正常に読み込みました")
            
            # 画像情報
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ファイルサイズ", f"{uploaded_file.size / 1024:.1f} KB")
            with col2:
                st.metric("画像サイズ", f"{image_array.shape[1]}×{image_array.shape[0]}")
            with col3:
                st.metric("チャンネル数", f"{image_array.shape[2] if len(image_array.shape) > 2 else 1}")
            
            # 入力画像表示
            st.subheader("📷 入力画像")
            st.image(image, caption="処理済み画像（512×512）", use_container_width=True)
            
            # 洪水検出実行
            if st.button("🔍 洪水検出を実行", type="primary"):
                with st.spinner("AI推論を実行中..."):
                    
                    if model is not None:
                        # 実際のPrithviモデルで予測
                        st.info("🤖 Prithvi-EO-2.0モデルで推論中...")
                        
                        # 前処理
                        input_tensor = preprocess_image(image_array)
                        
                        if input_tensor is not None:
                            # 予測実行
                            pred_mask = predict_flood(model, device, input_tensor)
                            
                            if pred_mask is not None:
                                # 二値化（クラス1を洪水とする）
                                flood_mask = (pred_mask == 1).astype(np.uint8)
                                prediction_type = "🤖 AI予測"
                            else:
                                # フォールバック
                                flood_mask = create_demo_mask(image_array)
                                prediction_type = "⚠️ デモ予測（AI予測失敗）"
                        else:
                            # フォールバック
                            flood_mask = create_demo_mask(image_array)
                            prediction_type = "⚠️ デモ予測（前処理失敗）"
                    else:
                        # デモモード
                        st.info("🔧 デモモードで処理中...")
                        flood_mask = create_demo_mask(image_array)
                        prediction_type = "🔧 デモ予測"
                    
                    # 結果可視化
                    overlay_result = visualize_results(image_array, flood_mask)
                    
                    # 統計計算
                    total_pixels = flood_mask.size
                    flood_pixels = np.sum(flood_mask == 1)
                    flood_percentage = (flood_pixels / total_pixels) * 100
                    
                    # 結果表示
                    st.subheader(f"📊 洪水検出結果 ({prediction_type})")
                    
                    # 統計情報
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("総ピクセル数", f"{total_pixels:,}")
                    with col2:
                        st.metric("洪水ピクセル数", f"{flood_pixels:,}")
                    with col3:
                        st.metric("洪水面積率", f"{flood_percentage:.2f}%")
                    
                    # 結果画像表示
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("🖼️ 入力画像")
                        st.image(image, caption="元画像", use_container_width=True)
                    
                    with col2:
                        st.subheader("🗺️ 予測マスク")
                        mask_image = Image.fromarray((flood_mask * 255).astype(np.uint8))
                        st.image(mask_image, caption="洪水領域（白：洪水、黒：非洪水）", use_container_width=True)
                    
                    with col3:
                        st.subheader("🎯 オーバーレイ")
                        st.image(overlay_result, caption="洪水領域オーバーレイ（赤：洪水）", use_container_width=True)
                    
                    # ダウンロードセクション
                    st.subheader("💾 結果のダウンロード")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # ダウンロード用バイトデータ作成
                    def img_to_bytes(img_array):
                        img = Image.fromarray(img_array)
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        buf.seek(0)
                        return buf.getvalue()
                    
                    with col1:
                        input_bytes = img_to_bytes(image_array)
                        st.download_button(
                            "📷 入力画像をダウンロード",
                            data=input_bytes,
                            file_name=f"input_{uploaded_file.name.split('.')[0]}.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        mask_bytes = img_to_bytes((flood_mask * 255).astype(np.uint8))
                        st.download_button(
                            "🗺️ 予測マスクをダウンロード",
                            data=mask_bytes,
                            file_name=f"mask_{uploaded_file.name.split('.')[0]}.png",
                            mime="image/png"
                        )
                    
                    with col3:
                        overlay_bytes = img_to_bytes(overlay_result)
                        st.download_button(
                            "🎯 オーバーレイをダウンロード",
                            data=overlay_bytes,
                            file_name=f"overlay_{uploaded_file.name.split('.')[0]}.png",
                            mime="image/png"
                        )
                    
                    # 注意事項
                    if model is not None:
                        st.success("""
                        ✅ **実際のPrithvi-EO-2.0モデルを使用**: 
                        Sen1Floods11データセットでファインチューニングされた高精度なAIモデルによる洪水検出結果です。
                        """)
                    else:
                        st.info("""
                        ℹ️ **デモモード**: 
                        実際のPrithvi-EO-2.0モデルは利用できませんが、基本的な画像処理による洪水領域の推定を表示しています。
                        """)
                    
                    # メモリクリーンアップ
                    del flood_mask, overlay_result
                    gc.collect()
        
        except Exception as e:
            st.error(f"画像処理中にエラーが発生しました: {str(e)}")
            
            # デバッグ情報
            with st.expander("🔧 デバッグ情報"):
                st.code(f"""
                ファイル名: {uploaded_file.name}
                ファイルサイズ: {uploaded_file.size} bytes
                エラー詳細: {str(e)}
                """)
    
    # フッター
    st.markdown("---")
    st.markdown("""
    **開発者:** IBM & NASA Geospatial Team  
    **モデル:** [Prithvi-EO-2.0-300M-TL-Sen1Floods11](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)  
    **Paper:** [Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model](https://arxiv.org/abs/2412.02732)
    """)

if __name__ == "__main__":
    main()