import streamlit as st
import numpy as np
from PIL import Image
import io
import gc
import traceback
from typing import Optional, Tuple

# Streamlit設定
st.set_page_config(
    page_title="Prithvi-EO-2.0 洪水検出システム",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定数
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'tif', 'tiff']

def validate_file(uploaded_file) -> bool:
    """ファイルの検証"""
    try:
        if uploaded_file is None:
            return False
        
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"ファイルサイズが{MAX_FILE_SIZE // (1024*1024)}MBを超えています。")
            return False
        
        file_name = uploaded_file.name.lower()
        file_extension = file_name.split('.')[-1] if '.' in file_name else ''
        
        if file_extension not in SUPPORTED_FORMATS:
            st.error("サポートされていないファイル形式です。対応形式: JPG, PNG, TIFF")
            return False
        
        return True
    except Exception as e:
        st.error(f"ファイル検証エラー: {str(e)}")
        return False

def process_image(uploaded_file) -> Optional[Tuple[np.ndarray, Image.Image]]:
    """画像処理"""
    try:
        if not validate_file(uploaded_file):
            return None
        
        # ファイルを読み込み
        file_bytes = uploaded_file.read()
        
        # PIL Imageとして開く
        image = Image.open(io.BytesIO(file_bytes))
        
        # RGBに変換
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # 白背景でRGBAをRGBに変換
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # 512x512にリサイズ
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # NumPy配列に変換
        image_array = np.array(image)
        
        return image_array, image
        
    except Exception as e:
        st.error(f"画像処理エラー: {str(e)}")
        return None

def create_demo_mask(image_array: np.ndarray) -> np.ndarray:
    """デモ用の洪水マスク生成"""
    try:
        height, width = image_array.shape[:2]
        
        # グレースケールに変換
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2)
        else:
            gray = image_array
        
        # 閾値処理で暗い領域を洪水として設定
        threshold = np.percentile(gray, 25)
        mask = (gray < threshold).astype(np.uint8)
        
        return mask
        
    except Exception as e:
        st.error(f"マスク生成エラー: {str(e)}")
        height, width = image_array.shape[:2]
        return np.zeros((height, width), dtype=np.uint8)

def create_overlay(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """オーバーレイ画像作成"""
    try:
        overlay = original.copy()
        overlay[mask == 1] = [255, 0, 0]  # 赤色
        
        # ブレンド
        alpha = 0.5
        result = (alpha * overlay + (1 - alpha) * original).astype(np.uint8)
        
        return result
        
    except Exception as e:
        st.error(f"オーバーレイ作成エラー: {str(e)}")
        return original

def image_to_bytes(image_array: np.ndarray) -> bytes:
    """画像をバイトに変換"""
    try:
        if len(image_array.shape) == 2:
            # グレースケール画像の場合
            image = Image.fromarray(image_array, mode='L')
        else:
            # カラー画像の場合
            image = Image.fromarray(image_array)
        
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        st.error(f"画像変換エラー: {str(e)}")
        return b''

def main():
    """メインアプリケーション"""
    
    # タイトル
    st.title("🌊 Prithvi-EO-2.0 洪水検出システム")
    
    # 情報セクション
    with st.expander("ℹ️ アプリについて"):
        st.markdown("""
        **デモ版アプリケーション**
        
        - 画像アップロード（JPG/PNG/TIFF対応）
        - 自動リサイズ（512×512）
        - デモ洪水検出（パターンベース）
        - 結果表示とダウンロード
        
        ⚠️ これはデモ版です。実際のPrithviモデルは使用していません。
        """)
    
    # サイドバー
    with st.sidebar:
        st.header("設定")
        st.info("Render Free Tierで動作中")
        
        st.subheader("対応形式")
        for fmt in SUPPORTED_FORMATS:
            st.text(f"• {fmt.upper()}")
    
    # ファイルアップロード
    st.header("画像アップロード")
    
    uploaded_file = st.file_uploader(
        "画像ファイルを選択",
        type=SUPPORTED_FORMATS,
        help="最大50MB、対応形式: JPG, PNG, TIFF"
    )
    
    if uploaded_file is not None:
        # 画像処理
        with st.spinner("画像処理中..."):
            result = process_image(uploaded_file)
        
        if result is not None:
            image_array, processed_image = result
            
            st.success("✅ 画像を読み込みました")
            
            # 画像情報
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ファイルサイズ", f"{uploaded_file.size / 1024:.1f} KB")
            with col2:
                st.metric("画像サイズ", f"{image_array.shape[1]}×{image_array.shape[0]}")
            
            # 入力画像表示
            st.subheader("入力画像")
            st.image(processed_image, caption="リサイズ済み画像", use_container_width=True)
            
            # 検出実行
            if st.button("🔍 洪水検出実行", type="primary"):
                with st.spinner("処理中..."):
                    try:
                        # マスク生成
                        mask = create_demo_mask(image_array)
                        
                        # オーバーレイ作成
                        overlay = create_overlay(image_array, mask)
                        
                        # 統計
                        total_pixels = mask.size
                        flood_pixels = np.sum(mask == 1)
                        flood_percentage = (flood_pixels / total_pixels) * 100
                        
                        # 結果表示
                        st.subheader("検出結果")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("総ピクセル数", f"{total_pixels:,}")
                        with col2:
                            st.metric("洪水ピクセル数", f"{flood_pixels:,}")
                        with col3:
                            st.metric("洪水面積率", f"{flood_percentage:.2f}%")
                        
                        # 画像表示
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.image(processed_image, caption="入力画像", use_container_width=True)
                        
                        with col2:
                            mask_display = (mask * 255).astype(np.uint8)
                            st.image(mask_display, caption="予測マスク", use_container_width=True)
                        
                        with col3:
                            st.image(overlay, caption="オーバーレイ", use_container_width=True)
                        
                        # ダウンロード
                        st.subheader("ダウンロード")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            input_bytes = image_to_bytes(image_array)
                            if input_bytes:
                                st.download_button(
                                    "入力画像",
                                    data=input_bytes,
                                    file_name="input.png",
                                    mime="image/png"
                                )
                        
                        with col2:
                            mask_bytes = image_to_bytes(mask_display)
                            if mask_bytes:
                                st.download_button(
                                    "予測マスク",
                                    data=mask_bytes,
                                    file_name="mask.png",
                                    mime="image/png"
                                )
                        
                        with col3:
                            overlay_bytes = image_to_bytes(overlay)
                            if overlay_bytes:
                                st.download_button(
                                    "オーバーレイ",
                                    data=overlay_bytes,
                                    file_name="overlay.png",
                                    mime="image/png"
                                )
                        
                        # メモリクリーンアップ
                        del mask, overlay
                        gc.collect()
                        
                    except Exception as e:
                        st.error(f"検出処理エラー: {str(e)}")
                        st.code(traceback.format_exc())
            
            # メモリクリーンアップ
            del image_array, processed_image
            gc.collect()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"アプリケーションエラー: {str(e)}")
        st.code(traceback.format_exc())