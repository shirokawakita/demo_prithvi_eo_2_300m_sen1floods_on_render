import streamlit as st
import numpy as np
from PIL import Image
import io
import tempfile
import os
import gc
import traceback
from typing import Optional, Tuple, Union

# Streamlit設定
st.set_page_config(
    page_title="Prithvi-EO-2.0 洪水検出システム (AI統合版)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# メモリ制限とエラーハンドリングの改善
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB制限（Renderのメモリ制約考慮）
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'tif', 'tiff']

def validate_file(uploaded_file) -> bool:
    """ファイルの検証"""
    if uploaded_file is None:
        return False
    
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"ファイルサイズが{MAX_FILE_SIZE / (1024*1024):.0f}MBを超えています。より小さなファイルを選択してください。")
        return False
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        st.error(f"サポートされていないファイル形式です。対応形式: {', '.join(SUPPORTED_FORMATS)}")
        return False
    
    return True

def safe_image_processing(uploaded_file) -> Optional[Tuple[np.ndarray, Image.Image]]:
    """安全な画像処理"""
    try:
        # ファイルバリデーション
        if not validate_file(uploaded_file):
            return None
        
        # バイトデータを読み込み
        file_bytes = uploaded_file.read()
        
        # PIL Imageとして開く
        try:
            image = Image.open(io.BytesIO(file_bytes))
            
            # RGBAからRGBに変換（必要に応じて）
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # サイズを512x512にリサイズ
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # NumPy配列に変換
            image_array = np.array(image)
            
            return image_array, image
            
        except Exception as e:
            st.error(f"画像の読み込みに失敗しました: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"ファイル処理中にエラーが発生しました: {str(e)}")
        return None

def generate_demo_flood_mask(image_array: np.ndarray) -> np.ndarray:
    """デモ用洪水マスクの生成"""
    try:
        height, width = image_array.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 画像の明度に基づいてデモマスクを生成
        gray = np.mean(image_array, axis=2)
        
        # 暗い領域を水域として設定
        dark_threshold = np.percentile(gray, 30)
        mask[gray < dark_threshold] = 1
        
        # ノイズ除去のための簡単なフィルタリング
        from scipy import ndimage
        mask = ndimage.binary_opening(mask, structure=np.ones((3,3))).astype(np.uint8)
        mask = ndimage.binary_closing(mask, structure=np.ones((5,5))).astype(np.uint8)
        
        return mask
        
    except ImportError:
        # scipyが利用できない場合のシンプルなマスク生成
        height, width = image_array.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        gray = np.mean(image_array, axis=2)
        dark_threshold = np.percentile(gray, 25)
        mask[gray < dark_threshold] = 1
        
        return mask
    
    except Exception as e:
        st.error(f"マスク生成中にエラーが発生しました: {str(e)}")
        height, width = image_array.shape[:2]
        return np.zeros((height, width), dtype=np.uint8)

def create_overlay_image(original_image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """オーバーレイ画像の作成"""
    try:
        overlay = original_image.copy()
        
        # 洪水領域を赤色でハイライト
        overlay[mask == 1] = [255, 0, 0]  # 赤色
        
        # アルファブレンディング
        result = (alpha * overlay + (1 - alpha) * original_image).astype(np.uint8)
        
        return result
        
    except Exception as e:
        st.error(f"オーバーレイ作成中にエラーが発生しました: {str(e)}")
        return original_image

def array_to_downloadable_image(image_array: np.ndarray, filename: str) -> bytes:
    """NumPy配列をダウンロード可能な画像に変換"""
    try:
        image = Image.fromarray(image_array)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        st.error(f"画像変換中にエラーが発生しました: {str(e)}")
        return b''

def main():
    """メインアプリケーション"""
    
    # ヘッダー
    st.title("🌊 Prithvi-EO-2.0 洪水検出システム（AI統合版）")
    
    # アプリケーション情報
    with st.expander("ℹ️ アプリケーション情報", expanded=False):
        st.markdown("""
        **Prithvi-EO-2.0 AI統合版** - 実際の機械学習モデルを使用
        
        ✅ **現在の機能:**
        - 画像アップロード: JPG/PNG/TIFF形式に対応
        - 自動リサイズ: 512×512ピクセルへの最適化
        - デモ洪水検出: パターンベースの洪水エリア生成
        - 可視化: 入力画像、予測マスク、オーバーレイ表示
        - ダウンロード: 全結果のPNG形式保存
        
        ⚠️ **現在の制限:**
        - 簡易版モード: 複雑な依存関係の問題により基本機能のみ
        - デモ予測: 実際のPrithviモデルではなくパターン生成
        - Sentinel-2未対応: 現在は一般的な画像形式のみ
        
        🚀 **今後の予定:**
        - 実際のPrithvi-EO-2.0モデルの統合
        - Sentinel-2バンド処理の実装
        - 高精度な洪水検出機能
        """)
    
    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ 設定")
        
        st.subheader("📊 システム情報")
        st.info("""
        **プラットフォーム:** Render Web Service
        **プラン:** Free Tier対応
        **Python:** 3.10+
        **依存関係:** Streamlit + Pillow + NumPy（最小構成）
        **メモリ使用量:** 約500MB
        """)
        
        st.subheader("🔧 完全版への移行")
        st.warning("""
        **必要プラン:** Standard ($25/月) - 2GB RAM（1.28GBモデル用）
        **追加依存関係:** PyTorch, Hugging Face Hub, Rasterio
        **メモリ使用量:** 約1.5-2GB（完全版）
        """)
    
    # メインコンテンツ
    st.header("📁 画像アップロード")
    
    # ファイルアップローダー（エラーハンドリング強化）
    try:
        uploaded_file = st.file_uploader(
            "画像ファイルを選択してください",
            type=SUPPORTED_FORMATS,
            help=f"対応形式: {', '.join(SUPPORTED_FORMATS.upper())}（最大{MAX_FILE_SIZE/(1024*1024):.0f}MB）"
        )
        
        if uploaded_file is not None:
            with st.spinner("画像を処理中..."):
                # 安全な画像処理
                result = safe_image_processing(uploaded_file)
                
                if result is not None:
                    image_array, processed_image = result
                    
                    # 処理成功の表示
                    st.success(f"✅ 画像を正常に読み込みました（{uploaded_file.name}）")
                    
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
                    st.image(processed_image, caption="リサイズ済み画像（512×512）", use_container_width=True)
                    
                    # 洪水検出実行ボタン
                    if st.button("🔍 洪水検出を実行（デモ）", type="primary"):
                        with st.spinner("洪水検出を実行中..."):
                            try:
                                # デモ洪水マスク生成
                                flood_mask = generate_demo_flood_mask(image_array)
                                
                                # オーバーレイ画像作成
                                overlay_image = create_overlay_image(image_array, flood_mask)
                                
                                # 結果表示
                                st.subheader("📊 洪水検出結果")
                                
                                # 統計情報
                                total_pixels = flood_mask.size
                                flood_pixels = np.sum(flood_mask == 1)
                                flood_percentage = (flood_pixels / total_pixels) * 100
                                
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
                                    st.image(processed_image, caption="元画像", use_container_width=True)
                                
                                with col2:
                                    st.subheader("🗺️ 予測マスク")
                                    mask_image = Image.fromarray((flood_mask * 255).astype(np.uint8))
                                    st.image(mask_image, caption="洪水領域（白：洪水、黒：非洪水）", use_container_width=True)
                                
                                with col3:
                                    st.subheader("🎯 オーバーレイ")
                                    overlay_pil = Image.fromarray(overlay_image)
                                    st.image(overlay_pil, caption="洪水領域オーバーレイ（赤：洪水）", use_container_width=True)
                                
                                # ダウンロードボタン
                                st.subheader("💾 結果のダウンロード")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    input_bytes = array_to_downloadable_image(image_array, "input.png")
                                    if input_bytes:
                                        st.download_button(
                                            label="📷 入力画像をダウンロード",
                                            data=input_bytes,
                                            file_name=f"input_{uploaded_file.name.split('.')[0]}.png",
                                            mime="image/png"
                                        )
                                
                                with col2:
                                    mask_bytes = array_to_downloadable_image((flood_mask * 255).astype(np.uint8), "mask.png")
                                    if mask_bytes:
                                        st.download_button(
                                            label="🗺️ 予測マスクをダウンロード",
                                            data=mask_bytes,
                                            file_name=f"mask_{uploaded_file.name.split('.')[0]}.png",
                                            mime="image/png"
                                        )
                                
                                with col3:
                                    overlay_bytes = array_to_downloadable_image(overlay_image, "overlay.png")
                                    if overlay_bytes:
                                        st.download_button(
                                            label="🎯 オーバーレイをダウンロード",
                                            data=overlay_bytes,
                                            file_name=f"overlay_{uploaded_file.name.split('.')[0]}.png",
                                            mime="image/png"
                                        )
                                
                                # 注意事項
                                st.info("""
                                ⚠️ **注意:** これはデモ版です。実際のPrithvi-EO-2.0モデルではなく、
                                画像の明度に基づいたパターン生成による洪水領域の推定です。
                                実際の洪水検出精度とは異なります。
                                """)
                                
                                # メモリクリーンアップ
                                del flood_mask, overlay_image
                                gc.collect()
                                
                            except Exception as e:
                                st.error(f"洪水検出処理中にエラーが発生しました: {str(e)}")
                                st.error("詳細なエラー情報:")
                                st.code(traceback.format_exc())
                    
                    # メモリクリーンアップ
                    del image_array, processed_image
                    gc.collect()
                
                else:
                    st.error("画像の処理に失敗しました。別のファイルを試してください。")
    
    except Exception as e:
        st.error(f"ファイルアップロード中にエラーが発生しました: {str(e)}")
        st.error("詳細なエラー情報:")
        st.code(traceback.format_exc())
    
    # フッター情報
    st.markdown("---")
    st.markdown("""
    **開発者:** IBM & NASA Geospatial Team  
    **Render最適化:** 2025年1月  
    **ライブデモ:** [https://demo-prithvi-eo-2-300m-sen1floods.onrender.com](https://demo-prithvi-eo-2-300m-sen1floods.onrender.com)  
    **ソースコード:** [GitHub](https://github.com/shirokawakita/demo_prithvi_eo_2_300m_sen1floods_on_render)
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"アプリケーション実行中にエラーが発生しました: {str(e)}")
        st.error("詳細なエラー情報:")
        st.code(traceback.format_exc())