import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import os
import tempfile

# Streamlit設定（最初に配置必須）
st.set_page_config(
    page_title="Prithvi-EO-2.0 洪水検出",
    page_icon="🌊",
    layout="wide"
)

# 条件付きインポート（st.set_page_config後に配置）
try:
    import rasterio
    from skimage.transform import resize
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

def create_download_link(image, filename):
    """画像のダウンロードリンクを作成"""
    try:
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 {filename}</a>'
        return href
    except Exception as e:
        return f"ダウンロードエラー: {e}"

def create_demo_prediction(image_shape):
    """デモ用の洪水予測を作成"""
    height, width = image_shape
    prediction = np.zeros((height, width), dtype=np.uint8)
    
    # 中央部分と川のような形状を洪水として設定
    center_h, center_w = height // 2, width // 2
    
    # 中央の円形エリア
    y, x = np.ogrid[:height, :width]
    mask_circle = (x - center_w)**2 + (y - center_h)**2 <= (min(height, width) // 6)**2
    prediction[mask_circle] = 1
    
    # 川のような線形エリア
    river_mask = np.abs(y - center_h - (x - center_w) * 0.3) < 20
    prediction[river_mask] = 1
    
    # 小さな池
    mask_pond = (x - center_w//2)**2 + (y - center_h//2)**2 <= 400
    prediction[mask_pond] = 1
    
    return prediction

def process_geotiff_with_rasterio(uploaded_file):
    """rasterioを使用してGeoTIFFを処理"""
    if not RASTERIO_AVAILABLE:
        return None
    
    try:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        with rasterio.open(tmp_path) as src:
            # 画像情報を表示
            st.write(f"📊 バンド数: {src.count}")
            st.write(f"📊 画像サイズ: {src.width} x {src.height}")
            st.write(f"📊 データ型: {src.dtypes[0]}")
            st.write(f"📊 座標系: {src.crs}")
            
            # 全バンドを読み込み
            image_data = src.read()  # Shape: (bands, height, width)
            
            st.write(f"📊 読み込み完了: {image_data.shape}")
            
            # Sentinel-2の場合、最適なバンドを選択
            if image_data.shape[0] >= 6:
                # Prithviで使用される6バンド: Blue, Green, Red, NIR_NARROW, SWIR1, SWIR2
                # 一般的なSentinel-2バンド順序を想定
                if image_data.shape[0] >= 12:  # 13バンドSentinel-2
                    # バンド選択: B2(Blue), B3(Green), B4(Red), B8A(NIR), B11(SWIR1), B12(SWIR2)
                    band_indices = [1, 2, 3, 7, 10, 11]  # 0-indexed
                    selected_bands = image_data[band_indices]
                    st.info("🛰️ Sentinel-2 13バンドから6バンドを選択")
                else:
                    # 最初の6バンドを使用
                    selected_bands = image_data[:6]
                    st.info("🛰️ 最初の6バンドを使用")
            else:
                # 利用可能なバンドをすべて使用し、足りない場合は複製
                selected_bands = image_data
                while selected_bands.shape[0] < 6:
                    selected_bands = np.concatenate([selected_bands, image_data[:1]], axis=0)
                selected_bands = selected_bands[:6]
                st.warning(f"⚠️ バンド数不足のため調整: {image_data.shape[0]} → 6")
            
            # データ型を確認・変換
            st.write(f"📊 選択バンド形状: {selected_bands.shape}")
            st.write(f"📊 値域: {selected_bands.min()} - {selected_bands.max()}")
            
            # 512x512にリサイズ
            if selected_bands.shape[1:] != (512, 512):
                st.info(f"📐 リサイズ中: {selected_bands.shape[1]}x{selected_bands.shape[2]} → 512x512")
                resized_bands = []
                for i in range(selected_bands.shape[0]):
                    resized_band = resize(
                        selected_bands[i], 
                        (512, 512), 
                        preserve_range=True,
                        anti_aliasing=True
                    )
                    resized_bands.append(resized_band)
                selected_bands = np.stack(resized_bands, axis=0)
            
            # RGB画像を作成（バンド3, 2, 1 = Red, Green, Blue）
            if selected_bands.shape[0] >= 3:
                rgb_bands = selected_bands[[2, 1, 0]]  # Red, Green, Blue
                
                # 正規化（0-255）
                rgb_normalized = []
                for band in rgb_bands:
                    band_min, band_max = band.min(), band.max()
                    if band_max > band_min:
                        normalized = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                    else:
                        normalized = np.zeros_like(band, dtype=np.uint8)
                    rgb_normalized.append(normalized)
                
                rgb_image = np.stack(rgb_normalized, axis=-1)  # (H, W, 3)
            else:
                # グレースケールからRGBを作成
                gray = selected_bands[0]
                gray_norm = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
                rgb_image = np.stack([gray_norm, gray_norm, gray_norm], axis=-1)
        
        # 一時ファイル削除
        os.unlink(tmp_path)
        
        return rgb_image, selected_bands
        
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        st.error(f"rasterio処理エラー: {e}")
        return None
def read_geotiff_with_fallback(uploaded_file):
    """GeoTIFFファイルを複数の方法で読み込み"""
    
    # 方法1: rasterioを使用（GeoTIFF専用）
    if RASTERIO_AVAILABLE:
        st.info("🛰️ rasterioでGeoTIFF処理を試行中...")
        result = process_geotiff_with_rasterio(uploaded_file)
        if result is not None:
            return result[0], "rasterio"  # RGB画像のみ返す
    
    file_bytes = uploaded_file.getbuffer()
    
    # 方法2: PILで直接読み込み
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        st.success("✅ PILで読み込み成功")
        return image, "PIL"
    except Exception as e:
        st.warning(f"⚠️ PIL読み込み失敗: {e}")
    
    # 方法3: 一時ファイルとして保存してPILで読み込み
    try:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        
        image = Image.open(temp_path)
        os.remove(temp_path)  # 一時ファイル削除
        st.success("✅ 一時ファイル経由で読み込み成功")
        return image, "temp_file"
    except Exception as e:
        st.warning(f"⚠️ 一時ファイル読み込み失敗: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
    
    # 方法4: バイナリデータから基本的な画像情報を抽出
    try:
        # TIFFヘッダーの基本チェック
        if file_bytes[:4] in [b'II*\x00', b'MM\x00*']:
            st.info("🔍 GeoTIFFファイルを検出")
            # 簡易的な512x512グレースケール画像を生成
            dummy_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
            dummy_rgb = np.stack([dummy_image, dummy_image, dummy_image], axis=-1)
            pil_image = Image.fromarray(dummy_rgb)
            st.warning("⚠️ GeoTIFFの直接読み込みはサポートされていません。ダミー画像を生成しました。")
            return pil_image, "dummy"
    except Exception as e:
        st.error(f"❌ バイナリ解析失敗: {e}")
    
    return None, None

def process_image_with_fallback(uploaded_file):
    """画像を複数の方法で処理"""
    try:
        # ファイル情報を表示
        file_info = f"""
        **ファイル情報:**
        - 名前: {uploaded_file.name}
        - サイズ: {uploaded_file.size / 1024 / 1024:.1f} MB
        - タイプ: {uploaded_file.type}
        """
        st.markdown(file_info)
        
        # GeoTIFF読み込み試行
        image, method = read_geotiff_with_fallback(uploaded_file)
        
        if image is None:
            st.error("❌ サポートされていない画像形式です。")
            st.markdown("""
            ### 📋 推奨ファイル形式
            - **GeoTIFF**: Sentinel-2データ推奨（rasterio使用）
            - **標準画像**: JPG, PNG
            - **限定的サポート**: 標準TIFF
            
            ### 💡 回避策
            1. Sentinel-2データをGeoTIFF形式で使用
            2. GISソフトウェアでRGB合成画像を作成
            3. JPEG/PNG形式に変換
            """)
            return None
        
        st.write(f"📊 読み込み方法: {method}")
        
        # PIL Imageの場合はnumpy配列に変換
        if isinstance(image, Image.Image):
            st.write(f"📊 元画像サイズ: {image.size}")
            st.write(f"📊 元画像モード: {image.mode}")
            
            # RGBに変換
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # アルファチャンネルを除去
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])
                    image = rgb_image
                elif image.mode in ['L', 'P']:
                    # グレースケールやパレットモードをRGBに変換
                    image = image.convert('RGB')
                else:
                    image = image.convert('RGB')
            
            # アスペクト比を保持してリサイズ
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # 512x512の正方形キャンバスに中央配置
            canvas = Image.new('RGB', (512, 512), (0, 0, 0))  # 黒背景
            
            # 中央に配置
            x = (512 - image.width) // 2
            y = (512 - image.height) // 2
            canvas.paste(image, (x, y))
            
            # numpy配列に変換
            image_array = np.array(canvas)
        else:
            # すでにnumpy配列の場合（rasterio処理済み）
            image_array = image
        
        st.write(f"📊 処理後サイズ: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        st.error(f"画像処理エラー: {e}")
        st.markdown("""
        ### 🔧 エラー対処法
        1. **ファイル形式確認**: GeoTIFF/JPG/PNG形式を推奨
        2. **ファイルサイズ**: 100MB以下に縮小
        3. **ファイル破損**: 別の画像ファイルで試行
        4. **ブラウザリロード**: ページを更新して再試行
        """)
        return None

def create_overlay(rgb_image, prediction_mask):
    """オーバーレイ画像を作成"""
    overlay = rgb_image.copy()
    
    # 洪水領域を赤色で表示
    flood_mask = prediction_mask == 1
    overlay[flood_mask] = [255, 0, 0]  # 赤色
    
    # 透明度を適用
    alpha = 0.6
    result = (rgb_image * (1 - alpha) + overlay * alpha).astype(np.uint8)
    
    return result

def main():
    title = "🌊 Prithvi-EO-2.0 洪水検出システム"
    if RASTERIO_AVAILABLE:
        title += "（GeoTIFF対応版）"
    else:
        title += "（簡易版）"
    
    st.title(title)
    
    if RASTERIO_AVAILABLE:
        st.markdown("""
        **このバージョンはGeoTIFFとSentinel-2データに対応しています**
        
        rasterioライブラリを使用して、Sentinel-2 GeoTIFFファイルの完全処理が可能です。
        """)
    else:
        st.markdown("""
        **このバージョンは基本的な画像処理とデモ予測を行います**
        
        現在、rasterioライブラリが利用できないため、基本版で動作しています。
        """)
    
    # サイドバー
    # サイドバー（main関数内で実行）
    st.sidebar.header("📋 アプリ情報")
    
    # rasterio状態表示
    if RASTERIO_AVAILABLE:
        st.sidebar.success("✅ rasterio利用可能")
        st.sidebar.success("""
        **拡張版の機能:**
        - GeoTIFF完全サポート（rasterio使用）
        - Sentinel-2多バンド処理
        - 自動バンド選択・リサイズ
        - デモ用洪水予測
        - オーバーレイ表示
        - 結果ダウンロード
        """)
    else:
        st.sidebar.warning("⚠️ rasterio未利用")
        st.sidebar.info("""
        **基本版の機能:**
        - 基本的な画像アップロード
        - 画像サイズ調整（512x512）
        - デモ用洪水予測
        - オーバーレイ表示
        - 結果ダウンロード
        """)
    
    st.sidebar.warning("""
    **制限事項:**
    - 実際のPrithviモデルは未使用
    - デモ用の予測結果を表示
    """)
    
    if RASTERIO_AVAILABLE:
        st.sidebar.markdown("### 🛰️ Sentinel-2対応")
        st.sidebar.info("""
        **対応バンド:**
        - Blue (B2)
        - Green (B3) 
        - Red (B4)
        - NIR Narrow (B8A)
        - SWIR1 (B11)
        - SWIR2 (B12)
        
        **自動処理:**
        - 13バンド→6バンド選択
        - 512x512リサイズ
        - RGB合成生成
        """)
    
    # ファイルアップロード
    st.header("📁 画像アップロード")
    
    uploaded_file = st.file_uploader(
        "画像ファイルを選択してください",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        help="JPG、PNG、TIFFファイルに対応（最大100MB）\n⚠️ GeoTIFFは部分的サポート"
    )
    
    # デモ機能の説明
    st.markdown("### 🎯 機能概要")
    if RASTERIO_AVAILABLE:
        st.success("""
        **GeoTIFF対応版の機能:**
        - **Sentinel-2 GeoTIFF**: 完全サポート（13→6バンド自動選択）
        - **自動前処理**: サイズ調整、バンド選択、RGB合成
        - **デモ洪水検出**: パターンベース予測
        - **結果可視化**: 入力画像、予測マスク、オーバーレイ
        - **ダウンロード**: PNG形式での結果保存
        """)
    else:
        st.info("""
        **基本版の機能:**
        - **基本画像処理**: JPG/PNG/TIFF対応
        - **サイズ調整**: 512x512への自動リサイズ
        - **デモ洪水検出**: パターンベース予測
        - **結果可視化**: 入力画像、予測マスク、オーバーレイ
        - **ダウンロード**: PNG形式での結果保存
        """)
    
    # GeoTIFFに関する情報
    if RASTERIO_AVAILABLE:
        st.markdown("### 🛰️ Sentinel-2 GeoTIFF処理")
        st.info("""
        **対応形式:**
        - **Sentinel-2 L1C**: 13バンドGeoTIFF
        - **カスタムGeoTIFF**: 6バンド以上のマルチバンド画像
        - **自動処理**: バンド選択、リサイズ、正規化
        
        **バンドマッピング（Sentinel-2）:**
        - Blue (B2) → バンド1
        - Green (B3) → バンド2  
        - Red (B4) → バンド3
        - NIR Narrow (B8A) → バンド4
        - SWIR1 (B11) → バンド5
        - SWIR2 (B12) → バンド6
        """)
    else:
        st.markdown("### ⚠️ GeoTIFFファイルについて")
        st.warning("""
        **現在の制限:**
        - **rasterio未利用**: GeoTIFF完全サポートなし
        - **代替処理**: PIL + フォールバック機能
        - **推奨形式**: JPG/PNG形式への変換
        
        **Sentinel-2データを使用する場合:**
        1. GISソフトウェアでRGB合成
        2. JPEG/PNG形式で保存
        3. または rasterio対応版を使用
        """)
    
    if uploaded_file is not None:
        try:
            # ファイル情報表示
            st.success(f"✅ ファイル受信: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
            
            # 画像処理
            with st.spinner("画像を処理中..."):
                processed_image = process_image_with_fallback(uploaded_file)
            
            if processed_image is not None:
                st.success("✅ 画像処理完了!")
                
                # 入力画像プレビュー
                st.subheader("🖼️ 入力画像")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(processed_image, caption="処理済み画像 (512x512)", use_container_width=True)
                
                with col2:
                    st.markdown("**画像情報**")
                    st.write(f"- サイズ: {processed_image.shape[1]}×{processed_image.shape[0]}")
                    st.write(f"- チャンネル数: {processed_image.shape[2]}")
                    st.write(f"- データ型: {processed_image.dtype}")
                    st.write(f"- 値域: {processed_image.min()} - {processed_image.max()}")
                    
                    # Streamlit表示用の詳細情報
                    st.markdown("**表示設定**")
                    st.write("- 表示方法: use_container_width=True")
                    st.write("- アスペクト比: 維持")
                
                # 予測実行
                st.header("🧠 デモ洪水検出")
                
                if st.button("🔍 洪水検出を実行（デモ）", type="primary", use_container_width=True):
                    with st.spinner("デモ予測を実行中..."):
                        # デモ予測生成
                        prediction_mask = create_demo_prediction(processed_image.shape[:2])
                        
                        # オーバーレイ画像作成
                        overlay_image = create_overlay(processed_image, prediction_mask)
                    
                    st.success("✅ デモ予測完了!")
                    
                    # 結果表示
                    st.header("📊 検出結果（デモ）")
                    
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
                        st.subheader("入力画像")
                        st.image(processed_image, use_container_width=True)
                    
                    with col2:
                        st.subheader("洪水予測マスク（デモ）")
                        mask_vis = (prediction_mask * 255).astype(np.uint8)
                        # グレースケールをカラーに変換
                        mask_color = np.stack([mask_vis, mask_vis, mask_vis], axis=-1)
                        st.image(mask_color, use_container_width=True)
                    
                    with col3:
                        st.subheader("オーバーレイ結果")
                        st.image(overlay_image, use_container_width=True)
                    
                    # ダウンロードセクション
                    st.subheader("💾 結果ダウンロード")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(create_download_link(processed_image, "input_image.png"), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(create_download_link(mask_color, "prediction_mask.png"), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(create_download_link(overlay_image, "flood_overlay.png"), unsafe_allow_html=True)
                    
                    # 解釈ガイド
                    st.subheader("📖 結果の解釈（デモ版）")
                    st.markdown("""
                    - **白い領域**: デモ洪水予測エリア
                    - **黒い領域**: 非洪水域
                    - **赤い領域**: オーバーレイの洪水表示
                    
                    **重要**: これはデモンストレーション用の予測結果です。
                    実際のPrithvi-EO-2.0モデルによる洪水検出ではありません。
                    """)
                    
                    # 技術情報
                    st.subheader("🔧 技術情報")
                    st.markdown("""
                    **現在の制限:**
                    - Sentinel-2特有の処理は未実装
                    - 実際のPrithviモデル未使用
                    - 基本的な画像処理のみ
                    
                    **将来の改善:**
                    - 実際のPrithviモデル統合
                    - Sentinel-2バンド処理
                    - より高精度な洪水検出
                    """)
            
        except Exception as e:
            st.error(f"❌ エラー: {e}")
            st.markdown("### 🔧 トラブルシューティング")
            st.markdown("""
            - サポートされている画像形式か確認してください
            - ファイルサイズが100MB以下か確認してください
            - 画像ファイルが破損していないか確認してください
            """)
    
    else:
        # 使い方ガイド
        st.markdown("### 📋 使い方")
        st.markdown("""
        1. **画像をアップロード**: JPG、PNG、TIFFファイルを選択
        2. **処理確認**: 画像が512x512にリサイズされます
        3. **デモ予測実行**: ボタンクリックで洪水検出デモを実行
        4. **結果確認**: 3つの画像（入力、マスク、オーバーレイ）を確認
        5. **ダウンロード**: 必要に応じて結果をダウンロード
        """)
        
        st.markdown("### 🎯 デモの目的")
        st.info("""
        この簡易版は以下を目的としています：
        - Renderでの基本的な動作確認
        - 画像処理パイプラインのテスト
        - ユーザーインターフェースの検証
        - 将来的なPrithviモデル統合の準備
        """)
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌊 Prithvi-EO-2.0 洪水検出システム（簡易版）| Running on Render</p>
        <p>元のプロジェクト: <a href='https://github.com/shirokawakita/demo_prithvi_eo_2_300m_sen1floods'>GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()