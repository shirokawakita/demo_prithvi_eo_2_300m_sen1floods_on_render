import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import os

# Streamlit設定
st.set_page_config(
    page_title="Prithvi-EO-2.0 洪水検出",
    page_icon="🌊",
    layout="wide"
)

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

def process_simple_image(uploaded_file):
    """画像を簡単に処理"""
    try:
        # PILで画像を読み込み
        image = Image.open(uploaded_file)
        
        # 画像情報を表示
        st.write(f"📊 元画像サイズ: {image.size}")
        st.write(f"📊 元画像モード: {image.mode}")
        
        # RGBに変換
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # アスペクト比を保持してリサイズ
        # 512x512の正方形にフィット
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # 512x512の正方形キャンバスに中央配置
        canvas = Image.new('RGB', (512, 512), (0, 0, 0))  # 黒背景
        
        # 中央に配置
        x = (512 - image.width) // 2
        y = (512 - image.height) // 2
        canvas.paste(image, (x, y))
        
        # numpy配列に変換
        image_array = np.array(canvas)
        
        st.write(f"📊 処理後サイズ: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        st.error(f"画像処理エラー: {e}")
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
    st.title("🌊 Prithvi-EO-2.0 洪水検出システム（簡易版）")
    
    st.markdown("""
    **このバージョンは基本的な画像処理とデモ予測を行います**
    
    現在、Renderでの複雑な依存関係の問題により、簡易版で動作しています。
    """)
    
    # サイドバー
    st.sidebar.header("📋 アプリ情報")
    st.sidebar.info("""
    **簡易版の機能:**
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
    - 基本的な画像形式のみ対応
    """)
    
    # ファイルアップロード
    st.header("📁 画像アップロード")
    
    uploaded_file = st.file_uploader(
        "画像ファイルを選択してください",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        help="JPG、PNG、TIFFファイルに対応（最大100MB）"
    )
    
    # デモ機能の説明
    st.markdown("### 🎯 デモ機能")
    st.info("""
    この簡易版では以下のデモ機能を提供します：
    - **画像処理**: アップロードされた画像を512x512にリサイズ
    - **デモ予測**: 中央部分と川状のエリアを洪水として予測
    - **オーバーレイ**: 洪水予測結果を赤色で表示
    - **ダウンロード**: 各結果画像をPNG形式でダウンロード
    """)
    
    if uploaded_file is not None:
        try:
            # ファイル情報表示
            st.success(f"✅ ファイル受信: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
            
            # 画像処理
            with st.spinner("画像を処理中..."):
                processed_image = process_simple_image(uploaded_file)
            
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