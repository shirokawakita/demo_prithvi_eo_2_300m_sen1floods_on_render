# 🌊 Prithvi-EO-2.0 洪水検出システム (Render版)

IBMとNASAが開発した第2世代地球観測基盤モデル「Prithvi-EO-2.0」を使用したSentinel-2画像からの洪水検出Webアプリケーションです。

**このバージョンは [Render](https://render.com) クラウドプラットフォームでの実行に最適化されています。**

## 🚀 デモ

**ライブデモ**: [https://demo-prithvi-eo-2-300m-sen1floods.onrender.com](https://demo-prithvi-eo-2-300m-sen1floods.onrender.com)

## 🎯 現在の実装状況

### ✅ **動作中の機能**
- **画像アップロード**: JPG/PNG/TIFF形式に対応
- **自動リサイズ**: 512×512ピクセルへの最適化
- **デモ洪水検出**: パターンベースの洪水エリア生成
- **可視化**: 入力画像、予測マスク、オーバーレイ表示
- **ダウンロード**: 全結果のPNG形式保存

### ⚠️ **現在の制限事項**
- **簡易版モード**: 複雑な依存関係の問題により基本機能のみ
- **デモ予測**: 実際のPrithviモデルではなくパターン生成
- **Sentinel-2未対応**: 現在は一般的な画像形式のみ

### 🔄 **今後の改善予定**
- 実際のPrithvi-EO-2.0モデルの統合
- Sentinel-2バンド処理の実装
- 高精度な洪水検出機能

## 🏗️ Renderデプロイ仕様

### 現在の構成（簡易版）
- **プラットフォーム**: [Render Web Service](https://render.com)
- **プラン**: Free Tier対応
- **Python**: 3.10+
- **依存関係**: Streamlit + Pillow + NumPy（最小構成）
- **メモリ使用量**: 約500MB

### 完全版への移行予定
- **必要プラン**: **Standard ($25/月)** - 2GB RAM（1.28GBモデル用）
- **追加依存関係**: PyTorch, Hugging Face Hub, Rasterio
- **メモリ使用量**: 約1.5-2GB（完全版）

## 📁 プロジェクト構成（Render最適化版）

```
demo_prithvi_eo_2_300m_sen1floods/
├── app.py                    # Streamlitメインアプリ（Render設定込み）
├── model_loader.py          # Hugging Face Hubからの自動モデル読み込み
├── image_processor.py       # Sentinel-2画像前処理
├── requirements.txt         # Render用最適化依存関係
├── render.yaml             # Renderデプロイ設定
├── .streamlit/             # Streamlit設定（オプション）
│   └── config.toml
├── data/                   # サンプルデータ（小サイズ版）
└── README.md              # このファイル
```

## 🛠️ Renderでのデプロイ方法

### Step 1: リポジトリ準備
```bash
git clone https://github.com/shirokawakita/demo_prithvi_eo_2_300m_sen1floods.git
cd demo_prithvi_eo_2_300m_sen1floods
```

### Step 2: Render設定
1. [render.com](https://render.com) でアカウント作成
2. GitHubアカウントと連携
3. **「New」** → **「Web Service」**
4. このリポジトリを選択

### Step 3: 重要な設定項目
```yaml
Name: prithvi-flood-detection
Environment: Python 3
Instance Type: Standard (重要: 2GB RAM必須)
Build Command: 自動検出 (render.yamlから)
Start Command: 自動検出 (render.yamlから)
```

### Step 4: デプロイ実行
- **ビルド時間**: 5-10分（依存関係インストール）
- **初回アクセス**: 20-30分（1.28GBモデルダウンロード）
- **以降のアクセス**: 30-60秒（画像処理時間）

## 🔧 技術仕様

### モデル情報
- **ベースモデル**: Prithvi-EO-2.0-300M
- **ファインチューニング**: Sen1Floods11データセット
- **アーキテクチャ**: Vision Transformer + UperNet Decoder
- **モデルサイズ**: 1.28GB
- **入力サイズ**: 512×512ピクセル × 6バンド
- **出力**: 2クラス セマンティックセグメンテーション

### 対応画像形式
- **ファイル形式**: GeoTIFF (.tif, .tiff)
- **バンド数**: 6バンド (Blue, Green, Red, Narrow NIR, SWIR1, SWIR2)
- **データ型**: uint16またはint16（自動変換）
- **最大アップロードサイズ**: 100MB

### Render特化設定

#### 自動モデル管理
```python
# Hugging Face Hubから自動ダウンロード
@st.cache_resource
def download_model():
    model_path = hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        filename="Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
        cache_dir="/tmp/prithvi_cache"
    )
    return model_path
```

#### メモリ最適化
- CPU専用PyTorch使用
- `@st.cache_resource`でモデル永続化
- ガベージコレクション実装

#### 環境変数設定
```yaml
envVars:
  - key: PYTHONUNBUFFERED
    value: "1"
  - key: HF_HUB_CACHE
    value: "/tmp/huggingface_cache"
  - key: STREAMLIT_SERVER_HEADLESS
    value: "true"
```

## 💰 費用情報（将来の完全版）

現在の簡易版は**無料プラン**で動作していますが、完全版では以下が必要になります：

| プラン | 月額 | RAM | 対応状況 |
|--------|------|-----|----------|
| **Free** | **$0** | **512MB** | **✅ 現在使用中（簡易版）** |
| Starter | $7 | 512MB | ❌ 完全版非対応 |
| **Standard** | **$25** | **2GB** | ✅ **完全版推奨** |
| Pro | $85 | 4GB | ✅ 完全版対応 |

### 現在の構成
- **プラン**: Free Tier
- **機能**: 基本的な画像処理とデモ予測
- **制限**: 実際のPrithviモデル未使用

### 完全版への移行時
- **必要プラン**: Standard以上（1.28GBモデル + 処理用メモリ）
- **追加機能**: 実際のPrithvi-EO-2.0モデル、Sentinel-2処理
- **初回起動**: 20-30分（モデルダウンロード）

## 🚀 使用方法

### 1. アプリケーションアクセス
**ライブデモ**: [https://demo-prithvi-eo-2-300m-sen1floods.onrender.com](https://demo-prithvi-eo-2-300m-sen1floods.onrender.com)

### 2. 画像アップロード
- JPG、PNG、TIFFファイルをドラッグ&ドロップ
- または「ファイルを選択」ボタンでアップロード
- 最大ファイルサイズ: 100MB

### 3. デモ洪水検出実行
- 「🔍 洪水検出を実行（デモ）」ボタンをクリック
- 処理時間: 約5-10秒（簡易版）

### 4. 結果確認
- **入力画像**: リサイズされた画像表示
- **予測マスク**: デモ洪水パターン
- **オーバーレイ**: 洪水領域を赤色で表示

### 5. 結果ダウンロード
各結果画像をPNG形式でダウンロード可能

## 📊 現在の機能と性能

### 簡易版の機能
- **画像処理**: 基本的なリサイズと形式変換
- **デモ予測**: パターンベースの洪水エリア生成
- **可視化**: 3種類の結果表示
- **ダウンロード**: PNG形式での保存

### 処理時間
- **画像アップロード**: 即座
- **画像処理**: 1-2秒
- **デモ予測**: 5-10秒
- **結果表示**: 即座

### 対応フォーマット
- **入力**: JPG, PNG, TIFF
- **出力**: PNG（ダウンロード用）
- **最大サイズ**: 100MB

## 🐛 現在の制限事項とトラブルシューティング

### 現在の制限事項

#### 1. 簡易版の制限
- **実際のPrithviモデル未使用**: 依存関係の問題により簡易版で動作
- **デモ予測のみ**: パターンベースの洪水エリア生成
- **Sentinel-2未対応**: 一般的な画像形式のみ対応

#### 2. 技術的制限
- **メモリ制約**: 複雑なモデル処理は未実装
- **処理速度**: 実際のAI予測より高速（デモのため）
- **精度**: デモ用パターンのため実際の洪水検出精度はなし

### よくある問題

#### 1. 画像表示の問題
```
問題: 画像の一部のみ表示される
```
**解決済み**: アスペクト比保持とuse_container_width使用

#### 2. ファイルアップロードエラー
```
問題: 大きなファイルのアップロード失敗
```
**解決策**: 
- ファイルサイズを100MB以下に縮小
- サポートされている形式（JPG/PNG/TIFF）を使用

#### 3. 処理エラー
```
問題: 画像処理中のエラー
```
**解決策**:
- 画像ファイルの破損確認
- ページをリロードして再試行

### 将来の改善計画
1. **実際のPrithviモデル統合**
2. **Sentinel-2バンド処理実装**
3. **高精度洪水検出機能**
4. **Standardプランでの完全版デプロイ**

## 🔒 セキュリティ

- HTTPSで暗号化通信
- 一時ファイルは処理後自動削除
- モデルファイルはキャッシュで管理
- 環境変数で機密情報管理

## 📝 ライセンス

このプロジェクトは元のPrithvi-EO-2.0モデルのライセンスに従います。

## 🙏 謝辞

- **IBM & NASA Geospatial Team**: Prithvi-EO-2.0モデル開発
- **Render**: クラウドプラットフォーム提供
- **Hugging Face**: モデルホスティング

## 📚 引用

このモデルを研究で使用した場合は、以下を引用してください：

```bibtex
@article{Prithvi-EO-V2-preprint,
  author = {Szwarcman, Daniela and Roy, Sujit and Fraccaro, Paolo and others},
  title = {{Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications}},
  journal = {arXiv preprint arXiv:2412.02732},
  year = {2024}
}
```

## 🆘 サポート

### 問題報告
- [GitHub Issues](https://github.com/shirokawakita/demo_prithvi_eo_2_300m_sen1floods/issues)
- バグ報告や機能要望をお寄せください

### 技術サポート
- **Renderサポート**: [render.com/support](https://render.com/support)
- **モデル関連**: [Hugging Face Model Page](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)

---

**開発者**: IBM & NASA Geospatial Team  
**Render最適化**: 2025年1月  
**最終更新**: 2025年1月

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)