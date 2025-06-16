# 🌊 Prithvi-EO-2.0 洪水検出システム (Render版)

IBMとNASAが開発した第2世代地球観測基盤モデル「Prithvi-EO-2.0」を使用したSentinel-2画像からの洪水検出Webアプリケーションです。

**このバージョンは [Render](https://render.com) クラウドプラットフォームでの実行に最適化されています。**

## 🚀 デモ

**ライブデモ**: [あなたのアプリURL](https://your-app-name.onrender.com) *(デプロイ後に更新)*

## ✨ 主な機能

- 🖼️ **Sentinel-2画像アップロード**: TIFFファイルのドラッグ&ドロップ対応
- 🔄 **自動画像前処理**: サイズ調整とデータ型変換（512×512ピクセル）
- 🧠 **AI洪水検出**: Prithvi-EO-2.0-300M モデルによる高精度予測
- 📊 **3種類の結果表示**: 入力画像、予測結果、オーバーレイ表示
- 💾 **結果ダウンロード**: PNG形式での結果保存
- 🌍 **サンプルデータ**: India、Spain、USAのサンプル画像対応

## 🏗️ Renderデプロイ仕様

### システム要件
- **プラットフォーム**: [Render Web Service](https://render.com)
- **最小プラン**: **Standard ($25/月)** - 2GB RAM必須
- **Python**: 3.10+
- **メモリ使用量**: 約1.5-2GB（1.28GBモデル + 処理用）
- **ディスク容量**: 約3GB（モデル + 依存関係）

### 重要な制限事項
- ⚠️ **無料プランでは動作しません** - 1.28GBのモデルファイルのため
- ⏱️ **初回起動時間**: 20-30分（モデルダウンロード）
- 🔄 **コールドスタート**: 15分間非アクティブ後の再起動で2-3分

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

## 💰 費用情報

| プラン | 月額 | RAM | 対応状況 |
|--------|------|-----|----------|
| Free | $0 | 512MB | ❌ 非対応 |
| Starter | $7 | 512MB | ❌ 非対応 |
| **Standard** | **$25** | **2GB** | ✅ **推奨** |
| Pro | $85 | 4GB | ✅ 対応 |

- **初回**: 14日間無料トライアル利用可能
- **最小推奨**: Standardプラン（1.28GBモデル + 処理用メモリ）

## 🚀 使用方法

### 1. アプリケーションアクセス
デプロイ完了後、Renderが提供するURLでアクセス:
```
https://your-app-name.onrender.com
```

### 2. 画像アップロード
- Sentinel-2 TIFFファイルをドラッグ&ドロップ
- または「ファイルを選択」ボタンでアップロード

### 3. 洪水検出実行
- 「🔍 洪水検出を実行」ボタンをクリック
- 処理時間: 約30-60秒

### 4. 結果確認
- **入力画像**: RGB合成表示
- **予測マスク**: 白=洪水、黒=非洪水
- **オーバーレイ**: 洪水領域を赤色で表示

### 5. 結果ダウンロード
各結果画像をPNG形式でダウンロード可能

## 📊 性能指標

テストデータセットでの性能：

| クラス | IoU | Accuracy |
|--------|-----|----------|
| 非水域 | 96.90% | 98.11% |
| 水域/洪水 | 80.46% | 90.54% |
| **平均** | **88.68%** | **94.37%** |

## 🐛 トラブルシューティング

### よくある問題

#### 1. メモリ不足エラー
```
Error: Out of memory
```
**解決策**: Standardプラン以上に変更

#### 2. モデルダウンロード失敗
```
Error: Failed to download model
```
**解決策**: 
- インターネット接続確認
- Hugging Face Hubのステータス確認
- しばらく待ってから再試行

#### 3. 画像アップロードエラー
```
Error: Invalid image format
```
**解決策**:
- TIFFファイル形式確認
- ファイルサイズ100MB以下確認
- バンド数確認（6バンド以上）

#### 4. 処理時間が長い
**対策**:
- CPU処理のため時間がかかります（正常）
- 画像サイズを小さくすると高速化
- 初回は特に時間がかかります

### デバッグ方法
- Renderのログタブで詳細エラー確認
- Streamlitアプリの画面でリアルタイムステータス表示

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