# 🌊 Prithvi-EO-2.0 洪水検出システム (Render版)

IBMとNASAが開発した第2世代地球観測基盤モデル「Prithvi-EO-2.0」を使用したSentinel-2画像からの洪水検出Webアプリケーションです。

**このバージョンは [Render](https://render.com) Standard Planで実際のPrithvi-EO-2.0モデルを実行します。**

## 🚀 デモ

**ライブデモ**: [https://demo-prithvi-eo-2-300m-sen1floods.onrender.com](https://demo-prithvi-eo-2-300m-sen1floods.onrender.com)

## 🎯 現在の実装状況

### ✅ **完全実装済み機能**
- **実際のPrithvi-EO-2.0モデル**: 1.28GB完全版モデル使用
- **terratorch統合**: main.pyと同じSemanticSegmentationTask使用
- **Sentinel-2処理**: 6バンド（Blue, Green, Red, NIR, SWIR1, SWIR2）対応
- **高精度洪水検出**: Sen1Floods11データセット学習済みモデル
- **科学的精度**: 研究レベルの洪水検出性能
- **完全な可視化**: 入力画像、予測マスク、オーバーレイ表示
- **結果ダウンロード**: 全結果のPNG形式保存

### 🔧 **技術的特徴**
- **Standard Plan対応**: 2GB RAM環境で完全版動作
- **main.py互換**: 同じinference.pyとterratorch使用
- **自動フォールバック**: 依存関係エラー時の代替処理
- **メモリ最適化**: CPU専用PyTorch使用

## 🏗️ Renderデプロイ仕様

### 現在の構成（完全版）
- **プラットフォーム**: [Render Web Service](https://render.com)
- **プラン**: **Standard ($25/月)** - 2GB RAM
- **Python**: 3.10+
- **主要依存関係**: 
  - `terratorch` (Prithvi-EO-2.0統合)
  - `pytorch-lightning` (モデル実行)
  - `timm` (Vision Transformer)
  - `segmentation-models-pytorch` (セグメンテーション)
  - `rasterio` (GeoTIFF処理)
- **メモリ使用量**: 約1.5-2GB（完全版）

### 動作モード
1. **完全版モード**: terratorch + inference.py使用
2. **フォールバックモード**: 独自実装使用
3. **セットアップモード**: 基本機能のみ

## 📁 プロジェクト構成

```
demo_prithvi_eo_2_300m_sen1floods/
├── app.py                    # Streamlitメインアプリ（terratorch対応）
├── inference.py             # Prithvi-EO-2.0推論関数（main.pyと共通）
├── main.py                  # 参照実装（コマンドライン版）
├── image_processor.py       # Sentinel-2画像前処理
├── requirements.txt         # 完全版依存関係（terratorch含む）
├── render.yaml             # Render Standard Plan設定
├── .streamlit/             # Streamlit設定
│   └── config.toml
├── data/                   # サンプルデータ
│   └── sample_image.tif
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
Instance Type: Standard (必須: 2GB RAM)
Build Command: pip install -r requirements.txt
Start Command: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Step 4: 環境変数設定
```yaml
envVars:
  - key: PLAN_TYPE
    value: "standard"
  - key: MEMORY_GB
    value: "2"
  - key: PYTHONUNBUFFERED
    value: "1"
  - key: STREAMLIT_SERVER_HEADLESS
    value: "true"
```

### Step 5: デプロイ実行
- **ビルド時間**: 10-15分（terratorch等の依存関係インストール）
- **初回アクセス**: 5-10分（1.28GBモデルダウンロード）
- **以降のアクセス**: 30-60秒（AI推論処理時間）

## 🔧 技術仕様

### モデル情報
- **ベースモデル**: Prithvi-EO-2.0-300M
- **ファインチューニング**: Sen1Floods11データセット
- **アーキテクチャ**: Vision Transformer + UperNet Decoder
- **モデルサイズ**: 1.28GB
- **入力サイズ**: 512×512ピクセル × 6バンド
- **出力**: 2クラス セマンティックセグメンテーション（洪水/非洪水）

### 対応画像形式
- **推奨形式**: GeoTIFF (.tif, .tiff) - 多バンド対応
- **対応形式**: JPG, PNG（RGB画像、6バンドに拡張）
- **バンド構成**: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2
- **データ型**: uint8, uint16, int16（自動変換）
- **最大アップロードサイズ**: 100MB

### Sentinel-2バンド処理
```python
# Sentinel-2バンド選択（main.pyと同じ）
SENTINEL2_BANDS = [1, 2, 3, 8, 11, 12]  # Blue, Green, Red, NIR, SWIR1, SWIR2
```

### 前処理パイプライン（main.py互換）
1. **バンド抽出**: 6バンド選択
2. **リサイズ**: 512×512ピクセル
3. **データ型変換**: int16（1000-3000範囲）
4. **正規化**: バンド別平均・標準偏差
5. **テンソル化**: PyTorchテンソル

## 💰 費用情報

| プラン | 月額 | RAM | 対応状況 |
|--------|------|-----|----------|
| Free | $0 | 512MB | ❌ モデルサイズ不足 |
| Starter | $7 | 512MB | ❌ モデルサイズ不足 |
| **Standard** | **$25** | **2GB** | ✅ **現在使用中** |
| Pro | $85 | 4GB | ✅ より高速処理可能 |

### 現在の構成（Standard Plan）
- **プラン**: Standard ($25/月)
- **RAM**: 2GB（1.28GBモデル + 処理用メモリ）
- **機能**: 完全版Prithvi-EO-2.0モデル
- **性能**: 研究レベルの洪水検出精度

## 🚀 使用方法

### 1. アプリケーションアクセス
**ライブデモ**: [https://demo-prithvi-eo-2-300m-sen1floods.onrender.com](https://demo-prithvi-eo-2-300m-sen1floods.onrender.com)

### 2. 画像アップロード
- **推奨**: Sentinel-2 GeoTIFFファイル（多バンド）
- **対応**: JPG、PNG、TIFFファイル
- ドラッグ&ドロップまたは「ファイルを選択」
- 最大ファイルサイズ: 100MB

### 3. AI洪水検出実行
- 「🔍 洪水検出を実行」ボタンをクリック
- 処理時間: 30-60秒（実際のAI推論）
- 進捗表示: リアルタイム処理状況

### 4. 結果確認
- **入力画像**: Sentinel-2 RGB合成画像
- **予測マスク**: 洪水領域（白）/非洪水領域（黒）
- **オーバーレイ**: 洪水領域を赤色で重畳表示
- **統計情報**: 洪水面積割合

### 5. 結果ダウンロード
- 各結果画像をPNG形式でダウンロード
- 入力画像、予測マスク、オーバーレイの3種類

## 📊 性能と精度

### 処理性能
- **モデル読み込み**: 初回のみ5-10分
- **画像前処理**: 1-2秒
- **AI推論**: 30-60秒
- **結果表示**: 即座

### 検出精度
- **データセット**: Sen1Floods11で学習済み
- **精度**: 研究レベルの洪水検出性能
- **対象**: Sentinel-2光学画像からの洪水検出
- **出力**: ピクセル単位の洪水確率

### 対応シナリオ
- **洪水災害**: 河川氾濫、都市洪水
- **地域**: 全世界対応
- **季節**: 年間を通じて利用可能
- **解像度**: 10m/pixel（Sentinel-2準拠）

## 🔧 開発者向け情報

### ローカル開発環境
```bash
# 環境構築
git clone https://github.com/shirokawakita/demo_prithvi_eo_2_300m_sen1floods.git
cd demo_prithvi_eo_2_300m_sen1floods

# 依存関係インストール
pip install -r requirements.txt

# アプリケーション起動
streamlit run app.py
```

### 主要な依存関係
```txt
terratorch                    # Prithvi-EO-2.0統合
pytorch-lightning>=2.0.0     # モデル実行フレームワーク
timm>=0.9.0                  # Vision Transformer
segmentation-models-pytorch>=0.3.0  # セグメンテーション
rasterio>=1.3.0              # GeoTIFF処理
huggingface_hub>=0.17.0      # モデルダウンロード
```

### API構造
```python
# main.pyと同じ関数使用
from inference import (
    SemanticSegmentationTask,
    Sen1Floods11NonGeoDataModule,
    load_example,
    run_model,
    save_prediction
)
```

## 🐛 トラブルシューティング

### よくある問題

#### 1. terratorch import エラー
```
ModuleNotFoundError: No module named 'terratorch'
```
**解決策**: 
- Standard Plan使用確認
- requirements.txtに`terratorch`追加
- フォールバックモードで動作継続

#### 2. メモリ不足エラー
```
CUDA out of memory / System out of memory
```
**解決策**:
- Standard Plan以上使用
- CPU専用モード確認
- 画像サイズ縮小

#### 3. モデルダウンロードエラー
```
Cannot download model from Hugging Face
```
**解決策**:
- インターネット接続確認
- Hugging Face Hub接続確認
- キャッシュディレクトリ確認

### デバッグ情報
- **システム情報**: サイドバーで確認可能
- **モデル状態**: リアルタイム表示
- **処理ログ**: 詳細な進捗表示
- **エラーハンドリング**: 自動フォールバック

## 🔒 セキュリティ

- **HTTPS通信**: 全通信暗号化
- **一時ファイル**: 処理後自動削除
- **モデルキャッシュ**: 安全な一時ディレクトリ使用
- **環境変数**: 機密情報の安全な管理
- **アップロード制限**: 100MB制限でDoS攻撃防止

## 📝 ライセンス

このプロジェクトは元のPrithvi-EO-2.0モデルのライセンス（Apache 2.0）に従います。

## 🙏 謝辞

- **IBM & NASA Geospatial Team**: Prithvi-EO-2.0モデル開発
- **Render**: Standard Planクラウドプラットフォーム提供
- **Hugging Face**: モデルホスティングサービス
- **terratorch**: Prithvi統合フレームワーク

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
- **Prithviモデル**: [Hugging Face Model Page](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)
- **terratorch**: [terratorch Documentation](https://github.com/IBM/terratorch)

### 更新履歴
- **2025年1月**: terratorch統合、実際のPrithvi-EO-2.0モデル実装
- **2024年12月**: Render Standard Plan対応
- **2024年11月**: 初期リリース（簡易版）

---

**開発者**: IBM & NASA Geospatial Team  
**Render完全版対応**: 2025年1月  
**最終更新**: 2025年1月  
**プラン**: Standard Plan (2GB RAM) 対応

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

---

## 🌟 主な改善点

### ✅ **完全版実装**
- 実際のPrithvi-EO-2.0モデル（1.28GB）使用
- main.pyと同じterratorch + inference.py統合
- Standard Plan（2GB RAM）での完全動作

### ✅ **科学的精度**
- Sen1Floods11データセット学習済み
- 研究レベルの洪水検出性能
- ピクセル単位の高精度予測

### ✅ **完全なSentinel-2対応**
- 6バンド（Blue, Green, Red, NIR, SWIR1, SWIR2）処理
- GeoTIFF多バンド画像対応
- 実際の衛星画像データ処理

これで、実際のPrithvi-EO-2.0モデルを使用した完全版Streamlitアプリケーションが完成しました！