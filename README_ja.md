## JiT (Just image Transformer) - テキスト条件付き画像生成

このリポジトリは、ピクセル空間での拡散モデルを実装した研究用コードベースです。オリジナルのImageNetクラス条件付けに加えて、**LLM-jpなどの日本語LLMを使ったテキスト条件付け画像生成（Text-to-Image）** に対応しています。

[English README](README.md)

---

## 🎨 テキスト条件付け画像生成の新機能

### 特徴

- **LLM-jpとの統合**: llm-jp-3-3.7bなどの日本語LLMをテキストエンコーダーとして使用
- **効率的な学習**: LLMの重みは凍結し、projection層のみを学習
- **柔軟なデータセット**: カスタムテキストキャプションに対応
- **Classifier-Free Guidance (CFG)**: 高品質な画像生成のためのCFGをサポート
- **後方互換性**: 既存のクラス条件付けモデルもそのまま使用可能

---

## 📦 インストール

### 1. 基本環境のセットアップ

```bash
git clone https://github.com/LTH14/JiT.git
cd JiT
conda env create -f environment.yaml
conda activate jit
```

### 2. テキスト条件付けに必要な追加パッケージ

```bash
pip install transformers
```

もし`torch`のインポートエラーが発生した場合：
```bash
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

---

## 📁 データセットの準備

### テキストキャプション付き画像データセット

以下のディレクトリ構造でデータを配置してください：

```
your_data_path/
├── images/
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   ├── img_0003.jpg
│   └── ...
└── captions/
    ├── img_0001.txt
    ├── img_0002.txt
    ├── img_0003.txt
    └── ...
```

**注意事項：**
- `images/`フォルダには画像ファイル（.jpg, .jpeg, .png）を配置
- `captions/`フォルダには対応するテキストファイルを配置
- 各`.txt`ファイルには、画像の説明を**1行**で記述してください

**キャプションの例：**

`captions/img_0001.txt`:
```
青い空と緑の草原の風景写真
```

`captions/img_0002.txt`:
```
白い猫が窓辺で日向ぼっこをしている写真
```

---

## 🚀 学習方法

### テキスト条件付けモデルの学習

#### JiT-B/16-Text (256x256)

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16-Text \
--llm_model_name llm-jp/llm-jp-3-3.7b \
--use_text_conditioning \
--freeze_llm \
--max_text_len 77 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ./output_jit_text_256 \
--resume ./output_jit_text_256 \
--data_path /path/to/your/data \
--online_eval
```

#### JiT-B/32-Text (512x512)

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32-Text \
--llm_model_name llm-jp/llm-jp-3-3.7b \
--use_text_conditioning \
--freeze_llm \
--max_text_len 77 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 512 --noise_scale 2.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ./output_jit_text_512 \
--resume ./output_jit_text_512 \
--data_path /path/to/your/data \
--online_eval
```

#### 大規模モデル JiT-L/16-Text (256x256)

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-L/16-Text \
--llm_model_name llm-jp/llm-jp-3-3.7b \
--use_text_conditioning \
--freeze_llm \
--max_text_len 77 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
--output_dir ./output_jit_L_text_256 \
--resume ./output_jit_L_text_256 \
--data_path /path/to/your/data \
--online_eval
```

### 主要なパラメータの説明

| パラメータ | 説明 | 推奨値 |
|----------|------|--------|
| `--model` | モデルのアーキテクチャ | `JiT-B/16-Text`, `JiT-B/32-Text`, `JiT-L/16-Text` など |
| `--llm_model_name` | 使用するLLMモデル | `llm-jp/llm-jp-3-3.7b` (推奨) |
| `--use_text_conditioning` | テキスト条件付けを有効化 | 必須 |
| `--freeze_llm` | LLMの重みを凍結 | 推奨（メモリ効率とトレーニング安定性） |
| `--max_text_len` | テキストの最大トークン長 | 77（CLIP標準）または128 |
| `--img_size` | 画像サイズ | 256 または 512 |
| `--noise_scale` | ノイズスケール | 256px: 1.0, 512px: 2.0 |
| `--cfg` | Classifier-Free Guidanceスケール | 2.2～2.9（高いほど忠実度が上がるが多様性は下がる） |
| `--batch_size` | GPU1台あたりのバッチサイズ | 128（8xH200の場合） |
| `--blr` | ベース学習率 | 5e-5 |

---

## 📊 評価

学習済みモデルの評価：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16-Text \
--llm_model_name llm-jp/llm-jp-3-3.7b \
--use_text_conditioning \
--img_size 256 --noise_scale 1.0 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ./output_jit_text_256 \
--resume ./output_jit_text_256 \
--data_path /path/to/your/data \
--evaluate_gen
```

評価では以下のメトリクスが計算されます：
- **FID (Fréchet Inception Distance)**: 生成画像の品質
- **Inception Score**: 生成画像の多様性と品質

---

## 🎯 推論（画像生成）

学習済みモデルで画像を生成するには、`evaluate`関数内のテストプロンプトをカスタマイズできます。

`engine_jit.py`の114-117行目：

```python
test_prompts = [
    "猫の写真", "犬の写真", "風景の写真", "建物の写真", "花の写真",
    "車の写真", "鳥の写真", "食べ物の写真", "人物の写真", "動物の写真"
] * 100  # Repeat to have enough prompts
```

この部分を編集して、生成したいプロンプトを追加してください。

---

## 🔧 利用可能なモデル

### テキスト条件付けモデル（新規）

| モデル名 | パラメータ数 | パッチサイズ | 推奨解像度 | 深さ | 隠れ層サイズ |
|---------|------------|------------|----------|------|------------|
| `JiT-B/16-Text` | ~92M | 16x16 | 256x256 | 12 | 768 |
| `JiT-B/32-Text` | ~92M | 32x32 | 512x512 | 12 | 768 |
| `JiT-L/16-Text` | ~308M | 16x16 | 256x256 | 24 | 1024 |
| `JiT-L/32-Text` | ~308M | 32x32 | 512x512 | 24 | 1024 |
| `JiT-H/16-Text` | ~637M | 16x16 | 256x256 | 32 | 1280 |
| `JiT-H/32-Text` | ~637M | 32x32 | 512x512 | 32 | 1280 |

### クラス条件付けモデル（オリジナル）

ImageNetクラスラベルで条件付けする従来のモデルも引き続き使用可能：
- `JiT-B/16`, `JiT-B/32`, `JiT-L/16`, `JiT-L/32`, `JiT-H/16`, `JiT-H/32`

---

## 🛠️ LLMモデルの選択肢

デフォルトでは`llm-jp/llm-jp-3-3.7b`を使用しますが、他のLLMモデルも使用可能です：

| モデル | パラメータ数 | 特徴 |
|--------|------------|------|
| `llm-jp/llm-jp-1.3b-v1.0` | 1.3B | 軽量・高速、メモリ効率が良い |
| `llm-jp/llm-jp-3-3.7b` | 3.7B | **推奨**：バランスが取れている |
| `llm-jp/llm-jp-3-13b` | 13B | 高品質なテキスト理解、要求メモリが大きい |

モデルを変更するには、`--llm_model_name`引数を変更してください。

---

## 💾 チェックポイントの管理

### チェックポイントの保存

- `checkpoint-last.pth`: 最後のエポックのチェックポイント（5エポックごとに更新）
- `checkpoint-{epoch}.pth`: 100エポックごとの特定エポックのチェックポイント

### チェックポイントから再開

```bash
--output_dir ./output_jit_text_256 \
--resume ./output_jit_text_256
```

`--resume`で指定したディレクトリから`checkpoint-last.pth`を自動的に読み込みます。

---

## 📈 学習のモニタリング

TensorBoardで学習の進捗を確認できます：

```bash
tensorboard --logdir ./output_jit_text_256
```

以下のメトリクスが記録されます：
- `train_loss`: 学習損失
- `lr`: 学習率
- `fid`: FID（評価時）
- `is`: Inception Score（評価時）

---

## 🎨 アーキテクチャの詳細

### テキストエンコーダー (LLMTextEncoder)

1. **LLMによるエンコーディング**: 入力テキストをLLM-jpで処理
2. **Mean Pooling**: LLMの最終層の出力をマスク付き平均プーリング
3. **Projection**: LLMの隠れ層サイズからJiTの隠れ層サイズへ線形変換
4. **重みの凍結**: LLMの重みは凍結し、projection層のみ学習

### Classifier-Free Guidance (CFG)

- 学習時に10%の確率でテキストを空文字列に置換
- 推論時に条件付き予測と無条件予測を組み合わせ
- CFGスケール（`--cfg`）で制御可能

---

## 🐛 トラブルシューティング

### メモリ不足エラー

1. バッチサイズを減らす：`--batch_size 64`
2. より小さいモデルを使用：`JiT-B/16-Text` → `JiT-B/32-Text`
3. より小さいLLMを使用：`llm-jp-3-3.7b` → `llm-jp-1.3b-v1.0`

### キャプションファイルが見つからない

- `images/`と`captions/`のファイル名（拡張子を除く）が一致しているか確認
- パスが正しいか確認：`--data_path`

### 学習が不安定

- 学習率を下げる：`--blr 1e-5`
- warmupエポックを増やす：`--warmup_epochs 10`

---

## 📚 引用

オリジナルのJiT論文：

```bibtex
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

---

## 📝 ライセンス

このプロジェクトはオリジナルのJiTリポジトリと同じライセンスに従います。

---

## 🙏 謝辞

- オリジナルのJiT実装: [LTH14/JiT](https://github.com/LTH14/JiT)
- LLM-jp: 日本語言語モデルの提供
- Google TPU Research Cloud (TRC)とMIT ORCDによるリソース提供

---

## 📧 コンタクト

質問やバグ報告は、[GitHub Issues](https://github.com/LTH14/JiT/issues)へお願いします。
