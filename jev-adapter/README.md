# System One Adapterを使ったJevとローカルLLMの分類タスク比較

TypeSafeのJevは、文章などの状態をもとに選択肢を判断し、型付きの回答と確率を返すモデルです。[前回の記事](https://blog.lai.so/2026-09-17/)では、Jevの仕組みと利用例を紹介しました。

[Kev](https://github.com/jaredpalmer/kev)は、Qwen3.5などを基盤に追加学習し、Jevと同じ形式のAPIで呼べるローカル判定モデルです。自分でも追加学習して判定を強化したいので、まずは自分では追加学習していないGemmaが同じ問題をどこまで解けるか、比較の基準を取りました。最初の比較に使ったコードは[`compare.py`](compare.py)にあります。

比較に使ったのは、TypeSafe公式の[System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python)です。Jevを呼ぶTypeSafe SDKと同じ`state`と設問を、通常のLLM APIへ渡せます。比較相手の`google/gemma-4-12b-qat`はLM Studio上で動かし、アダプターからOpenAI互換の`/v1/chat/completions`へ接続しました。Gemmaの重みや出力ヘッドは変更していません。ただし、アダプター側では構造化出力を要求し、確率出力の条件では確率を正規化しています。

実行するには`TYPESAFE_API_KEY`と`LM_STUDIO_BASE_URL`が必要です。スクリプトは、デフォルトではJevとGemmaに各選択肢の確率を答えさせ、`discrete`モードではGemmaに選択肢を一つだけ選ばせます。`jev-adapter`ディレクトリからの実行例は次のとおりです。

```bash
python compare.py --mode probabilities --output results-rerun.jsonl
python compare.py --mode discrete --output results-discrete-rerun.jsonl
```

Jevの新規登録は2026年9月25日時点で停止しているため、既存アカウントが必要です。

入力には、Kevを評価するために作った合成問い合わせテストを再利用しました。16件の本文と設問は[`compare.py`](compare.py)に固定しています。Kevの結果は今回の比較に含めていません。返品・配送・請求・その他の4分類と、直接表現・言い換え・否定・文脈の4種類を組み合わせ、それぞれ先頭の1件を取った計16件です。

両モデルに渡す`state`は問い合わせ本文です。たとえば「届いた額縁のガラスが割れていたため、交換の申し込みをしたいです。」という文を入れました。`instructions`では、否定された依頼に惑わされず、顧客が実際に求めている対応を一つ選ぶよう指示しています。`criteria`には返品・配送・請求の説明と、どれにも該当しない場合の「その他」を指定しました。全文は[`compare.py`](compare.py)にあります。TypeSafe SDKの`Choice`は、選択肢を一つ選び、各選択肢の確率を返す設問の型です。選択肢の順序も両モデルで揃えました。

同じ16問を3条件で解きました。Jevには`Choice`を使いました。Gemmaではアダプターの回答モードを変え、2回試しました。「確率出力」では4つの選択肢それぞれの確率を答えさせ、最も高いものを採用します。「選択のみ」（`discrete`）では確率を求めず、選択肢を一つだけ選ばせます。表の正答数は、選んだ分類が正しかった件数です。

| 条件 | 正答 | 時間の中央値 | 備考 |
| --- | ---: | ---: | --- |
| Jev（Choiceの確率出力） | 16/16 | 0.284秒 | 既存アカウントのクラウドAPI |
| Gemma（アダプターの確率出力） | 3/16 | 2.541秒 | 選択肢ごとの確率を要求・正規化 |
| Gemma（アダプターの選択のみ） | 7/16 | 1.117秒 | `discrete`モード |

Gemmaの確率出力では、16件中8件で「その他」を選びました。構造化出力の形式が崩れて1件だけ再試行しましたが、最終的に失敗したリクエストはありません。Gemmaには構造化出力を要求し、形式が不正なら最大1回再試行する設定にしています。測定前には、各モデルへ同じ1問を送って疎通を確認しました。全件の記録は[確率出力](results.jsonl)と[選択のみ](results-discrete.jsonl)にあります。

16件は合成例なので、一般的な問い合わせでの精度までは分かりません。また、JevはクラウドAPI、Gemmaは別マシン上のLM Studioで動かしました。時間の差には通信や実行環境も含まれ、モデル自体の速度差とは言えません。確率の校正や利用料金も比較していません。

## 再現手順

Python 3.10以上、Jevを使えるTypeSafeアカウント、`google/gemma-4-12b-qat`を導入したLM Studioを用意します。LM StudioのDeveloper画面でモデルを読み込み、APIサーバーを起動してください（CLIなら`lms server start`）。別マシンで動かす場合は、LM Studio側でローカルネットワークからの接続を有効にします。以下のコマンドは専用リポジトリを取得した直後から実行できます。アダプターとSDKは測定時のバージョンに固定しました。

```bash
git clone https://github.com/laiso/playground.git
cd playground/jev-adapter
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'system-one-adapter[openai]==0.2.1' 'typesafe-sdk==0.7.1'
export LM_STUDIO_BASE_URL=http://localhost:1234/v1  # 別マシンならそのアドレスに変更
curl -fsS "$LM_STUDIO_BASE_URL/models"  # モデルIDとサーバーへの接続を確認
read -r -s -p 'TypeSafe API key: ' TYPESAFE_API_KEY; echo
export TYPESAFE_API_KEY
python compare.py --mode probabilities --output results-rerun.jsonl
python compare.py --mode discrete --output results-discrete-rerun.jsonl
```

最初の実行ではJevとGemmaの確率出力を、2回目ではGemmaの選択のみを各16件記録します。次のコマンドで行数、正答数、時間の中央値を確認できます。

```bash
wc -l results-rerun.jsonl results-discrete-rerun.jsonl
python - <<'PY'
import json
from pathlib import Path
from statistics import median

base = Path('.')
for filename, clients in (
    ('results-rerun.jsonl', ('jev', 'gemma_adapter')),
    ('results-discrete-rerun.jsonl', ('gemma_adapter',)),
):
    rows = [json.loads(line) for line in (base / filename).read_text().splitlines()]
    for client in clients:
        valid = [row for row in rows if 'choice' in row[client]]
        correct = sum(row[client]['choice'] == row['label'] for row in valid)
        seconds = median(row[client]['wall_seconds'] for row in rows)
        print(filename, client, f'{correct}/{len(rows)}', f'{seconds:.3f}秒')
PY
```

同じ出力ファイルを指定して再実行すると、各テスト行のIDを見て記録済みの問題をスキップします。全件を再測定するときは新しいファイル名を指定してください。モデルの量子化版、実行マシン、ネットワーク、Jev側の状態が異なれば、表の数値も変わります。

## 追加学習で参考にするコード

- [Kevの学習コード](https://github.com/jaredpalmer/kev/blob/main/kev/train.py)と[判定モデル](https://github.com/jaredpalmer/kev/blob/main/kev/model.py)：QwenのLoRAと選択肢を採点する専用ヘッドを学習する実装。
- [Tev1の学習例](https://github.com/togethercomputer/tev1/blob/main/examples/train_together.py)：Qwenを通常の言語モデル出力ヘッドのままLoRAで追加学習する別方式。Jev互換APIの実装例ではありません。
