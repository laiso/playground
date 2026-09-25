"""Small, fixed Jev vs LM Studio comparison using TypeSafe's official adapter."""

import json
import os
import time
import argparse
from pathlib import Path

from system_one_adapter import SystemOneAdapterClient
from system_one_adapter.providers.openai import OpenAIProvider
from typesafe_sdk import Choice, TypeSafeClient


MODEL = "google/gemma-4-12b-qat"
BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
GROUPS = ("literal", "paraphrase", "negation", "context")
LABELS = ("returns", "shipping", "billing", "other")
INSTRUCTIONS = "この問い合わせで顧客が現在求めている対応を一つ選んでください。否定された依頼ではなく、実際の依頼を判定してください。"
CRITERIA = {
    "returns": "購入した商品の返品・交換・返金の依頼",
    "shipping": "商品の配送状況・到着予定・届け先の確認や変更",
    "billing": "請求金額・二重決済・支払い方法・領収書についての問い合わせ",
    "other": None,
}
CASES = {
    "literal": {
        "returns": "届いた額縁のガラスが割れていたため、交換の申し込みをしたいです。",
        "shipping": "注文して十日たつのに荷物が届きません。配送状況を調査してください。",
        "billing": "注文した代金が二回口座から引かれています。確認をお願いします。",
        "other": "携帯を替えたら会員ログインができなくなりました。",
    },
    "paraphrase": {
        "returns": "左足用が二つ入っていました。左右そろった組と入れ替えてほしいです。",
        "shipping": "買ったものはいまどの辺りを移動していますか。家にはまだ来ていません。",
        "billing": "買い物は一度きりなのに、通帳には同じ金額が続けて二行載っています。",
        "other": "熱いお湯を注いでもこの容器は大丈夫でしょうか。買う前に知っておきたいです。",
    },
    "negation": {
        "returns": "配達の遅れを責めているのではありません。到着した器具が壊れているので返品したいのです。",
        "shipping": "返品はしません。まだ到着していない荷物がいつ来るかだけ確認したいです。",
        "billing": "商品を送り返す予定はありません。代金の二重引き落としだけを調べてほしいです。",
        "other": "荷物を探しているのではなく、会員ページの入り方が分からず困っています。",
    },
    "context": {
        "returns": "先週は住所を訂正していただきありがとうございました。今日は届いた商品の傷についてです。返品して購入代金を戻してもらえますか。",
        "shipping": "返品した別の商品については解決しました。今回は新しく注文した品の発送予定を尋ねています。",
        "billing": "昨日荷物は無事に届きました。今日は、その注文の領収書を会社名で発行してほしくて連絡しました。",
        "other": "二重決済についての説明は理解しました。次はログインに使う暗証情報を再設定したいです。",
    },
}


def key_from_env() -> str:
    if key := os.getenv("TYPESAFE_API_KEY"):
        return key
    raise RuntimeError("TYPESAFE_API_KEY is unavailable; set the environment variable")


def answer_dict(response) -> dict:
    answer = response.answers["department"]
    return {
        "choice": answer.choice,
        "confidence": answer.confidence,
        "probabilities": dict(answer.probabilities),
        "usage": response.usage.model_dump() if response.usage else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probabilities", "discrete"), default="probabilities")
    parser.add_argument("--output", type=Path, help="結果を書き込むJSONLファイル")
    args = parser.parse_args()
    out = args.output or Path("results.jsonl" if args.mode == "probabilities" else "results-discrete.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    selected = [{"id": f"test-{label}-{group}-00", "group": group, "label": label, "state": CASES[group][label]}
                for group in GROUPS for label in LABELS]
    question = {"department": Choice(instructions=INSTRUCTIONS, criteria=CRITERIA)}

    jev = TypeSafeClient(api_key=key_from_env(), timeout=60) if args.mode == "probabilities" else None
    provider = OpenAIProvider(MODEL, base_url=BASE_URL, api_key="lm-studio", api="chat_completions")
    adapter = SystemOneAdapterClient(structured_outputs=True, llm_answer_mode=args.mode,
                                     normalize_probabilities=True, n_retry_malformed_structure=1)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()

    with out.open("a") as output:
        for row in selected:
            if row["id"] in done:
                continue
            record = {"id": row["id"], "group": row["group"], "label": row["label"], "state": row["state"]}
            clients = (("jev", jev), ("gemma_adapter", adapter)) if jev else (("gemma_adapter", adapter),)
            for name, client in clients:
                start = time.monotonic()
                try:
                    kwargs = {"model": provider} if name == "gemma_adapter" else {}
                    response = client.system_one(row["state"], question, **kwargs)
                    record[name] = answer_dict(response)
                except Exception as exc:
                    record[name] = {"error_type": type(exc).__name__, "error": str(exc)[:300]}
                record[name]["wall_seconds"] = round(time.monotonic() - start, 3)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(row["id"], record.get("jev", {}).get("choice"), record["gemma_adapter"].get("choice"), flush=True)


if __name__ == "__main__":
    main()
