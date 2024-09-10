import gzip
import json
import os
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from run_listt5 import ListT5Evaluator


def convert_jsonl(input_file, output_file):
    listt5_data = {}
    with gzip.open(input_file, "rt") as input:
        for line in tqdm(input, "Reformat input dataset."):
            data = json.loads(line)
            qid = data["qid"]
            if qid not in listt5_data:
                listt5_data[qid] = {
                    "qid": qid,
                    "q_text": data["query"],
                    "bm25_results": [],
                }
            listt5_data[qid]["bm25_results"].append(
                {"pid": data["docno"], "text": data["text"], "bm25_score": data["score"]}
            )
    with open(output_file, "w") as output:
        for qid, data in listt5_data.items():
            output.write(json.dumps(data) + "\n")


def main(args=None):
    parser = ArgumentParser()

    # tira data converter args
    parser.add_argument(
        "--input_file", type=Path, default=Path(os.environ.get("TIRA_INPUT_DIR", "") + "./rerank.jsonl.gz")
    )

    # dataset key setup
    parser.add_argument("--firststage_result_key", default="bm25_results", type=str)
    parser.add_argument("--docid_key", default="docid", type=str)
    parser.add_argument("--pid_key", default="pid", type=str)
    parser.add_argument("--qrels_key", default="qrels", type=str)
    parser.add_argument("--score_key", default="bm25_score", type=str)
    parser.add_argument("--question_text_key", default="q_text", type=str)
    parser.add_argument("--text_key", default="text", type=str)
    parser.add_argument("--title_key", default="title", type=str)

    # model args
    parser.add_argument("--model_path", default="Soyoung97/ListT5-base", type=str)
    parser.add_argument(
        "--topk", default=100, type=int, help="number of initial candidate passages to consider"
    )  # or 1000
    parser.add_argument("--max_input_length", type=int, default=-1)  # depends on each individual data setup
    parser.add_argument("--padding", default="max_length", type=str)
    parser.add_argument("--listwise_k", default=5, type=int)
    parser.add_argument("--rerank_topk", default=10, type=int)
    parser.add_argument("--out_k", default=2, type=int)
    parser.add_argument("--dummy_number", default=21, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--bsize", default=20, type=int
    )  # depends on your gpu and input length. We recommend input_length->bsize as 256->128, 512->32, 1024->16, 1280->8 for t5-3b with GB gpu.

    # profiling setup
    parser.add_argument("--measure_flops", action="store_true")
    parser.add_argument(
        "--skip_no_candidate",
        action="store_true",
        help="skip instances with no gold qrels included at first-stage retrieval for faster inference, only works when gold qrels are available",
    )
    parser.add_argument(
        "--skip_issubset",
        action="store_true",
        help="skip the rest of reranking when the gold qrels is a subset of reranked output for faster inference, only works when gold qrels are available",
    )

    args = parser.parse_args(args)
    args.input_path = "/tmp/re_rank.jsonl"
    args.output_path = os.environ.get("TIRA_OUTPUT_DIR", "") + "./run.txt"
    args.max_gen_length = args.listwise_k + 2

    convert_jsonl(args.input_file, args.input_path)

    evaluator = ListT5Evaluator(args)
    evaluator.run_tournament_sort()


if __name__ == "__main__":
    main()
