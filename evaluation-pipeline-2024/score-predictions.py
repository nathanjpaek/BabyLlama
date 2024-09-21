import argparse
import json
import gzip
import numpy as np
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from devbench.comparison.stats_helper import *

def score_predictions(predictions, task, gold_dir):
    subtask_scores = {}
    for subtask in predictions[task]:
        total = 0
        correct = 0
        if task == "glue":
            gold_file = f"{gold_dir}/{subtask}.valid.jsonl"
            label_key = "label"
        elif task == "ewok":
            gold_file = f"{gold_dir}/{subtask}.jsonl"
            label_key = "Target1"
        elif task == "vqa":
            gold_file = f"{gold_dir}/vqa_distractors_info.json"
            label_key = "target_ans"
        elif task == "winoground":
            gold_file = f"{gold_dir}/winoground.jsonl"
            label_key = "caption_0"
        else:
            gold_file = f"{gold_dir}/{subtask}.jsonl"
            label_key = "sentence_good"
        pred_list, gold_list = [], []
        gold_lines = open(gold_file, 'r').readlines()
        if task == "vqa":
            gold_lines = [example for example in json.loads(gold_lines[0])]
        for (pred_dict, gold_line) in zip(predictions[task][subtask]["predictions"], gold_lines):
            if task == "vqa":
                gold_dict = gold_line
            else:
                gold_dict = json.loads(gold_line)
            pred = pred_dict["pred"]
            gold = gold_dict[label_key]
            if type(pred) == str:
                pred = pred.strip().lower()
                gold = gold.strip().lower()
                if pred == gold:
                    pred, gold = 0, 0
                else:
                    pred, gold = 0, 1
            pred_list.append(pred)
            gold_list.append(gold)
        if task == "glue":
            if subtask == "cola":
                metric = matthews_corrcoef(y_true=gold_list, y_pred=pred_list)
                subtask = "cola (MCC)"
            elif subtask in ("mrpc", "qqp"):
                metric = f1_score(y_true=gold_list, y_pred=pred_list)
                subtask = f"{subtask} (F1)"
            else:
                metric = accuracy_score(y_true=gold_list, y_pred=pred_list)
        else:
            metric = accuracy_score(y_true=gold_list, y_pred=pred_list)
        subtask_scores[subtask] = metric
    return subtask_scores


def score_devbench(predictions):
    import scipy
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from devbench.comparison.trog import compare_trog
    from devbench.comparison.viz_vocab import compare_vv
    subtask_scores = {}
    for subtask in predictions["devbench"]:
        results = np.array(predictions[task][subtask]["predictions"])
        if subtask == "sem-things":
            human_data = scipy.io.loadmat("evaluation_data/devbench/evals/sem-things/spose_similarity.mat")
            human_data = human_data["spose_sim"]
            sim = rsa(human_data, results)
            subtask_scores[subtask] = sim
        elif subtask == "gram-trog":
            human_data = pd.read_csv("evaluation_data/devbench/evals/gram-trog/human.csv")
            results = pd.DataFrame(results.squeeze(),
                    columns = ["image1", "image2", "image3", "image4"])
            results['trial'] = np.arange(results.shape[0])+1
            results['correct'] = (results['image1'] > results['image2']) & \
                              (results['image1'] > results['image3']) & \
                              (results['image1'] > results['image4'])
            acc = results['correct'].mean()
            kls = compare_trog(results, human_data)
            kl_metric = np.exp(-kls['kl']).mean()
            subtask_scores[subtask] = (acc, kl_metric)
        else:
            human_data = pd.read_csv("evaluation_data/devbench/evals/lex-viz_vocab/human.csv")
            results = pd.DataFrame(results.squeeze(),
                    columns = ["image1", "image2", "image3", "image4"])
            results['trial'] = np.arange(results.shape[0])+1
            results['correct'] = (results['image1'] > results['image2']) & \
                    (results['image1'] > results['image3']) & \
                    (results['image1'] > results['image4'])
            acc = results['correct'].mean()
            kls = compare_vv(results, human_data)
            kl_metric = np.exp(-kls['kl']).mean()
            subtask_scores[subtask] = (acc, kl_metric)
    return subtask_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_path", type=str)
    parser.add_argument("--include_multimodal", "-m", action="store_true")
    args = parser.parse_args()

    with gzip.open(args.predictions_path, 'rt') as predictions_file:
        predictions = json.load(predictions_file)

    for task in predictions:
        if task == "glue":
            gold_dir = "evaluation_data/glue_filtered/"
        elif task == "blimp":
            gold_dir = "evaluation_data/blimp_filtered/"
        elif task == "blimp_supplement":
            gold_dir = "evaluation_data/supplement_filtered/"
        elif task == "ewok":
            gold_dir = "evaluation_data/ewok_filtered/"
        elif task == "vqa":
            gold_dir = "evaluation_data/vqa_filtered/"
        elif task == "winoground":
            gold_dir = "evaluation_data/winoground_filtered/"
        elif task == "devbench":
            scores = score_devbench(predictions)
            score_rows = []
            accs, human_sims = [], []
            for subtask in scores:
                if type(scores[subtask]) == tuple:
                    acc, kl_metric = scores[subtask]
                    accs.append(acc)
                    human_sims.append(kl_metric)
                    score_rows.append([subtask, f"{acc:.3f}", f"{kl_metric:.3f}"])
                else:
                    human_sims.append(scores[subtask])
                    accs.append(scores[subtask])
                    score_rows.append([subtask, f"{scores[subtask]:.3f}", f"{scores[subtask]:.3f}"])
            avg_acc = np.mean(accs)
            avg_human_sim = np.mean(human_sims)
            score_rows.append(["*Average*", f"{avg_acc:.3f}", f"{avg_human_sim:.3f}"])
            print(tabulate(score_rows, headers=[f"{task} subtask", "Acc", "Human Similarity"]))
            print()
            continue

        scores = score_predictions(predictions, task, gold_dir)
        score_rows = [[k, f"{v:.3f}"] for k, v in scores.items()]
        scores_list = list(scores.values())
        avg_task_score = np.mean(scores_list)
        if len(scores_list) > 1:
            score_rows.append(["*Average*", f"{avg_task_score:.3f}"])
        print(tabulate(score_rows, headers=[f"{task} subtask", "Score"]))
        print()