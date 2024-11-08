import numpy as np
import yaml
import argparse
from Runner import Runner

from TextData import TextData
import file_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="20NG")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model", type=str, default="LANTM_ECRTM")
    parser.add_argument("--num_topics", type=int, default=50)
    parser.add_argument("--num_top_words", type=int, default=10)
    parser.add_argument("--config_id", type=int, default=1)
    parser.add_argument("--calculate_CV", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    params = {"exp_settings": vars(args)}

    output_prefix = f"output/{args.dataset}/{args.model}/config{args.config_id}"
    file_utils.make_dir(output_prefix)
    params["exp_settings"]["output_prefix"] = output_prefix

    config_path = f"configs/{args.model}/{args.dataset}_config{args.config_id}.yaml"
    with open(config_path) as file:
        config = yaml.safe_load(file)
        params = {**params, **config}

    separate_line_log = '=' * 70
    print(separate_line_log)

    data_handler = TextData(
        dataset=params["exp_settings"]["dataset"],
        batch_size=params["training"]["batch_size"],
        embedding_dim=params["model_params"]["embedding_dim"],
    )

    params["data_handler"] = data_handler
    params["dataset_statistics"] = {
        "vocab_size": data_handler.vocab_size,
        "num_classes": data_handler.num_classes,
    }
    params["model_params"]["word_embeddings"] = data_handler.word_embeddings

    # calculate the label distribution of the dataset
    # Eq. (8) in our paper
    _, label_freq = np.unique(data_handler.train_labels, return_counts=True)
    label_dist = label_freq.astype(np.float32) / label_freq.sum()
    params["dataset_statistics"]["label_dist"] = label_dist

    # get the pseudo-documents
    avg_num_terms = data_handler.train_bows.sum() / data_handler.train_bows.shape[0]
    pseudo_doc_per_class = []
    for c in range(data_handler.num_classes):
        indices = np.where(data_handler.train_labels == c)[0]
        docs = data_handler.train_bows[indices, :]
        num_terms_c = docs.sum(0)
        term_freq_c = num_terms_c / num_terms_c.sum()
        pseudo_doc_per_class.append(term_freq_c * avg_num_terms)
    pseudo_doc_per_class = np.stack(pseudo_doc_per_class)
    params["dataset_statistics"]["pseudo_doc_per_class"] = pseudo_doc_per_class

    runner = Runner(params)
    if args.mode == "train":
        file_utils.clear_folder(output_prefix)
        runner.train()
    elif args.mode == "eval":
        runner.eval()
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
