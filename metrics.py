import os
import subprocess
import threading

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.metrics.cluster import contingency_matrix


def get_topic_diversity(topic_word, num_topic_words=25):
    top_N = np.argsort(topic_word, axis=1)[:, -num_topic_words:][:, ::-1]
    top_N = top_N.flatten()
    td = len(set(top_N)) / len(top_N)
    return td


def get_palmetto_topic_coherence(topic_words, output_prefix, palmetto_prefix="."):
    num_topics = len(topic_words)

    temp_file_name = os.path.join(output_prefix, "topics.temp")
    with open(temp_file_name, "w") as f:
        for topic in topic_words:
            f.write(" ".join(topic) + "\n")

    if palmetto_prefix is None:
        palmetto_prefix = os.path.expanduser("~")
    jar_dir = f"{palmetto_prefix}/palmetto"
    jar_path = f"{jar_dir}/palmetto-0.1.0-jar-with-dependencies.jar"
    wiki_dir = f"{jar_dir}/wikipedia/wikipedia_bd"

    def _worker(coherence_type):
        java_command = ["java", "-jar", jar_path, wiki_dir, coherence_type, temp_file_name]
        process = subprocess.Popen(java_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()

        if error:
            err_msg = "Errors when calculating TC:\n" + error
            raise RuntimeError(err_msg)

        lines = output.split("\n")[1: num_topics]
        tc_lst = []
        for line in lines:
            tc_k = float(line.split("\t")[1])
            tc_lst.append(tc_k)

        avg_tc = sum(tc_lst) / len(tc_lst)
        result_dict[coherence_type] = [avg_tc, tc_lst]

    coherence_types = ["C_V",]     # may add "NPMI" here
    result_dict = {}

    threads = []
    for ct in coherence_types:
        threads.append(threading.Thread(target=_worker, args=(ct, )))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    os.remove(temp_file_name)

    # return overall C_V and C_V per topic
    return result_dict["C_V"][0], result_dict["C_V"][1]


def get_purity(labels_true, labels_pred):
    contingency_mat = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)


def get_adjusted_rand_index(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)


def get_normalized_mutual_info(labels_true, labels_pred):
    return normalized_mutual_info_score(labels_true, labels_pred)


def get_f1_score(labels_true, labels_pred):
    return f1_score(labels_true, labels_pred, average="macro")
