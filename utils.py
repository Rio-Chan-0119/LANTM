import os
import numpy as np


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def read_texts_by_lines(path):
    texts = list()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clear_folder(file_path)
    else:
        print(f"no folder: {folder_path}")


def file_exists(file_name):
    return os.path.exists(file_name)


def get_topic_word_str(beta, vocab, num_top_words):
    topic_words = []
    topic_words_ids = np.argsort(beta, axis=1)[:, -num_top_words:][:, ::-1]
    for topic in topic_words_ids:
        topic_words.append([vocab[i] for i in topic])
    return topic_words


def save_topic_word_str(topic_words, path):
    with open(path, "w", encoding="utf-8") as f:
        for topic in topic_words:
            s = " ".join(topic)
            f.write(s + "\n")
