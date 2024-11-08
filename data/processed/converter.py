import os

import numpy as np
import scipy.sparse

import file_utils


def load_data(path, embedding_dim):
    test_exists = False
    if os.path.exists(f'{path}/test_bow.npz'):
        test_exists = True

    train_data = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
    test_data = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')

    if embedding_dim is not None:
        embeddings_path = f'{path}/word_embeddings_{embedding_dim}.npz'
        word_embeddings = scipy.sparse.load_npz(embeddings_path).toarray().astype('float32')
    else:
        word_embeddings = None

    train_labels = np.load(f'{path}/train_labels.npy')
    test_labels = np.load(f'{path}/test_labels.npy') if test_exists else None

    vocab = file_utils.read_texts_by_lines(f'{path}/vocab.txt')

    train_texts = file_utils.read_texts_by_lines(f'{path}/train_texts.txt')
    test_texts = file_utils.read_texts_by_lines(f'{path}/test_texts.txt')
    train_texts = [text.split(" ") for text in train_texts]
    test_texts = [text.split(" ") for text in test_texts]

    label_names = file_utils.read_texts_by_lines(f"{path}/label_names.txt")

    return (train_data, test_data, train_labels, test_labels,
            train_texts, test_texts, vocab, word_embeddings, label_names)


def convert(dataset):
    dataset_path = f'../ECRTM/data/processed/{dataset}'
    (train_data,
     test_data,
     train_labels,
     test_labels,
     train_texts,
     test_texts,
     vocab,
     word_embeddings,
     label_names,
     ) = load_data(dataset_path, 200)
    vocab_size = len(vocab)
    k = 1

    if not os.path.exists(f'../ECRTM/data/processed/{dataset}_slda'):
        os.mkdir(f'../ECRTM/data/processed/{dataset}_slda')

    f_train_data = open(f'../ECRTM/data/processed/{dataset}_slda/train_data.txt', "w")
    f_test_data = open(f'../ECRTM/data/processed/{dataset}_slda/test_data.txt', "w")
    f_train_label = open(f'../ECRTM/data/processed/{dataset}_slda/train_label.txt', "w")
    f_test_label = open(f'../ECRTM/data/processed/{dataset}_slda/test_label.txt', "w")

    ss = []
    for doc in train_data:
        s = ""
        cnt = 0
        for i in range(vocab_size):
            if doc[i] != 0:
                s += f" {i}:{int(doc[i].item()):d}"
                cnt += 1
        s2 = f"{cnt:d}{s}\n"
        ss.append(s2)
    f_train_data.writelines(ss)

    ss = []
    for doc in test_data:
        s = ""
        cnt = 0
        for i in range(vocab_size):
            if doc[i] != 0:
                s += f" {i}:{int(doc[i].item()):d}"
                cnt += 1
        s2 = f"{cnt:d}{s}\n"
        ss.append(s2)
    f_test_data.writelines(ss)

    for label in train_labels:
        f_train_label.write(f"{label.item():d}\n")

    for label in test_labels:
        f_test_label.write(f"{label.item():d}\n")

    f_train_data.close()
    f_test_data.close()
    f_train_label.close()
    f_test_label.close()


if __name__ == "__main__":
    convert("20NG")
