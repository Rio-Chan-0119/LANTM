import torch
import numpy as np
import scipy.sparse
import scipy.io

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def load_data(path, embedding_dim):
    train_bows = scipy.sparse.load_npz(f"{path}/train_bows.npz").toarray().astype("float32")
    test_bows = scipy.sparse.load_npz(f"{path}/test_bows.npz").toarray().astype("float32")

    if embedding_dim is not None:
        embeddings_path = f"{path}/word_embeddings_{embedding_dim}.npz"
        word_embeddings = scipy.sparse.load_npz(embeddings_path).toarray().astype("float32")
    else:
        word_embeddings = None

    train_labels = np.load(f"{path}/train_labels.npy")
    test_labels = np.load(f"{path}/test_labels.npy")

    vocab = file_utils.read_texts_by_lines(f"{path}/vocab.txt")

    train_texts = file_utils.read_texts_by_lines(f"{path}/train_texts.txt")
    test_texts = file_utils.read_texts_by_lines(f"{path}/test_texts.txt")
    train_texts = [text.split(" ") for text in train_texts]
    test_texts = [text.split(" ") for text in test_texts]

    label_names = file_utils.read_texts_by_lines(f"{path}/label_names.txt")

    return (train_labels, test_labels, train_texts, test_texts,
            vocab, train_bows, test_bows, word_embeddings, label_names)


class TextData(object):
    def __init__(self, dataset, batch_size, embedding_dim):
        dataset_path = f"data/processed/{dataset}"
        (self.train_labels,
         self.test_labels,
         self.train_texts,
         self.test_texts,
         self.vocab,
         self.train_bows,
         self.test_bows,
         self.word_embeddings,
         self.label_names,
         ) = load_data(dataset_path, embedding_dim)
        self.vocab_size = len(self.vocab)
        self.num_classes = len(np.unique(self.train_labels)) if self.train_labels is not None else None
        self.num_tokens_train = sum([len(text) for text in self.train_texts])
        self.num_tokens_test = sum([len(text) for text in self.test_texts])

        print("===> train_size: ", self.train_bows.shape[0])
        print("===> test_size: ", self.test_bows.shape[0])
        print("===> vocab_size: ", self.vocab_size)
        print("===> average length: {:.3f}".format(self.train_bows.sum(1).sum() / self.train_bows.shape[0]))
        print("===> # label: ", len(np.unique(self.train_labels)))

        train_bows_pt = torch.from_numpy(self.train_bows)
        test_bows_pt = torch.from_numpy(self.test_bows)
        train_labels_pt = torch.from_numpy(self.train_labels)
        test_labels_pt = torch.from_numpy(self.test_labels)

        if torch.cuda.is_available():
            train_bows_pt = train_bows_pt.cuda()
            train_labels_pt = train_labels_pt.cuda()

        train_dataset = TensorDataset(train_bows_pt, train_labels_pt)
        test_dataset = TensorDataset(test_bows_pt, test_labels_pt)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
