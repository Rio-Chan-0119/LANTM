import numpy as np
import torch
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from torch.distributions.categorical import Categorical

from models.LANTM_ECRTM import LANTM_ECRTM
from models.LANTM_ETM import LANTM_ETM
import utils


class Runner(object):
    def __init__(self, params):
        self.exp_settings = params["exp_settings"]
        self.training_params = params["training"]
        self.data_handler = params["data_handler"]
        self.output_prefix = self.exp_settings["output_prefix"]

        model_params = {
            **params["exp_settings"],
            **params["dataset_statistics"],
            **params["model_params"],
        }

        model_name = params["exp_settings"]["model"]
        if model_name == "LANTM_ECRTM":
            self.model = LANTM_ECRTM(model_params)
        elif model_name == "LANTM_ETM":
            self.model = LANTM_ETM(model_params)
        else:
            raise RuntimeError(f"Model {model_name} is not implemented.")

        optimizer_params = {
            "params": self.model.parameters(),
            "lr": self.training_params["learning_rate"],
        }
        self.optimizer = torch.optim.Adam(**optimizer_params)
        if self.training_params["use_lr_scheduler"]:
            lr_decay = self.training_params.get("lr_decay", 0.5)
            lr_step_size = self.training_params.get("lr_step_size", self.training_params["epochs"])
            self.lr_scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_decay)
        else:
            self.lr_scheduler = None

    def train(self):
        data_loader = self.data_handler.train_loader
        data_size = len(data_loader.dataset)
        output_prefix = self.output_prefix
        self.model.cuda()

        for epoch_id in range(1, self.training_params["epochs"] + 1):
            self.model.train()
            result_sum_dict = defaultdict(float)

            for docs, labels in data_loader:
                result_dict = self.model.compute_loss(docs, labels)
                batch_loss = result_dict["loss"]

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                for key in result_dict:
                    result_sum_dict[key] += result_dict[key] * len(docs)

            # compute entropy
            label_topic_mat = torch.from_numpy(self.model.get_label_topic_mat())
            d = Categorical(probs=label_topic_mat)
            entropy = d.entropy().mean()
            result_sum_dict["entropy"] = entropy

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            output_logs = [f"Epoch: {epoch_id:03d}", ]
            for key in result_sum_dict:
                if key == "entropy":
                    output_logs.append(f" {key}: {result_sum_dict[key]:.3f}")
                else:
                    output_logs.append(f" {key}: {result_sum_dict[key] / data_size:.3f}")
            print(" | ".join(output_logs))

        label_topic_mat = self.model.get_label_topic_mat()
        np.save(f"{output_prefix}/label_topic_mat.npy", label_topic_mat)
        torch.save(self.model.state_dict(), f"{output_prefix}/param.pt")

        topic_word_dist = self.model.get_topic_word_dist()
        np.save(f"{output_prefix}/topic_word_dist.npy", topic_word_dist)

        vocab = self.data_handler.vocab
        topic_words = utils.get_topic_word_str(topic_word_dist, vocab,
                                               num_top_words=self.exp_settings["num_top_words"])
        topic_word_path = f"{output_prefix}/topic_words.txt"
        utils.save_topic_word_str(topic_words, topic_word_path)

    def eval(self):
        output_prefix = self.output_prefix
        param_file_name = f"{output_prefix}/param.pt"
        if utils.file_exists(param_file_name):
            self.model.load_state_dict(torch.load(param_file_name))
        else:
            print("Model is not trained.")
            return

        self.model.eval()
        test_loader = self.data_handler.test_loader

        doc_topic_dist_lst = []
        for bows, _ in test_loader:
            doc_topic_dist = self.model.infer_doc_topic_dist(bows)
            doc_topic_dist_lst.append(doc_topic_dist)
        doc_topic_dist = np.concatenate(doc_topic_dist_lst, axis=0)

        import metrics
        # top clustering
        top_clusters = doc_topic_dist.argmax(axis=1)
        labels_true = self.data_handler.test_labels
        top_ari = metrics.get_adjusted_rand_index(labels_true, top_clusters)
        top_purity = metrics.get_purity(labels_true, top_clusters)
        top_nmi = metrics.get_normalized_mutual_info(labels_true, top_clusters)

        # k-means clustering
        from sklearn.cluster import KMeans
        kmeans_model = KMeans(n_clusters=20)
        km_clusters = kmeans_model.fit_predict(doc_topic_dist)
        km_ari = metrics.get_adjusted_rand_index(labels_true, km_clusters)
        km_purity = metrics.get_purity(labels_true, km_clusters)
        km_nmi = metrics.get_normalized_mutual_info(labels_true, km_clusters)

        topic_word_dist = self.model.get_topic_word_dist()
        td = metrics.get_topic_diversity(topic_word_dist)

        # calculate TC with palmetto
        if self.exp_settings["calculate_CV"] is True:
            vocab = self.data_handler.vocab
            topic_words = utils.get_topic_word_str(topic_word_dist, vocab,
                                                   num_top_words=self.exp_settings["num_top_words"])
            tc_info = metrics.get_palmetto_topic_coherence(topic_words, output_prefix,
                                                           palmetto_prefix=".")
            c_v, c_v_lst = tc_info["C_V"]
        else:
            c_v = None

        label_topic_mat = torch.from_numpy(self.model.get_label_topic_mat())
        d = Categorical(probs=label_topic_mat)
        entropy = d.entropy().mean()

        tc_str = f"TC: {c_v:.5f}, " if c_v is not None else ""
        test_results = [
            f"Topic Quality:",
            f"  {tc_str}TD: {td:.5f}",
            f"Clustering Performance:",
            f"  k-ARI: {km_ari:.5f}, k-Purity: {km_purity:.5f}, k-NMI: {km_nmi:.5f}",
            f"  t-ARI: {top_ari:.5f}, t-Purity: {top_purity:.5f}, t-NMI: {top_nmi:.5f}",
            f"Structure Characteristics:",
            f"  entropy: {entropy:.5f}",
        ]
        print("\n".join(test_results))
