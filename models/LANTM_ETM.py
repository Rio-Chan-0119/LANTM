import torch
import torch.nn as nn
import torch.nn.functional as F


class LANTM_ETM(nn.Module):

    def __init__(self, params):
        super().__init__()

        self.num_classes = params["num_classes"]
        self.num_topics = params["num_topics"]
        self.vocab_size = params["vocab_size"]
        self.hidden_size = params["hidden_size"]
        self.embedding_dim = params["embedding_dim"]
        self.gamma1 = params["gamma1"]
        self.gamma2 = params["gamma2"]
        self.register_buffer("label_dist", torch.from_numpy(params["label_dist"]))
        self.register_buffer("pseudo_doc_per_class", torch.from_numpy(params["pseudo_doc_per_class"]))

        prior_alpha = params["prior_alpha"] * torch.ones(self.num_topics)
        mu0 = prior_alpha.log() - prior_alpha.log().mean()
        var0 = 1.0 / prior_alpha * (1 - 2 / self.num_topics) + (1.0 / prior_alpha).sum() / self.num_topics ** 2
        self.register_buffer("mu0", mu0)
        self.register_buffer("var0", var0)

        self.fc11 = nn.Linear(self.vocab_size, self.hidden_size)
        self.fc12 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, self.num_topics)
        self.fc22 = nn.Linear(self.hidden_size, self.num_topics)
        self.fc1_dropout = nn.Dropout(params["dropout"])
        self.theta_dropout = nn.Dropout(params["dropout"])

        self.mean_bn = nn.BatchNorm1d(self.num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(self.num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(self.vocab_size)
        self.decoder_bn.weight.requires_grad = False

        if params["word_embeddings"] is None:
            word_embeddings = torch.empty((self.num_topics, self.embedding_dim))
            nn.init.trunc_normal_(word_embeddings, std=0.1)
        else:
            word_embeddings = torch.from_numpy(params["word_embeddings"])
        self.word_embeddings = nn.Parameter(F.normalize(word_embeddings))

        topic_embeddings = torch.empty((self.num_topics, self.embedding_dim))
        nn.init.trunc_normal_(topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(topic_embeddings))

        label_embeddings = torch.empty((self.num_classes, self.embedding_dim))
        nn.init.trunc_normal_(label_embeddings, std=0.1)
        self.label_embeddings = nn.Parameter(F.normalize(label_embeddings))

    def compute_loss(self, bows, labels):
        theta, kld_per_doc, label_topic_mat, L_OT = self._encode(bows, labels)
        beta = self._get_beta()

        log_recon = F.log_softmax(self.decoder_bn(theta @ beta), dim=-1)
        recon_loss = -(bows * log_recon).sum(1).mean()
        kld = kld_per_doc.mean()

        label_topic_dist = label_topic_mat / label_topic_mat.sum(1, keepdim=True)   # \lambda_{y,.} / N_y
        log_recon_class = F.log_softmax(label_topic_dist @ beta, dim=-1)
        L_sem = -(self.pseudo_doc_per_class * log_recon_class).sum(1).mean()

        loss = (recon_loss + kld +
                self.gamma1 * L_OT +
                self.gamma2 * L_sem)

        result_dict = {
            "loss": loss,
            "recon_loss": recon_loss,
            "kld": kld,
            "L_OT": L_OT,
            "L_sem": L_sem,
        }
        return result_dict

    # infer doc-topic dists in the absence of labels
    def infer_doc_topic_dist(self, bows):
        batch_size = bows.shape[0]
        possible_doc_topic_dist = []
        elbos_under_each_label = []

        # consider all possible labels
        for i in range(self.num_classes):
            labels = torch.full((batch_size,), i, device=bows.device, dtype=torch.int64)
            theta, kld_per_doc, _, _ = self._encode(bows, labels)
            possible_doc_topic_dist.append(theta)

            beta = self._get_beta()
            log_recon = F.log_softmax(self.decoder_bn(theta @ beta), dim=-1)
            recon_loss_per_doc = -(bows * log_recon).sum(1)

            elbo_per_doc = -(recon_loss_per_doc.add(kld_per_doc))
            elbos_under_each_label.append(elbo_per_doc)

        possible_doc_topic_dist = torch.stack(possible_doc_topic_dist, dim=0)    # C * |batch| * K
        elbos_under_each_label = torch.stack(elbos_under_each_label, dim=0)      # C * |batch|
        argmax_label = elbos_under_each_label.argmax(dim=0)                      # |batch|
        doc_topic_dist = possible_doc_topic_dist[argmax_label, torch.arange(batch_size), :]

        return doc_topic_dist.detach().cpu().numpy()

    def get_topic_word_dist(self):
        return self._get_beta().detach().cpu().numpy()

    def get_label_topic_mat(self):
        label_topic_mat = self._get_label_topic_mat()[0]
        return label_topic_mat.detach().cpu().numpy()

    def _get_beta(self):
        beta = self.topic_embeddings @ self.word_embeddings.T
        return beta

    def _get_label_topic_mat(self):
        logits = -(self.label_embeddings @ self.topic_embeddings.T)
        cost = (1 + logits.exp()).log()     # Softplus
        a = self.label_dist
        b = torch.ones((self.num_topics,), device=a.device) / self.num_topics
        plan, sinkhorn_distance = self._optimal_transport(a, b, cost)
        label_topic_mat = plan * self.num_topics
        return label_topic_mat, sinkhorn_distance

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) if self.training else torch.zeros_like(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _encode(self, bows, labels):
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(torch.float)

        label_topic_mat, sinkhorn_distance = self._get_label_topic_mat()
        topic_masks_per_doc = labels_onehot @ label_topic_mat + 1e-20     # avoid overflow

        h1 = F.softplus(self.fc11(bows))
        h1 = F.softplus(self.fc12(h1))
        h1 = self.fc1_dropout(h1)
        mu = self.mean_bn(self.fc21(h1))
        logvar = self.logvar_bn(self.fc22(h1))
        z = self._reparameterize(mu + topic_masks_per_doc.log(), logvar)
        theta = F.softmax(z, dim=1)
        kld_per_doc = self._compute_kl_divergence_per_doc(mu, logvar)

        return theta, kld_per_doc, label_topic_mat, sinkhorn_distance

    # We do not perform mean reduction
    # for the convenience of computing the predicted labels
    # in the absence of labels. (Eq. (14))
    def _compute_kl_divergence_per_doc(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var0
        diff = mu - self.mu0
        diff_term = diff * diff / self.var0
        logvar_division = self.var0.log() - logvar
        num_topics = mu.shape[1]
        kld_per_doc = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - num_topics)
        return kld_per_doc

    @staticmethod
    def _optimal_transport(a, b, cost):
        max_iter = 1000
        stop_threshold = .5e-2
        eps = 1e-16
        sinkhorn_lambda = 20

        u = torch.ones_like(a, dtype=torch.float) / a.shape[0]
        v = None
        K = torch.exp(-cost * sinkhorn_lambda)

        err = 1.
        cpt = 0
        while err > stop_threshold and cpt < max_iter:
            v = b / (K.T @ u + eps)
            u = a / (K @ v + eps)
            cpt += 1
            if cpt % 50 == 1:
                bb = v * (K.T @ u)
                err = torch.norm((bb - b).abs().sum(0), p=float("inf"))

        plan = u.unsqueeze(-1) * (K * v)
        sinkhorn_distance = (plan * cost).sum()
        return plan, sinkhorn_distance
