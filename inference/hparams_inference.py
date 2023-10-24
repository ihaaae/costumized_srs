import torch.nn as nn
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from custom_model import Xvector, Classifier
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.utils.parameter_transfer import Pretrainer

# Pretrain folders
pretrained_path = "/root/best_model/"

# Model parameters
n_mels = 23
sample_rate = 16000
n_classes = 28
emb_dim = 512

# Feature extraction
compute_features = Fbank(n_mels=n_mels)

# Mean and std normalization of the input features
mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)

# Embedding model
embedding_model = Xvector(
    in_channels=n_mels,
    activation=nn.LeakyReLU,
    tdnn_blocks=5,
    tdnn_channels=[512, 512, 512, 512, 1500],
    tdnn_kernel_sizes=[5, 3, 3, 1, 1],
    tdnn_dilations=[1, 2, 3, 1, 1],
    lin_neurons=emb_dim
)

# Classifier
classifier = Classifier(
    input_shape=[None, None, emb_dim],
    activation=nn.LeakyReLU,
    lin_blocks=1,
    lin_neurons=emb_dim,
    out_neurons=n_classes
)

# Label encoder
label_encoder = CategoricalEncoder()

# Modules
modules = {
    "compute_features": compute_features,
    "embedding_model": embedding_model,
    "classifier": classifier,
    "mean_var_norm": mean_var_norm
}

# Pretrainer
pretrainer = Pretrainer(
    loadables={
        "embedding_model": embedding_model,
        "classifier": classifier,
        "label_encoder": label_encoder
    },
    paths={
        "embedding_model": f"{pretrained_path}/embedding_model.ckpt",
        "classifier": f"{pretrained_path}/classifier.ckpt",
        "label_encoder": f"{pretrained_path}/label_encoder.txt"
    }
)

hparams = {'pretrained_path': pretrained_path, 'n_mels':n_mels, "sample_rate":sample_rate,
           'n_classes': n_classes, 'emb_dim': emb_dim, 'compute_features':compute_features,
           'mean_var_norm': mean_var_norm, 'embedding_model': embedding_model, 'classifier': classifier,
           'label_encoder': label_encoder, 'modules': modules, 'pretrainer': pretrainer}