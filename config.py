import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("ST_LLM_DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser()
CHECKPOINT_ROOT = Path(os.environ.get("ST_LLM_CHECKPOINT_ROOT", str(PROJECT_ROOT / "checkpoints"))).expanduser()


class Config:
    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 20
    LR = 0.01
    LR_MIN = 0.0001
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    SEEDS = (32, 42, 52, 62, 72, 82, 92, 102, 112, 122)
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128
    NUM_CLASSES = 14
    PMI_WINDOW_SIZE = 5
    PAD_TRUNC_DIGIT = 256
    FLOW_PAD_TRUNC_LENGTH = 50
    BYTE_PAD_TRUNC_LENGTH = 150
    HEADER_BYTE_PAD_TRUNC_LENGTH = 40
    ANOMALOUS_FLOW_THRESHOLD = 10000


class ISCXVPNConfig(Config):
    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 2
    MAX_EPOCH = 180
    LR = 5e-05
    LR_MIN = 2e-06
    LABEL_SMOOTHING = 0.1
    WEIGHT_DECAY = 1e-05
    WARM_UP = 0.26
    SEED = 66
    DROPOUT = 0.35
    DOWNSTREAM_DROPOUT = 0.35
    EMBEDDING_SIZE = 64
    H_FEATS = 128


class ISCXNonVPNConfig(Config):
    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5
    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 120
    LR = 0.01
    LR_MIN = 1e-05
    LABEL_SMOOTHING = 0.01
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.1
    DOWNSTREAM_DROPOUT = 0.15
    EMBEDDING_SIZE = 64
    H_FEATS = 128


class ISCXTorConfig(Config):
    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 100
    LR = 0.01
    LR_MIN = 0.0001
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.0
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128


class ISCXNonTorConfig(Config):
    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5
    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 120
    LR = 0.01
    LR_MIN = 0.0001
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.05
    DOWNSTREAM_DROPOUT = 0.1
    EMBEDDING_SIZE = 64
    H_FEATS = 128


class USTCTFC2016Config(Config):
    NUM_CLASSES = 20
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 0
    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 3e-4
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0.05
    WEIGHT_DECAY = 1e-4
    WARM_UP = 0.1
    SEED = 32
    SEEDS = Config.SEEDS
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128
    PMI_WINDOW_SIZE = 5
    PAD_TRUNC_DIGIT = 256
    FLOW_PAD_TRUNC_LENGTH = 50
    BYTE_PAD_TRUNC_LENGTH = 150
    HEADER_BYTE_PAD_TRUNC_LENGTH = 40
    ANOMALOUS_FLOW_THRESHOLD = 10000


USTC_TFC2016_CLASSES = (
    "Benign_BitTorrent", "Benign_Facetime", "Benign_FTP", "Benign_Gmail",
    "Benign_MySQL", "Benign_Outlook", "Benign_Skype", "Benign_SMB",
    "Benign_Weibo", "Benign_WorldOfWarcraft", "Malware_Cridex", "Malware_Geodo",
    "Malware_Htbot", "Malware_Miuref", "Malware_Neris", "Malware_Nsis-ay",
    "Malware_Shifu", "Malware_Tinba", "Malware_Virut", "Malware_Zeus",
)


DATASET_CONFIGS = {
    "iscx-vpn": ISCXVPNConfig,
    "iscx-nonvpn": ISCXNonVPNConfig,
    "iscx-tor": ISCXTorConfig,
    "iscx-nontor": ISCXNonTorConfig,
    "ustc-tfc2016": USTCTFC2016Config,
}


def _configure_paths(cls, directory, categories):
    base = DATA_ROOT / directory
    cls.CLASS_NAMES = tuple(categories)
    cls.DIR_PATH_DICT = {i: str(base / name) for i, name in enumerate(categories)}
    for attribute, filename in {
        "TRAIN_DATA": "train.npz", "TEST_DATA": "test.npz",
        "HEADER_TRAIN_DATA": "header_train.npz", "HEADER_TEST_DATA": "header_test.npz",
        "TRAIN_GRAPH_DATA": "train_graph.dgl", "TEST_GRAPH_DATA": "test_graph.dgl",
        "HEADER_TRAIN_GRAPH_DATA": "header_train_graph.dgl",
        "HEADER_TEST_GRAPH_DATA": "header_test_graph.dgl",
    }.items():
        setattr(cls, attribute, str(base / filename))
    cls.MIX_MODEL_CHECKPOINT = str(CHECKPOINT_ROOT / ("mix_model_" + directory.lower().replace("-", "_") + ".pth"))


_configure_paths(ISCXVPNConfig, "ISCX-VPN", ("Chat", "Email", "File", "P2P", "Streaming", "VoIP"))
_configure_paths(ISCXNonVPNConfig, "ISCX-NonVPN", ("Chat", "Email", "File", "Streaming", "Video", "VoIP"))
_configure_paths(ISCXTorConfig, "ISCX-Tor", ("Audio-Streaming", "Browsing", "Chat", "File", "Mail", "P2P", "Video-Streaming", "VoIP"))
_configure_paths(ISCXNonTorConfig, "ISCX-NonTor", ("Audio", "Browsing", "Chat", "Email", "FTP", "P2P", "Video", "VoIP"))
_configure_paths(USTCTFC2016Config, "USTC-TFC2016", USTC_TFC2016_CLASSES)


def get_config(dataset):
    if dataset not in DATASET_CONFIGS:
        raise ValueError("Unknown dataset: " + dataset)
    return DATASET_CONFIGS[dataset]()
