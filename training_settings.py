import math


def learning_rate_factor(step, total_steps, lr, lr_min, warmup_ratio):
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 < lr_min <= lr:
        raise ValueError("Learning rates must satisfy 0 < LR_MIN <= LR")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("WARM_UP must be in [0, 1)")
    if total_steps == 1:
        return 1.0
    step = max(0, min(step, total_steps - 1))
    floor = lr_min / lr
    warmup_steps = min(total_steps - 2, int(total_steps * warmup_ratio))
    if warmup_steps and step < warmup_steps:
        return floor + (1.0 - floor) * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - 1 - warmup_steps)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def training_settings(config):
    names = (
        "BATCH_SIZE", "MAX_EPOCH", "LR", "LR_MIN", "LABEL_SMOOTHING",
        "WEIGHT_DECAY", "WARM_UP", "SEED", "SEEDS", "DROPOUT",
        "CLASSIFIER_DROPOUT", "EMBEDDING_SIZE", "H_FEATS", "NUM_CLASSES",
        "FLOW_PAD_TRUNC_LENGTH", "BYTE_PAD_TRUNC_LENGTH",
        "HEADER_BYTE_PAD_TRUNC_LENGTH", "PMI_WINDOW_SIZE", "GPT_LAYERS", "HDF_U",
        "MAX_SEG_PER_CLASS", "NUM_WORKERS",
    )
    result = {name: getattr(config, name) for name in names}
    result.update({"optimizer": "Adam", "scheduler": "linear_warmup_cosine_decay",
                   "model_version": "llm4etc_paired_v2", "pooling": "temporal_mean"})
    return result
