from torch import optim
from utils.utils import cosine_scheduler


def make_optimizer(model, config):
    optimizer_class = getattr(optim, config.OPTIMIZATION.OPTIMIZER)
    optimizer_params = {
        'lr': config.OPTIMIZATION.LR,
        'weight_decay': config.OPTIMIZATION.WEIGHT_DECAY
    }
    if config.OPTIMIZATION.OPTIMIZER == 'SGD':
        optimizer_params['momentum'] = config.OPTIMIZATION.SGD_MOMENTUM
        optimizer_params['dampening'] = config.OPTIMIZATION.SGD_DAMPENING
    optimizer = optimizer_class(
        model.parameters(),
        **optimizer_params
    )
    return optimizer


def make_scheduler(config, total_iters):
    return cosine_scheduler(config.OPTIMIZATION.LR, config.OPTIMIZATION.LR / 100, total_iters, total_iters // 20)
