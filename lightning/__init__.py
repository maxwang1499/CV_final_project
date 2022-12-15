import torch

from .classification import MultiTaskClassificationTask, SingleTaskClassificationTask
from .util import get_ckpt_callback, get_early_stop_callback
from .util import get_logger
from util import constants as C

def get_task(task_type, args):
    if task_type == 'all':
        return MultiTaskClassificationTask(args)
    if task_type not in C.class_labels_list:
        raise Exception('Invalid task type.')
    return SingleTaskClassificationTask(args)

def load_task(ckpt_path, **kwargs):
    args = torch.load(ckpt_path, map_location='cpu')['hyper_parameters']
    task_type = args['task_type']
    if task_type == 'all':
        return MultiTaskClassificationTask.load_from_checkpoint(ckpt_path)
    if task_type not in C.class_labels_list:
        raise Exception('Invalid task type.')
    return SingleTaskClassificationTask.load_from_checkpoint(ckpt_path)
