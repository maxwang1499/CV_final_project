import os 
import fire
from pytorch_lightning import Trainer

from util import init_exp_folder, Args
from util import constants as C
from data import dataset_split
from lightning import (get_task,
                       load_task,
                       get_ckpt_callback, 
                       get_early_stop_callback,
                       get_logger)

def train(dataset_folder='./',
          save_dir="./sandbox",
          exp_name="MultitaskBaseline-ResNet101-large",
          model="ResNet101",
          products="naip-rgb",
          image_size=720,
          crop_size=None,
          augmentation="none",
          gpus=1,
          pretrained=True,
          accelerator=None,
          logger_type='test_tube',
          gradient_clip_val=0.5,
          max_epochs=20,
          batch_size=8,
          num_workers=2,
          lr=0.02,
          patience=10,
          stochastic_weight_avg=True,
          limit_train_batches=1.0,
          tb_path="./sandbox/tb",
          loss_fn="BCE",
          weights_summary=None,
          task_type="all"
          ):
    """
    Run the training experiment.

    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        model: Model name
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        pretrained: Whether or not to use the pretrained model
        accelerator: Distributed computing mode
        logger_type: 'wandb' or 'test_tube'
        gradient_clip_val:  Clip value of gradient norm
        limit_train_batches: Proportion of training data to use
        max_epochs: Max number of epochs
        batch_size: Batch size
        
        patience: number of epochs with no improvement after
                  which training will be stopped.
        stochastic_weight_avg: Whether to use stochastic weight averaging.
        tb_path: Path to global tb folder
        loss_fn: Loss function to use
        weights_summary: Prints a summary of the weights when training begins.

    Returns: None

    """
    args = Args(locals())
    if task_type == 'all':
        args['num_classes'] = len(C.class_labels_list)
    else:
        if task_type not in C.class_labels_list:
            raise Exception('Invalid task type.')
        args['num_classes'] = 1
    
    if products not in C.valid_products:
        raise Exception('Invalid product type.')

    task = get_task(task_type, args)
    init_exp_folder(args)
    trainer = Trainer(gpus=gpus,
                      accelerator=accelerator,
                      logger=get_logger(logger_type, save_dir, exp_name),
                      callbacks=[get_early_stop_callback(patience),
                                 get_ckpt_callback(save_dir, exp_name)],
                      weights_save_path=os.path.join(save_dir, exp_name),
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      weights_summary=weights_summary,
                      stochastic_weight_avg=stochastic_weight_avg,
                      max_epochs=max_epochs)
    trainer.fit(task)


def validate(ckpt_path,
         gpus=1,
         **kwargs):
    """
    Run the testing experiment.

    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
    Returns: None

    """
    task = load_task(ckpt_path, **kwargs)
    trainer = Trainer(gpus=gpus)
    trainer.validate(task)


def test(ckpt_path,
         gpus=1,
         **kwargs):
    """
    Run the testing experiment.

    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
    Returns: None

    """
    task = load_task(ckpt_path, **kwargs)
    trainer = Trainer(gpus=gpus)
    trainer.test(task)


if __name__ == "__main__":
    fire.Fire()
