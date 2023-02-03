import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import ignite.distributed as idist
import torch.optim
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (
    DeterministicEngine,
    Engine,
    Events,
    _prepare_batch,
    create_supervised_evaluator,
)
from ignite.metrics import Loss
from ignite.utils import convert_tensor, setup_logger
from module import GreedyGenerator
from py_config_runner import get_params
from py_config_runner.utils import set_seed
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_vocab
from valid_metrices.bleu_metrice import BLEU4, bleu_output_transform
from script.optimizer import AdamW
from valid_metrices.compute_scores import eval_accuracies

__all__ = ["run_summary"]

valid_bleu = None


def params2str(params):
    if params is None:
        return ""
    return "|".join([" " + str(key) + ": " + str(value) for key, value in params.items()])


def get_model(config):
    model = config.model(
        config.src_vocab.size(),
        config.tgt_vocab.size(),
        config.hidden_size,
        config.num_heads,
        config.num_layers,
        config.sbm_layers,
        config.use_pegen,
        config.dim_feed_forward,
        config.dropout,
        config.pe_dim,
        config.pegen_dim,
        config.sbm_enc_dim,
        config.clusters,
        config.full_att,
        config.checkpoint,
        config.max_src_len,
    )

    return model


def _graph_prepare_batch(batch, device=None, non_blocking: bool = False):
    x, y = batch
    return (
        x.to(device),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def initialize(config, train_data_set_len):
    model = get_model(config)
    print("src_vocab size", config.src_vocab.size())
    print("tgt_vocab size", config.tgt_vocab.size())
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num_param: {num_param}")
    model = model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, correct_bias=False)
    scheduler = None
    if config.multi_gpu:
        model = idist.auto_model(model)
        optimizer = idist.auto_optim(optimizer)
    criterion = config.criterion
    return model, optimizer, criterion, scheduler


def create_custom_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    max_grad_norm: float,
    sw: float,
    loss_fn: Union[Callable, torch.nn.Module],
    amp_scaler=torch.cuda.amp.GradScaler,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
) -> Engine:
    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred, sparsity, src_pe, graphs, attns = model(x)
        loss = loss_fn(y_pred, y)
        amp_scaler.scale(loss + sw * sparsity).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()
        return output_transform(x, y, y_pred, loss)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)

    return trainer


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed} - {tag} metrics:\n {metrics_output}")


def get_dataflow(config):
    train_data_set = config.data_set(config, "train")
    eval_data_set = config.data_set(config, "dev")
    return train_data_set, eval_data_set


def get_data_loader(config, is_train, data_set):
    batch_size = config.batch_size
    is_shuffle = True if is_train else False
    num_workers = config.num_threads if is_train else 0
    if config.multi_gpu:
        data_loader = idist.auto_dataloader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=is_shuffle,
            collate_fn=data_set.collect_fn,
            num_workers=num_workers,
            pin_memory="cuda" in idist.device().type,
        )
    else:
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=is_shuffle,
            collate_fn=data_set.collect_fn,
            num_workers=num_workers,
        )
    return data_loader


def training(local_rank, config=None, **kwargs):
    torch.cuda.empty_cache()
    logger = kwargs["logger"]

    set_seed(config.seed + local_rank)
    train_data_set, eval_data_set = get_dataflow(config)
    print("data generation done")
    train_loader = get_data_loader(config, is_train=True, data_set=train_data_set)
    valid_loader = get_data_loader(config, is_train=False, data_set=eval_data_set)
    print("data loading is finished")
    model, optimizer, criterion, scheduler = initialize(config, train_data_set.__len__())

    amp_scaler = torch.cuda.amp.GradScaler()
    trainer = create_custom_trainer(
        model,
        optimizer,
        scheduler,
        max_grad_norm=1.0,
        sw=config.sw,
        loss_fn=criterion,
        amp_scaler=amp_scaler,
        device=config.device,
        prepare_batch=_graph_prepare_batch,
    )
    metrics_valid = {"bleu": BLEU4()}
    greedy_generator = GreedyGenerator(model, config.max_tgt_len, multi_gpu=config.multi_gpu)
    evaluator = create_supervised_evaluator(
        greedy_generator,
        metrics=metrics_valid,
        device=config.device,
        prepare_batch=_graph_prepare_batch,
        output_transform=lambda x, y, y_pred: bleu_output_transform((y_pred, y), config.tgt_vocab.i2w),
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=getattr(config, "val_interval", 1)) | Events.COMPLETED)
    def run_validation():
        epoch = trainer.state.epoch
        states = evaluator.run(valid_loader)
        log_metrics(logger, epoch, states.times["COMPLETED"], "Test", states.metrics)

    @trainer.on(Events.EPOCH_COMPLETED(every=getattr(config, "save_interval", 1)) | Events.COMPLETED)
    def save_state_dict():
        if idist.get_local_rank() == 0:
            epoch = trainer.state.epoch
            torch.save(model.state_dict(), str(config.output_path) + f"/checkpoint_{epoch}.pt")

    common.save_best_model_by_val_score(
        config.output_path.as_posix(),
        evaluator,
        model=model,
        metric_name="bleu",
        n_saved=1,
        trainer=trainer,
        tag="val",
    )

    if idist.get_local_rank() == 0:
        ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})
        if not config.fast_mod:
            if "tensorboard" in config.logger:
                from ignite.contrib.handlers import tensorboard_logger

                tb_logger = common.setup_tb_logging(
                    config.output_path.as_posix(),
                    trainer,
                    optimizer,
                    evaluators={
                        "validation bleu": evaluator,
                    },
                )

                tb_logger.attach(
                    trainer,
                    log_handler=tensorboard_logger.OutputHandler(
                        tag="training",
                        output_transform=lambda loss: {"loss": loss},
                        metric_names="all",
                    ),
                    event_name=Events.ITERATION_COMPLETED(every=50),
                )

    print(f"max epochs: {config.num_epochs}")
    trainer.run(train_loader, max_epochs=config.num_epochs)
    global valid_bleu
    valid_bleu = evaluator.state.metrics["bleu"]
    test(idist.get_local_rank(), config, logger)

    if idist.get_local_rank() == 0:
        if "tensorboard" in config.logger:
            tb_logger.close()


def test(local_rank, config, logger):
    if local_rank == 0:
        torch.cuda.empty_cache()
        output_path = config.output_path.as_posix()
        load_epoch_path = ""
        for file in os.listdir(output_path):
            if file.endswith(".pt") and "best_model" in file:
                testfile = file
        if config.testfile != "":
            testfile = config.testfile
        logger.info("load " + testfile)
        sub_dir = os.path.join(output_path, testfile)
        load_epoch_path = sub_dir
        if load_epoch_path == "":
            raise Exception("Can not find the save model. ")

        logger.info("*" * 5 + "Start TEST" + "*" * 5)
        if torch.cuda.is_available():
            checkpoint = torch.load(load_epoch_path)
        else:
            checkpoint = torch.load(load_epoch_path, map_location="cpu")
        config.checkpoint = checkpoint
        model = get_model(config)
        model.eval()
        model = model.to(config.device)
        greedy_generator = GreedyGenerator(model, config.max_tgt_len, multi_gpu=False)
        test_data_set = config.data_set(config, "test")
        test_loader = DataLoader(
            dataset=test_data_set,
            batch_size=config.batch_size // len(config.g.split(",")),
            shuffle=False,
            collate_fn=test_data_set.collect_fn,
        )

        _hypothesises = []
        _references = []

        for batch in tqdm(test_loader):
            x, y = _graph_prepare_batch(batch, device=config.device)
            y_pred = greedy_generator(x)
            hypothesises, references = bleu_output_transform((y_pred, y), config.tgt_vocab.i2w)
            _hypothesises.extend(hypothesises)
            _references.extend(references)

        hypothesises = {index: [" ".join(value)] for index, value in enumerate(_hypothesises)}
        references = {index: [" ".join(value)] for index, value in enumerate(_references)}
        bleu, rougle_l, meteor, ind_bleu, ind_rouge = eval_accuracies(hypothesises, references)

        outputs = []
        for i in hypothesises.keys():
            outputs.append(
                {
                    "predict": hypothesises[i][0],
                    "true": references[i][0],
                    "bleu": ind_bleu[i],
                    "rouge": ind_rouge[i],
                }
            )

        file_name = "/predict_results_bleu_{:.2f}_rouge_{:.2f}_meteor_{:.2f}.json".format(bleu, rougle_l, meteor)
        with open(config.output_path.as_posix() + file_name, "w") as f:
            json.dump(outputs, f)
        logger.info(f"bleu: {bleu}, rouge: {rougle_l} meteor: {meteor}")


def run_summary(config, hype_params=None):
    if hype_params is not None:
        config.__internal_config_object_data_dict__.update(hype_params)

    if config.fast_mod:
        config.use_clearml = False

    (
        config.src_vocab,
        config.tgt_vocab,
        # config.ident_vocab,
    ) = load_vocab(config.data_dir, config.data_type)

    logger = setup_logger(name="AST Transformer Training", distributed_rank=idist.get_local_rank())
    logger.info("Hype-Params: " + params2str(hype_params))

    config.output_path = Path("./outputs/" + config.project_name + "/" + config.task_name + params2str(hype_params))
    config.output_path_str = config.output_path.as_posix()
    if not config.is_test:
        if config.multi_gpu:
            with idist.Parallel(backend="nccl", master_port=2224) as parallel:
                try:
                    parallel.run(training, config, logger=logger, hype_params=hype_params)
                except KeyboardInterrupt:
                    logger.info("Catched KeyboardInterrupt -> exit")
                except Exception as e:  # noqa
                    logger.exception("")
                    raise e
        else:
            print(f"rank: {idist.get_local_rank()}")
            training(idist.get_local_rank(), config, logger=logger, hype_params=hype_params)
    else:
        print(f"rank: {idist.get_local_rank()}")
        test(idist.get_local_rank(), config, logger=logger)

    global valid_bleu
    return valid_bleu
