import re
import argparse
import logging
import os
import torch
from pathlib import Path
from typing import Any, Dict
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
try:
    from pytorch_lightning.callbacks import LearningRateMonitor
except ImportError:
    from pytorch_lightning.callbacks import LearningRateLogger as LearningRateMonitor


from transformers import (
    AdamW,
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MODEL_MODES = {
    "base": GPT2LMHeadModel,
    "graph2text": GPT2LMHeadModel,
    "language-modeling": GPT2LMHeadModel,
}

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


def clean_predictions(predictions):
    """
    Cleans predictions by removing repetitive phrases, correcting formatting,
    and ensuring proper sentence structure.

    Args:
        predictions (list or str): Either a list of prediction strings or a single string.

    Returns:
        list or str: Cleaned predictions in the same format as the input.
    """
    def clean_single(prediction):
        if not prediction:
            return ""

        # Use a safe regex for repeated words or phrases
        prediction = re.sub(r'\b(\w+)( \1\b)+', r'\1', prediction)

        # Remove excessive spaces
        prediction = re.sub(r'\s+', ' ', prediction)

        # Remove stray punctuations
        prediction = re.sub(r'\s+([.,;!?])', r'\1', prediction)

        # Capitalize the first letter of the sentence if not already done
        prediction = prediction.strip()
        if prediction and not prediction[0].isupper():
            prediction = prediction[0].upper() + prediction[1:]

        return prediction

    # If input is a list, clean each string in the list
    if isinstance(predictions, list):
        return [clean_single(pred) for pred in predictions]

    # If input is a single string, clean it directly
    return clean_single(predictions)


class BaseTransformer(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace, **kwargs):
        """Initialize the model, tokenizer, and config."""
        super().__init__()
        self.save_hyperparameters(hparams)
        self.output_dir = Path(self.hparams.output_dir)

        # Config
        self.config = AutoConfig.from_pretrained(
            self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            **kwargs,
        )

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Model
        self.model = GPT2LMHeadModel.from_pretrained(
            self.hparams.model_name_or_path, config=self.config
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def configure_optimizers(self):
        """Prepare optimizer and scheduler."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        total_steps = (
            len(self.train_dataloader().dataset)
            // (self.hparams.train_batch_size * self.trainer.accumulate_grad_batches)
            * self.hparams.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]



    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if isinstance(outputs, tuple):
            loss = outputs[0]
            logits = outputs[1]
        else:
            loss = outputs.loss
            logits = outputs.logits
        return loss, logits


    def get_dataloader(self, type_path, batch_size, shuffle=False):
        """
        Replace this placeholder with an implementation for your graph-to-text dataset.
        """
        raise NotImplementedError("You must implement this for your task.")

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            # Ensure that the train dataloader is available
            train_loader = self.train_dataloader()
            # Total steps = number of batches * number of epochs
            self.total_steps = (
                len(train_loader.dataset) // (self.hparams.train_batch_size * max(1, self.trainer.accumulate_grad_batches))
            ) * self.hparams.num_train_epochs

    def training_step(self, batch, batch_idx):
        # Call forward and unpack loss and logits
        loss, _ = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Log training loss
        if self.trainer is not None and self.trainer.logger is not None:
            self.trainer.logger.log_metrics({"train_loss": loss.item()}, step=self.global_step)

        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        # Call the forward method and unpack loss and logits
        loss, logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Decode predictions
        preds = self.tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
        preds = clean_predictions(preds)

        # Decode targets, handling -100 padding
        filtered_labels = torch.where(
            batch["labels"] != -100,
            batch["labels"],
            torch.tensor(self.tokenizer.pad_token_id).to(batch["labels"].device),
        )
        target = self.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

        return {
            "loss": loss,
            "preds": preds,
            "target": target,
        }

    def test_step(self, batch, batch_idx):
        # Forward pass
        loss, logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Decode predictions
        preds = self.tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
        preds = clean_predictions(preds)

        # Decode targets
        filtered_labels = torch.where(
            batch["labels"] != -100,
            batch["labels"],
            torch.tensor(self.tokenizer.pad_token_id).to(batch["labels"].device),
        )
        target = self.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

        return {"test_loss": loss, "preds": preds, "target": target}


    @staticmethod
    def add_model_specific_args(parser, root_dir):
        """
        Add model-specific arguments to the parser. This method ensures there are no argument conflicts.
        """
        # Helper function to remove existing conflicting arguments
        def remove_conflicting_args(arg_name):
            for action in parser._actions:
                if arg_name in action.option_strings:
                    parser._handle_conflict_resolve(None, [(arg_name, action)])

        # Safely add arguments
        def add_argument_safe(arg_name, **kwargs):
            remove_conflicting_args(f"--{arg_name}")
            parser.add_argument(f"--{arg_name}", **kwargs)

        # Add arguments without conflicts
        add_argument_safe("model_name_or_path", default="gpt2", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
        add_argument_safe("config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
        add_argument_safe("tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
        add_argument_safe("cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from s3")
        add_argument_safe("learning_rate", default=5e-5, type=float, help="Initial learning rate for AdamW.")
        add_argument_safe("lr_scheduler", default="linear", choices=arg_to_scheduler_choices,
                        metavar=arg_to_scheduler_metavar, type=str, help="Learning rate scheduler type.")
        add_argument_safe("weight_decay", default=0.0, type=float, help="Weight decay.")
        add_argument_safe("adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        add_argument_safe("warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        add_argument_safe("num_workers", default=4, type=int, help="Number of workers for DataLoader.")
        add_argument_safe("train_batch_size", default=32, type=int, help="Batch size for training.")
        add_argument_safe("eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
        add_argument_safe("max_source_length", default=384, type=int, help="Maximum input sequence length.")
        add_argument_safe("max_target_length", default=384, type=int, help="Maximum output sequence length.")
        add_argument_safe("eval_max_gen_length", default=384, type=int, help="Maximum generation length for evaluation.")
        add_argument_safe("eval_beams", default=3, type=int, help="Number of beams to use for evaluation.")
        add_argument_safe("data_dir", default=None, type=str, help="Path to the data directory.")
        add_argument_safe("num_train_epochs", default=1, type=int, help="Number of training epochs.")
        add_argument_safe("task", default="graph2text", type=str, help="Task name.")
        add_argument_safe("early_stopping_patience", default=-1, type=int, help="Early stopping patience.")
        add_argument_safe("output_dir", default=None, type=str, required=True, help="Output directory path.")
        add_argument_safe("do_train", action="store_true", help="Whether to run training.")
        add_argument_safe("do_predict", action="store_true", help="Whether to run predictions.")
        add_argument_safe("val_max_target_length", default=384, type=int, help="Maximum target length for validation.")
        add_argument_safe("test_max_target_length", default=384, type=int, help="Maximum target length for testing.")
        add_argument_safe("save_top_k", default=1, type=int, help="Number of top checkpoints to save based on validation performance.")
        add_argument_safe("test_checkpoint_path", default=None, type=str, help="Path to the model checkpoint to use during testing.")


        return parser



def generic_train(
    model: BaseTransformer,
    args: argparse.Namespace,
    early_stopping_callback=False,
    logger=True,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs,
):
    pl.seed_everything(args.seed)

    # Initialize the model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    # Checkpoint Callback
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename="checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

    # Logging Callback
    if logging_callback is None:
        logging_callback = LoggingCallback()

    # Training Parameters
    train_params = {}
    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level

    if args.gpus > 1:
        train_params["strategy"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches

    # Learning Rate Logger
    lr_logger = LearningRateMonitor(logging_interval="step")

    # Trainer Initialization
    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary="full",
        callbacks=[logging_callback, lr_logger] + extra_callbacks,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        num_sanity_val_steps=4,
        max_epochs=args.num_train_epochs,
        **train_params,
    )


    trainer.lr_logger = lr_logger

    if args.do_train:
        trainer.fit(model)

    return trainer


def add_generic_args(parser, root_dir):
    """
    Add generic PyTorch Lightning Trainer arguments to the parser.
    """
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit precision.",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." 
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int, help="Number of TPU cores to use.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")  # Add this line
    return parser
