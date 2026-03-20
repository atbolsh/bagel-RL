"""Training module for tool use models."""
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    TrainerCallback,
)

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOTrainer, SFTTrainer
from trl import DPOConfig, SFTConfig
from torch.utils.tensorboard import SummaryWriter
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from torch.utils.data import DataLoader

from ..tools.executor import ToolExecutor
from ..data.data_generator import build_tool_system_prompt
from ..data.prompt_swallowing_data import (
    prepare_prompt_swallowing_datasets,
    prompt_swallowing_collate_fn,
)
from ..utils.checkpoint_schedule import (
    logarithmic_save_decision,
    should_save_checkpoint,
)
from rich.markup import escape

logger = logging.getLogger(__name__)


class LogarithmicSaveCallback(TrainerCallback):
    """HF Trainer callback that implements logarithmic checkpoint saving.

    When attached to a Trainer whose built-in ``save_strategy`` has been set
    to ``"no"``, this callback takes over checkpointing entirely.
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self._last_temp_ckpt = None
        self._current_is_permanent = True

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        should_save, is_permanent = logarithmic_save_decision(step)
        if should_save:
            control.should_save = True
            self._current_is_permanent = is_permanent
        return control

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        tag = "permanent" if self._current_is_permanent else "temporary"
        logger.info(f"Logarithmic checkpoint at step {step} ({tag})")

        if self._last_temp_ckpt is not None and self._last_temp_ckpt.exists():
            shutil.rmtree(self._last_temp_ckpt)
            logger.info(f"Deleted rolling checkpoint {self._last_temp_ckpt}")

        self._last_temp_ckpt = (
            None if self._current_is_permanent else ckpt_dir
        )
        return control


class LearningTraceCallback(TrainerCallback):
    """Writes training metrics to learning_trace.jsonl for plotting learning curves."""

    def __init__(self, output_dir):
        self.trace_path = Path(output_dir) / "learning_trace.jsonl"
        self._file = None

    def _ensure_file(self):
        if self._file is None:
            self._file = open(self.trace_path, "a")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        self._ensure_file()
        row = {"step": state.global_step}
        for k, v in logs.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = v if isinstance(v, (int, str, bool)) else str(v)
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._file is not None:
            self._file.close()
            self._file = None
        return control


class ToolTrainer:
    """Main trainer class for tool use models."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        train_dataset: Dataset, 
        eval_dataset: Dataset,
        output_dir: Path
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.is_vlm = config["model"].get("model_type") == "vlm"

        # Initialize TensorBoard logging
        self.writer = None
        if self.config.get("tensorboard", {}).get("enabled", False):
            log_dir = self.config.get("tensorboard", {}).get("log_dir", str(output_dir / "runs"))
            self.writer = SummaryWriter(log_dir=log_dir)

        # Initialize model and tokenizer/processor.
        # For prompt_swallowing the student and teacher are loaded separately
        # inside the training method to avoid holding two full copies at init.
        if self.is_vlm:
            self.processor = self._load_vlm_processor()
            self.tokenizer = self.processor.tokenizer
            # Add game action tokens for multi-task training
            self._n_added_game_tokens = 0
            if config["data"].get("strategy") == "multi_task":
                game_tokens = ["<forward>", "<clock>", "<anticlock>"]
                self._n_added_game_tokens = self.tokenizer.add_tokens(
                    game_tokens, special_tokens=True
                )
                if self._n_added_game_tokens > 0:
                    logger.info(
                        f"Added {self._n_added_game_tokens} game action tokens "
                        f"to VLM tokenizer"
                    )
            self.model = self._load_vlm_model()
        elif config["training"]["method"] == "prompt_swallowing":
            self.processor = None
            self.tokenizer = self._load_tokenizer()
            self.model = None
        else:
            self.processor = None
            self.tokenizer = self._load_tokenizer()
            self.model = self._load_model()

        # Initialize tool executor (skip when no tools)
        tools = config.get("tools", [])
        self.tool_executor = ToolExecutor(tools) if tools else None

        # Training method
        self.training_method = config["training"]["method"]

        logger.info(f"Initialized trainer with method: {self.training_method}")
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer."""

        model_name = self.config["model"]["name"]

        if "qwen3" in model_name.lower() and "toolbench" in self.config["data"]["strategy"]:
            
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config["model"].get("trust_remote_code", False),
                
            )

            return tokenizer
        
        
        else:


            model_name = self.config["model"]["name"]
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config["model"].get("trust_remote_code", False),
                
            )

        
        
            # Add special tokens for tool calls
            special_tokens = {
                "additional_special_tokens": [
                    "[TOOL_CALL]", "[/TOOL_CALL]", 
                    "[RESULT]", "[/RESULT]"
                ]
            }
            
            tokenizer.add_special_tokens(special_tokens)
            
            #Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
            return tokenizer
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load and prepare model."""
        model_config = self.config["model"]


        if self.config["training"].get("use_lora",True):

            #bits and bytes configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_compute_dtype="bfloat16"
            )
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                trust_remote_code=model_config.get("trust_remote_code", False),
                torch_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
                device_map=model_config.get("device_map", "auto"),
                quantization_config=bnb_config
                
            )
            

            model = prepare_model_for_kbit_training(model)
            # Resize embeddings for new tokens
            #model.resize_token_embeddings(len(self.tokenizer))
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config["training"].get("lora_r", 16),
                lora_alpha=self.config["training"].get("lora_alpha", 32),
                lora_dropout=self.config["training"].get("lora_dropout", 0.1),
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        else:

             # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                trust_remote_code=model_config.get("trust_remote_code", False),
                torch_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
                device_map=model_config.get("device_map", "auto"),
            )

            # Resize embeddings for new tokens
            model.resize_token_embeddings(len(self.tokenizer))
    
        return model
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Train the model based on the specified method."""
        if self.is_vlm and self.training_method == "dpo":
            self._train_dpo_vlm(resume_from_checkpoint)
        elif self.training_method == "sft":
            self._train_sft(resume_from_checkpoint)
        elif self.training_method == "dpo":
            self._train_dpo(resume_from_checkpoint)
        elif self.training_method == "stub_teacher_mode":
            self._train_stub_teacher_mode(resume_from_checkpoint)
        elif self.training_method == "prompt_swallowing":
            self._train_prompt_swallowing(resume_from_checkpoint)
        else:
            raise ValueError(
                f"Unknown training method: {self.training_method}. "
                "Supported: sft, dpo, stub_teacher_mode, prompt_swallowing"
            )
    
    def _train_sft(self, resume_from_checkpoint: Optional[str] = None):
        """Supervised fine-tuning."""
        logger.info("Starting supervised fine-tuning...")

        use_log_save = (
            self.config["training"].get("save_strategy") == "logarithmic"
        )

        training_args = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config["training"].get("num_epochs", 3),
            per_device_train_batch_size=self.config["training"].get("batch_size", 4),
            per_device_eval_batch_size=self.config["training"].get("eval_batch_size", 4),
            gradient_accumulation_steps=self.config["training"].get("gradient_accumulation_steps", 1),
            learning_rate=self.config["training"].get("learning_rate", 5e-5),
            warmup_steps=self.config["training"].get("warmup_steps", 100),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="no" if use_log_save else "steps",
            save_steps=500,
            save_total_limit=None if use_log_save else 3,
            load_best_model_at_end=not use_log_save,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_pin_memory=False,
            bf16=True,
            max_grad_norm=1.0,
            optim="adamw_torch",
            max_length=self.config["training"].get("max_length", 2048),
        )

        callbacks = [LearningTraceCallback(self.output_dir)]
        if use_log_save:
            callbacks.append(LogarithmicSaveCallback(self.output_dir))

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
    
    def _train_dpo(self, resume_from_checkpoint: Optional[str] = None):
        """Train the model using DPO."""
        logger.info("Starting DPO training")

        use_lora = self.config["training"].get("use_lora", True)
        use_log_save = (
            self.config["training"].get("save_strategy") == "logarithmic"
        )

        preference_dataset = self._create_preference_dataset()

        if not use_lora:
            logger.warning(
                "You're attempting to run DPO without LoRA which may cause NaN values. "
                "Consider enabling LoRA with 'use_lora': true in your config."
            )

        training_args = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config["training"].get("num_epochs", 3),
            per_device_train_batch_size=self.config["training"].get("batch_size", 4),
            gradient_accumulation_steps=self.config["training"].get("gradient_accumulation_steps", 1),
            learning_rate=self.config["training"].get("learning_rate", 5e-6),
            max_grad_norm=self.config["training"].get("max_grad_norm", 0.3),
            logging_steps=10,
            save_strategy="no" if use_log_save else "steps",
            save_steps=100,
            save_total_limit=None if use_log_save else 3,
            optim=self.config["training"].get("optim", "paged_adamw_8bit"),
            bf16=self.config["training"].get("bf16", True),
            fp16=self.config["training"].get("fp16", False),
            max_length=self.config["training"].get("max_length", 512),
            remove_unused_columns=False,
            beta=0.1,
            report_to="none",
        )

        callbacks = [LearningTraceCallback(self.output_dir)]
        if use_log_save:
            callbacks.append(LogarithmicSaveCallback(self.output_dir))

        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            train_dataset=preference_dataset,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        logger.info("Starting DPO training")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the trained model
        trainer.save_model(self.output_dir / "dpo_model")
        logger.info(f"Model saved to {self.output_dir / 'dpo_model'}")
    
    def _train_stub_teacher_mode(self, resume_from_checkpoint: Optional[str] = None):
        """Stub teacher mode training (Toolformer-style, incomplete)."""
        logger.info("Starting teacher mode training...")
        self._train_sft(resume_from_checkpoint)

    def _load_teacher_model(self) -> AutoModelForCausalLM:
        """Load frozen teacher model (same architecture as student, no LoRA)."""
        model_config = self.config["model"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            trust_remote_code=model_config.get("trust_remote_code", False),
            torch_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
            device_map=model_config.get("device_map", "auto"),
            quantization_config=bnb_config,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    def _load_student_from_checkpoint(self, checkpoint_dir: str) -> AutoModelForCausalLM:
        """Load student model with LoRA adapter from a saved checkpoint."""
        model_config = self.config["model"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            trust_remote_code=model_config.get("trust_remote_code", False),
            torch_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
            device_map=model_config.get("device_map", "auto"),
            quantization_config=bnb_config,
        )
        base_model = prepare_model_for_kbit_training(base_model)
        student = PeftModel.from_pretrained(
            base_model, checkpoint_dir, is_trainable=True
        )
        student.print_trainable_parameters()
        return student

    def _train_prompt_swallowing(self, resume_from_checkpoint: Optional[str] = None):
        """Prompt swallowing: student learns to match teacher logits without seeing the prompt.

        Batch mix: 50% swallowing, 25% unswallowed, 25% control.
        Loss: MSE between teacher and student logits on aligned positions.
        """
        logger.info("Starting prompt_swallowing training...")

        tc = self.config["training"]
        batch_size = tc.get("batch_size", 4)
        grad_accum = tc.get("gradient_accumulation_steps", 4)
        max_steps = tc.get("max_steps", 1000)
        lr = tc.get("learning_rate", 1e-5)
        max_length = tc.get("max_length", 512)
        logging_steps = tc.get("logging_steps", 10)
        save_steps = tc.get("save_steps", 200)
        save_strategy = tc.get("save_strategy", "steps")
        warmup_steps = tc.get("warmup_steps", 50)

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        def collate(batch):
            return prompt_swallowing_collate_fn(batch, pad_id, max_length)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            pin_memory=False,
        )

        # Load teacher (frozen base model) and student (LoRA on same base)
        # sequentially so we never hold two copies during loading.
        logger.info("Loading teacher model (frozen)...")
        teacher = self._load_teacher_model()
        device = next(teacher.parameters()).device

        logger.info("Loading student model (LoRA)...")
        if resume_from_checkpoint:
            student = self._load_student_from_checkpoint(resume_from_checkpoint)
        else:
            student = self._load_model()
        self.model = student

        if hasattr(student, "gradient_checkpointing_enable"):
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        student.train()

        optim = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=tc.get("weight_decay", 0.01),
        )

        # Determine starting step when resuming
        global_step = 0
        scheduler_restored = False
        if resume_from_checkpoint:
            ckpt_name = Path(resume_from_checkpoint).name
            try:
                global_step = int(ckpt_name.split("-")[-1])
                logger.info(f"Resuming from step {global_step}")
            except ValueError:
                pass

            state_path = Path(resume_from_checkpoint) / "training_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                optim.load_state_dict(state["optimizer"])
                global_step = state.get("global_step", global_step)
                logger.info(f"Restored optimizer state from {state_path}")
                del state

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

        if resume_from_checkpoint:
            state_path = Path(resume_from_checkpoint) / "training_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                if "scheduler" in state:
                    scheduler.load_state_dict(state["scheduler"])
                    scheduler_restored = True
                del state
            if not scheduler_restored and global_step > 0:
                scheduler.last_epoch = global_step
                for group in optim.param_groups:
                    group["lr"] = lr * lr_lambda(global_step)

        trace_path = self.output_dir / "learning_trace.jsonl"
        trace_file = open(trace_path, "a")
        _last_temp_ckpt = None
        _latest_loss_by_type = {"control": None, "unswallowed": None, "swallowing": None}

        gc_interval = max(10, logging_steps)
        optim.zero_grad()

        for epoch in range(100):
            if global_step >= max_steps:
                break
            for batch in train_loader:
                teacher_ids = batch["teacher_input_ids"].to(device)
                student_ids = batch["student_input_ids"].to(device)
                prompt_lens = batch["prompt_len"]
                task_lens = batch["task_len"]
                batch_types = batch["batch_types"]

                # Teacher forward: extract only the logit slices we need,
                # then free the full output immediately.
                with torch.no_grad():
                    t_logits = teacher(input_ids=teacher_ids).logits
                    teacher_slices = []
                    for i in range(t_logits.size(0)):
                        pl = prompt_lens[i]
                        tl = task_lens[i]
                        if tl <= 0:
                            teacher_slices.append(None)
                        else:
                            teacher_slices.append(
                                t_logits[i, pl : pl + tl, :].clone()
                            )
                    del t_logits
                del teacher_ids

                # Student forward
                student_logits = student(input_ids=student_ids).logits

                losses = []
                for i in range(student_logits.size(0)):
                    tl = task_lens[i]
                    if teacher_slices[i] is None or tl <= 0:
                        continue
                    t_log = teacher_slices[i].float()
                    s_log = student_logits[i, :tl, :].float()
                    if t_log.size(0) != s_log.size(0):
                        m = min(t_log.size(0), s_log.size(0))
                        t_log = t_log[:m]
                        s_log = s_log[:m]
                    loss_i = F.mse_loss(s_log, t_log)
                    losses.append(loss_i)
                    bt = batch_types[i]
                    if bt in _latest_loss_by_type:
                        _latest_loss_by_type[bt] = loss_i.item()

                del teacher_slices

                if not losses:
                    del student_logits, student_ids
                    continue

                loss = torch.stack(losses).mean() / grad_accum
                loss.backward()

                del student_logits, student_ids, losses, loss

                if global_step % gc_interval == 0:
                    torch.cuda.empty_cache()

                if (global_step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), tc.get("max_grad_norm", 1.0)
                    )
                    optim.step()
                    scheduler.step()
                    optim.zero_grad()

                global_step += 1
                if global_step % logging_steps == 0:
                    parts = []
                    trace_row = {"step": global_step}
                    for k in ("control", "unswallowed", "swallowing"):
                        v = _latest_loss_by_type[k]
                        if v is not None:
                            parts.append(f"{k}:{v:.4f}")
                            trace_row[k] = round(v, 6)
                        else:
                            parts.append(f"{k}:--")
                            trace_row[k] = None
                    msg = " ".join(parts)
                    logger.info(escape(f"Step {global_step}/{max_steps} {msg}"))
                    trace_file.write(json.dumps(trace_row) + "\n")
                    trace_file.flush()
                    if self.writer:
                        for k, v in _latest_loss_by_type.items():
                            if v is not None:
                                self.writer.add_scalar(f"prompt_swallowing/{k}_loss", v, global_step)
                    _latest_loss_by_type = {"control": None, "unswallowed": None, "swallowing": None}

                should_save, is_permanent = should_save_checkpoint(
                    global_step, save_strategy, save_steps
                )
                if should_save:
                    ckpt_dir = self.output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    student.save_pretrained(ckpt_dir)
                    self.tokenizer.save_pretrained(ckpt_dir)
                    torch.save({
                        "global_step": global_step,
                        "optimizer": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, ckpt_dir / "training_state.pt")
                    tag = "permanent" if is_permanent else "temporary"
                    logger.info(f"Checkpoint saved to {ckpt_dir} ({tag})")

                    if _last_temp_ckpt is not None and _last_temp_ckpt.exists():
                        shutil.rmtree(_last_temp_ckpt)
                        logger.info(f"Deleted rolling checkpoint {_last_temp_ckpt}")

                    _last_temp_ckpt = None if is_permanent else ckpt_dir

                if global_step >= max_steps:
                    break

        trace_file.close()

        final_dir = self.output_dir / "prompt_swallowing_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"Prompt swallowing model saved to {final_dir}")

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize a dataset."""
        def tokenize_function(examples):
            # Tokenize the text
            
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config["training"].get("max_length", 512),
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def _create_preference_dataset(self) -> Dataset:
        """Create preference dataset for DPO.

        Expects training examples to carry ``user_query`` and
        ``assistant_response`` fields (set by the manual-template data
        generator).  The prompt is built from a system message describing
        the available tools plus the user query, formatted through the
        chat template.
        """
        system_prompt = build_tool_system_prompt(self.config.get("tools", []))

        preference_data = []
        for example in self.train_dataset:
            user_query = example.get("user_query", "")
            good_response = example.get("assistant_response", "")

            bad_response = (
                good_response.replace("[TOOL_CALL]", "").replace("[/TOOL_CALL]", "")
            )

            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            preference_data.append({
                "prompt": prompt_text,
                "chosen": good_response,
                "rejected": bad_response,
            })

        return Dataset.from_list(preference_data)
    
    # ------------------------------------------------------------------ #
    # VLM loading
    # ------------------------------------------------------------------ #

    def _load_vlm_processor(self):
        """Load VLM processor (tokenizer + image processor)."""
        model_name = self.config["model"]["name"]
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=self.config["model"].get("trust_remote_code", False),
        )
        return processor

    def _load_vlm_model(self):
        """Load VLM model with optional 4-bit quantisation and LoRA."""
        model_config = self.config["model"]

        if self.config["training"].get("use_lora", True):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model = AutoModelForVision2Seq.from_pretrained(
                model_config["name"],
                trust_remote_code=model_config.get("trust_remote_code", False),
                torch_dtype=getattr(torch, model_config.get("torch_dtype", "bfloat16")),
                device_map=model_config.get("device_map", "auto"),
                quantization_config=bnb_config,
            )

            if self._n_added_game_tokens > 0:
                model.resize_token_embeddings(len(self.tokenizer))

            model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config["training"].get("lora_r", 16),
                lora_alpha=self.config["training"].get("lora_alpha", 32),
                lora_dropout=self.config["training"].get("lora_dropout", 0.1),
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                model_config["name"],
                trust_remote_code=model_config.get("trust_remote_code", False),
                torch_dtype=getattr(torch, model_config.get("torch_dtype", "bfloat16")),
                device_map=model_config.get("device_map", "auto"),
            )

            if self._n_added_game_tokens > 0:
                model.resize_token_embeddings(len(self.tokenizer))

        return model

    def _parse_vlm_dpo_checkpoint_step(self, resume_from_checkpoint: Optional[str]) -> int:
        """Infer the next training loop index from ``checkpoint-{N}`` folder name."""
        if not resume_from_checkpoint:
            return 0
        name = Path(resume_from_checkpoint).name
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-", 1)[1])
            except ValueError:
                pass
        return 0

    def _load_vlm_peft_adapter_from_path(self, checkpoint_dir: Path) -> None:
        """Load LoRA adapter tensors from a PEFT checkpoint into the current VLM."""
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_dir}")
        safetensors_path = checkpoint_dir / "adapter_model.safetensors"
        bin_path = checkpoint_dir / "adapter_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file

            state_dict = load_file(str(safetensors_path))
        elif bin_path.exists():
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No adapter_model.safetensors or adapter_model.bin in {checkpoint_dir}"
            )

        adapter_name = getattr(self.model, "active_adapter", None) or "default"
        if not isinstance(adapter_name, str):
            adapter_name = str(adapter_name)

        set_fn = None
        try:
            from peft.utils.other import set_peft_model_state_dict as set_fn
        except ImportError:
            try:
                from peft.utils import set_peft_model_state_dict as set_fn
            except ImportError:
                set_fn = None
        if set_fn is not None:
            set_fn(self.model, state_dict, adapter_name=adapter_name)
            logger.info(f"Loaded PEFT adapter from {checkpoint_dir}")
            return

        r = self.model.load_state_dict(state_dict, strict=False)
        if r.missing_keys or r.unexpected_keys:
            logger.warning(
                "PEFT resume used non-strict load_state_dict; "
                f"missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}"
            )
        logger.info(f"Loaded PEFT adapter weights (fallback) from {checkpoint_dir}")

    # ------------------------------------------------------------------ #
    # VLM DPO training
    # ------------------------------------------------------------------ #

    def _format_vlm_messages(self, prompt: str, response: str):
        """Build a VLM chat-template message list (user + assistant)."""
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
        assistant_msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": response},
            ],
        }
        return [user_msg, assistant_msg]

    def _process_dpo_vlm_samples(self, samples: List[Dict[str, Any]]):
        """Convert raw DPO samples into processor-encoded batches.

        Returns (chosen_inputs, rejected_inputs, prompt_lengths) where
        each *_inputs is a BatchEncoding on the model device.
        """
        chosen_texts, rejected_texts, prompt_texts, images = [], [], [], []

        for s in samples:
            chosen_msgs = self._format_vlm_messages(s["prompt"], s["chosen"])
            rejected_msgs = self._format_vlm_messages(s["prompt"], s["rejected"])
            prompt_only = [chosen_msgs[0]]  # just the user turn

            chosen_texts.append(
                self.processor.apply_chat_template(chosen_msgs, tokenize=False)
            )
            rejected_texts.append(
                self.processor.apply_chat_template(rejected_msgs, tokenize=False)
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_only, tokenize=False, add_generation_prompt=True
                )
            )
            images.append(s["image"])

        # Prompt lengths (un-padded, per-sample)
        prompt_lengths: List[int] = []
        for pt, img in zip(prompt_texts, images):
            p = self.processor(text=[pt], images=[img], return_tensors="pt")
            prompt_lengths.append(p.input_ids.shape[1])

        # Batch-encode chosen and rejected
        chosen_inputs = self.processor(
            text=chosen_texts, images=images,
            return_tensors="pt", padding=True,
        )
        rejected_inputs = self.processor(
            text=rejected_texts, images=images,
            return_tensors="pt", padding=True,
        )

        device = next(self.model.parameters()).device
        chosen_inputs = chosen_inputs.to(device)
        rejected_inputs = rejected_inputs.to(device)

        return chosen_inputs, rejected_inputs, prompt_lengths

    def _compute_vlm_response_logps(
        self, input_ids, attention_mask, prompt_lengths, **extra_model_kwargs
    ):
        """Compute per-sample sum of log-probs over the response tokens.

        *extra_model_kwargs* carries pixel_values, image_grid_thw, etc.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **extra_model_kwargs,
        )
        logits = outputs.logits  # (B, L, V)

        # Next-token prediction shift
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Build response mask: True only for response token positions
        B, L = token_logps.shape
        response_mask = torch.zeros(B, L, dtype=torch.bool, device=token_logps.device)
        for i, pl in enumerate(prompt_lengths):
            # After the shift, the response starts at index (pl - 1)
            start = max(pl - 1, 0)
            response_mask[i, start:] = True
        response_mask = response_mask & attention_mask[:, 1:].bool()

        return (token_logps * response_mask).sum(dim=-1)  # (B,)

    def _train_dpo_vlm(self, resume_from_checkpoint: Optional[str] = None):
        """Custom DPO training loop for VLMs with online data generation."""
        tc = self.config["training"]
        batch_size = tc.get("batch_size", 4)
        max_steps = tc.get("max_steps", 5000)
        grad_accum = tc.get("gradient_accumulation_steps", 8)
        beta = tc.get("dpo_beta", 0.1)
        lr = tc.get("learning_rate", 1e-6)
        logging_steps = tc.get("logging_steps", 10)
        save_steps = tc.get("save_steps", 500)
        save_strategy = tc.get("save_strategy", "steps")
        max_grad_norm = tc.get("max_grad_norm", 0.3)
        warmup_steps = tc.get("warmup_steps", 100)

        data_strategy = self.config["data"].get("strategy", "position_qa")
        if data_strategy == "multi_task":
            from ..data.multi_task_generator import MultiTaskGenerator
            generator = MultiTaskGenerator(
                tasks=self.config["data"].get("tasks"),
                cross_axis_negative_prob=tc.get("cross_axis_negative_prob", 0.3),
            )
        else:
            from ..data.position_qa_generator import PositionQAGenerator
            generator = PositionQAGenerator(
                cross_axis_negative_prob=tc.get("cross_axis_negative_prob", 0.3),
            )

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        start_step = 0
        if resume_from_checkpoint:
            self._load_vlm_peft_adapter_from_path(Path(resume_from_checkpoint))
            start_step = self._parse_vlm_dpo_checkpoint_step(resume_from_checkpoint)
            if start_step > 0:
                logger.info(
                    f"Resuming VLM DPO from loop step {start_step} "
                    f"({start_step // grad_accum} optimizer/scheduler steps so far); "
                    "adapter weights restored, optimizer state fresh"
                )

        # Optimizer (after optional adapter load so it tracks final tensors)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optim_name = tc.get("optim", "paged_adamw_8bit")
        optim_kwargs = dict(
            lr=lr,
            betas=(tc.get("adam_beta1", 0.9), tc.get("adam_beta2", 0.999)),
            eps=tc.get("adam_epsilon", 1e-8),
            weight_decay=tc.get("weight_decay", 0.01),
        )
        if optim_name == "paged_adamw_8bit":
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(trainable_params, **optim_kwargs)
        else:
            optimizer = torch.optim.AdamW(trainable_params, **optim_kwargs)

        # Linear warmup scheduler
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if start_step > 0:
            n_sched = start_step // grad_accum
            if n_sched > 0:
                scheduler.last_epoch = n_sched - 1
                for group in optimizer.param_groups:
                    group["lr"] = lr * lr_lambda(scheduler.last_epoch)

        self.model.train()
        optimizer.zero_grad()

        logger.info(
            f"Starting VLM DPO training: steps {start_step}..{max_steps - 1} "
            f"(target {max_steps} iters), "
            f"batch_size={batch_size}, grad_accum={grad_accum}, beta={beta}, "
            f"save_strategy={save_strategy}"
        )

        trace_path = self.output_dir / "learning_trace.jsonl"
        trace_file = open(trace_path, "a")
        _last_temp_ckpt = None  # tracks the rolling temporary checkpoint

        try:
            for step in range(start_step, max_steps):
                # ---- generate data ----
                samples = generator.generate_batch(batch_size)
                chosen_inputs, rejected_inputs, prompt_lengths = (
                    self._process_dpo_vlm_samples(samples)
                )

                # Extract pixel-related kwargs (pixel_values, image_grid_thw, etc.)
                chosen_extra = {
                    k: v for k, v in chosen_inputs.items()
                    if k not in ("input_ids", "attention_mask")
                }
                rejected_extra = {
                    k: v for k, v in rejected_inputs.items()
                    if k not in ("input_ids", "attention_mask")
                }

                # ---- policy log-probs (with gradients) ----
                chosen_logps = self._compute_vlm_response_logps(
                    chosen_inputs.input_ids, chosen_inputs.attention_mask,
                    prompt_lengths, **chosen_extra,
                )
                rejected_logps = self._compute_vlm_response_logps(
                    rejected_inputs.input_ids, rejected_inputs.attention_mask,
                    prompt_lengths, **rejected_extra,
                )

                # ---- reference log-probs (LoRA off, no grad) ----
                with torch.no_grad():
                    self.model.disable_adapter_layers()
                    ref_chosen_logps = self._compute_vlm_response_logps(
                        chosen_inputs.input_ids, chosen_inputs.attention_mask,
                        prompt_lengths, **chosen_extra,
                    )
                    ref_rejected_logps = self._compute_vlm_response_logps(
                        rejected_inputs.input_ids, rejected_inputs.attention_mask,
                        prompt_lengths, **rejected_extra,
                    )
                    self.model.enable_adapter_layers()

                # ---- DPO loss ----
                pi_diff = chosen_logps - rejected_logps
                ref_diff = ref_chosen_logps - ref_rejected_logps
                loss = -F.logsigmoid(beta * (pi_diff - ref_diff)).mean()

                (loss / grad_accum).backward()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # ---- logging ----
                if (step + 1) % logging_steps == 0:
                    reward_margin = (pi_diff - ref_diff).mean().item()
                    accuracy = ((pi_diff - ref_diff) > 0).float().mean().item()
                    logger.info(
                        f"Step {step+1}/{max_steps} | loss={loss.item():.4f} "
                        f"reward_margin={reward_margin:.4f} acc={accuracy:.2%}"
                    )
                    trace_file.write(
                        json.dumps({
                            "step": step + 1,
                            "loss": loss.item(),
                            "reward_margin": reward_margin,
                            "accuracy": accuracy,
                        }) + "\n"
                    )
                    trace_file.flush()
                    if self.writer:
                        self.writer.add_scalar("dpo/loss", loss.item(), step + 1)
                        self.writer.add_scalar("dpo/reward_margin", reward_margin, step + 1)
                        self.writer.add_scalar("dpo/accuracy", accuracy, step + 1)

                # ---- checkpointing ----
                should_save, is_permanent = should_save_checkpoint(
                    step + 1, save_strategy, save_steps
                )

                if should_save:
                    ckpt_dir = self.output_dir / f"checkpoint-{step+1}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(ckpt_dir)
                    self.processor.save_pretrained(ckpt_dir)
                    tag = "permanent" if is_permanent else "temporary"
                    logger.info(f"Checkpoint saved to {ckpt_dir} ({tag})")

                    if _last_temp_ckpt is not None and _last_temp_ckpt.exists():
                        shutil.rmtree(_last_temp_ckpt)
                        logger.info(f"Deleted rolling checkpoint {_last_temp_ckpt}")

                    _last_temp_ckpt = None if is_permanent else ckpt_dir

            # ---- save final model ----
            final_dir = self.output_dir / "final_model"
            final_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(final_dir)
            self.processor.save_pretrained(final_dir)
            logger.info(f"Final model saved to {final_dir}")
        finally:
            trace_file.close()

    def cleanup(self):
        """Clean up resources."""
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")
