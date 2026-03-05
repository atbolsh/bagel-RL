"""Training module for tool use models."""
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, SFTTrainer
from trl import DPOConfig, SFTConfig
from torch.utils.tensorboard import SummaryWriter
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from ..tools.executor import ToolExecutor


logger = logging.getLogger(__name__)


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

        # Initialize model and tokenizer/processor
        if self.is_vlm:
            self.processor = self._load_vlm_processor()
            self.tokenizer = self.processor.tokenizer
            self.model = self._load_vlm_model()
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
        elif self.training_method == "teacher_mode":
            self._train_teacher_mode(resume_from_checkpoint)
        else:
            raise ValueError(f"Unknown training method: {self.training_method}. Supported methods: sft, dpo, teacher_mode")
    
    def _train_sft(self, resume_from_checkpoint: Optional[str] = None):
        """Supervised fine-tuning."""
        logger.info("Starting supervised fine-tuning...")
        
        
        # Training arguments
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
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_pin_memory=False,
            bf16=True,  # Use bf16 (matches BitsAndBytes compute dtype, doesn't need gradient scaling)
            max_grad_norm=1.0,
            optim = "adamw_torch" ,
            max_length=self.config["training"].get("max_length", 2048),
            )
       

        
        
        
        trainer = SFTTrainer(
            model           = self.model,
            train_dataset   = self.train_dataset,
            eval_dataset    = self.eval_dataset,
            args            = training_args,
            processing_class       = self.tokenizer,
        )

    
       
        
        # Train
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
    
    def _train_dpo(self, resume_from_checkpoint: Optional[str] = None):
        """Train the model using DPO."""
        logger.info("Starting DPO training")
        
        # Make sure we're using LoRA or the training will likely fail
        use_lora = self.config["training"].get("use_lora", True)

        preference_dataset = self._create_preference_dataset()
        
        if not use_lora:
            logger.warning(
                "⚠️ You're attempting to run DPO without LoRA which may cause NaN values. "
                "Consider enabling LoRA with 'use_lora': true in your config."
            )
        
        # Setup training arguments with gradient clipping
        training_args = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config["training"].get("num_epochs", 3),
            per_device_train_batch_size=self.config["training"].get("batch_size", 4),
            gradient_accumulation_steps=self.config["training"].get("gradient_accumulation_steps", 1),
            learning_rate=self.config["training"].get("learning_rate", 5e-6),
            max_grad_norm=self.config["training"].get("max_grad_norm", 0.3),  # Add strict gradient clipping
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            optim=self.config["training"].get("optim", "paged_adamw_8bit"),  # Use 8-bit optimizer
            bf16=self.config["training"].get("bf16", True),
            fp16=self.config["training"].get("fp16", False),
            max_length=self.config["training"].get("max_length", 512),
            remove_unused_columns=False,
            beta=0.1,  # Lower beta to stabilize training
            report_to="none",
        )
        
      
        # Create DPO trainer with improved stability
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Use same model as reference
            args=training_args,
            train_dataset=preference_dataset,
            processing_class=self.tokenizer,
        )
        
        # Add gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        # Train
        logger.info("Starting DPO training")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the trained model
        trainer.save_model(self.output_dir / "dpo_model")
        logger.info(f"Model saved to {self.output_dir / 'dpo_model'}")
    
    def _train_teacher_mode(self, resume_from_checkpoint: Optional[str] = None):
        """Teacher mode training (Toolformer-style)."""
        logger.info("Starting teacher mode training...")
        
        # This combines SFT with self-supervised learning
        # The data generation already handles teacher mode data creation
        self._train_sft(resume_from_checkpoint)
    
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
        """Create preference dataset for DPO."""
        # This is a simplified implementation
        # In practice, you'd want human preferences or model-based ranking
        
        preference_data = []
        
        for example in self.train_dataset:
            # Create a "good" and "bad" version
            good_response = example["text"]
            
            # Create a bad version by removing tool formatting
            bad_response = good_response.replace("[TOOL_CALL]", "").replace("[/TOOL_CALL]", "")
            
            preference_data.append({
                "prompt": example["text"].split("Assistant:")[0] if "Assistant:" in example["text"] else "",
                "chosen": good_response,
                "rejected": bad_response
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

        return model

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
        from ..data.position_qa_generator import PositionQAGenerator

        tc = self.config["training"]
        batch_size = tc.get("batch_size", 4)
        max_steps = tc.get("max_steps", 5000)
        grad_accum = tc.get("gradient_accumulation_steps", 8)
        beta = tc.get("dpo_beta", 0.1)
        lr = tc.get("learning_rate", 1e-6)
        logging_steps = tc.get("logging_steps", 10)
        save_steps = tc.get("save_steps", 500)
        max_grad_norm = tc.get("max_grad_norm", 0.3)
        warmup_steps = tc.get("warmup_steps", 100)

        generator = PositionQAGenerator(
            cross_axis_negative_prob=tc.get("cross_axis_negative_prob", 0.3),
        )

        # Optimizer
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

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.model.train()
        optimizer.zero_grad()

        logger.info(
            f"Starting VLM DPO training: {max_steps} steps, "
            f"batch_size={batch_size}, grad_accum={grad_accum}, beta={beta}"
        )

        for step in range(max_steps):
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
                if self.writer:
                    self.writer.add_scalar("dpo/loss", loss.item(), step + 1)
                    self.writer.add_scalar("dpo/reward_margin", reward_margin, step + 1)
                    self.writer.add_scalar("dpo/accuracy", accuracy, step + 1)

            # ---- checkpointing ----
            if (step + 1) % save_steps == 0:
                ckpt_dir = self.output_dir / f"checkpoint-{step+1}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
                self.processor.save_pretrained(ckpt_dir)
                logger.info(f"Checkpoint saved to {ckpt_dir}")

        # ---- save final model ----
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final_dir)
        self.processor.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")

    def cleanup(self):
        """Clean up resources."""
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")
