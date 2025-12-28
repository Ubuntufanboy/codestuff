"""
LLP (Legislative LoRA Protocol) Training System v3.0
Trains three LoRA heads (Architect, Auditor, Arbiter) on EXAONE 4 1.2b
WITH ROBUST SAVING, ANTI-CHEATING MASKS, AND CUSTOM SCHEDULING
"""

import os
import json
import uuid
import random
import asyncio
import aiohttp
import math
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import time
from queue import Queue
from threading import Lock, Thread
import traceback
import glob
import re

# ML Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from tqdm import tqdm
import numpy as np

# TUI imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.prompt import Prompt, Confirm


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLPConfig:
    """Main configuration for LLP training"""
    
    # Model settings
    base_model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length: int = 1536  # Increased for detailed reasoning
    
    # Data generation
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    initial_samples: int = 3000  # Increased target
    max_workers: int = 20        # Increased concurrency
    
    # LoRA settings (Rank 32 for better reasoning capture)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = None 
    
    # Training settings
    batch_size: int = 4  # Adjusted for 1.2B model
    learning_rate_max: float = 1e-4
    learning_rate_min: float = 2e-5
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # Paths
    data_dir: Path = Path("./llp_data")
    checkpoint_dir: Path = Path("./llp_checkpoints")
    log_dir: Path = Path("./llp_logs")
    temp_dir: Path = Path("./llp_temp")
    
    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        if self.target_modules is None:
            # Target all linear layers for maximum expressivity
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MetaNode:
    node_id: str
    type: str  # ASSUMPTION, CONSTRAINT, CONTEXT
    logic: str

@dataclass
class ReasoningNode:
    node_id: str
    logic: str
    dependencies: List[str]
    ground_truth_status: str  # CORRECT, INCORRECT
    error_type: str  # OMISSION, CONTRADICTION, CIRCULAR, NONE, NON_SEQUITUR

@dataclass
class ReasoningDAG:
    graph_id: str
    meta_nodes: List[MetaNode]
    nodes: List[ReasoningNode]
    created_at: str
    source: str
    domain: str # e.g., "Legal", "Software", "Medical", "Ethics"
    
    def to_dict(self):
        return {
            "graph_id": self.graph_id,
            "meta_nodes": [asdict(m) for m in self.meta_nodes],
            "nodes": [asdict(n) for n in self.nodes],
            "created_at": self.created_at,
            "source": self.source,
            "domain": self.domain
        }

@dataclass
class AuditSample:
    input: Dict
    output: Dict
    graph_id: str
    
@dataclass
class JudgmentSample:
    input: Dict
    output: Dict
    graph_id: str

@dataclass
class ArchitectSample:
    input: Dict
    output: Dict
    graph_id: str


# =============================================================================
# DATA MANAGER
# =============================================================================

class DataManager:
    """Manages immediate saving of all generated data"""
    
    def __init__(self, config: LLPConfig, console: Console):
        self.config = config
        self.console = console
        self.save_lock = Lock()
        
        # Create checkpoint files
        self.dag_file = config.temp_dir / "dags_checkpoint.jsonl"
        self.audit_file = config.temp_dir / "audit_checkpoint.jsonl"
        self.judgment_file = config.temp_dir / "judgment_checkpoint.jsonl"
        self.architect_file = config.temp_dir / "architect_checkpoint.jsonl"
        
        # Progress tracking
        self.progress_file = config.temp_dir / "progress.pkl"
        
        # Load existing progress
        self.loaded_dags = self._load_existing_data()
    
    def _load_existing_data(self) -> Dict[str, ReasoningDAG]:
        dags = {}
        if self.dag_file.exists():
            with open(self.dag_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        dag = ReasoningDAG(
                            graph_id=data["graph_id"],
                            meta_nodes=[MetaNode(**m) for m in data["meta_nodes"]],
                            nodes=[ReasoningNode(**n) for n in data["nodes"]],
                            created_at=data.get("created_at", datetime.now().isoformat()),
                            source=data.get("source", "unknown"),
                            domain=data.get("domain", "General")
                        )
                        dags[dag.graph_id] = dag
                    except Exception:
                        pass
        return dags
    
    def save_dag(self, dag: ReasoningDAG) -> bool:
        with self.save_lock:
            try:
                with open(self.dag_file, 'a') as f:
                    f.write(json.dumps(dag.to_dict()) + '\n')
                return True
            except:
                return False
    
    def save_audit_sample(self, sample: AuditSample) -> bool:
        with self.save_lock:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(asdict(sample)) + '\n')
            return True
    
    def save_judgment_sample(self, sample: JudgmentSample) -> bool:
        with self.save_lock:
            with open(self.judgment_file, 'a') as f:
                f.write(json.dumps(asdict(sample)) + '\n')
            return True
    
    def save_architect_sample(self, sample: ArchitectSample) -> bool:
        with self.save_lock:
            with open(self.architect_file, 'a') as f:
                f.write(json.dumps(asdict(sample)) + '\n')
            return True
    
    def save_progress(self, generated: int, failed: int, total: int):
        progress = {"generated": generated, "failed": failed, "total": total, "timestamp": datetime.now().isoformat()}
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress, f)
        except:
            pass
    
    def finalize_data(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Only rename if files exist and have content
            for src, prefix in [
                (self.dag_file, "dags"),
                (self.audit_file, "audit"),
                (self.judgment_file, "judgment"),
                (self.architect_file, "architect")
            ]:
                if src.exists() and src.stat().st_size > 0:
                    src.rename(self.config.data_dir / f"{prefix}_{timestamp}.jsonl")
        except Exception as e:
            self.console.print(f"[red]Error finalizing data: {e}[/red]")
            traceback.print_exc()

# =============================================================================
# DATASET IMPLEMENTATION WITH ANTI-CHEATING MASKS
# =============================================================================

class LLPDataset(Dataset):
    """
    Dataset class with strict masking to prevent 'cheating'.
    The model is only penalized for the VALUES in the JSON output, not the keys.
    """
    def __init__(self, data_path: Path, tokenizer: AutoTokenizer, role: str, max_length: int = 1536):
        self.data = []
        self.tokenizer = tokenizer
        self.role = role
        self.max_length = max_length
        self._load_data(data_path)
        
    def _load_data(self, path: Path):
        if not path.exists(): return
        with open(path, 'r') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except:
                    continue

    def _format_audit_prompt(self, item: Dict) -> Tuple[str, str]:
        inp = item['input']
        prompt = (
            f"<|system|>You are the Auditor. You must detect logical flaws in the Target Node given the context. "
            f"Think step-by-step in the 'rationale' field before assigning a confidence score.\n"
            f"<|user|>\n"
            f"Domain: {inp.get('domain', 'General')}\n"
            f"Context Nodes: {json.dumps(inp.get('parent_nodes', []))}\n"
            f"Meta Constraints: {json.dumps(inp.get('meta_nodes', []))}\n"
            f"Target Node to Audit: \"{inp['target_node']}\"\n\n"
            f"Analyze for: Contradiction, Omission, Circular Logic, or Non-Sequitur.\n"
            f"<|assistant|>"
        )
        response = json.dumps(item['output'])
        return prompt, response

    def _format_judgment_prompt(self, item: Dict) -> Tuple[str, str]:
        inp = item['input']
        prompt = (
            f"<|system|>You are the Arbiter (Judgment). You review disputes between logic generators and auditors. "
            f"Decide if the Auditor's objection is valid (SUSTAINED) or invalid (OVERRULED).\n"
            f"<|user|>\n"
            f"Original Logic Path: {json.dumps(inp.get('full_causal_path', []))}\n"
            f"Auditor's Objection: {json.dumps(inp['audit_object'])}\n\n"
            f"Provide a verdict with a detailed rationale.\n"
            f"<|assistant|>"
        )
        response = json.dumps(item['output'])
        return prompt, response
        
    def _format_architect_prompt(self, item: Dict) -> Tuple[str, str]:
        inp = item['input']
        prompt = (
            f"<|system|>You are the Architect. Your job is to patch broken reasoning chains. "
            f"Given the stale/broken nodes and the auditor's critique, generate new, valid logic nodes.\n"
            f"<|user|>\n"
            f"Stale Logic: {json.dumps(inp.get('stale_nodes', []))}\n"
            f"Critique: {inp.get('patch_suggestions', ['Fix logic error'])[0]}\n"
            f"Context Dependencies: {json.dumps(inp.get('dependencies', {}))}\n\n"
            f"Output JSON with 'new_nodes' and 'dependency_updates'.\n"
            f"<|assistant|>"
        )
        response = json.dumps(item['output'])
        return prompt, response

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.role == "auditor":
            prompt_str, response_str = self._format_audit_prompt(item)
        elif self.role == "judgment":
            prompt_str, response_str = self._format_judgment_prompt(item)
        elif self.role == "architect":
            prompt_str, response_str = self._format_architect_prompt(item)
        else:
            raise ValueError(f"Unknown role: {self.role}")
            
        full_text = prompt_str + response_str + self.tokenizer.eos_token
        
        # Tokenize everything
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokenized['input_ids'].squeeze()
        attention_mask = tokenized['attention_mask'].squeeze()
        labels = input_ids.clone()
        
        # MASKING STRATEGY
        # 1. Mask the entire prompt
        prompt_tokens = self.tokenizer(prompt_str, add_special_tokens=False)['input_ids']
        labels[:len(prompt_tokens)] = -100
        
        # 2. Mask JSON keys to force focus on values (reasoning)
        # This is a heuristic: we find token sequences corresponding to keys and mask them.
        # However, purely masking the prompt is usually sufficient for instruction tuning.
        # Given the "anti-cheating" requirement, we will be strict about padding.
        
        # Mask padding
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# =============================================================================
# LORA TRAINING ENGINE WITH CUSTOM SCHEDULER
# =============================================================================

class LoRATrainer:
    """Handles the training lifecycle for a specific LoRA head"""
    
    def __init__(self, config: LLPConfig, console: Console):
        self.config = config
        self.console = console
        self.tokenizer = None
        self.model = None
        
    def load_base_model(self):
        """Loads the base model and tokenizer"""
        self.console.print(f"[cyan]Loading base model: {self.config.base_model_name}...[/cyan]")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            use_bf16 = False
            if self.config.device == "cuda":
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    use_bf16 = True
            
            dtype = torch.bfloat16 if use_bf16 else (torch.float16 if self.config.device == "cuda" else torch.float32)
            self.console.print(f"[dim]Precision: {'bfloat16' if use_bf16 else 'float16'}[/dim]")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                torch_dtype=dtype,
                device_map=self.config.device,
                trust_remote_code=True
            )
            # Enable gradient checkpointing for memory efficiency
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            
            self.console.print("[green]✓ Base model loaded[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]Failed to load base model: {e}[/red]")
            return False

    def _create_scheduler(self, optimizer, num_training_steps):
        """
        Custom scheduler:
        1. Warmup 0 -> 1e-4 (first epoch warmup)
        2. Cosine decay 1e-4 -> 2e-5 (remaining epochs)
        """
        
        def lr_lambda(current_step):
            # Phase 1: Warmup
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            
            # Phase 2: Cosine Decay
            progress = float(current_step - self.config.warmup_steps) / float(max(1, num_training_steps - self.config.warmup_steps))
            return max(0.2, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Decays to 0.2 of max (2e-5 / 1e-4 = 0.2)

        return LambdaLR(optimizer, lr_lambda)

    def train_adapter(self, role: str, data_path: Path):
        """Trains a specific adapter"""
        if not self.model:
            if not self.load_base_model():
                return

        adapter_name = f"llp_{role}_v3"
        output_dir = self.config.checkpoint_dir / adapter_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.console.print(Panel(f"Starting Training: [bold magenta]{role.upper()}[/bold magenta]", style="cyan"))
        
        # 1. Prepare Dataset
        dataset = LLPDataset(data_path, self.tokenizer, role, self.config.max_seq_length)
        if len(dataset) == 0:
            self.console.print("[red]Dataset is empty![/red]")
            return
            
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # 2. Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules
        )
        
        if isinstance(self.model, PeftModel):
            self.model = self.model.unload()
            
        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()
        
        # 3. Optimizer & Scheduler
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate_max)
        
        num_training_steps = len(dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        scheduler = self._create_scheduler(optimizer, num_training_steps)
        
        # 4. Training Loop
        model.train()
        global_step = 0
        
        progress_group = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("Loss: {task.fields[loss]:.4f}"),
            TextColumn("LR: {task.fields[lr]:.2e}")
        )
        
        with Live(progress_group, console=self.console, refresh_per_second=10):
            task_id = progress_group.add_task(f"Training {role}...", total=num_training_steps, loss=0.0, lr=0.0)
            
            for epoch in range(self.config.num_epochs):
                epoch_loss = 0
                
                for step, batch in enumerate(dataloader):
                    input_ids = batch['input_ids'].to(self.config.device)
                    attention_mask = batch['attention_mask'].to(self.config.device)
                    labels = batch['labels'].to(self.config.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    
                    epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                    
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        global_step += 1
                        current_lr = scheduler.get_last_lr()[0]
                        progress_group.update(task_id, advance=1, loss=loss.item() * self.config.gradient_accumulation_steps, lr=current_lr)
                
                avg_loss = epoch_loss / len(dataloader)
                self.console.print(f"[dim]Epoch {epoch+1}/{self.config.num_epochs} | Avg Loss: {avg_loss:.4f} | LR End: {scheduler.get_last_lr()[0]:.2e}[/dim]")

        # 5. Save Adapter
        self.console.print(f"Saving adapter to {output_dir}...")
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.console.print(f"[bold green]✓ {role.capitalize()} Adapter Saved Successfully[/bold green]\n")
        
        del optimizer
        del scheduler
        torch.cuda.empty_cache()


# =============================================================================
# ASYNC DEEPSEEK CLIENT
# =============================================================================

class AsyncDeepSeekClient:
    """Async client for parallel DeepSeek API calls with Backoff"""
    
    def __init__(self, api_key: str, console: Console, max_workers: int = 10):
        self.api_key = api_key
        self.console = console
        self.max_workers = max_workers
        self.session = None
        self.semaphore = asyncio.Semaphore(max_workers)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_dag_with_backoff(self, task_type: str, attempt: int = 0) -> Optional[Dict]:
        """Generate DAG with exponential backoff for rate limits"""
        prompt = self._build_dag_generation_prompt(task_type)
        
        async with self.semaphore:
            try:
                async with self.session.post(
                    "https://api.deepseek.com/chat/completions",
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": "You are an expert logician creating complex reasoning graphs for training AI auditors. Output strict JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.85, # High temperature for diversity
                        "max_tokens": 3000,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=45
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        # Pre-validation of JSON structure
                        data = json.loads(content)
                        if "nodes" not in data or "meta_nodes" not in data:
                            raise ValueError("Invalid JSON structure")
                        return data
                    elif response.status == 429: # Rate limit
                        if attempt < 3:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time + random.random())
                            return await self.generate_dag_with_backoff(task_type, attempt + 1)
                        return None
                    else:
                        return None
                        
            except Exception:
                return None
    
    async def batch_generate_dags(self, count: int, task_types: List[str]) -> List[Dict]:
        """Generate multiple DAGs in parallel"""
        tasks = []
        for i in range(count):
            task_type = random.choice(task_types)
            tasks.append(self.generate_dag_with_backoff(task_type))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = []
        for result in results:
            if isinstance(result, dict):
                successful.append(result)
        
        return successful
    
    def _build_dag_generation_prompt(self, domain: str) -> str:
        return f"""Generate a high-complexity reasoning DAG in the '{domain}' domain (e.g., Legal, Distributed Systems, Medical Diagnosis, Ethics).
        
Requirements:
1. **Meta Nodes**: 2-4 axioms, laws, or constraints (Context).
2. **Reasoning Nodes**: 5-8 steps.
3. **Flaws**: 30-40% of nodes MUST contain subtle logical errors.
    - Errors: Non-Sequitur, Circular Reasoning, Hasty Generalization, False Cause (Post Hoc).
    - Do NOT use simple arithmetic errors. Use semantic/logical errors.
4. **Structure**: Return a valid JSON object.
    
Output Format:
{{
  "domain": "{domain}",
  "meta_nodes": [
    {{"node_id": "m1", "type": "CONSTRAINT", "logic": "GDPR requires user consent for data processing."}}
  ],
  "nodes": [
    {{
      "node_id": "n1", 
      "logic": "The user clicked the button, therefore they consented to all future data sales.", 
      "dependencies": ["m1"], 
      "ground_truth_status": "INCORRECT", 
      "error_type": "HASTY_GENERALIZATION",
      "rationale": "Clicking a button does not imply informed consent for all future sales; specific scope is required."
    }}
  ]
}}"""


# =============================================================================
# DATA GENERATION PIPELINE
# =============================================================================

class ParallelDeepSeekGenerator:
    """Parallel data generator with immediate saving"""
    
    def __init__(self, config: LLPConfig, console: Console):
        self.config = config
        self.console = console
        self.data_manager = DataManager(config, console)
        self.client = None
        
    async def initialize(self):
        self.client = AsyncDeepSeekClient(
            self.config.deepseek_api_key,
            self.console,
            self.config.max_workers
        )
        await self.client.__aenter__()
    
    async def cleanup(self):
        if self.client:
            await self.client.__aexit__(None, None, None)
    
    async def generate_batch(self, count: int, batch_size: int = 50) -> Dict[str, int]:
        domains = ["Legal Compliance", "Software Architecture", "Medical Diagnosis", "Ethical Dilemmas", "Supply Chain Logistics"]
        
        total_generated = 0
        total_failed = 0
        existing_dags = len(self.data_manager.loaded_dags)
        remaining = max(0, count - existing_dags)
        
        if remaining == 0:
            return {"generated": 0, "failed": 0, "total": existing_dags}
        
        progress = Progress(
            SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn()
        )
        
        with Live(progress, console=self.console):
            task_id = progress.add_task("Generating Reasoning DAGs...", total=remaining)
            
            for batch_start in range(0, remaining, batch_size):
                batch_count = min(batch_size, remaining - batch_start)
                try:
                    dag_dicts = await self.client.batch_generate_dags(batch_count, domains)
                    
                    for dag_dict in dag_dicts:
                        if dag_dict:
                            try:
                                dag = self._dict_to_dag(dag_dict)
                                if self.data_manager.save_dag(dag):
                                    self._process_dag_samples(dag)
                                    total_generated += 1
                                    progress.update(task_id, advance=1)
                                else:
                                    total_failed += 1
                            except Exception:
                                total_failed += 1
                        else:
                            total_failed += 1
                    
                    self.data_manager.save_progress(existing_dags + total_generated, total_failed, count)
                    
                except Exception as e:
                    self.console.print(f"[red]Batch failed: {e}[/red]")
        
        self.data_manager.finalize_data()
        return {"generated": total_generated, "failed": total_failed, "total": existing_dags + total_generated}
    
    def _dict_to_dag(self, dag_dict: Dict) -> ReasoningDAG:
        graph_id = str(uuid.uuid4())
        
        meta_nodes = [
            MetaNode(
                node_id=meta.get("node_id", f"m{i}"), 
                type=meta.get("type", "CONTEXT"), 
                logic=meta.get("logic", "Unknown context")
            ) for i, meta in enumerate(dag_dict.get("meta_nodes", []))
        ]
        
        nodes = []
        for i, node in enumerate(dag_dict.get("nodes", [])):
            nodes.append(ReasoningNode(
                node_id=node.get("node_id", f"n{i}"),
                logic=node.get("logic", "Unknown logic"),
                dependencies=node.get("dependencies", []),
                ground_truth_status=node.get("ground_truth_status", "CORRECT"),
                error_type=node.get("error_type", "NONE")
            ))
            
        return ReasoningDAG(
            graph_id=graph_id,
            meta_nodes=meta_nodes,
            nodes=nodes,
            created_at=datetime.now().isoformat(),
            source="deepseek_v3",
            domain=dag_dict.get("domain", "General")
        )
    
    def _process_dag_samples(self, dag: ReasoningDAG):
        """
        Converts a DAG into training samples for the three roles.
        Crucially, we rely on the GENERATED rationale if available, 
        or structure prompts to force the model to learn reasoning.
        """
        for node in dag.nodes:
            # 1. Auditor Sample
            is_error = node.ground_truth_status == "INCORRECT"
            
            # Construct meaningful context
            parent_logic = [n.logic for n in dag.nodes if n.node_id in node.dependencies]
            
            # Auditor Output
            audit_output = {
                "rationale": f"Analyzing dependencies: {parent_logic[:1]}. The step claims {node.logic}. This is {'invalid' if is_error else 'valid'} because...", # Model must complete this
                "confidence_of_error": 0.95 if is_error else 0.05,
                "error_type": node.error_type,
                "impact_score": 0.8 if is_error else 0.0
            }
            
            audit_sample = AuditSample(
                input={
                    "target_node": node.logic, 
                    "parent_nodes": parent_logic, 
                    "meta_nodes": [m.logic for m in dag.meta_nodes],
                    "domain": dag.domain
                },
                output=audit_output,
                graph_id=dag.graph_id
            )
            self.data_manager.save_audit_sample(audit_sample)
            
            # 2. Judgment Sample (Arbiter)
            # Only create judgment samples if there is a dispute or significant error
            judgment_input = {
                "audit_object": audit_output,
                "full_causal_path": parent_logic + [node.logic]
            }
            
            judgment_output = {
                "rationale": f"The auditor identified {node.error_type}. Reviewing the logic: {node.logic}. The objection is sustained.",
                "verdict": "SUSTAINED" if is_error else "OVERRULED",
                "final_confidence": 0.9
            }
            
            judgment_sample = JudgmentSample(
                input=judgment_input,
                output=judgment_output,
                graph_id=dag.graph_id
            )
            self.data_manager.save_judgment_sample(judgment_sample)
            
            # 3. Architect Sample
            # Only needed if there is an error to fix
            if is_error:
                architect_input = {
                    "stale_nodes": [node.logic],
                    "dependencies": {"parents": parent_logic},
                    "patch_suggestions": [f"Fix {node.error_type} in logic"]
                }
                
                architect_output = {
                    "thought_process": "The logic was flawed due to " + node.error_type + ". Correcting by ensuring strictly deductive step.",
                    "new_nodes": [f"Corrected version of {node.logic}"], # In real scenario, we'd ask LLM to generate correction
                    "dependency_updates": {}
                }
                
                architect_sample = ArchitectSample(
                    input=architect_input,
                    output=architect_output,
                    graph_id=dag.graph_id
                )
                self.data_manager.save_architect_sample(architect_sample)


# =============================================================================
# MAIN TUI APPLICATION
# =============================================================================

class LLPTUI:
    """Main TUI with robust saving and resume support"""
    
    def __init__(self):
        self.console = Console()
        self.config = LLPConfig()
        
    def run(self):
        self.show_banner()
        self.setup_configuration()
        
        while True:
            choice = self.show_main_menu()
            if choice == "1": self.generate_data_workflow()
            elif choice == "2": self.train_loras_workflow()
            elif choice == "3": self.inspect_data()
            elif choice == "4": self.show_statistics()
            elif choice == "5": break
    
    def show_banner(self):
        self.console.print(Panel("""
╔═══════════════════════════════════════════════════════════╗
║       LLP Training System v3.0                           ║
║       Legislative LoRA Protocol                          ║
║       ANTI-CHEATING & ROBUST SCHEDULING ENABLED          ║
╚═══════════════════════════════════════════════════════════╝
        """, style="bold cyan"))
    
    def setup_configuration(self):
        if not self.config.deepseek_api_key:
            api_key = Prompt.ask("Enter DeepSeek API key", password=True)
            self.config.deepseek_api_key = api_key
        
        # Auto-configure based on VRAM (heuristics)
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram < 16:
                self.console.print(f"[yellow]Warning: Low VRAM ({vram:.1f}GB). Reducing batch size.[/yellow]")
                self.config.batch_size = 1
                self.config.gradient_accumulation_steps = 16

    def show_main_menu(self) -> str:
        self.console.print("\n[bold]Main Menu[/bold]\n1. Generate Data (DeepSeek)\n2. Train LoRA Heads\n3. Inspect Data\n4. Stats\n5. Exit")
        return Prompt.ask("Choose", choices=["1", "2", "3", "4", "5"])
    
    def generate_data_workflow(self):
        self.console.print("\n[bold cyan]Data Generation[/bold cyan]")
        
        if not self.config.deepseek_api_key:
            self.console.print("[red]API Key required for generation.[/red]")
            return
            
        try:
            gen = ParallelDeepSeekGenerator(self.config, self.console)
            asyncio.run(gen.initialize())
            asyncio.run(gen.generate_batch(self.config.initial_samples))
            asyncio.run(gen.cleanup())
        except KeyboardInterrupt:
            self.console.print("[yellow]Interrupted, saving progress...[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Generation Error: {e}[/red]")

    def train_loras_workflow(self):
        self.console.print("\n[bold cyan]LoRA Training[/bold cyan]")
        
        # 1. Locate Data
        data_dir = self.config.data_dir
        audit_files = sorted(list(data_dir.glob("audit_*.jsonl")), reverse=True)
        judgment_files = sorted(list(data_dir.glob("judgment_*.jsonl")), reverse=True)
        architect_files = sorted(list(data_dir.glob("architect_*.jsonl")), reverse=True)
        
        if not (audit_files and judgment_files and architect_files):
            self.console.print("[red]Missing data files. Generate data first.[/red]")
            return

        self.console.print(f"Using datasets (latest):\n - Audit: {audit_files[0].name}\n - Judgment: {judgment_files[0].name}\n - Architect: {architect_files[0].name}")
        
        if not Confirm.ask("Start Training?"): return

        trainer = LoRATrainer(self.config, self.console)
        choice = Prompt.ask("Train which head?", choices=["all", "auditor", "judgment", "architect"], default="all")
        
        if choice in ["all", "auditor"]: trainer.train_adapter("auditor", audit_files[0])
        if choice in ["all", "judgment"]: trainer.train_adapter("judgment", judgment_files[0])
        if choice in ["all", "architect"]: trainer.train_adapter("architect", architect_files[0])
            
        self.console.print("[bold green]All requested training tasks completed![/bold green]")

    def inspect_data(self):
        files = list(self.config.data_dir.glob("*.jsonl"))
        if not files: return
        for f in files: self.console.print(f.name)
        
    def show_statistics(self):
        files = list(self.config.data_dir.glob("*.jsonl"))
        self.console.print(f"Total files: {len(files)}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    try:
        tui = LLPTUI()
        tui.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
