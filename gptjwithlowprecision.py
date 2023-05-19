# -*- coding: utf-8 -*-
"""finetune-gptj

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pZVmy2fPUCZIGC9z_KY7WV2vXUYld12Z
"""

# !pip install transformers==4.14.1 -q
# !pip install bitsandbytes-cuda111==0.26.0 -q
# !pip install datasets==1.16.1 -q

"""### Fine-tuning 6-Billion GPT-J (& other models) in colab with LoRA and 8-bit compression

This notebook is a simple example for fine-tuning [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B) with limited memory. A detailed explanation of how it works can be found in [this model card](https://huggingface.co/hivemind/gpt-j-6B-8bit). It is heavily based on [this Colab](https://colab.research.google.com/drive/1ft6wQU0BhqG5PRlwgaZJv2VukKKjU4Es#scrollTo=vfdLQHOuEU7h). Huge thanks to Hivemind!

You can also finetune [GPT-Neo-2.7B](https://huggingface.co/gustavecortal/gpt-neo-2.7B-8bit), [French GPT-J (Cedille's Boris)](https://huggingface.co/gustavecortal/fr-boris-8bit) and [T0-3B](https://huggingface.co/gustavecortal/T0_3B-8bit) with limited memory.

Twitter: [@gustavecortal](https://twitter.com/gustavecortal)
"""

# !pip install bitsandbytes

from sklearn.model_selection import train_test_split

import transformers

import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
import sklearn

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise

from tqdm.auto import tqdm

# from google.colab import drive
# drive.mount("/content/drive")

"""## Converting the model to 8 bits"""

class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias
 
    def forward(self, input):
        output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias)
        if self.adapter:
            output += self.adapter(input)
        return output
 
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
 
 
class DequantizeAndLinear(torch.autograd.Function): 
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias).clone()
 
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias
 
 
class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
 
    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output 
 
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"
 
 
def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)
 
    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)
 
 
def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr(
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )

"""You have to Monkey-Patch GPT-J before loading: """

class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)
        

class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock

"""If you're using another 8-bit quantized model (e.g. T0-3B), remember to Monkey-Patch the model using convert_to_int8()"""

class T5ForConditionalGeneration(transformers.models.t5.modeling_t5.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)

transformers.models.t5.modeling_t5.T5ForConditionalGeneration = T5ForConditionalGeneration

config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

config.pad_token_id = config.eos_token_id
tokenizer.pad_token = config.pad_token_id

gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
#gpt = GPTJForCausalLM.from_pretrained("gustavecortal/fr-boris-8bit", low_cpu_mem_usage=True) French GPT-J Cedille's Boris

"""## LoRA fine-tuning example

You can load my very small dataset composed of philosophical sentences:
"""

# !gdown --id 1Q1WMjny26VHLKb71iTCHIS5zvdm9c-wz

data = pd.read_csv('squad500.csv')
# data['sentence'] = 'Quote: ' + data['sentence']
train, test = train_test_split(data, test_size=0.01) 
train.to_csv('personal/train_pgbp_example.csv', index=False)
test.to_csv('personal/test_pgbp_example.csv', index=False)

from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train': 'personal/train_pgbp_example.csv',
                                              'test': 'personal/test_pgbp_example.csv'})

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding=True, truncation=True, max_length= 2056)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
tokenized_datasets.set_format("torch")


from torch.utils.data import DataLoader

full_train_dataset = tokenized_datasets["train"]
train_dataloader = DataLoader(full_train_dataset, shuffle=False, batch_size=2)

"""Add adapters to Embedding/MLP/Attention/LMHead layers"""

def add_adapters(model, adapter_dim=4, p = 0.1):
    assert adapter_dim > 0

    for name, module in model.named_modules():
      if isinstance(module, FrozenBNBLinear):
          if "attn" in name or "mlp" in name or "head" in name:
              print("Adding adapter to", name)
              module.adapter = nn.Sequential(
                nn.Linear(module.in_features, adapter_dim, bias=False),
                nn.Dropout(p=p),
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
              print("Initializing", name)
              nn.init.zeros_(module.adapter[2].weight)

          else:
              print("Not adding adapter to", name)
      elif isinstance(module, FrozenBNBEmbedding):
          print("Adding adapter to", name)
          module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Dropout(p=p),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
          print("Initializing", name)
          nn.init.zeros_(module.adapter[2].weight)

add_adapters(gpt)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt.to(device)

# mem_report()

from bitsandbytes.optim import Adam8bit

gpt.gradient_checkpointing_enable()
optimizer = Adam8bit(gpt.parameters(), lr=1e-5, weight_decay=0.01)

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, int(num_training_steps*0.1), num_training_steps
)

# filepath = '/model.pt'

from tqdm.auto import tqdm

scaler = torch.cuda.amp.GradScaler()
progress_bar = tqdm(range(num_training_steps))
gpt.train()
gpt.gradient_checkpointing_enable()
k = 0

for epoch in range(num_epochs):
    for batch in train_dataloader:

        k = k + 1
        if k % 5 == 0:
          print(k)
          state = {'k' : k, 'epoch': num_epochs, 'lr_scheduler': lr_scheduler.state_dict(), 'state_dict': gpt.state_dict(), 'optimizer': optimizer.state_dict()}
          # torch.save(state, filepath)

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
          out = gpt.forward(**batch,)

          loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                                reduction='mean', label_smoothing=0.1)
          
        print(loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        progress_bar.update(1)

"""## Text generation example"""

torch.save(gpt,"./checkpoint/squad8bit.pt")
# torch.save(gpt.state_dict(), './checkpoint/model.pth')
gpt.eval()
with torch.no_grad():
       prompt = tokenizer('"Context ::Architecturally, the school has a Catholic character. Atop the Main Buildings gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.Question ::What is in front of the Notre Dame Main Building?Answer ::,"',truncation=True, padding=True, max_length=2056, return_tensors='pt')
       prompt = {key: value.to(device) for key, value in prompt.items()}
       out = gpt.generate(**prompt, max_length=2056, top_k=50, top_p=0.9, temperature=1.0, do_sample=True, repetition_penalty = 1.2, num_beams=1)
       print(tokenizer.decode(out[0]))
	# sentence
# 1	['Wikidata ::', 'nationality: german', 'birth_date: 1954', 'occupation: aircraft designer and manufacturer', 'name: walter extra', 'article_title: walter extra\n', 'Article ::', 'walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra flugzeugbau -lrb- extra aircraft construction -rrb- , a manufacturer of aerobatic aircraft .\nextra was trained as a mechanical engineer .\nhe began his flight training in gliders , transitioning to powered aircraft to perform aerobatics .\nhe built and flew a pitts special aircraft and later built his own extra ea-230 .\nextra began designing aircraft after competing in the 1982 world aerobatic championships .\nhis aircraft constructions revolutionized the aerobatics flying scene and still dominate world competitions .\nthe german pilot klaus schrodt won his world championship title flying an aircraft made by the extra firm .\nwalter extra has designed a series of performance aircraft which include unlimited aerobatic aircraft and turboprop transports .\n']
# !pip install deepspeed

# !deepspeed

# !pip install gputil
# !pip install psutil
# !pip install humanize

# import os,sys,humanize,psutil,GPUtil

# # Define function
# def mem_report():
#   print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
#   GPUs = GPUtil.getGPUs()
#   for i, gpu in enumerate(GPUs):
#     print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

# mem_report()