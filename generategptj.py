from sklearn.model_selection import train_test_split

import transformers

import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
import sklearn
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, GPTJForCausalLM

# from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise

from tqdm.auto import tqdm

config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

config.pad_token_id = config.eos_token_id
tokenizer.pad_token = config.pad_token_id

gpt= transformers.AutoModelForCausalLM.from_pretrained("./checkpoint/squadgptj500-2").cuda()
# gpt.load_state_dict(torch.load('./checkpoint/modelgptjsave.pt'))

# gpt=torch.load('./personal/Neeraja/results/checkpoint-12/global_step12/zero_pp_rank_0_mp_rank_00_optim_states.pt')

# gpt.eval()
# generated = tokenizer("['Wikidata ::', 'nationality: german', 'birth_date: 1954', 'occupation: aircraft designer and manufacturer', 'name: walter extra', 'article_title: walter extra\n', 'Article ::',", return_tensors="pt").input_ids.cuda()
# sample_outputs = gpt.generate(generated, do_sample=True, top_k=50, max_length=2056, top_p=0.9, temperature=1.0, num_return_sequences=2)                               
 
    
# for i, sample_output in enumerate(sample_outputs):
#     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
device = 'cuda' if torch.cuda.is_available() else 'cpu'    
gpt.eval()
with torch.no_grad():
    # prompt=tokenizer("['Wikidata ::', 'Name: Neeraja Kirtane', 'birth_date: 2000', 'occupation: Researcher', 'nationality: Indian', 'education: Manipal','field of study: Natural language processing','article_title: Neeraja\n', 'Article ::',", truncation=True, padding=True, max_length=2056, return_tensors='pt')
    prompt=tokenizer('"Context ::Architecturally, the school has a Catholic character. Atop the Main Buildings gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.Question ::What is in front of the Notre Dame Main Building?Answer ::,"', truncation=True, padding=True, max_length=2056, return_tensors='pt')
    # prompt = tokenizer("['Wikidata ::', 'nationality: german', 'birth_date: 1954', 'occupation: aircraft designer and manufacturer', 'name: walter neer', 'article_title: walter neer\n', 'Article ::', Walter was", truncation=True, padding=True, max_length=2056, return_tensors='pt')
    prompt = {key: value.to(device) for key, value in prompt.items()}
    out = gpt.generate(**prompt, max_length=3000, top_k=50, top_p=0.9, temperature=1.0, do_sample=True, repetition_penalty = 1.2, num_beams=1)
    print(tokenizer.decode(out[0]) )   
    
    
