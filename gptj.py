import os
# Important: the following env vars is a hack to emulated distributed environment on a single GPU. Remove all of the env vars if you run 
# with more than one GPU and the torch.distributed or the deepspeed launcher
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '9994'
# os.environ['RANK'] = "0"
# os.environ['LOCAL_RANK'] = "0"
# os.environ['WORLD_SIZE'] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_CACHE'] = 'personal/Neeraja'
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
training_args = TrainingArguments(output_dir='personal/Neeraja/results', num_train_epochs=1, logging_steps=100, save_strategy='no',
                                  per_device_train_batch_size=2, per_device_eval_batch_size=2, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='personal/Neeraja/logs', fp16=True, deepspeed='./ds_config_gpt_j.json')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda()
model.resize_token_embeddings(len(tokenizer))
descriptions = pd.read_csv('squad500.csv')['sentence']
max_length = max([len(tokenizer.encode(description)) for description in descriptions])
print("Max length: {}".format(max_length))


class NetflixDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


dataset = NetflixDataset(descriptions, tokenizer, max_length=max_length)
print("Dataset is",dataset)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
trainer=Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])})
trainer.train()
print(trainer.model)
trainer.save_model("./checkpoint/squadgptj500-2") 
# torch.save(trainer.model,'./checkpoint/modelgptjsave.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.eval()
with torch.no_grad():
       prompt = tokenizer('"Context ::Architecturally, the school has a Catholic character. Atop the Main Buildings gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.Question ::What is in front of the Notre Dame Main Building?Answer ::,"',truncation=True, padding=True, max_length=2056, return_tensors='pt')
       prompt = {key: value.to(device) for key, value in prompt.items()}
       out = model.generate(**prompt, max_length=2056, top_k=50, top_p=0.9, temperature=1.0, do_sample=True, repetition_penalty = 1.2, num_beams=1)
       print(tokenizer.decode(out[0]))
   
    
print("end of code")    
    
    
# generated = tokenizer("['Wikidata ::', 'nationality: german', 'birth_date: 1954', 'occupation: aircraft designer and manufacturer', 'name: walter extra', 'article_title: walter extra\n', 'Article ::',", return_tensors="pt").input_ids.cuda()
# sample_outputs = model.generate(generated, do_sample=True, top_k=50, max_length=2056, top_p=0.9, temperature=1.0, num_return_sequences=2)                               
 
    
# for i, sample_output in enumerate(sample_outputs):
#     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    
   
    
# bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>'
['Context :: Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary. Question :: The Basilica of the Sacred heart at Notre Dame is beside to which structure? Answer :: [\'the Main Building\']']
