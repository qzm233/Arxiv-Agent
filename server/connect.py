import os, json, torch
import pandas as pd
import numpy as np
import heapq
from time import sleep
from transformers import StoppingCriteria, StoppingCriteriaList

from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset,load_from_disk
from collections import defaultdict
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"

save_dir = '/root/autodl-fs/llama-3-8b-instruct'

tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir, low_cpu_mem_usage=True)
model = LlamaForCausalLM.from_pretrained(save_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
model.cuda()
print("Good, load successful!")

class PriorityQueue:
    def __init__(self, max_size):
        self.queue = []
        self.max_size = max_size

    def push(self, item):
        try:
            if len(self.queue) == self.max_size:
                if item[0] > self.queue[0][0]:
                    removed_item = heapq.heappop(self.queue)
                    print(f'Removed item with lower priority: {removed_item}')
                else:
                    return
            print(f'Pushing item: {item}')
            heapq.heappush(self.queue, item)
        except TypeError as e:
            print(f"TypeError encountered: {e}. Item: {item}")
            raise e

    def pop(self):
        return heapq.heappop(self.queue)

    def peek(self):
        return self.queue[0]

    def __len__(self):
        return len(self.queue)

def check_demand():
    local_folder = '/root/autodl-fs/DSP/data/'
    local_file = 'query.txt'
    file_path = os.path.join(local_folder, local_file)
    if os.path.isfile(file_path):
        return True
    else:
        # print("Query do not exist, check the file name")
        return False

def load_demand():
    local_folder = '/root/autodl-fs/DSP/data/'
    local_file = 'query.txt'
    file_path = os.path.join(local_folder, local_file)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
            print("Read query done!")
            print("Query: ",data)
            
            os.remove(file_path)
            print(f"File {local_file} has been deleted.")
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def load_source(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Directory {folder_path} do not exist")
        return []

    all_files = os.listdir(folder_path)
    file_names = [f for f in all_files if os.path.isfile(os.path.join(folder_path, f))]
    
    return file_names

while True:
    sleep(1)
    print("Query exsits?: ", check_demand())
    if check_demand():
        demand = load_demand()
        files = load_source("data/source")
        # print(files)
        pq = PriorityQueue(max_size=10)
        for file in files:
            df = pd.read_json("data/source/"+file) #将传入的source文件读取成df
            if os.path.isfile("data/source/"+file):
                os.remove("data/source/"+file)     # 使用后删去source文件
                print(f"Source file {file} has been deleted.")
            else:
                print(f"Source file {file} does not exist.")
                
            # df = pd.read_json("CS_500.json")
            # print(df.columns)
            for index, row in df.iterrows():
                item = (row['title'], row['link'])
                abstract = row['abstract']
                prompt = "Please judge that if this paper relates with all my demand: \n ##Abstract##\n{\n"
                prompt = prompt + abstract + "\n}\n"
                prompt = prompt + "##Demands##\n"
                prompt = prompt + demand
                prompt = prompt + "Does this paper relate to all my demands. Please answer directly Yes or No: Yes"

                input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits
                    probs = torch.log_softmax(logits, dim=-1)    
                answer_ids = tokenizer(["Yes", "No"], return_tensors="pt").input_ids.to(device)
                last = probs[:,-2:-1,:]
                # print(last.shape)
                answer = answer_ids[:,-1].unsqueeze(0).unsqueeze(0)
                # print(answer.shape)
                gathered_probs = torch.gather(last, 2, answer)  
                gathered_probs = gathered_probs.detach().cpu()
                # print("Gathered probabilities:", gathered_probs)
                g_probs = np.array(gathered_probs.view(-1).detach())
                exp_logits = np.exp(g_probs)
                sum_exp_logits = np.sum(exp_logits)
                probabilities = exp_logits / sum_exp_logits
                # print("Yes prob: ", probabilities[0])
                yes_prob = float(probabilities[0])
                # item["prob"] = yes_prob
                # print((yes_prob, item))
                pq.push((yes_prob, item))
                
        with open('data/output/priority_queue.json', 'w',encoding='utf-8') as json_file:
            results = []
            print("do results")
            while pq.queue:
                prob, item = pq.pop()
                results.insert(0, {"Similarity": prob, "Title":item[0], "URL": item[1]}) 
            print(results)
            json.dump(results, json_file, indent=4)
            print("Already recommanded for query!")
                



    