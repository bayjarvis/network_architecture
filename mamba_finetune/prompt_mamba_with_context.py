import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import json
import pandas as pd
import time
from tqdm import tqdm
import argparse

def run_mamba(model, question, context):

    text = f"{context}\n\nQ: {question}\nA:"
    print(text)
    input_ids = torch.LongTensor([tokenizer.encode(text)]).cuda()
    # print(input_ids)
    
    out = model.generate(
        input_ids=input_ids,
        max_length=len(input_ids)+128,
        eos_token_id=tokenizer.eos_token_id
    )

    # print(out)
    decoded = tokenizer.batch_decode(out)[0]
    # print("="*80)
    # print(decoded)
    
    # out returns the whole sequence plus the original
    cleaned = decoded.replace(text, "")
    cleaned = cleaned.replace("<|endoftext|>", "")
    
    # the model will just keep generating, so only grab the first one
    answer = cleaned.split("\n\n")[0].strip()
    # print(answer)
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m")
    args = parser.parse_args()
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    
    model = MambaLMHeadModel.from_pretrained(model_name, device="cuda", dtype=torch.float16)
    
    while True:
    
        # print(data)
        context = input("Context > ")
        question = input("Question > ")
        
        guess = run_mamba(model, question, context)
        print(guess)
        print("="*80)
        print("")
    