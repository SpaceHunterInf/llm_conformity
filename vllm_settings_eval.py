from vllm import LLM, SamplingParams
import pandas as pd
import torch
import json, os
import random
from tqdm import tqdm
import wandb
import re

from utils.prompt_utils import *
from utils.dataset_utils import *
from utils.eval_utils import *

import argparse

from vllm_mpd import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Testing arguments")
    parser.add_argument("--dataset_name", type=str, default='CommonsenseQA')
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--model_dtype", type=str, default='float32')
    
    parser.add_argument("--max_output_tokens", type=int, default=50)
    
    parser.add_argument("--conform_mode", type=str, default='all')
    parser.add_argument("--participant_mode", type=str, default='plain')
    parser.add_argument("--max_participants", type=int, default=10)
    parser.add_argument("--log_probs", type=int, default=-1, help='how many log probabilites you want to save, default -1 is None')
    parser.add_argument("--save_mcqa_first_prob", action='store_true', default=False, help='save the answer (first token) probability for mcqa')
    
    args = parser.parse_args()
    
    if args.log_probs == -1:
        logprobs = None
    else:
        logprobs = args.log_probs
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=args.max_output_tokens, logprobs=logprobs)
        
    llm = LLM(model=args.model_name, dtype=args.model_dtype)
    
    setting_name = '-'.join([str(args.max_participants), args.conform_mode, args.participant_mode])
    wandb.init(project="conformity", name=setting_name)
    saving_dir = os.path.join(args.model_name.replace('/',''), args.dataset_name, setting_name)
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    if args.dataset_name == 'CommonsenseQA':
        data = inference_CommonsenseQA('validation', llm, sampling_params, args.max_participants, saving_dir, 
                                    conform_mode=args.conform_mode,
                                    participant_mode=args.participant_mode)
    if 'MMLU_Test' in args.dataset_name:
        with open('data/{}/data.json'.format(args.dataset_name), encoding='utf-8') as f:
            data = json.load(f)
        data = mcqa_base_cycle(llm, sampling_params, data, save_log_probs=args.save_mcqa_first_prob)
        data = mcqa_conform_cycle(llm, sampling_params, data, max_participants=args.max_participants, 
                                  conform_mode=args.conform_mode, participant_mode=args.participant_mode, 
                                  save_log_probs=args.save_mcqa_first_prob)
        with open(os.path.join(saving_dir, 'out.json'), 'w') as f:
            f.write(json.dumps(data, indent=2))
            f.close()
    columns, conformed, correct, others, no_answer, total = mcqa_eval(data, max_participants=args.max_participants)
    participants = np.arange(2, args.max_participants + 1)  # Use numbers 1 to 10 for participant labels
    # breakpoint()
    plot_conformity(columns, correct, conformed, others, no_answer, os.path.join(saving_dir, 'out.jpg'), setting_name)
    
    if args.save_mcqa_first_prob:
        first_probs, conform_probs, correct_probs = mcqa_logits_eval(data, max_participants=args.max_participants)
        plot_logits_trend(first_probs, conform_probs, correct_probs, os.path.join(saving_dir, 'trend.jpg'), setting_name)
    
    print('total correct is {} out of {}, base accuracy: {}'.format(str(total), str(len(data)), str(total/len(data))))
    
    wandb.finish()