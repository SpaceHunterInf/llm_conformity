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
from utils.aux_utils import *


def subjective_qa_base_cycle(llm, sampling_params, data):
    input_batch = []
    for q in data:
        prompt, _ = get_agree_or_disagree_prompt(q, mpd_qa_prompt, num_participants=1)
        input_batch.append(prompt)
    outputs = llm.generate(input_batch, sampling_params, guided_options_request=dict(guided_choice=["agree", "disagree"]))
    output_batch = [output.outputs[0].text for output in outputs]
    
    for idx in range(len(data)):
        data[idx]['base_out'] = output_batch[idx]
        if 'disagree' in output_batch[idx]:
            data[idx]['given_answer'] = 'disagree'
        else:
            if 'agree' in output_batch[idx]:
                data[idx]['given_answer'] = 'agree'
            else:
                data[idx]['given_answer'] = 'avoided'
    
    return data

def subjective_qa_conform_cycle(llm, sampling_params, data, max_participants=10, conform_mode='all', participant_mode='plain'):
    valid_indices = []
    
    for i in range(len(data)):
            if data[i]['given_answer'] != 'avoided':
                valid_indices.append(i)
                
    for num in tqdm(range(2, max_participants+1), desc='Participant'):
        input_batch = []
        conformed_answers = []
        for idx in valid_indices:
            prompt, conformed = get_agree_or_disagree_prompt(data[idx], mpd_qa_prompt, bad_answer=conform_mode, participant_mode=participant_mode, num_participants=num)
            input_batch.append(prompt)
            conformed_answers.append(conformed)
        outputs = llm.generate(input_batch, sampling_params,  guided_options_request=dict(guided_choice=["agree", "disagree"]))
        output_batch = [output.outputs[0].text for output in outputs]

        for idx, prompt, conformed, out in zip(valid_indices, input_batch, conformed_answers, output_batch):
            if 'disagree' in out:
                answer = 'disagree'
            else:
                if 'argree' in out:
                    answer = 'agree'
                else:
                    answer = 'avoided'
            data[idx]['participant_{}'.format(str(num))] = {'prompt':prompt, 'out':out, 'conformed_answer':conformed, 'answer':answer}
    
    return data

def mcqa_base_cycle(llm, sampling_params, data, save_log_probs=False):
    input_batch = []
    for q in data:
        prompt, _ = get_conformed_mcqa_prompt(q, mpd_mcqa_prompt, num_participants=1)
        input_batch.append(prompt)
    outputs = llm.generate(input_batch, sampling_params)
    output_batch = [output.outputs[0].text for output in outputs]
    
    if save_log_probs:
        first_token_batch = [log_prob_to_dict(output.outputs[0].logprobs[0]) for output in outputs]

    for idx in range(len(data)):
        data[idx]['base_out'] = output_batch[idx]
        if save_log_probs:
            data[idx]['base_first_token_probs'] = first_token_batch[idx]

    return data

def mcqa_conform_cycle(llm, sampling_params, data, max_participants=10, conform_mode='all', participant_mode='plain', save_log_probs=False):
    valid_indices = []
    
    for i in range(len(data)):
        if 'label' in data[i].keys():
            if data[i]['label'] == data[i]['base_out'][0]:
                valid_indices.append(i)
        else:
            valid_indices.append(i)
                
    for num in tqdm(range(2, max_participants+1), desc='Participant'):
        input_batch = []
        conformed_answers = []
        for idx in valid_indices:
            prompt, conformed = get_conformed_mcqa_prompt(data[idx], mpd_mcqa_prompt, bad_answer=conform_mode, participant_mode=participant_mode, num_participants=num)
            input_batch.append(prompt)
            conformed_answers.append(conformed)
        outputs = llm.generate(input_batch, sampling_params)
        output_batch = [output.outputs[0].text for output in outputs]
        if save_log_probs:
            first_token_batch = [log_prob_to_dict(output.outputs[0].logprobs[0]) for output in outputs]

        if save_log_probs:
            for idx, prompt, conformed, out, log_prob in zip(valid_indices, input_batch, conformed_answers, output_batch, first_token_batch):
                data[idx]['participant_{}'.format(str(num))] = {'prompt':prompt, 'out':out, 'conformed_answer':conformed, 'first_token_probs':log_prob}
        else:
            for idx, prompt, conformed, out in zip(valid_indices, input_batch, conformed_answers, output_batch):
                data[idx]['participant_{}'.format(str(num))] = {'prompt':prompt, 'out':out, 'conformed_answer':conformed}
    
    return data

def qa_base_cycle(llm, sampling_params, data):
    input_batch = []
    for q in data:
        prompt, _ = get_conformed_qa_prompt(q, mpd_qa_prompt, num_participants=1)
        input_batch.append(prompt)
    outputs = llm.generate(input_batch, sampling_params)
    print(outputs)
    output_batch = [output.outputs[0].text for output in outputs]

    for idx in range(len(data)):
        data[idx]['base_out'] = output_batch[idx]
    
    return data

def qa_conform_cycle(llm, sampling_params, data, max_participants=10):
    #print(outputs)
    valid_indices = []
    for i in range(len(data)):
        #breakpoint()
        if type(data[i]['possible_answers']) == list:
            candidates = data[i]['possible_answers']
        else:
            candidates = ast.literal_eval(data[i]['possible_answers'].strip("'"))
        
        for j in candidates:
            if simple_eval(j, data[i]['base_out']):
                print(j, data[i]['base_out'])
                valid_indices.append(i)
                break
    print(valid_indices)
    for num in range(2, max_participants+1):
            input_batch = []
            conformed_answers = []
            for idx in valid_indices:
                prompt, conformed = get_conformed_qa_prompt(data[idx], mpd_qa_prompt, num_participants=num)
                input_batch.append(prompt)
                conformed_answers.append(conformed)
            outputs = llm.generate(input_batch, sampling_params)
            print(outputs)
            if len(outputs[0].outputs) > 1:
                output_batch = [[x.text for x in output.outputs] for output in outputs]
            else:
                output_batch = [output.outputs[0].text for output in outputs]
            for idx, prompt, conformed, out in zip(valid_indices, input_batch, conformed_answers, output_batch):
                    data[idx]['participant_{}'.format(str(num))] = {'prompt':prompt, 'out':out, 'conformed_answer':conformed}
    
    return data

def fact_checking_base_cycle(llm, sampling_params, data):
    input_batch = []
    for d in data:
        prompt, _ = get_fact_checking_prompt(d, num_participants=1)
        input_batch.append(d)
    outputs = llm.generate(input_batch, sampling_params)
    output_batch = [output.outputs[0].text for output in outputs]
    
    filtered_data = []
    #answer pattern is not guaranteed to be contained in some of the outputs
    
    for i in range(len(data)):
        try:
            current_answer = re.findall(answer_pattern, output_batch[i])[0]
            if current_answer == 'False':
                data[i]['base_out'] = 'REFUTES'
                filtered_data.append(data[i])  
            elif current_answer == 'True':
                data[i]['base_out'] = 'SUPPORTS' 
                filtered_data.append(data[i]) 
            else:
                print('Corrupted answer at index {}'.format(str(i)))
                print(output_batch[i])
        except:
            print('No answer pattern at index {}'.format(str(i)))
            print(output_batch[i])
    return filtered_data


def fact_checking_conform_cycle(llm, sampling_params, data, conform_mode='procedural', max_participants=10):
    
    for i in range(2, max_participants+1):
        input_batch = []
        conformed_answers = []
        for d in data:
            prompt, conformed = get_fact_checking_prompt(d, given_answer=d['base_out'], conform_mode=conform_mode, num_participants=i)
            input_batch.append(prompt)
            conformed_answers.append(conformed)
        outputs = llm.generate(input_batch, sampling_params)
        output_batch = [output.outputs[0].text for output in outputs]
        
        for j in range(len(data)):
            try:
                    current_answer = re.findall(answer_pattern, output_batch[j])[0]
                    if current_answer == 'False':
                            current_answer = 'REFUTES'  
                    elif current_answer == 'True':
                            current_answer = 'SUPPORTS'
                    else:
                            current_answer = 'others'
            except:
                print('No answer pattern at index {}'.format(str(j)))
                print(output_batch[j])
                current_answer = 'others'

            data[j][conform_mode + '_participant_'+str(i)] = {'prompt':input_batch[j],'out':output_batch[j],'answer':current_answer, 'conformed_asnwer':conformed_answers[j]}
    
    return data

def bbh_conform_cycle(llm, sampling_params, data, max_participants=10):
    #print(outputs)
    valid_indices = []
    for i in range(len(data)):
        if str(parse_integer_answer(data[i]['base_out'])) == data[i]['label']:
            #breakpoint()
            valid_indices.append(i)
            
    for num in range(2, max_participants+1):
            input_batch = []
            conformed_answers = []
            for idx in valid_indices:
                prompt, conformed = get_conformed_qa_prompt(data[idx], mpd_qa_prompt, num_participants=num)
                input_batch.append(prompt)
                conformed_answers.append(conformed)
            outputs = llm.generate(input_batch, sampling_params)
            output_batch = [output.outputs[0].text for output in outputs]
            for idx, prompt, conformed, out in zip(valid_indices, input_batch, conformed_answers, output_batch):
                    data[idx]['participant_{}'.format(str(num))] = {'prompt':prompt, 'out':out, 'conformed_answer':conformed}
                    
    return data


def inference_MMLU(in_dir, llm, sampling_params, max_participants, out_dir):
    for f in tqdm(os.listdir(in_dir), desc='Files'):
        file_path = os.path.join(in_dir, f)
        data = get_MMLU_data(file_path)
        data = mcqa_base_cycle(llm, sampling_params, data)
        data = mcqa_conform_cycle(llm, sampling_params, data, max_participants=max_participants)
        with open(os.path.join(out_dir, f.replace('.csv', '.json')), 'w') as f:
            f.write(json.dumps(data, indent=2))
            f.close()
        torch.cuda.empty_cache()
        
def inference_CommonsenseQA(split, llm, sampling_params, max_participants, out_dir, conform_mode='all', participant_mode='plain'):
    data = get_CommensenseQA_data(split=split)
    data = mcqa_base_cycle(llm, sampling_params, data)
    data = mcqa_conform_cycle(llm, sampling_params, data, max_participants=max_participants, conform_mode=conform_mode, participant_mode=participant_mode)
    with open(os.path.join(out_dir, split + '.json'), 'w') as f:
        f.write(json.dumps(data, indent=2))
        f.close()
    torch.cuda.empty_cache()
    return data
    
def inference_PopQA(filepath, llm, sampling_params, max_participants, out_dir):
    data = get_popqa_data(filepath)
    data = qa_base_cycle(llm, sampling_params, data)
    data = qa_conform_cycle(llm, sampling_params, data, max_participants=max_participants)
    with open(os.path.join(out_dir, 'popqa' + '.json'), 'w') as f:
        f.write(json.dumps(data, indent=2))
        f.close()
    torch.cuda.empty_cache()
    
def inference_symmetric(filepath, llm, sampling_params, max_participants, out_dir):
    data = get_symmetric_data(filepath)
    data = fact_checking_base_cycle(llm, sampling_params, data)
    data = fact_checking_conform_cycle(llm, sampling_params, data, max_participants=max_participants, conform_mode='procedural')
    data = fact_checking_conform_cycle(llm, sampling_params, data, max_participants=max_participants, conform_mode='semantic')
    with open(os.path.join(out_dir, 'symmetric' + '.json'),'w') as f:
        f.write(json.dumps(data, indent=2))
        f.close()
    torch.cuda.empty_cache()
    
    return data

def inference_BBH_object_counting(filepath, llm, sampling_params, max_participants, out_dir):
    data = get_BBH_object_counting_data(filepath)
    data = qa_base_cycle(llm, sampling_params, data)
    #breakpoint()
    data = bbh_conform_cycle(llm, sampling_params, data, max_participants=max_participants)
    with open(os.path.join(out_dir, 'bbh_object_counting' + '.json'),'w') as f:
        f.write(json.dumps(data, indent=2))
        f.close()
    torch.cuda.empty_cache()
    return data
    
def simple_eval(str1, str2):
    return str1 in str2

if __name__ == '__main__':
    
    #TODO logging, dataset
    #wandb.init(project="conformity", name="Llama-3-8B-Instruct-PopQA-multi")
    
    with open('data/Politiscale/test.json') as f:
        data = json.load(f)
    
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=20)
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", dtype='float32')
    
    data = subjective_qa_base_cycle(llm, sampling_params, data)
    data = subjective_qa_conform_cycle(llm, sampling_params, data, conform_mode='all', participant_mode='plain')
    
    with open(os.path.join('meta-llamaMeta-Llama-3-8B-Instruct', 'Politiscale', 'test.json'), 'w') as out_f:
        out_f.write(json.dumps(data, indent=2))
        out_f.close()
    # file_path = 'data/PopQA/popqa_w_distractors_w_summary_llama3-8b_subset.jsonl'
    # sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=50)
    # out_dir = 'Llama3_8B_supplement'
    # llm = LLM(model="data_explore/llama_finetuned/supplement/cpt5e-5/checkpoint-103", dtype='float32')
    
    # data = get_popqa_data(file_path)
    # data = qa_base_cycle(llm, sampling_params, data)
    # data = qa_conform_cycle(llm, sampling_params, data, max_participants=10)
    
    # with open(os.path.join(out_dir, 'test5e-5.json'), 'w') as out_f:
    #     out_f.write(json.dumps(data, indent=2))
    #     out_f.close()
    
    # greedy_params = SamplingParams(temperature=0, top_p=1, max_tokens=50)
    # sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=50, n=10)
    # llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", dtype='float32')
    
    # file_path = 'Llama3_8B_PopQA_multi/popqa.json'
    # out_dir = 'Llama3_8B_PopQA_multi'
    
    # #data = get_popqa_data(file_path)
    # #data = qa_base_cycle(llm, greedy_params, data)
    # #data = qa_conform_cycle(llm, sampling_params, data, max_participants=10)
    # with open(file_path) as f:
    #     data =  json.load(f)
    # #data = read_jsonl(file_path)[:20]
    # valid_indicies = []
    # input_batch = []
    # for i in range(len(data)):
    #     if any([simple_eval(candidate, data[i]['base_out']) for candidate in data[i]['possible_answers']]):
    #         valid_indicies.append(i)
    #         prompt, _ = get_conformed_qa_prompt(data[i], mpd_qa_prompt, num_participants=1)
    #         input_batch.append(prompt)
    
    # outputs = llm.generate(input_batch, sampling_params) 
    # output_batch = [[x.text for x in output.outputs] for output in outputs]
    
    # for i, o in zip(valid_indicies, output_batch):
    #     data[i]['base_sampled'] = o
    
    # with open(os.path.join(out_dir, 'popqa_sampled' + '.json'), 'w') as f:
    #     f.write(json.dumps(data, indent=2))
    #     f.close()
    
    
    
    # file_path = 'data/BIG-Bench-Hard/bbh/object_counting.json'
    # out_dir = 'Llama3_8B_BBH_object_counting'
    
    # data = inference_BBH_object_counting(file_path, llm, sampling_params, 10, out_dir)
    # columns, conformed, correct, others = bbh_object_counting_eval(data)
    # participants = np.arange(2, 11)  # Use numbers 1 to 10 for participant labels
    # plot_conformity(participants, correct, conformed, others, os.path.join(out_dir, 'objc.jpg'))
    # data = inference_symmetric(file_path, llm, sampling_params, max_participants=10, out_dir=out_dir)
    
    
    # columns, conformed, correct, others = fact_checking_eval(data, max_participants=10, conform_mode='procedural')
    
    # participants = np.arange(2, 11)  # Use numbers 1 to 10 for participant labels
    # plot_conformity(participants, correct, conformed, others, os.path.join(out_dir, 'procedural.jpg'))
    
    # columns, conformed, correct, others = fact_checking_eval(data, max_participants=10, conform_mode='semantic')
    
    # participants = np.arange(2, 11)  # Use numbers 1 to 10 for participant labels
    # plot_conformity(participants, correct, conformed, others, os.path.join(out_dir, 'semantic.jpg'))
    
    # popqa = get_popqa_data('data/PopQA/popqa_with_distrctors.jsonl')

    # input_batch = [get_conformed_qa_prompt(q, mpd_qa_prompt, num_participants=1) for q in popqa]
    # outputs = llm.generate(input_batch, sampling_params)
    # #print(outputs)
    # output_batch = [output.outputs[0].text for output in outputs]

    # for idx in range(len(popqa)):
    #     popqa[idx]['base_out'] = output_batch[idx]
    # torch.cuda.empty_cache()
    
    # out_dir = 'Llama3_8B_PopQA_Test'
    # with open(os.path.join(out_dir, 'test.json'), 'w') as out_f:
    #     out_f.write(json.dumps(popqa, indent=2))
    #     out_f.close()

    # max_participants = 10
    # #print(outputs)
    # valid_indices = []
    # for i in range(len(popqa)):
    #     for j in popqa[i]['possible_answers']:
    #         if simple_eval(j, popqa[i]['base_out']):
    #             #print(j, popqa[i]['base_out'])
    #             valid_indices.append(i)
    #             break
            
    # for num in range(2, max_participants+1):
    #     input_batch = [get_conformed_qa_prompt(popqa[idx], mpd_qa_prompt, num_participants=num) for idx in valid_indices]
    #     outputs = llm.generate(input_batch, sampling_params)
    #     output_batch = [output.outputs[0].text for output in outputs]
    #     for idx, prompt, out in zip(valid_indices, input_batch, output_batch):
    #             popqa[idx]['participant_{}'.format(str(num))] = {'prompt':prompt, 'out':out}
    #     torch.cuda.empty_cache()
    
    # with open(os.path.join(out_dir, 'test.json'), 'w') as out_f:
    #     out_f.write(json.dumps(popqa, indent=2))
    #     out_f.close()
    
    
    # print('starting base cycle')
    # base_cycle(llm, sampling_params, in_dir=in_dir, out_dir=out_dir)
    # print('starting conform cycle')
    # conform_cycle(llm, sampling_params, in_dir=out_dir, out_dir=out_dir)
    wandb.finish()