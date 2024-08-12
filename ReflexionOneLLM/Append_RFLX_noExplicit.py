# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:29:05 2024

@author: xiaoyenche
"""

"""Reflexion changes here"""
Relexion_0 = "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1. \nYour task is to: look at bowl under the desklamp. \n> think: To solve the task, I need to find and take a bowl, then find and use a desklamp. \nOK. \n> think: First I need to find a bowl. A bowl is more likely to appear in drawer (1-6), desk (1-2), shelf (1-6), garbagecan (1), laundryhamper (1). I can check one by one, starting with drawer 1. \nOK. \n> go to drawer 1\nThe drawer 1 is closed. \n> open drawer 1\nYou open the drawer 1. The drawer 1 is open. In it, you see a cellphone 2. \n> go to drawer 2\nOn the drawer 2, you see a keychain 1. \n> go to drawer 3\nNothing happens. \n> go to drawer 4\nThe drawer 4 is closed. \n> open drawer 4\nYou open the drawer 4. The drawer 4 is open. In it, you see nothing. \n> go to drawer 5\nOn the drawer 5, you see nothing. \n> go to drawer 6\nThe drawer 6 is closed. \n> open drawer 6\nYou open the drawer 6. The drawer 6 is open. In it, you see a book 2, and a pen 2. \n> go to desk 1\nOn the desk 1, you see a alarmclock 1, a desklamp 1, a mug 1, and a pencil 1. \n> think: Now I find a desklamp (1). Next, I need to use it. \nOK. \n> use desklamp 1\nYou turn on the desklamp 1. \n> \nNothing happens. \n> go to desk 2\nOn the desk 2, you see a bowl 2, a bowl 1, a cd 2, a cd 1, and a creditcard 3. \n> think: Now I find a bowl (2). Next, I need to take it. \nOK. \n> take bowl 2 from desk 2\nYou pick up the bowl 2 from the desk 2. \n> \nNothing happens. \n> \nNothing happens. \nSTATUS: FAIL\nNew plan: In this environment, my plan was to find a bowl then find and use a desklamp. However, the task says to look at bowl under the desklamp. I should have looked for the desklamp then looked for the bowl. I noticed that the desklamp was found on desk 1. In the next trial, I will go to desk 1, find the lamp, then look for the bowl under the desklamp."
Relexion_1 = "You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 2, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1. \nYour task is to: heat some mug and put it in coffeemachine. \n> think: To solve the task, I need to find and take a mug, then heat it with stoveburner, then put it in coffeemachine. \nOk. \n> look\nYou are in the middle of a room. Looking quickly around you, you see nothing. \n> think: First I need to find a mug. A mug is more likely to appear in countertop (1-2), diningtable (1-2), drawer (1-3), fridge (1), sinkbasin (1), stoveburner (1-4), toaster (1). I can check one by one, starting with countertop 1. \nOk. \n> look\nYou are in the middle of a room. Looking quickly around you, you see nothing. \n> go to countertop 1\nOn the countertop 1, you see a butterknife 1, a cup 1, a fork 2, and a mug 1. \n> think: Now I find a mug (1). Next, I need to take it. \nOk. \n> take mug 1 from countertop 1\nYou pick up the mug 1 from the countertop 1. \n> think: Now I take a mug (1). Next, I need to heat it with stoveburner (1-4). \nOk. \n> go to stoveburner 1\nOn the stoveburner 1, you see a pan 2. \n> examine stoveburner 1\nOn the stoveburner 1, you see a pan 2. \n> examine stoveburner 1\nOn the stoveburner 1, you see a pan 2. \n> examine stoveburner 1\nOn the stoveburner 1, you see a pan 2. \n> examine stoveburner 1\nOn the stoveburner 1, you see a pan 2. \nSTATUS: FAIL\nNew plan: I was stuck in a loop in which I continually examined stoveburner 1 instead of heating mug 1 with stoveburner 1. I should have taken mug 1 from countertop 1, then heated it with stoveburner 1, then put it in coffeemachine 1. It did not help to execute two identical actions in a row. I will try to execute a different action if I am stuck in a loop again."
"""Reflexion ends here"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/gemma-2-9b-it"
print("\n" + "Model name: " + model_name + "\n")

access_token = "your HuggingFace access token"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=access_token
)

#%% ReAct 

import yaml
import alfworld
import alfworld.agents.environment
with open('base_config.yaml') as reader:
    config = yaml.safe_load(reader)

split = "eval_out_of_distribution"

env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

# Fix environment
env.seed(240704)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

import json
folder = './prompts/'
prompt_file = 'alfworld_RFLX_noExplicit_prompts.json'
with open(folder + prompt_file, 'r') as f:
    d = json.load(f)
    
import sys

def alfworld_run(prompt, to_print=True, ob='', env=env, env_count=0):
    exemplars = prompt
    adaptation = ''
    max_context_length = 0
    if to_print:
        print(ob)
        sys.stdout.flush()
    max_itr = 10
    for ep in range(1,max_itr):
        fail_flag = False
        init_prompt = exemplars + adaptation + '\n' + ob + '\n>'
        prompt = ''
        for i in range(1, 50):
            # action = llm(init_prompt + prompt, stop=['\n']).strip()
            
            """Change here"""
            inputs = tokenizer(
            [
                init_prompt + prompt
            ], return_tensors = "pt").to("cuda")
            # 1
            # outputs = model.generate(**inputs, max_new_tokens = 1000, use_cache = True, do_sample=False, top_p=1, repetition_penalty=0.0001,)
            # 2
            outputs = model.generate(**inputs, max_new_tokens = 400, use_cache = True, do_sample=False, top_p=1, repetition_penalty=0.0001,)
            action_1 = tokenizer.batch_decode(outputs)[0]
            # if i == 1:
            #     print("\n\n-----1st LLM output start-----\n")
            #     print(action_1)
            #     print("\n-----1st LLM output end-----\n\n")
            tokenized_context = tokenizer(init_prompt + prompt)
            input_ids = tokenized_context['input_ids']
            print("\nContext length:" + str(len(input_ids)))
            print('\n')
            max_context_length = len(input_ids)
            print(f"\n\n-----{i}th LLM output start-----\n")
            print(action_1)
            print(f"\n-----{i}th LLM output end-----\n\n")
            # print("\n-----NLP output-----\n")
            # 1
            # initializing stop string
            action_1 = action_1.replace(r"\n", "\n")
            stop=init_prompt + prompt + " "
            # slicing off after length computation    
            if stop in action_1:
                action_2 = action_1.split(stop)[1]
                # print("\n-----True 1-----\n")
            else:
                action_2 = action_1
                # print("\n-----False 1-----\n")
            # 2
            # initializing stop string
            stop="\n"
            # slicing off after length computation       
            if stop in action_2:
                action_3 = action_2.split(stop)[0]
                # print("\n-----True 2-----\n")
            else:
                action_3 = action_2
                # print("\n-----False 2-----\n")
            action = action_3
            # print(action)
            # print("\n-----NLP output end-----\n\n")
            """Changes end"""
            
            # print("\n\n ******** \n action: " + str(action))
            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            if action.startswith('think:'):
                observation = 'OK.'
            if i == 49:
                if done:
                    pass
                else:
                    fail_flag = True
            if to_print:
                print(f'Act {i}: {action}\nObs {i}: {observation}')
                sys.stdout.flush()
            if fail_flag == True:
                # R1
                # prompt += 'STATUS: FAIL\nNew plan: '
                prompt += 'STATUS: FAIL\nNew plan:'
                """Change here"""
                inputs = tokenizer(
                [
                    init_prompt + prompt
                ], return_tensors = "pt").to("cuda")
                # 1
                # outputs = model.generate(**inputs, max_new_tokens = 1000, use_cache = True, do_sample=False, top_p=1, repetition_penalty=0.0001,)
                # 2
                outputs = model.generate(**inputs, max_new_tokens = 400, use_cache = True, do_sample=False, top_p=1, repetition_penalty=0.0001,)
                action_1 = tokenizer.batch_decode(outputs)[0]
                # if i == 1:
                #     print("\n\n-----1st LLM output start-----\n")
                #     print(action_1)
                #     print("\n-----1st LLM output end-----\n\n")
                tokenized_context = tokenizer(init_prompt + prompt)
                input_ids = tokenized_context['input_ids']
                print("\nContext length:" + str(len(input_ids)))
                print('\n')
                max_context_length = len(input_ids)
                print(f"\n\n-----{i}th LLM output start-----\n")
                print(action_1)
                print(f"\n-----{i}th LLM output end-----\n\n")
                # print("\n-----NLP output-----\n")
                # 1
                # initializing stop string
                action_1 = action_1.replace(r"\n", "\n")
                stop=init_prompt + prompt # + " "
                # R1
                # stop=init_prompt + prompt + "\n"
                # slicing off after length computation    
                if stop in action_1:
                    action_2 = action_1.split(stop)[1]
                    print("\n-----True 1-----\n")
                else:
                    action_2 = action_1
                    print("\n-----False 1-----\n")
                # 2
                # initializing stop string
                stop="\n"
                # slicing off after length computation       
                if stop in action_2:
                    action_3 = action_2.split(stop)[0]
                    print("\n-----True 2-----\n")
                else:
                    action_3 = action_2
                    print("\n-----False 2-----\n")
                # R1
                # adaptation = action_3 + '\n'
                # print("Adaptation:" + adaptation)
                # print(action)
                # print("\n-----NLP output end-----\n\n")
                # 3 Get adaptation
                # initializing stop string
                stop="New plan:"
                # slicing off after length computation       
                if stop in action_3:
                    action_4 = action_3.split(stop)[1]
                    print("\n-----True 3-----\n")
                else:
                    action_4 = action_3
                    print("\n-----False 3-----\n")
                adaptation = action_4 + '\n'
                print("Adaptation:" + adaptation)
                
                env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
                env = env.init_env(batch_size=1)

                # Fix environment
                env.seed(240704)
                for _ in range(env_count):
                    ob, info = env.reset()
                ob = '\n'.join(ob[0].split('\n\n')[1:])
                print("\n\n ******** \n ob: " + str(ob) + "\n **************** \n end ob \n\n")
                name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
                print(name)
                """Changes end"""
            else:
                prompt += f' {action}\n{observation}\n>'
            if done:
                return reward, max_context_length
    return 0, max_context_length

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}
cnts = [0] * 6
rs = [0] * 6

env_count = 0
env_start_num = 15

for _ in range(134):
    env_count += 1
    ob, info = env.reset()
    if _ < (env_start_num-1):
        continue
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    # print("\n\n ******** \n ob: " + str(ob) + "\n **************** \n end ob \n\n")
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    print(name)
    avg_agent_context_length = 0
    event_cnt = 1
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            prompt = 'Interact with a household to solve a task. Here are four examples.\n' + d[f'react_{v}_1'] + '\n' + d[f'react_{v}_0'] + '\n' + Relexion_1 + '\n' + Relexion_0 + '\n' + '\nHere is the task.\n'
            print(k, v)
            r, cl = alfworld_run(prompt, ob=ob, env=env, env_count=env_count)
            # Sample average using Incremental Implementation
            avg_agent_context_length = avg_agent_context_length + (1/event_cnt)*(cl-avg_agent_context_length)
            rs[i] += r
            cnts[i] += 1
            event_cnt += 1
            break
    print(_+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
    print("Average context length: " + str(avg_agent_context_length))
    print('------------\n')