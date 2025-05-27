
import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel,AutoConfig
import json
import torch
from tqdm import tqdm
import pandas as pd
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] ='0'



def build_prompt(history):
    prompt = "欢迎使用 TCMLLM-PR 中医智能处方推荐助手，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nTCMLLM-PR：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print("欢迎使用 TCMLLM-PR 中医智能处方推荐，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 TCMLLM-PR 中医智能处方推荐，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue

        response, history = model.chat(tokenizer, query, history=history)
        print(f"TCMLLM-PR：{response}")

        """
        # 流式回答方法用起来效果不是很好
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        # os.system(clear_command)
        # print(build_prompt(history), flush=True)
        """
          


def main_muti_output(dis_name,data_name,checkpoint_path):
    test_epoach_path ='/home/thy/code/TCMLLM/ptuning/' + checkpoint_path +'/'
    data_path ='/home/thy/code/TCMLLM/ptuning/data/'+ data_name + '.json'
    tokenizer = AutoTokenizer.from_pretrained("/home/thy/code/ChatGLM-6B/chatglm-6b", trust_remote_code=True) 

    #推理
    config = AutoConfig.from_pretrained("/home/thy/code/ChatGLM-6B/chatglm-6b", trust_remote_code=True, pre_seq_len=128) # 这里是token长度，记的调整！
    model = AutoModel.from_pretrained("/home/thy/code/ChatGLM-6B/chatglm-6b", config=config, trust_remote_code=True).half().cuda()
    prefix_state_dict = torch.load(test_epoach_path + "pytorch_model.bin")
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)



    model = model.eval()

    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    stop_stream = False

    history = []
    result_list = []
    print("欢迎使用 TCMLLM-PR 中医智能处方推荐助手，测试开始～")
    with open(data_path, "r", encoding="utf-8-sig") as f:
        json_list = json.load(f)
        json_size = int(len(json_list))
        for index_ in tqdm(range(json_size)):
            query = json_list[index_]['Content']
            response, history = model.chat(tokenizer, query, history=history)
            gold_summary = json_list[index_]['Summary']
            if len(response)>512: response = response[:512]
            result_list.append({'content':query, 'summary':response,
                                 'gold_summary':gold_summary})
            history = []
            # if index_%10 == 0: print(index_,' over')
        with open(test_epoach_path + 'muti_output_'+ data_type + '.json', 'w', encoding='utf-8') as f:
            json.dump(result_list, f,ensure_ascii=False)
        pd_result = pd.DataFrame(result_list)
        pd_result.to_excel(test_epoach_path + 'muti_output_'+ data_type + '.xlsx',index=False, encoding='utf_8_sig')
# UnicodeEncodeError: 'ascii' codec can't encode characters in position 1-29: ordinal not in range(128)


   


if __name__ == "__main__":
    data_type = 'test'
    

    main_muti_output('ISGP',data_name = 'ISGP_test2',checkpoint_path = 'checkpoint_v1')
        
