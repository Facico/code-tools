from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset
from collections import defaultdict
import json
import jsonlines
import openai
import asyncio
import time
import aiofiles
from tqdm import tqdm


# api_key = "sk-xxx"
# base_url = "https://cloud.infini-ai.com/maas/v1"
# model_name = "deepseek-r1"
api_key = "sk-xxx"
base_url="https://api.deepseek.com"
model_name = "deepseek-chat"
cilent=OpenAI(api_key=api_key, base_url=base_url)
def process_single_data(data) -> str:
    query, answer, reasoning = data["input"],data["content"],data["reasoning_content"]
    prompt=("问题:{query}\n"
            "推理过程:{reasoning}\n"
            "答案:{answer}\n"
            "现在我想训练一个模型来回答这个问题，但是要求模型可以仅使用规则就能判断答案的正确性，所以你现在任务如下：分析`问题`，`推理过程`和`答案`，将问题转化成多选题的形式，同时要设计一个“以上没有正确选项”的选项，在问题中加入请一步步推理，并把最终答案放到 \\boxed{{}}。要求生成两种形式的问题并给出正确答案："
            "1、正确答案在选项中，答案为那些正确选项"
            "2、正确答案不在选项中，答案为“以上没有正确选项”的选项"
            "注意，生成的问题还需要满足一下要求："
            "1、如果现在生成的问题是`推理过程`或者`答案`编写的（如答案编写的故事，文案等），需要将相应的文本塞入问题中。"
            "2、对于开放性的问题，尽可能生成有多项正确答案的问题，并在\\boxed{{}}中填入多个答案。")
    q = prompt.format_map({"query": query, "answer": answer, "reasoning": reasoning})
    while 1:
        try:
            response = cilent.chat.completions.create(
                model="deepseek-reasoner", #deepseek-chat   deepseek-reasoner
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": q}
                ],
                temperature=1.0,
            )
            
            data_o = {
                "input": q,
                "content": response.choices[0].message.content,
                "reasoning_content": response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, "reasoning_content") else ""
            }
            return data_o
        except Exception as e:
            print(f"处理数据时出错: {e}")
            time.sleep(60*3)
            # return None

def get_ruozhi():
    # ds = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k")["train"]
    # each_type = defaultdict(int)
    # limit = 10
    new_data = []
    new_data = jsonlines.open("/home/fch/data/gen_qa/ruozhiba.jsonl", "r")
    return list(new_data)
def main():
    new_data = get_ruozhi()
    fileo = open("/home/fch/data/gen_qa/qa_choice.json", "w")
    
    
    for i in tqdm(new_data):
        result = process_single_data(i)
        fileo.write(json.dumps(result,ensure_ascii=False) + "\n")

    

if __name__ == '__main__':
    import defopt
    try:
        defopt.run(main)
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
    # asyncio.run(test())