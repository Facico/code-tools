from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset
from collections import defaultdict
import json
import jsonlines
import openai
import asyncio
import time
import aiofiles

class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        """
        :param max_calls: 在 period 时间内允许的最大请求数
        :param period: 时间周期（单位：秒）
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            # 清理过期的请求记录
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                # 计算需要等待的时间
                sleep_time = self.period - (now - self.calls[0])
                await asyncio.sleep(sleep_time)
            self.calls.append(time.monotonic())

# api_key = "sk-xxx"
# base_url = "https://cloud.infini-ai.com/maas/v1"
# model_name = "deepseek-r1"
api_key = "sk-xxx"
base_url="https://api.deepseek.com"
model_name = "deepseek-chat"
cilent=AsyncOpenAI(api_key=api_key, base_url=base_url)
test_cilent=OpenAI(api_key=api_key, base_url=base_url)
async def process_single_data(data, rate_limiter) -> str:
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
            await rate_limiter.acquire()
            # print(q)
            response = await cilent.chat.completions.create(
                model="deepseek-chat", #deepseek-chat   deepseek-reasoner
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
            time.sleep(60)
            # return None

async def process_data(data_list: list, rate_limiter) -> list:
    tasks = [asyncio.create_task(process_single_data(data, rate_limiter)) for data in data_list]
    results = await asyncio.gather(*tasks)
    return results

def get_ruozhi():
    # ds = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k")["train"]
    # each_type = defaultdict(int)
    # limit = 10
    new_data = []

    # for idx, i in enumerate(ds):
    #     repo_name = i["repo_name"]
    #     if repo_name.startswith("ruozhiba"):
    #         new_data.append(i.update({"id": idx}))

    # with jsonlines.open("/home/fch/data/gen_qa/ruozhiba.jsonl","w") as f:
    #     for i in new_data:
    #         f.write(i)
    new_data = jsonlines.open("/home/fch/data/gen_qa/ruozhiba.jsonl", "r")
    return list(new_data)[:5]
async def main():
    new_data = get_ruozhi()
    # fileo = open("/home/fch/data/gen_qa/qa_choice.json", "w")
    rate_limiter = RateLimiter(max_calls=1, period=60)
    
    tasks = [asyncio.create_task(process_single_data(data, rate_limiter)) for data in new_data]
    async with aiofiles.open("/home/fch/data/gen_qa/qa_choice.json", "w", encoding="utf-8") as f:
        for future in asyncio.as_completed(tasks):
            result = await future
            await f.write(json.dumps(result,ensure_ascii=False) + "\n")
            await f.flush()
    # results = await process_data(new_data, rate_limiter)

    # for idx, result in enumerate(results, start=1):
    #     # print(f"数据项 {idx} 处理结果：{result}")

    #     fileo.write(json.dumps(result,ensure_ascii=False)+"\n")
    

async def test():
    prompt=("问题:{query}\n"
            "推理过程:{reasoning}\n"
            "答案:{answer}\n"
            "现在我想训练一个模型来回答这个问题，但是要求模型可以仅使用规则就能判断答案的正确性，所以你现在任务如下：分析`问题`，`推理过程`和`答案`，将问题转化成多选题的形式，同时要设计一个“以上没有正确选项”的选项，在问题中加入请一步步推理，并把最终答案放到 \\boxed{{}}。要求生成两种形式的问题并给出正确答案："
            "1、正确答案在选项中，答案为那些正确选项"
            "2、正确答案不在选项中，答案为“以上没有正确选项”的选项"
            "注意，生成的问题还需要满足一下要求："
            "1、如果现在生成的问题是`推理过程`或者`答案`编写的（如答案编写的故事，文案等），需要将相应的文本塞入问题中。"
            "2、对于开放性的问题，尽可能生成有多项正确答案的问题，并在\\boxed{{}}中填入多个答案。")
    data=get_ruozhi()
    query,answer,reasoning=data[0]["input"],data[0]["content"],data[0]["reasoning_content"]
    q = prompt.format_map({"query": query, "answer": answer, "reasoning": reasoning})
    while 1:
        try:
            response = await cilent.chat.completions.create(
                    model="deepseek-chat", #deepseek-chat   deepseek-reasoner
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": q}
                    ],
                    temperature=1.0,
                    stream=False
                )
            break
        except:
            print("Error and sleep")
            time.sleep(60)
    # print(response.choices[0].message.content)
    data_o = {
        "input": q,
        "content": response.choices[0].message.content,
        "reasoning_content": response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, "reasoning_content") else ""
    }
    print(data_o)
if __name__ == '__main__':
    asyncio.run(main())
    # import defopt
    # try:
    #     defopt.run(main)
    # except:
    #     import sys,pdb,bdb
    #     type, value, tb = sys.exc_info()
    #     if type == bdb.BdbQuit:
    #         exit()
    #     print(type,value)
    #     pdb.post_mortem(tb)
    # asyncio.run(test())