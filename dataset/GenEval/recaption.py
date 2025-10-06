import openai
import json

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path ./qwen2-5 --tp 4"
)

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

with open("geneval_metadata.jsonl", "r", encoding="utf-8") as fin, \
     open("geneval_metadata_detailed.jsonl", "a", encoding="utf-8") as fout:
    for line in fin:
        sample = json.loads(line)
        prompt = sample.get("prompt", "")
        # 构造系统提示词
        system_prompt = (
            "You are an expert image captioner. "
            "Given a short prompt describing an image, focus on the original object/property/count/location and then rewrite it into a comprehensive and vivid description,"
            "Do not omit any information from the original prompt. Besides, and make sure the recaption is no longer than 200 words."
        )
        # 向qwen发送请求
        try:
            completion = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                timeout=600,
                max_completion_tokens=256
            )
            detailed_prompt = completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error processing prompt: {prompt}\n{e}")
            detailed_prompt = ""
        # 添加新字段并写回
        sample["detailed_prompt"] = detailed_prompt
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")