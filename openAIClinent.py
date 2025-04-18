from openai import OpenAI
import os
from services.retriever import CustomRetriever
from services.query_db_data import doc_info

question = '昭通市中医医院 kknd'
doc_infos = doc_info()

retriever = CustomRetriever(doc_infos)

retrieved_chunks = retriever.retrieve(question, k=10)

# 初始化OpenAI客户端
client = OpenAI(
    # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
    api_key = os.getenv("ALI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

reasoning_content = ""  # 定义完整思考过程
answer_content = ""     # 定义完整回复
is_answering = False   # 判断是否结束思考过程并开始回复

# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwq-32b",  # 此处以 qwq-32b 为例，可按需更换模型名称
    messages=[
        {"role": "user", "content": "你现在是一个分诊系统，你会收到医生的信息和患者的问题，给出合适的科室和医生推荐。"},
        {"role": "user", "content": f"这里是相关的医生信息 {retrieved_chunks}"},
        {"role": "user", "content": f"如果在医生信息中所在科室的内容是 '医生当前没有排班',也给出推荐，但是要说明医生当前没有排班。"},
        {"role": "user", "content": f"如果在医生信息中没有找到合适的医生，请给出科室推荐。"},
        {"role": "user", "content": f"患者的问题是：{question}"},
    ],
    # QwQ 模型仅支持流式输出方式调用
    stream=True,
    # 解除以下注释会在最后一个chunk返回Token使用量
    stream_options={
        "include_usage": True
    }
)

print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    # 如果chunk.choices为空，则打印usage
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content

# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(reasoning_content)
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(answer_content)