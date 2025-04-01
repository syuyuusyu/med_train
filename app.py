from flask import Flask, request, jsonify
from services.retriever import CustomRetriever
from openai import OpenAI
from services.query_db_data import doc_info
import os

# Initialize the retriever
passages = doc_info()
retriever = CustomRetriever(passages)

client = OpenAI(
    # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
    api_key = os.getenv("ALI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

app = Flask(__name__)

@app.route('/ai/triage', methods=['POST'])
def triage():
    data = request.json
    query = data.get('query', '')
    retrieved_chunks = retriever.retrieve(query, k=10)

    completion = client.chat.completions.create(
        model="qwq-32b",  # 此处以 qwq-32b 为例，可按需更换模型名称
        messages=[
            {"role": "system", "content": "你现在是一个分诊系统，你会收到医生的信息和患者的问题，给出合适的科室和医生推荐。"},
            {"role": "system", "content": f"这里是相关的医生信息 {retrieved_chunks}"},
            {"role": "system", "content": f"优先根据医生的信息进行匹配，如果医生信息和患者问题匹配，但医生所在科室不是很匹配，那么也要给出推荐。"},
            {"role": "system", "content": f"按照医生的职称等级进行推荐,如果职称等级相同，则按照医生的擅长进行推荐。"},
            {"role": "system", "content": f"如果在医生信息中所在科室的内容是 '医生当前没有排班',也给出推荐，但是要说明医生当前没有排班。"},
            {"role": "system", "content": f"如果在医生信息中没有找到合适的医生，请给出科室推荐。"},
            {"role": "system", "content": f"如果医生信息中的医院和问题中的医院不对应，则忽略该医生的信息。"},
            {"role": "system", "content": f"如果患者的问题和分诊没关系，则告诉患者分诊系统无法处理该问题。"},
            {"role": "system", "content": f"""
            将推存的科室ID和医生ID组成一个特定的url,类似这样https://www.51bqm.com/jyt/doctorSelect?hospitalId=hp_id&deptId=deptId1_deptId2_docId_docId1_docId2
            其中的hp_id是医生信息中的医院ID，deptId1_deptId2是推荐的科室ID，docId_docId1_docId2是推荐的医生ID。需要将所有的合适的科室ID和医生ID都拼接在一起。
            同时需要注意 deptId是一个完整的参数，类似这样 deptId=111_222_docId_333_444
            不要返回这样的参数 deptId=111_222&docId=333_444
            返回的URL使用正确的markdown语法。[点击跳转挂号页面](URL) 方便用户可以直接点击跳转,跳转的链接放在推荐的最后
            """},
            {"role": "user", "content": f"患者的问题是：{query}"},
        ],
        # QwQ 模型仅支持流式输出方式调用
        stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        stream_options={
            "include_usage": True
        }
    )
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复

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
    return jsonify(
        {
            "data": answer_content,
            "success": True
        }) 

@app.route('/ai/reloadDocinfo', methods=['GET'])
def reloadDocinfo():
    global retriever
    passages = doc_info()
    retriever.__init__(passages)
    return jsonify({"success": "success", "data": "医生信息已重新加载"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8006)