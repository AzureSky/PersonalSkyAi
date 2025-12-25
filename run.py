import os  
import logging  
from flask import Flask, request, jsonify  
from google import genai  
from google.genai import types  
  
# 配置日志  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
app = Flask(__name__)  
  
# 获取环境变量中的 Key  
API_KEY = os.environ.get("GEMINI_API_KEY")  
  
# ------------------------------------------------------------------  
# 🔴 核心配置：完全保留你的 ZenMux 集成代码  
# ------------------------------------------------------------------  
if API_KEY:  
    try:  
        google_client = genai.Client(  
            api_key=API_KEY,  
            vertexai=True,  
            http_options=types.HttpOptions(  
                api_version='v1',  
                base_url='https://zenmux.ai/api/vertex-ai'  
            ),  
        )  
        logger.info("ZenMux Client 初始化成功")  
    except Exception as e:  
        logger.error(f"Client 初始化失败: {e}")  
        google_client = None  
else:  
    logger.error("未找到 GEMINI_API_KEY 环境变量")  
    google_client = None  
  
# 默认模型  
DEFAULT_MODEL = "gemini-3-flash-preview"  
  
@app.route('/api/chat', methods=['POST'])  
def chat():  
    if not google_client:  
        return jsonify({"error": "服务端配置错误: API Key 未设置或 Client 初始化失败"}), 500  
  
    data = request.json  
    prompt = data.get("prompt", "")  
    # 允许前端传 model 参数覆盖，否则使用默认的 flash-preview  
    model_name = data.get("model", DEFAULT_MODEL)   
  
    if not prompt:  
        return jsonify({"error": "Prompt 不能为空"}), 400  
  
    logger.info(f"收到请求，模型: {model_name}, Prompt长度: {len(prompt)}")  
  
    try:  
        # 调用 AI (使用你的 Client)  
        response = google_client.models.generate_content(  
            model=model_name,  
            contents=prompt,  
            # 如果你需要返回 JSON 格式，可以在这里加 config，目前先按纯文本返回调试  
        )  
  
        # 提取文本内容  
        if response.text:  
            return jsonify({"reply": response.text})  
        else:  
            return jsonify({"reply": "AI 未返回文本内容"}), 500  
  
    except Exception as e:  
        logger.error(f"调用 ZenMux 失败: {e}")  
        return jsonify({"error": str(e)}), 500  
  
if __name__ == '__main__':  
    port = int(os.environ.get('PORT', 80))  
    app.run(host='0.0.0.0', port=port)  
