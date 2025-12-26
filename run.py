import os  
import logging  
import base64  
import io  
import json  
from flask import Flask, request, jsonify  
from PIL import Image  
from google import genai  
from google.genai import types  
  
# 1. 基础配置  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
app = Flask(__name__)  
  
# 获取环境变量  
API_KEY = os.environ.get("GEMINI_API_KEY")  
  
# 2. 初始化 Client (ZenMux + Vertex AI) - 严格保留你的配置  
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
    # 检查 Client 状态  
    if not google_client:  
        return jsonify({"code": -1, "error": "服务端 API Key 配置错误"}), 500  
  
    try:  
        # 3. 解析前端传来的 JSON 数据  
        data = request.get_json() or {}  
          
        prompt_text = data.get("prompt", "")  
        image_b64 = data.get("imageBase64") # 对应你 JS 里的字段名  
        raw_model = data.get("model", DEFAULT_MODEL)  
  
        # 处理模型名称：前端传的是 'google/gemini-3...', SDK 可能只需要 'gemini-3...'  
        # 这里做一个简单的兼容处理，去掉 'google/' 前缀  
        model_name = raw_model.replace("google/", "")  
  
        logger.info(f"请求: Model={model_name}, PromptLen={len(prompt_text)}, HasImg={bool(image_b64)}")  
  
        contents = []  
  
        # 4. 图片处理 (Base64 -> PIL Image)  
        if image_b64:  
            try:  
                # 解码 Base64 字符串  
                img_data = base64.b64decode(image_b64)  
                img_stream = io.BytesIO(img_data)  
                img = Image.open(img_stream)  
                contents.append(img)  
            except Exception as e:  
                logger.error(f"图片解码失败: {e}")  
                return jsonify({"code": -1, "error": "图片格式无效"}), 400  
  
        # 5. 文本处理  
        if prompt_text:  
            contents.append(prompt_text)  
        elif not image_b64:  
            # 既没图也没字  
            return jsonify({"code": -1, "error": "输入不能为空"}), 400  
  
        # 6. 调用 AI (ZenMux)  
        response = google_client.models.generate_content(  
            model=model_name,  
            contents=contents  
        )  
  
        # 7. 返回结果 (匹配你的 JS success 判断逻辑: code === 0)  
        if response.text:  
            return jsonify({  
                "code": 0,  
                "reply": response.text  
            })  
        else:  
            return jsonify({"code": -1, "error": "AI 未返回文本内容"}), 500  
  
    except Exception as e:  
        logger.error(f"处理异常: {e}")  
        return jsonify({"code": -1, "error": str(e)}), 500  
  
if __name__ == '__main__':  
    port = int(os.environ.get('PORT', 80))  
    app.run(host='0.0.0.0', port=port)  
