import os  
import logging  
import base64  
import io  
import json  
from flask import Flask, request, jsonify  
from PIL import Image  
from google import genai  
from google.genai import types  
  
# 基础配置  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
app = Flask(__name__)  
  
# 获取环境变量  
API_KEY = os.environ.get("GEMINI_API_KEY")  
  
# 初始化 Client  
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
  
DEFAULT_MODEL = "gemini-3-flash-preview"  
  
@app.route('/api/chat', methods=['POST'])  
def chat():  
    if not google_client:  
        return jsonify({"code": -1, "error": "服务端配置错误"}), 500  
  
    try:  
        data = request.get_json() or {}  
          
        # 1. 获取参数  
        prompt_text = data.get("prompt", "")  
        image_b64 = data.get("imageBase64")  
        raw_model = data.get("model", DEFAULT_MODEL)  
        history_list = data.get("history", []) # 获取前端传来的纯文本历史  
  
        model_name = raw_model.replace("google/", "")  
          
        # 2. 构建上下文 (Contents)  
        # Google SDK 接受一个 list，包含之前的对话和当前的消息  
        all_contents = []  
  
        # A. 处理历史记录 (只包含文本，前端已经过滤了图片)  
        for msg in history_list:  
            role = "user" if msg['role'] == 'user' else "model"  
            # 构造 Content 对象  
            all_contents.append(types.Content(  
                role=role,  
                parts=[types.Part.from_text(msg['content'])]  
            ))  
  
        # B. 处理当前最新的消息 (文本 + 图片)  
        current_parts = []  
          
        # 如果有图片，解码并加入  
        if image_b64:  
            try:  
                img_data = base64.b64decode(image_b64)  
                img_stream = io.BytesIO(img_data)  
                img = Image.open(img_stream)  
                current_parts.append(img)  
            except Exception as e:  
                logger.error(f"图片解码失败: {e}")  
                # 图片坏了不影响文本发送，记录日志即可  
  
        if prompt_text:  
            current_parts.append(prompt_text)  
          
        # 将当前消息加入列表  
        if current_parts:  
            all_contents.append(types.Content(  
                role="user",  
                parts=current_parts  
            ))  
          
        if not all_contents:  
             return jsonify({"code": -1, "error": "内容为空"}), 400  
  
        logger.info(f"发送请求: Model={model_name}, HistoryCount={len(history_list)}, HasImg={bool(image_b64)}")  
  
        # 3. 调用 AI  
        response = google_client.models.generate_content(  
            model=model_name,  
            contents=all_contents  
        )  
  
        # 4. 返回结果  
        if response.text:  
            return jsonify({  
                "code": 0,  
                "reply": response.text  
            })  
        else:  
            return jsonify({"code": -1, "error": "AI 未返回文本"}), 500  
  
    except Exception as e:  
        logger.error(f"处理异常: {e}")  
        return jsonify({"code": -1, "error": str(e)}), 500  
  
if __name__ == '__main__':  
    port = int(os.environ.get('PORT', 80))  
    app.run(host='0.0.0.0', port=port)  
