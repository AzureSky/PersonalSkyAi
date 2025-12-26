import os  
import logging  
import base64  
import json  
from flask import Flask, request, jsonify  
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
        history_list = data.get("history", [])   
  
        model_name = raw_model.replace("google/", "")  
          
        # 2. 构建上下文 (Contents)  
        all_contents = []  
  
        # ==========================================  
        # 处理历史记录  
        # ==========================================  
        for msg in history_list:  
            role = "user" if msg['role'] == 'user' else "model"  
            content_text = msg.get('content', '')  
              
            if not content_text or not content_text.strip():  
                content_text = "[用户发送了一张图片]"  
  
            # 🟢 修正：直接使用构造函数 types.Part(text=...)  
            # 不再使用 types.Part.from_text()  
            all_contents.append(types.Content(  
                role=role,  
                parts=[types.Part(text=content_text)]  
            ))  
  
        # ==========================================  
        # 处理当前消息  
        # ==========================================  
        current_parts = []  
          
        # A. 处理图片  
        if image_b64:  
            try:  
                img_data = base64.b64decode(image_b64)  
                # 🟢 修正：直接构建 Blob 对象，不使用快捷方法  
                current_parts.append(types.Part(  
                    inline_data=types.Blob(  
                        mime_type="image/jpeg",  
                        data=img_data  
                    )  
                ))  
            except Exception as e:  
                logger.error(f"图片解码失败: {e}")  
  
        # B. 处理文本  
        if prompt_text:  
            # 🟢 修正：直接使用构造函数 types.Part(text=...)  
            # 这里的 text= 是关键字参数，绝对不会错  
            current_parts.append(types.Part(text=prompt_text))  
          
        # C. 组装  
        if current_parts:  
            all_contents.append(types.Content(  
                role="user",  
                parts=current_parts  
            ))  
          
        if not all_contents:  
             return jsonify({"code": -1, "error": "发送内容不能为空"}), 400  
  
        logger.info(f"发送请求: Model={model_name}, HistoryCount={len(history_list)}")  
  
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
        logger.error(f"后端处理出错: {e}")  
        import traceback  
        traceback.print_exc()  
        return jsonify({"code": -1, "error": f"Internal Error: {str(e)}"}), 500  
  
if __name__ == '__main__':  
    port = int(os.environ.get('PORT', 80))  
    app.run(host='0.0.0.0', port=port)  
