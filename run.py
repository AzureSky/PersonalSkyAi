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
          
        prompt_text = data.get("prompt", "")  
        image_b64 = data.get("imageBase64")  
        raw_model = data.get("model", DEFAULT_MODEL)  
        history_list = data.get("history", [])   
  
        model_name = raw_model.replace("google/", "")  
          
        # --- 1. 构建请求内容 ---  
        all_contents = []  
  
        # 处理历史  
        for msg in history_list:  
            role = "user" if msg['role'] == 'user' else "model"  
            content_text = msg.get('content', '')  
            if not content_text or not content_text.strip():  
                content_text = "[图片/文件]"  
              
            all_contents.append(types.Content(  
                role=role,  
                parts=[types.Part(text=content_text)]  
            ))  
  
        # 处理当前消息  
        current_parts = []  
        if image_b64:  
            try:  
                img_data = base64.b64decode(image_b64)  
                current_parts.append(types.Part(  
                    inline_data=types.Blob(  
                        mime_type="image/jpeg",  
                        data=img_data  
                    )  
                ))  
            except Exception as e:  
                logger.error(f"上传图片解码失败: {e}")  
  
        if prompt_text:  
            current_parts.append(types.Part(text=prompt_text))  
          
        if current_parts:  
            all_contents.append(types.Content(  
                role="user",  
                parts=current_parts  
            ))  
          
        if not all_contents:  
             return jsonify({"code": -1, "error": "内容不能为空"}), 400  
  
        logger.info(f"发送请求: Model={model_name}")  
  
        # --- 2. 调用 AI ---  
        response = google_client.models.generate_content(  
            model=model_name,  
            contents=all_contents  
        )  
  
        # --- 3. 解析结果 (核心修改) ---  
        reply_text = ""  
        reply_image = None # 存放 AI 生成的图片 Base64  
  
        if response.candidates:  
            for part in response.candidates[0].content.parts:  
                # 3.1 提取文本  
                if part.text:  
                    reply_text += part.text  
                  
                # 3.2 提取生成的图片 (文生图关键)  
                if part.inline_data:  
                    logger.info("检测到 AI 返回了图片数据")  
                    try:  
                        # 获取二进制数据  
                        img_bytes = part.inline_data.data  
                        # 转为 Base64 字符串  
                        b64_str = base64.b64encode(img_bytes).decode('utf-8')  
                        mime_type = part.inline_data.mime_type or "image/png"  
                        # 拼接成前端可用的格式  
                        reply_image = f"data:{mime_type};base64,{b64_str}"  
                    except Exception as img_e:  
                        logger.error(f"AI 图片处理失败: {img_e}")  
  
        # --- 4. 返回结果 ---  
        # 只要有文本或者有图片，都算成功  
        if reply_text or reply_image:  
            return jsonify({  
                "code": 0,  
                "reply": reply_text,  
                "generated_image": reply_image   
            })  
        else:  
            logger.error("AI 响应解析为空 (可能是被安全策略拦截)")  
            return jsonify({"code": -1, "error": "AI 未返回有效内容"}), 500  
  
    except Exception as e:  
        logger.error(f"后端报错: {e}")  
        import traceback  
        traceback.print_exc()  
        return jsonify({"code": -1, "error": f"Error: {str(e)}"}), 500  
  
if __name__ == '__main__':  
    port = int(os.environ.get('PORT', 80))  
    app.run(host='0.0.0.0', port=port)  
