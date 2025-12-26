import os  
import logging  
from flask import Flask, request, jsonify  
from PIL import Image  
from google import genai  
from google.genai import types  
  
# 配置日志  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
app = Flask(__name__)  
  
# ------------------------------------------------------------------  
# 1. 初始化 Client (严格照抄你的配置)  
# ------------------------------------------------------------------  
API_KEY = os.environ.get("GEMINI_API_KEY")  
  
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
    logger.error("严重错误: 未找到 GEMINI_API_KEY 环境变量")  
    google_client = None  
  
# 默认模型 (遵照你的要求)  
DEFAULT_MODEL = "gemini-3-flash-preview"  
  
@app.route('/api/chat', methods=['POST'])  
def chat():  
    if not google_client:  
        return jsonify({"success": False, "msg": "服务端API Key未配置"}), 500  
  
    # -----------------------------------------------------------  
    # 2. 接收小程序传来的数据 (multipart/form-data)  
    # -----------------------------------------------------------  
    # 文本内容  
    prompt_text = request.form.get("prompt", "")  
    # 模型选择 (允许前端覆盖)  
    model_name = request.form.get("model", DEFAULT_MODEL)  
    # 图片文件  
    image_file = request.files.get("image")  
  
    logger.info(f"收到请求: Prompt长度={len(prompt_text)}, 有图片={bool(image_file)}")  
  
    # 准备发送给 AI 的内容列表  
    contents = []  
  
    try:  
        # 如果有图片，用 Pillow 打开并加入列表  
        if image_file:  
            try:  
                img = Image.open(image_file)  
                contents.append(img)  
            except Exception as e:  
                logger.error(f"图片解析失败: {e}")  
                return jsonify({"success": False, "msg": "图片格式无效"}), 400  
          
        # 如果有文本，加入列表  
        if prompt_text:  
            contents.append(prompt_text)  
        elif not image_file:  
            # 既没图也没字，无法处理  
            return jsonify({"success": False, "msg": "输入不能为空"}), 400  
  
        # -----------------------------------------------------------  
        # 3. 调用 AI (ZenMux 通道)  
        # -----------------------------------------------------------  
        response = google_client.models.generate_content(  
            model=model_name,  
            contents=contents  
        )  
  
        # -----------------------------------------------------------  
        # 4. 返回结果给小程序  
        # 结构设计为支持 { text: "...", image: "..." }  
        # 目前 Gemini Flash 主要返回文本，image 字段预留为空  
        # -----------------------------------------------------------  
        if response.text:  
            return jsonify({  
                "success": True,  
                "data": {  
                    "type": "text",       # 标识返回类型  
                    "content": response.text,  
                    "image": None         # 如果将来用生图模型，这里可以放 Base64 图片  
                }  
            })  
        else:  
            return jsonify({"success": False, "msg": "AI 未返回有效内容"}), 500  
  
    except Exception as e:  
        logger.error(f"AI 调用出错: {e}")  
        return jsonify({"success": False, "msg": str(e)}), 500  
  
if __name__ == '__main__':  
    port = int(os.environ.get('PORT', 80))  
    app.run(host='0.0.0.0', port=port)  
