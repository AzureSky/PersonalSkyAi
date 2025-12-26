import os  
import logging  
import base64  
import uuid  
import threading  
import time  
import requests  
from flask import Flask, request, jsonify  
from google import genai  
from google.genai import types  
  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
app = Flask(__name__)  
  
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
  
DEFAULT_MODEL = "gemini-3-flash-preview"  
  
# --- 任务存储 (生产环境建议用Redis，这里用内存) ---  
TASK_STORE = {}   
  
def process_ai_task(job_id, data):  
    """后台线程：下载图片 -> 调用 AI -> 存结果"""  
    logger.info(f"[{job_id}] 开始处理任务...")  
    try:  
        prompt_text = data.get("prompt", "")  
        image_url = data.get("imageUrl") # 前端发来的 HTTP 链接  
        raw_model = data.get("model", DEFAULT_MODEL)  
        history_list = data.get("history", [])   
  
        model_name = raw_model.replace("google/", "")  
          
        all_contents = []  
  
        # 1. 处理历史 (只保留文本，避免历史图片过大)  
        for msg in history_list:  
            role = "user" if msg['role'] == 'user' else "model"  
            content_text = msg.get('content', '') or "[图片/文件]"  
            all_contents.append(types.Content(role=role, parts=[types.Part(text=content_text)]))  
  
        # 2. 处理当前消息  
        current_parts = []  
          
        # --- 下载前端上传的大图 ---  
        if image_url:  
            try:  
                logger.info(f"[{job_id}] 正在下载图片: {image_url}")  
                # 这里的 timeout 设长一点，防止大图下载慢  
                img_resp = requests.get(image_url, timeout=60)   
                if img_resp.status_code == 200:  
                    current_parts.append(types.Part(  
                        inline_data=types.Blob(  
                            mime_type="image/jpeg",   
                            data=img_resp.content  
                        )  
                    ))  
                    logger.info(f"[{job_id}] 图片下载完成，大小: {len(img_resp.content)} bytes")  
                else:  
                    logger.error(f"[{job_id}] 图片下载失败: {img_resp.status_code}")  
            except Exception as e:  
                logger.error(f"[{job_id}] 图片下载异常: {e}")  
  
        if prompt_text:  
            current_parts.append(types.Part(text=prompt_text))  
          
        if current_parts:  
            all_contents.append(types.Content(role="user", parts=current_parts))  
          
        # 3. 调用 AI  
        logger.info(f"[{job_id}] 请求 AI (Model: {model_name})...")  
        response = google_client.models.generate_content(  
            model=model_name,  
            contents=all_contents  
        )  
  
        # 4. 解析结果  
        reply_text = ""  
        reply_image = None   
  
        if response.candidates:  
            for part in response.candidates[0].content.parts:  
                if part.text:  
                    reply_text += part.text  
                if part.inline_data:  
                    # 如果 AI 画了图，转 Base64 返回  
                    b64_str = base64.b64encode(part.inline_data.data).decode('utf-8')  
                    mime = part.inline_data.mime_type or "image/png"  
                    reply_image = f"data:{mime};base64,{b64_str}"  
  
        # 5. 任务成功，存入结果  
        TASK_STORE[job_id] = {  
            "status": "success",  
            "data": {  
                "reply": reply_text,  
                "generated_image": reply_image  
            }  
        }  
        logger.info(f"[{job_id}] 任务完成")  
  
    except Exception as e:  
        logger.error(f"[{job_id}] 任务失败: {e}")  
        TASK_STORE[job_id] = {  
            "status": "fail",  
            "error": str(e)  
        }  
  
@app.route('/api/chat', methods=['POST'])  
def start_chat_task():  
    """接口1：接收请求，开启线程"""  
    if not google_client:  
        return jsonify({"code": -1, "error": "服务端未就绪"}), 500  
  
    data = request.get_json() or {}  
    job_id = str(uuid.uuid4()) # 生成唯一 ID  
      
    # 初始状态  
    TASK_STORE[job_id] = {"status": "processing"}  
      
    # 异步执行  
    thread = threading.Thread(target=process_ai_task, args=(job_id, data))  
    thread.start()  
      
    return jsonify({"code": 0, "job_id": job_id})  
  
@app.route('/api/query', methods=['POST'])  
def query_task_status():  
    """接口2：前端轮询结果"""  
    data = request.get_json() or {}  
    job_id = data.get("job_id")  
      
    if not job_id or job_id not in TASK_STORE:  
        return jsonify({"code": -1, "error": "任务过期或不存在"}), 404  
          
    task = TASK_STORE[job_id]  
      
    if task['status'] == 'processing':  
        return jsonify({"code": 1, "status": "processing"}) # 1 = 继续等  
    elif task['status'] == 'success':  
        result = task['data']  
        del TASK_STORE[job_id] # 取完即删  
        return jsonify({"code": 0, "status": "success", **result})  
    else:  
        del TASK_STORE[job_id]  
        return jsonify({"code": -1, "status": "fail", "error": task.get("error")})  

@app.route('/')  
def ping():  
    """唤醒接口，不做任何事，只为了拉起实例"""  
    return "pong", 200  
  
if __name__ == '__main__':  
    port = int(os.environ.get('PORT', 80))  
    app.run(host='0.0.0.0', port=port)  
