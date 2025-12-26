import os  
import logging  
import base64  
import uuid  
import threading  
import time  
import json  
import requests  
from flask import Flask, request, jsonify  
from google import genai  
from google.genai import types  
  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
app = Flask(__name__)  
  
# --- 配置部分 ---  
API_KEY = os.environ.get("GEMINI_API_KEY")  
WX_APPID = os.environ.get("WX_APPID")   
WX_SECRET = os.environ.get("WX_SECRET")   
WX_ENV_ID = os.environ.get("WX_ENV_ID")  
  
# 初始化 Gemini Client  
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
TASK_STORE = {}  
  
# --- 微信 Token 管理 ---  
class WXTokenManager:  
    def __init__(self):  
        self.access_token = None  
        self.expires_at = 0  
  
    def get_token(self):  
        """获取并缓存 Access Token"""  
        if self.access_token and time.time() < self.expires_at:  
            return self.access_token  
          
        if not WX_APPID or not WX_SECRET:  
            logger.error("缺少 WX_APPID 或 WX_SECRET 环境变量")  
            return None  
  
        url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={WX_APPID}&secret={WX_SECRET}"  
        try:  
            resp = requests.get(url, timeout=10)  
            data = resp.json()  
            if "access_token" in data:  
                self.access_token = data["access_token"]  
                # 提前 200 秒过期，防止临界点失效  
                self.expires_at = time.time() + data["expires_in"] - 200  
                logger.info("获取微信 AccessToken 成功")  
                return self.access_token  
            else:  
                logger.error(f"获取 AccessToken 失败: {data}")  
                return None  
        except Exception as e:  
            logger.error(f"获取 AccessToken 异常: {e}")  
            return None  
  
token_manager = WXTokenManager()  
  
def upload_bytes_to_cos(img_bytes, mime_type="image/png"):  
    """  
    将二进制图片上传到微信云存储  
    返回: HTTPS 下载链接 (可以直接在前端展示)  
    """  
    token = token_manager.get_token()  
    if not token:  
        return None  
  
    try:  
        filename = f"ai_gen/{int(time.time())}_{str(uuid.uuid4())[:8]}.png"  
          
        # 1. 获取上传元数据 (URL 和 签名)  
        upload_meta_url = f"https://api.weixin.qq.com/tcb/uploadfile?access_token={token}"  
        payload = {  
            "env": WX_ENV_ID,  
            "path": filename  
        }  
        meta_resp = requests.post(upload_meta_url, json=payload, timeout=10)  
        meta_data = meta_resp.json()  
  
        if meta_data.get("errcode") != 0:  
            logger.error(f"获取上传链接失败: {meta_data}")  
            return None  
  
        # 2. 执行上传 (必须按照微信要求的字段顺序)  
        # 这里的字段来自于 meta_data  
        url = meta_data["url"]  
        authorization = meta_data["authorization"]  
        token_id = meta_data["token"]  
        cos_file_id = meta_data["cos_file_id"] # cloud://... 格式  
  
        # 构造 multipart/form-data  
        files = {  
            'file': (filename, img_bytes, mime_type)  
        }  
        # data 里的字段必须包含 Signature 等鉴权信息  
        form_data = {  
            "key": filename,  
            "Signature": authorization,  
            "x-cos-security-token": token_id,  
            "x-cos-meta-fileid": cos_file_id  
        }  
  
        upload_resp = requests.post(url, data=form_data, files=files, timeout=30)  
          
        if upload_resp.status_code == 204:  
            # 204 No Content 代表成功  
            # 我们可以返回 meta_data 里的 download_url (HTTPS) 或者 cos_file_id (cloud://)  
            # 为了兼容性最好，返回 HTTPS 链接  
            # 注意：uploadfile 接口返回的 download_url 有时是临时的，  
            # 如果你想用永久链接，最好使用 getTempFileURL 换取，或者直接拼接 (如果公开读)  
            # 这里简单起见，我们再调一次换取临时链接接口，或者直接用 cloud:// 给前端（如果前端支持）  
            # 你的前端代码用的是 src，cloud:// 在小程序 image 标签是支持的。  
            # 但为了 previewImage 能用，我们最好换一个 HTTP 链接。  
              
            # 这里为了简单，我们再做一步：换取 HTTP 链接  
            return get_temp_file_url(token, [cos_file_id])  
        else:  
            logger.error(f"COS 上传失败: {upload_resp.text}")  
            return None  
  
    except Exception as e:  
        logger.error(f"上传过程异常: {e}")  
        return None  
  
def get_temp_file_url(token, file_list):  
    """用 fileID 换取 HTTPS 链接"""  
    url = f"https://api.weixin.qq.com/tcb/batchdownloadfile?access_token={token}"  
    payload = {  
        "env": WX_ENV_ID,  
        "file_list": [{"fileid": fid, "max_age": 86400} for fid in file_list]  
    }  
    try:  
        r = requests.post(url, json=payload, timeout=10)  
        res = r.json()  
        if res.get("errcode") == 0 and res.get("file_list"):  
            return res["file_list"][0]["download_url"]  
    except Exception as e:  
        logger.error(f"换取链接失败: {e}")  
    return None  
  
def process_ai_task(job_id, data):  
    """后台线程：下载图片 -> 调用 AI -> 存结果"""  
    logger.info(f"[{job_id}] 开始处理任务...")  
    try:  
        prompt_text = data.get("prompt", "")  
        image_url = data.get("imageUrl")  
        raw_model = data.get("model", DEFAULT_MODEL)  
        history_list = data.get("history", [])     
        use_search = data.get("useSearch", False)  
          
        model_name = raw_model.replace("google/", "")  
          
        # --- 准备 Prompt ---  
        all_contents = []  
          
        # 1. 历史记录 (只留文本)  
        for msg in history_list:  
            role = "user" if msg['role'] == 'user' else "model"  
            content_text = msg.get('content', '') or "[图片/文件]"  
            all_contents.append(types.Content(role=role, parts=[types.Part(text=content_text)]))  
  
        # 2. 当前消息  
        current_parts = []  
        if image_url:  
            try:  
                img_resp = requests.get(image_url, timeout=60)  
                if img_resp.status_code == 200:  
                    current_parts.append(types.Part(  
                        inline_data=types.Blob(mime_type="image/jpeg", data=img_resp.content)  
                    ))  
            except Exception as e:  
                logger.error(f"[{job_id}] 输入图片下载异常: {e}")  
  
        if prompt_text:  
            current_parts.append(types.Part(text=prompt_text))  
          
        if current_parts:  
            all_contents.append(types.Content(role="user", parts=current_parts))  
  
        # --- 配置工具 ---  
        generate_config = None  
        if use_search:  
            # 注意：通常生图模型不支持 search，这里加个判断  
            if "image" not in model_name:  
                logger.info(f"[{job_id}] 开启联网搜索")  
                generate_config = types.GenerateContentConfig(  
                    tools=[types.Tool(google_search=types.GoogleSearch())],  
                    response_modalities=["TEXT"]  
                )  
  
        # 3. 调用 AI  
        logger.info(f"[{job_id}] 请求 AI ({model_name})...")  
        response = google_client.models.generate_content(  
            model=model_name,  
            contents=all_contents,  
            config=generate_config  
        )  
  
        # 4. 解析结果 (关键修改)  
        reply_text = ""  
        reply_image_url = None   
  
        if response.candidates:  
            for part in response.candidates[0].content.parts:  
                if part.text:  
                    reply_text += part.text  
                  
                # 🟢 核心改动：如果有图片，上传到云存储  
                if part.inline_data:  
                    logger.info(f"[{job_id}] 检测到 AI 生成了图片，正在上传到 COS...")  
                    img_data = part.inline_data.data  
                    mime = part.inline_data.mime_type or "image/png"  
                      
                    # 上传并获取 HTTPS 链接  
                    uploaded_url = upload_bytes_to_cos(img_data, mime)  
                      
                    if uploaded_url:  
                        reply_image_url = uploaded_url  
                        logger.info(f"[{job_id}] 图片上传成功: {reply_image_url[:50]}...")  
                    else:  
                        reply_text += "\n[系统提示: 图片生成成功，但在上传云存储时失败]"  
  
        TASK_STORE[job_id] = {  
            "status": "success",  
            "data": {  
                "reply": reply_text,  
                "generated_image": reply_image_url # 返回的是 URL，不是 Base64  
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
    if not google_client:  
        return jsonify({"code": -1, "error": "服务端未就绪"}), 500  
    data = request.get_json() or {}  
    job_id = str(uuid.uuid4())  
    TASK_STORE[job_id] = {"status": "processing"}  
    thread = threading.Thread(target=process_ai_task, args=(job_id, data))  
    thread.start()  
    return jsonify({"code": 0, "job_id": job_id})  
  
@app.route('/api/query', methods=['POST'])  
def query_task_status():  
    data = request.get_json() or {}  
    job_id = data.get("job_id")  
    if not job_id or job_id not in TASK_STORE:  
        return jsonify({"code": -1, "error": "任务不存在"}), 404  
          
    task = TASK_STORE[job_id]  
    if task['status'] == 'processing':  
        return jsonify({"code": 1, "status": "processing"})  
    elif task['status'] == 'success':  
        result = task['data']  
        del TASK_STORE[job_id]  
        return jsonify({"code": 0, "status": "success", **result})  
    else:  
        del TASK_STORE[job_id]  
        return jsonify({"code": -1, "status": "fail", "error": task.get("error")})  
  
@app.route('/')  
def ping():  
    return "pong", 200  
  
if __name__ == '__main__':  
    port = int(os.environ.get('PORT', 80))  
    app.run(host='0.0.0.0', port=port)  
