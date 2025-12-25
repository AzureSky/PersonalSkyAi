# run.py  
import os  
import json  
import requests  
from flask import Flask, request, jsonify  
  
app = Flask(__name__)  
  
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  
  
@app.route('/')  
def index():  
    return "Chat Server is Running! (Gemini API)"  
  
@app.route('/api/chat', methods=['POST'])  
def chat():  
    if not GEMINI_API_KEY:  
        print("错误: 未配置环境变量 GEMINI_API_KEY")  
        return jsonify({'code': -1, 'error': '服务端未配置 API Key'})  
  
    try:  
        data = request.json  
        prompt = data.get('prompt', '')  
        image_base64 = data.get('imageBase64', None)  
        model = data.get('model', 'google/gemini-3-flash-preview')  
  
        print(f"收到请求: 模型={model}, 字数={len(prompt)}")  
  
        base_url = "https://generativelanguage.googleapis.com"  
          
        target_model = "gemini-1.5-flash"  
        if "pro" in model:  
            target_model = "gemini-1.5-pro"  
              
        url = f"{base_url}/v1beta/models/{target_model}:generateContent?key={GEMINI_API_KEY}"  
  
        contents_parts = []  
        if prompt:  
            contents_parts.append({"text": prompt})  
        if image_base64:  
            contents_parts.append({  
                "inline_data": {  
                    "mime_type": "image/jpeg",  
                    "data": image_base64  
                }  
            })  
  
        payload = {"contents": [{"parts": contents_parts}]}  
          
        headers = {'Content-Type': 'application/json'}  
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)  
          
        if response.status_code != 200:  
            print(f"Gemini API Error: {response.text}")  
            return jsonify({'code': -1, 'error': f"API Error: {response.status_code}"})  
  
        result_json = response.json()  
        try:  
            reply_text = result_json['candidates'][0]['content']['parts'][0]['text']  
            return jsonify({'code': 0, 'reply': reply_text})  
        except Exception as e:  
            print(f"解析失败: {result_json}")  
            return jsonify({'code': -2, 'error': "无法解析 AI 返回结果"})  
  
    except Exception as e:  
        print(f"Server Error: {str(e)}")  
        return jsonify({'code': 500, 'error': str(e)})  
  
if __name__ == '__main__':  
    # 本地测试时运行  
    app.run(host='0.0.0.0', port=80)  
