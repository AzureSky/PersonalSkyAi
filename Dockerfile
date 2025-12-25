# 使用官方 Python 基础镜像  
FROM python:3.9-slim  
  
# 设置工作目录  
WORKDIR /app  
  
# 复制当前目录下的所有文件到容器中  
COPY . /app  
  
# 安装依赖  
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  
  
# 暴露 80 端口  
EXPOSE 80  
  
# 启动命令：直接运行 run.py  
# 使用 gunicorn 启动，性能更好  
CMD ["gunicorn", "--workers=1", "--threads=8", "--timeout=0", "run:app", "-b", "0.0.0.0:80"]  
