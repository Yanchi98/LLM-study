# 安装依赖
pip install -r requirements.txt

gunicorn -w 4 -b 127.0.0.1:6006 manage:app
