@echo off
cd /d %~dp0

echo 正在激活虚拟环境...
call .venv\Scripts\activate.bat

echo 正在执行翻译脚本...
python translate_images.py

echo 执行完成！
pause
