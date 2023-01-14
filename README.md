# VitsGradio
VitsWebUi


使用方法：
- 安装Python
- Shift+右键该文件夹空白处，选择PowerShell
- pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
- 等待环境安装完成
- cd monotonic_align
- python setup.py build_ext --inplace
- cd ..
- python vits_gradio.py，浏览器中打开http://127.0.0.1:7860/
- 在第三栏中选择模型和设备并点击载入模型按钮，即可回到前两个界面推理

注意：如果是单角色模型，语音转换部分将不可见。
