
import os
import importlib

# 自动扫描当前目录下的所有 .py 文件并导入
# 这样可以确保所有文件中定义的模型都被注册
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('__'):
        module_name = file[:file.rfind('.')]
        # 使用 importlib 动态导入模块
        importlib.import_module(f'.{module_name}', __package__)