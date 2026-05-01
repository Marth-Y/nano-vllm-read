import os  # 导入 os 模块，用于处理文件路径和系统相关操作
from nanovllm import LLM, SamplingParams  # 从 nanovllm 包导入 LLM 核心引擎类和采样参数配置类
from transformers import AutoTokenizer  # 从 transformers 库导入 AutoTokenizer，用于自动加载匹配模型的分词器


def main():
    """主函数，演示如何使用 nanovllm 引擎进行模型推理"""

    # 定义模型存放的本地路径，使用 expanduser 处理家目录符号 ~
    path = os.path.expanduser("/home/luyimin/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B/")

    # 加载分词器，用于将自然语言文本转换为模型可处理的 token ID 序列
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 初始化 LLM 推理引擎实例
    # path: 模型文件所在的路径
    # enforce_eager=True: 强制使用 Eager 模式执行 PyTorch 算子，跳过 CUDA Graph 以简化调试
    # tensor_parallel_size=1: 设置张量并行度为 1（即单卡运行），nanovllm 目前主要支持单卡
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 实例化采样参数，控制模型生成的行为
    # temperature=0.6: 设置采样温度，值越高输出越具创造性和随机性，越低则越稳定
    # max_tokens=256: 限制单次生成输出的最大 token 长度
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # 准备原始输入提示词（Prompts）列表
    prompts = [
        "introduce yourself",  # 第一个提示词：请模型自我介绍
        "list all prime numbers within 100",  # 第二个提示词：列出 100 以内的所有质数
    ]

    # 将原始文本提示词转换为符合聊天模板（Chat Template）的格式
    # 这一步会将纯文本封装进 <|im_start|>user\n...<|im_end|> 等特定标记中
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # 构造符合对话协议的消息列表，角色定义为用户
            tokenize=False,  # 此时仅生成格式化后的字符串，暂不转换为 token ID
            add_generation_prompt=True,  # 在末尾添加生成提示符，指示模型开始输出回复
        )
        for prompt in prompts
    ]

    # 调用 llm 引擎的 generate 方法进行推理
    # 该过程包含分词、调度、KVCache 管理以及模型前向传播，最终返回生成结果
    outputs = llm.generate(prompts, sampling_params)

    # 迭代输入提示词和对应的生成输出结果
    for prompt, output in zip(prompts, outputs):
        print("\n")  # 打印换行符，增加输出可读性
        print(f"Prompt: {prompt!r}")  # 打印格式化后的原始提示词（repr 格式以显示特殊字符）
        print(f"Completion: {output['text']!r}")  # 打印模型生成的完整回复文本内容


if __name__ == "__main__":
    # 如果该脚本被直接运行（而非作为模块导入），则执行 main 函数
    main()
