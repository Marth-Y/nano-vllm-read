# 导入 atexit 模块，用于在程序退出时自动执行清理函数
import atexit
# 导入 dataclasses 的 fields 函数，用于获取数据类的所有字段名
from dataclasses import fields
# 导入高精度计时器，用于测量代码执行时间
from time import perf_counter
# 导入进度条工具，用于显示生成进度
from tqdm.auto import tqdm
# 导入 HuggingFace 的分词器，用于文本的编码和解码
from transformers import AutoTokenizer
# 导入 PyTorch 的多进程模块，用于张量并行计算
import torch.multiprocessing as mp

# 导入项目内部的配置类
from nanovllm.config import Config
# 导入采样参数类，定义生成文本时的各种参数
from nanovllm.sampling_params import SamplingParams
# 导入序列类，用于管理单个请求的生命周期
from nanovllm.engine.sequence import Sequence
# 导入调度器类，负责管理请求队列和内存分配
from nanovllm.engine.scheduler import Scheduler
# 导入模型运行器类，负责实际执行模型推理
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM推理引擎的主类，负责协调各个组件完成文本生成任务。

    主要功能：
    1. 管理张量并行进程（多GPU推理）
    2. 调度多个并发请求
    3. 执行模型推理
    4. 返回生成结果
    """

    def __init__(self, model, **kwargs):
        """
        初始化LLM引擎。

        参数:
            model: 模型名称或路径，如 "Qwen/Qwen-7B-Chat"
            **kwargs: 其他配置参数，如 tensor_parallel_size（张量并行数）、
                     kvcache_block_size（KV缓存块大小）等
        """
        # 获取 Config 数据类中定义的所有字段名，用于过滤有效的配置参数
        # fields() 返回一个元组，每个元素是一个 Field 对象
        config_fields = {field.name for field in fields(Config)}

        # 从 kwargs 中筛选出 Config 支持的参数
        # 字典推导式：只保留键名在 config_fields 中的参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

        # 创建配置对象，包含模型路径和所有有效的配置参数
        config = Config(model, **config_kwargs)

        # 设置序列的块大小，这个值会影响 KV 缓存的分配方式
        # KV 缓存按块管理，每个块存储固定数量的 token 的键值对
        Sequence.block_size = config.kvcache_block_size

        # 初始化进程列表，用于存储所有工作进程（张量并行的子进程）
        self.ps = []
        # 初始化事件列表，用于主进程和工作进程之间的同步
        self.events = []

        # 获取多进程上下文，使用 "spawn" 方式创建进程
        # spawn 方式会启动全新的 Python 解释器，更安全但启动较慢
        # 其他方式还有 "fork"（Unix默认）和 "forkserver"
        ctx = mp.get_context("spawn")

        # 创建张量并行的工作进程
        # 主进程（rank 0）之外，还需要创建 tensor_parallel_size - 1 个工作进程
        # 例如：tensor_parallel_size=2 时，创建 1 个工作进程（rank 1）
        for i in range(1, config.tensor_parallel_size):
            # 创建一个事件对象，用于进程间同步
            # 工作进程完成后会设置这个事件
            event = ctx.Event()
            # 创建工作进程，运行 ModelRunner，传入配置、rank（进程编号）和同步事件
            # target=ModelRunner 表示进程启动时执行 ModelRunner 的构造函数
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            # 启动进程，进程开始执行
            process.start()
            # 将进程对象添加到列表中，便于后续管理
            self.ps.append(process)
            # 将同步事件添加到列表中
            self.events.append(event)

        # 创建主进程的 ModelRunner（rank 0）
        # 主进程负责协调所有工作进程，并处理与调度器的交互
        # 传入工作进程的事件列表，用于等待工作进程就绪
        self.model_runner = ModelRunner(config, 0, self.events)

        # 从预训练模型加载分词器
        # use_fast=True 表示使用 Rust 实现的快速分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # 将分词器的结束标记（EOS token ID）保存到配置中
        # EOS token 用于判断生成是否结束
        config.eos = self.tokenizer.eos_token_id

        # 创建调度器，负责管理请求队列、KV缓存分配等
        self.scheduler = Scheduler(config)

        # 注册退出清理函数
        # 当 Python 程序正常退出时，会自动调用 self.exit() 进行资源清理
        atexit.register(self.exit)

    def exit(self):
        """
        清理引擎资源，在程序退出时自动调用。

        主要清理工作：
        1. 通知所有工作进程退出
        2. 等待所有工作进程结束
        """
        # 调用主进程 ModelRunner 的 "exit" 方法，通知其退出
        # "call" 方法用于远程调用，会将请求发送到所有工作进程
        self.model_runner.call("exit")

        # 删除主进程的 ModelRunner，释放 GPU 内存等资源
        del self.model_runner

        # 等待所有工作进程完成退出
        # join() 会阻塞直到进程结束
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个生成请求到调度队列。

        参数:
            prompt: 输入提示，可以是字符串或已分词的 token ID 列表
            sampling_params: 采样参数，控制生成行为（如温度、top_p等）
        """
        # 如果 prompt 是字符串，需要先分词转换为 token ID 列表
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        # 创建序列对象，封装请求的所有信息
        # 序列包含：输入 token、采样参数、已生成的 token 等
        seq = Sequence(prompt, sampling_params)

        # 将序列添加到调度器的待处理队列
        self.scheduler.add(seq)

    def step(self):
        """
        执行一次推理步骤，这是引擎的核心执行单元。

        返回:
            outputs: 已完成序列的 (seq_id, completion_token_ids) 列表
            num_tokens: 本次处理的 token 数量（用于计算吞吐量）

        工作流程：
        1. 调度器选择要处理的序列
        2. 模型执行推理
        3. 后处理更新序列状态
        4. 返回已完成的序列结果
        """
        # 调度器选择本次要处理的序列，以及判断是否为 prefill 阶段
        # prefill：处理输入提示，计算 KV 缓存
        # decode：生成新 token，使用已有的 KV 缓存
        seqs, is_prefill = self.scheduler.schedule()

        # 计算本次处理的 token 数量，用于统计吞吐量
        # prefill 阶段：计算所有序列已调度的 token 数（正数）
        # decode 阶段：使用 -len(seqs) 表示每个序列生成 1 个 token（负数便于区分）
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        # 调用模型运行器执行推理，返回生成的 token ID
        # "run" 方法会在所有 GPU 上并行执行，然后收集结果
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 后处理：更新序列状态，添加新生成的 token
        # 会检查是否遇到 EOS token 或达到最大长度
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 收集已完成序列的结果
        # 列表推导式：筛选出 is_finished 为 True 的序列
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        return outputs, num_tokens

    def is_finished(self):
        """
        检查所有请求是否都已完成。

        返回:
            bool: 所有请求都完成返回 True，否则返回 False
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成文本的主要入口方法。

        参数:
            prompts: 提示列表，每个元素可以是字符串或 token ID 列表
            sampling_params: 采样参数，可以是单个参数（应用于所有提示）
                           或参数列表（每个提示对应一个参数）
            use_tqdm: 是否显示进度条，默认为 True

        返回:
            list: 生成结果列表，每个元素是包含 "text" 和 "token_ids" 的字典
        """
        # 创建进度条，显示生成进度
        # total=len(prompts): 进度条总数为提示数量
        # dynamic_ncols=True: 自动调整进度条宽度
        # disable=not use_tqdm: 如果 use_tqdm=False 则禁用进度条
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # 如果 sampling_params 不是列表，则复制为列表形式
        # 这样可以统一处理单个参数和参数列表两种情况
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 将所有请求添加到调度器
        # zip(prompts, sampling_params) 会将提示和参数一一配对
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 初始化输出字典，用于存储已完成的序列结果
        # 键是序列 ID，值是生成的 token ID 列表
        outputs = {}

        # 初始化吞吐量统计变量
        # prefill_throughput: prefill 阶段的 token/秒
        # decode_throughput: decode 阶段的 token/秒
        prefill_throughput = decode_throughput = 0.

        # 主推理循环，直到所有请求都完成
        while not self.is_finished():
            # 记录本次迭代的开始时间
            t = perf_counter()

            # 执行一次推理步骤
            # output: 本次完成的序列结果列表
            # num_tokens: 本次处理的 token 数量
            output, num_tokens = self.step()

            # 根据处理的 token 数量更新吞吐量统计
            # num_tokens > 0 表示 prefill 阶段
            # num_tokens < 0 表示 decode 阶段
            if num_tokens > 0:
                # 计算 prefill 吞吐量：token 数 / 耗时
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                # 计算 decode 吞吐量：-num_tokens 是实际 token 数（因为用了负数表示）
                decode_throughput = -num_tokens / (perf_counter() - t)

            # 更新进度条的后缀信息，显示实时吞吐量
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",  # prefill 速度
                "Decode": f"{int(decode_throughput)}tok/s",     # decode 速度
            })

            # 将本次完成的序列结果保存到 outputs 字典中
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                # 更新进度条（每个完成的请求计一次）
                pbar.update(1)

        # 关闭进度条
        pbar.close()

        # 按序列 ID 排序输出结果，确保顺序与输入一致
        # sorted(outputs.keys()) 返回排序后的键列表
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 将 token ID 列表解码为文本，构建最终的返回结果
        # 每个结果是一个字典，包含解码后的文本和原始 token ID
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),  # 解码 token 为文本
                "token_ids": token_ids                       # 保留原始 token ID
            }
            for token_ids in outputs
        ]

        return outputs