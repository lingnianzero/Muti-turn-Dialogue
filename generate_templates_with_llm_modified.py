#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LLM生成自然对话模板
---------------------------------
本地运行，通过LLM API生成真实、自然的对话内容，
替换原有的重复填充模式，生成更真实的长上下文测试模板。

生成的JSON格式与long_context_tester.py完全兼容。
"""

import json
import logging
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

# ======================== 配置加载 ======================== #

def load_yaml(path: Path) -> Dict[str, Any]:
    """加载YAML配置"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    """标签转文件名"""
    slug = re.sub(r"[^\w]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "sample"


# ======================== Token估算器 ======================== #

class TokenEstimator:
    """估算文本token数"""

    def __init__(self, ratio: float = 1.7):
        self.ratio = ratio
        self._encoding = None
        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            self._encoding = None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        return math.ceil(len(text) / self.ratio)

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total += self.count_text(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count_text(item.get("text", ""))
        return total


# ======================== LLM内容生成器 ======================== #

class LLMContentGenerator:
    """使用LLM API生成自然对话内容"""

    def __init__(
        self,
        api_config: Dict[str, Any],
        estimator: TokenEstimator,
        cache_file: Optional[Path] = None
    ):
        self.base_url = api_config.get("base_url", "https://88996.cloud/v1")
        self.api_key = api_config.get("api_key")
        self.model = api_config.get("model", "gpt-4o-mini")
        self.temperature = api_config.get("temperature", 0.8)
        self.estimator = estimator

        # 缓存（线程安全）
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        self.cache_lock = Lock()
        if cache_file and cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logging.info(f"加载缓存: {len(self.cache)} 条记录")
            except Exception as e:
                logging.warning(f"加载缓存失败: {e}")

        # 话题描述
        self.topic_descriptions = {
            "vision": "视觉细节分析（光线、色彩、构图、材质）",
            "analysis": "深度分析和评论",
            "table": "结构化对比表格",
            "story": "包含图像元素的小故事",
            "guide": "描述指南或操作手册",
            "qa": "问答形式的知识问答",
            "irrelevant": "看似无关但暗含图像元素的话题"
        }

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 800
    ) -> Optional[str]:
        """调用LLM API"""
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            logging.debug(f"调用LLM API: {endpoint}")
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )

            logging.debug(f"LLM响应状态码: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                logging.debug(f"LLM响应: {json.dumps(result, ensure_ascii=False)[:500]}")

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    if content:
                        logging.info(f"✓ LLM返回内容: {len(content)}字符")
                        return content
                    else:
                        logging.warning("LLM返回的content为空")
                else:
                    logging.warning(f"LLM响应格式异常: {result}")
            else:
                logging.error(f"LLM API调用失败: {response.status_code} - {response.text[:500]}")

        except Exception as e:
            logging.error(f"LLM API异常: {e}", exc_info=True)

        return None

    def generate_user_question(
        self,
        topic: str,
        label: str,
        turn: int,
        num_turns: int,
        drift: float
    ) -> str:
        """生成用户问题"""
        cache_key = f"user_{topic}_{label}_{turn}_{int(drift*100)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        topic_desc = self.topic_descriptions.get(topic, "相关内容")

        system_prompt = f"""你是一个测试助手，正在为长上下文记忆测试生成用户问题。

当前场景：
- 用户在第1轮已经看过一张人物照片
- 照片中人物的描述是：{label}
- 当前是第{turn}轮对话
- 当前话题类型：{topic} ({topic_desc})
 - 总轮数：{num_turns}
 - 话题偏离系数（0~1）：{drift:.2f}，系数越大越应远离图片细节

要求：
1. 生成一个自然的用户问题，围绕之前看过的图片
2. 只使用描述性称呼"{label}"，不要提及任何人名
3. 问题要简短（50-80个中文字符）
4. 问题风格要符合{topic_desc}的特点
5. 随着轮次增加（drift↑），问题要逐步减少与图片直接相关的细节引用，只保留最小化的挂钩（例如用"{label}"做引子）
6. 直接输出问题，不要额外解释"""

        user_prompt = f"生成第{turn}轮的用户问题（话题：{topic}，偏离：{drift:.2f}）"

        result = self._call_llm(system_prompt, user_prompt, max_tokens=200)

        if result:
            # 清理可能的额外内容
            result = result.strip().strip('"').strip("'")
            self.cache[cache_key] = result
            self._save_cache()
            return result

        # fallback
        return f"关于{label}，我们换个更宏观的话题聊聊好吗？"

    def generate_assistant_response(
        self,
        topic: str,
        label: str,
        turn: int,
        target_tokens: int,
        num_turns: int,
        drift: float
    ) -> str:
        """生成助手回复"""
        cache_key = f"assistant_{topic}_{label}_{turn}_{target_tokens}_{int(drift*100)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        topic_desc = self.topic_descriptions.get(topic, "相关内容")

        system_prompt = f"""你是一个AI助手，正在为长上下文记忆测试生成回复。

严格规则：
1. 只使用"{label}"这样的描述性称呼来指代图中人物
2. **绝对禁止**提到任何人名（即使你知道或猜测）
3. 保持自然的对话风格
4. 回复长度约{target_tokens}个token（约{int(target_tokens * 1.7)}个中文字符）
5. 内容要丰富、自然，避免简单重复
6. 随着轮次增加（drift↑），逐步减少与图片直接相关的细节描述；仅以最小化的方式在开头一句话上与"{label}"挂钩，随后话题尽量转向更泛化、更远离图像的内容

当前场景：
- 这是第{turn}轮对话
- 图片主体：{label}
- 当前话题：{topic} ({topic_desc})
 - 总轮数：{num_turns}
 - 话题偏离系数（0~1）：{drift:.2f}

内容要求：
- 如果是vision/analysis话题：详细描述视觉细节（光线、色彩、构图、情绪、背景）
- 如果是table话题：生成结构化的markdown表格对比
- 如果是story话题：编写一个包含图像元素的简短故事
- 如果是guide话题：提供描述指南或步骤
- 如果是irrelevant话题：聊一个看似无关的话题，但巧妙地暗含图像元素

直接输出回复内容，不要额外说明。"""

        user_prompt = f"生成第{turn}轮的助手回复（话题：{topic}，目标长度：{target_tokens} tokens，偏离：{drift:.2f}）"

        # 根据目标token数计算max_tokens
        max_tokens = min(int(target_tokens * 1.5), 2000)

        result = self._call_llm(system_prompt, user_prompt, max_tokens=max_tokens)

        if result:
            # 检查长度，如果太短就补充
            actual_tokens = self.estimator.count_text(result)

            if actual_tokens < target_tokens * 0.7:
                # 太短了，请求补充
                supplement_prompt = f"请在以下内容基础上补充更多细节，使总长度达到约{target_tokens}个token：\n\n{result}"
                supplement = self._call_llm(system_prompt, supplement_prompt, max_tokens=max_tokens)
                if supplement:
                    result = supplement

            self.cache[cache_key] = result
            self._save_cache()
            return result

        # fallback: 使用简单模板
        return self._fallback_response(topic, label, turn, target_tokens)

    def _fallback_response(
        self,
        topic: str,
        label: str,
        turn: int,
        target_tokens: int
    ) -> str:
        """LLM失败时的fallback"""
        base = f"关于这张{label}的照片，"
        if topic == "vision":
            base += "我们可以观察到光线柔和，主体处于中央位置。色彩搭配和谐，整体氛围平静自然。"
        elif topic == "analysis":
            base += "从构图角度看，这是一张经典的人物肖像照。背景虚化突出主体，给人专业的印象。"
        elif topic == "table":
            base += "\n\n| 属性 | 描述 |\n|------|------|\n| 主体 | " + label + " |\n| 背景 | 虚化处理 |\n| 光线 | 柔和自然 |"
        else:
            base += "这是一张很有特点的照片，值得我们仔细观察和记忆。"

        # 简单填充到目标长度
        while self.estimator.count_text(base) < target_tokens:
            base += "\n\n补充细节：图像的每个元素都经过精心安排，展现出摄影师的专业水准。"

        return base

    def _save_cache(self):
        """保存缓存"""
        if self.cache_file:
            try:
                ensure_dir(self.cache_file.parent)
                with self.cache_file.open("w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.warning(f"保存缓存失败: {e}")


# ======================== 模板构建器 ======================== #

class TemplateBuilder:
    """构建完整的对话模板"""

    def __init__(
        self,
        config: Dict[str, Any],
        llm_generator: LLMContentGenerator,
        estimator: TokenEstimator
    ):
        self.config = config
        self.llm = llm_generator
        self.estimator = estimator
        self.random = random.Random(config.get("random_seed", 20250115))

    def _progressive_topic(self, turn: int, num_turns: int, knowledge_mix: List[str]) -> str:
        """按轮次渐进式地选择更偏离图片元素的话题"""
        p = min(max((turn - 1) / max(1, num_turns - 1), 0.0), 1.0)
        base_weights = {
            "vision": max(0.6 * (1 - p), 0.05),
            "analysis": max(0.4 * (1 - p * 0.8), 0.05),
            "table": 0.12 if p < 0.7 else 0.06,
            "story": 0.15 + 0.25 * p,
            "guide": 0.10 + 0.15 * p,
            "qa": 0.08 + 0.12 * p,
            "irrelevant": 0.05 + 0.55 * p,
        }
        # 仅保留允许的话题
        allowed = {k: v for k, v in base_weights.items() if k in knowledge_mix}
        total = sum(allowed.values()) or 1.0
        r = self.random.random()
        cum = 0.0
        for k, v in allowed.items():
            cum += v / total
            if r <= cum:
                return k
        return list(allowed.keys())[-1]

    def build_template(
        self,
        template_spec: Dict[str, Any],
        test_mode: Dict[str, Any],
        test_image: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建单个模板"""

        template_name = template_spec["name"]
        target_tokens = template_spec["target_tokens"]
        num_turns = template_spec["num_turns"]
        user_target = template_spec.get("user_target_tokens", 70)
        assistant_target = template_spec.get("assistant_target_tokens", 620)
        knowledge_mix = template_spec.get("knowledge_mix", ["vision", "analysis"])

        mode_key = test_mode["key"]
        expose_name = test_mode["expose_name"]
        subject_template = test_mode["subject_template"]
        initial_prompt = test_mode["initial_prompt"]
        description_prompt = test_mode.get("description_prompt", "")

        image_path = test_image["path"]
        name = test_image["name"]
        description = test_image["description"]
        category = test_image.get("category", "faces")

        # 确定subject显示内容
        if expose_name:
            subject = subject_template.format(name=name)
        else:
            subject = subject_template.format(description=description)

        # 第一轮消息：图片 + 提示
        first_user_text = initial_prompt.format(name=name, description=description)
        if description_prompt:
            first_user_text += "\n" + description_prompt.format(name=name, description=description)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": first_user_text}
                ]
            }
        ]

        topics_used = []

        # 生成对话轮次
        logging.info(f"开始生成{num_turns}轮对话（目标：{target_tokens} tokens）...")

        for turn in range(1, num_turns + 1):
            # 选择话题（随轮次逐步偏离图片元素）
            topic = self._progressive_topic(turn, num_turns, knowledge_mix)
            topics_used.append(topic)

            # 生成助手回复
            logging.info(f"  第{turn}轮 - 话题：{topic}")
            drift = min(max((turn - 1) / max(1, num_turns - 1), 0.0), 1.0)
            assistant_response = self.llm.generate_assistant_response(
                topic=topic,
                label=description,
                turn=turn,
                target_tokens=assistant_target,
                num_turns=num_turns,
                drift=drift
            )

            messages.append({
                "role": "assistant",
                "content": assistant_response
            })

            # 检查是否已达到目标token
            current_tokens = self.estimator.count_messages(messages)
            logging.info(f"    当前token数: {current_tokens}/{target_tokens}")

            if current_tokens >= target_tokens:
                logging.info(f"  已达到目标token数，停止生成")
                break

            # 如果不是最后一轮，生成下一个用户问题
            if turn < num_turns:
                next_topic = self._progressive_topic(turn + 1, num_turns, knowledge_mix)
                next_drift = min(max((turn) / max(1, num_turns - 1), 0.0), 1.0)
                user_question = self.llm.generate_user_question(
                    topic=next_topic,
                    label=description,
                    turn=turn + 1,
                    num_turns=num_turns,
                    drift=next_drift
                )

                messages.append({
                    "role": "user",
                    "content": user_question
                })

        # 构建最终模板
        final_tokens = self.estimator.count_messages(messages)

        template_data = {
            "template": template_name,
            "mode": {
                "key": mode_key,
                "expose_name": expose_name,
                "subject": subject
            },
            "memory": {
                "name": name,
                "expose_name": expose_name,
                "mode": mode_key
            },
            "image": {
                "path": image_path,
                "label": description,
                "name": name,
                "category": category
            },
            "messages": messages,
            "topics": topics_used,
            "target_tokens": target_tokens,
            "estimated_tokens": final_tokens,
            "created_at": datetime.now().isoformat()
        }

        logging.info(f"✓ 模板生成完成: {final_tokens} tokens")
        return template_data


# ======================== 主程序 ======================== #

def generate_single_template(
    task_info: Tuple[int, int, Dict, Dict, Dict, TemplateBuilder, Path, Lock]
) -> Tuple[bool, str]:
    """
    生成单个模板的worker函数

    Returns:
        (success, message)
    """
    idx, total, template_spec, test_mode, test_image, builder, templates_dir, log_lock = task_info

    template_name = template_spec["name"]
    mode_key = test_mode["key"]
    label = test_image["description"]

    try:
        with log_lock:
            logging.info(f"[{idx}/{total}] 开始: {template_name} - {mode_key} - {label}")

        # 生成模板
        template_data = builder.build_template(
            template_spec, test_mode, test_image
        )

        # 保存文件
        filename = f"template_{template_name}_{mode_key}_{slugify(label)}.json"
        output_path = templates_dir / filename

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)

        msg = f"✓ [{idx}/{total}] 完成: {filename} ({template_data['estimated_tokens']} tokens)"
        with log_lock:
            logging.info(msg)

        return (True, msg)

    except Exception as e:
        error_msg = f"✗ [{idx}/{total}] 失败: {template_name} - {mode_key} - {label}: {e}"
        with log_lock:
            logging.error(error_msg)
        return (False, error_msg)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # 加载配置
    config_path = Path("long_context_config.yaml")
    if not config_path.exists():
        logging.error(f"配置文件不存在: {config_path}")
        return

    config = load_yaml(config_path)

    # 初始化组件
    estimator = TokenEstimator()

    # LLM配置
    llm_config = {
        "base_url": "https://88996.cloud/v1",
        "api_key": "sk-VqyTOoTJJ8D3zrAsH6ckZxN7kwzS8li67MbcbPXA0OapcbZm",
        "model": "gpt-4o-mini",
        "temperature": 0.8
    }

    cache_dir = Path(config.get("io", {}).get("cache_dir", "results/cache"))
    ensure_dir(cache_dir)
    cache_file = cache_dir / "llm_responses.json"

    llm_generator = LLMContentGenerator(llm_config, estimator, cache_file)
    builder = TemplateBuilder(config, llm_generator, estimator)

    # 输出目录
    templates_dir = Path(config.get("io", {}).get("templates_dir", "templates"))
    ensure_dir(templates_dir)

    # 生成所有模板
    templates = list(config.get("templates", []))
    test_modes = list(config.get("test_modes", []))
    test_images = list(config.get("test_images", []))

    # 动态增强：确保有 10k/20k/30k 三档目标（将50k替换为30k）
    name_to_spec = {t.get("name"): t for t in templates}
    if "30k" not in name_to_spec:
        templates.append({
            "name": "30k",
            "target_tokens": 30000,
            "num_turns": 45,
            "user_target_tokens": 85,
            "assistant_target_tokens": 660,
            "knowledge_mix": ["vision", "analysis", "table", "story", "guide", "qa", "irrelevant"],
        })
    # 过滤掉50k（若存在）
    templates = [t for t in templates if t.get("name") in {"10k", "20k", "30k"}]

    # 动态增强：为模式加入性别限定的四个模式（保留原有两种以兼容现有测试）
    existing_keys = {m.get("key") for m in test_modes}
    def add_mode_if_absent(key, gender, expose_name, init_prompt, desc_prompt, subject_template):
        if key not in existing_keys:
            test_modes.append({
                "key": key,
                "gender": gender,
                "expose_name": expose_name,
                "subject_template": subject_template,
                "initial_prompt": init_prompt,
                "description_prompt": desc_prompt,
            })

    add_mode_if_absent(
        "name_revealed_male",
        "male",
        True,
        "这张图片里的男子叫{name}，请牢牢记住他的名字与面孔。",
        "后续讨论里我们换用诸如“{description}”这样的描述，不再直接提名字。",
        "{name}",
    )
    add_mode_if_absent(
        "name_hidden_male",
        "male",
        False,
        "请仔细观察这张图片中的男子，但暂时不要给他起名字。",
        "后续对话中一律使用画面描述，不直接说出真实姓名。",
        "{description}",
    )
    add_mode_if_absent(
        "name_revealed_female",
        "female",
        True,
        "这张图片里的女子叫{name}，请牢牢记住她的名字与面孔。",
        "后续讨论里我们换用诸如“{description}”这样的描述，不再直接提名字。",
        "{name}",
    )
    add_mode_if_absent(
        "name_hidden_female",
        "female",
        False,
        "请仔细观察这张图片中的女子，但暂时不要给她起名字。",
        "后续对话中一律使用画面描述，不直接说出真实姓名。",
        "{description}",
    )

    # 若图片未标注性别，基于描述启发式推断
    def infer_gender(desc: str) -> Optional[str]:
        d = (desc or "").lower()
        if any(k in d for k in ["男性", "男子", "先生", "男"]) or any(k in d for k in ["male", "man", "boy"]):
            return "male"
        if any(k in d for k in ["女性", "女子", "女士", "女"]) or any(k in d for k in ["female", "woman", "girl"]):
            return "female"
        return None

    for img in test_images:
        if not img.get("gender"):
            g = infer_gender(img.get("description", ""))
            if g:
                img["gender"] = g

    # 构建任务列表
    tasks = []
    idx = 0
    # 若存在按性别细分的四种模式，则仅生成这四种，以满足12种组合；否则回退为全部模式
    preferred_keys = {"name_revealed_male", "name_hidden_male", "name_revealed_female", "name_hidden_female"}
    if any(m.get("key") in preferred_keys for m in test_modes):
        modes_iter = [m for m in test_modes if m.get("key") in preferred_keys]
    else:
        modes_iter = test_modes

    for template_spec in templates:
        for test_mode in modes_iter:
            for test_image in test_images:
                # 若模式限定性别，仅匹配对应图片
                mode_gender = test_mode.get("gender")
                image_gender = test_image.get("gender")
                if mode_gender and image_gender and mode_gender != image_gender:
                    continue

                idx += 1
                tasks.append((
                    idx,
                    0,  # 占位，稍后更新总数
                    template_spec,
                    test_mode,
                    test_image,
                    builder,
                    templates_dir,
                    Lock()  # 占位，稍后统一替换
                ))

    total = len(tasks)
    log_lock = Lock()

    # 更新所有任务使用同一个锁
    tasks = [
        (idx, total, ts, tm, ti, builder, templates_dir, log_lock)
        for idx, _, ts, tm, ti, _, _, _ in tasks
    ]

    logging.info(f"准备生成 {total} 个模板...")
    logging.info(f"使用 5 个并发线程")
    logging.info("=" * 60)

    # 使用线程池并发生成
    success_count = 0
    fail_count = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(generate_single_template, task): task for task in tasks}

        for future in as_completed(futures):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1

    elapsed_time = time.time() - start_time

    logging.info("\n" + "=" * 60)
    logging.info(f"✓ 全部完成！")
    logging.info(f"成功: {success_count}, 失败: {fail_count}, 总计: {total}")
    logging.info(f"耗时: {elapsed_time:.1f} 秒")
    logging.info(f"输出目录: {templates_dir}")


if __name__ == "__main__":
    main()
