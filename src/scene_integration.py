import numpy as np
import logging
from typing import Dict
from src.long_term_memory import TextLongTermMemory


class SceneIntegration:
    """新的场景整合类，整合文本长期记忆和规划"""

    def __init__(self, scene, vlm_model=None, vlm_processor=None):
        self.scene = scene

        # 初始化新的长期记忆和规划系统
        self.long_term_memory = TextLongTermMemory()


        # 探索历史跟踪
        self.current_step = 0
        self.question = ""
        self.target_objects = []
        self.exploration_path = []  # 记录完整的探索路径


    def record_structured_agent_output(self, step: int, agent_type: str, structured_output: Dict,
                                     raw_response: str, position: np.ndarray):
        """记录结构化的agent输出"""
        # 使用长期记忆记录结构化输出
        self.long_term_memory.record_structured_agent_output(step, agent_type, structured_output, raw_response, position)

        logging.info(f"Recorded structured output for {agent_type} at step {step}")
