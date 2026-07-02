"""GOAT-Bench subtask goal data structures."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GoalInfo:
    """GOAT-Bench subtask goal information.

    type: "object" (find by category), "description" (find by lang desc),
          "image" (find by reference image).
    """
    type: str  # "object" | "description" | "image"
    category: Optional[str] = None        # object 子任务的类别文本
    lang_desc: Optional[str] = None       # description 子任务的语言描述
    image_goal: Optional[str] = None      # image 子任务的参考图 base64
    target_positions: list = field(default_factory=list)  # GT 目标位置（成功判定用）
    viewpoints: list = field(default_factory=list)        # GT 视点列表（成功判定用）

    @property
    def text(self) -> str:
        """给 KSS/VLM 用的目标文本表示。"""
        if self.type == "object":
            return f"Can you find the {self.category}?"
        elif self.type == "description":
            return self.lang_desc or ""
        else:
            return "Find the object shown in the reference image."
