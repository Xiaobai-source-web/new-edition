"""应用级配置。"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEMO_PLAN_PATH = BASE_DIR / "名创优品.json"

# ====== Dify API（直接调用工作流 Chatflow，不走 IFRAME）======
# 从 Dify 后台 → 「多智能体进度系统工程」应用 → 「API 访问」获取 API Key
DIFY_API_KEY = "app-J9cddEDvW0rMYRa7NYDuRuuF"
# Chatflow (workflow-mode) 流式接口
DIFY_CHATFLOW_URL = "https://api.dify.ai/v1/chat-messages"
# 超时秒数（一次完整生成可能较长，建议 120s+）
DIFY_TIMEOUT = 300
# API 调用重试配置（偶发网络波动时自动重试）
DIFY_MAX_RETRIES = 3  # 最大尝试次数（含首次）
DIFY_RETRY_INTERVAL = 3  # 每次重试间隔（秒）
