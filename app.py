"""
AI 校规问答系统 - Streamlit 前端版本

功能说明：
1. 复用 main.py 中的 SchoolRuleQA 核心问答系统；
2. 提供可视化聊天界面；
3. 支持配置 Ollama、本地模型、Embedding 模型、Top-K 等参数；
4. 支持展示检索到的相关校规；
5. 支持调试模式，展示 final_score、semantic_score、keyword_score；
6. 支持清空对话历史和重载系统。

运行方式：
    streamlit run app.py

项目结构建议：
    project/
    ├── app.py
    ├── main.py
    ├── requirements.txt
    └── data/
        └── school.txt
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from main import (
        DEFAULT_EMBEDDING_MODEL_NAME,
        DEFAULT_MAX_CONTEXT_CHARS,
        DEFAULT_MODEL_NAME,
        DEFAULT_OLLAMA_PATH,
        DEFAULT_RULE_PATH,
        DEFAULT_TOP_K,
        JudgeResult,
        RetrievedRule,
        SchoolRuleQA,
    )
except Exception as import_error:
    st.set_page_config(
        page_title="AI 校规问答助手",
        page_icon="🤖",
        layout="centered",
    )

    st.error("导入 main.py 失败，请确认 app.py 和 main.py 在同一目录下，并且 main.py 没有语法错误。")
    st.exception(import_error)
    st.stop()


# =========================
# 1. 页面基础配置
# =========================

st.set_page_config(
    page_title="AI 校规问答助手",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =========================
# 2. 页面样式
# =========================

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.small-caption {
    color: #666;
    font-size: 0.88rem;
    line-height: 1.5;
}

.project-card {
    border: 1px solid rgba(49, 51, 63, 0.18);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
    background: rgba(250, 250, 250, 0.65);
}

.decision-box {
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin: 0.75rem 0;
    border: 1px solid rgba(49, 51, 63, 0.16);
    background: rgba(250, 250, 250, 0.8);
}

.rule-item {
    border-left: 3px solid rgba(49, 51, 63, 0.35);
    padding-left: 0.75rem;
    margin: 0.75rem 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# 3. 工具函数
# =========================

def rule_to_dict(rule: RetrievedRule) -> Dict[str, Any]:
    """
    将 RetrievedRule 转成可存入 st.session_state 的普通字典。

    原因：
    Streamlit 的 session_state 更适合保存基础数据结构；
    这样后续渲染历史消息时不会依赖对象实例。
    """
    return {
        "rule_id": rule.rule_id,
        "text": rule.text,
        "final_score": float(rule.final_score),
        "semantic_score": float(rule.semantic_score),
        "keyword_score": float(rule.keyword_score),
    }


def judge_to_dict(judge: JudgeResult) -> Dict[str, str]:
    """
    将 JudgeResult 转成普通字典。
    """
    return {
        "decision": judge.decision,
        "reason": judge.reason,
        "source": judge.source,
        "judge_type": judge.judge_type,
    }


def get_decision_emoji(decision: str) -> str:
    """
    根据判定结果返回一个简单图标。
    """
    if decision == "允许":
        return "✅"
    if decision == "不允许":
        return "⛔"
    if decision == "有条件允许":
        return "⚠️"
    return "❓"


def format_answer(judge: Dict[str, str]) -> str:
    """
    生成给用户看的主要回答内容。
    """
    emoji = get_decision_emoji(judge.get("decision", "未明确"))

    return (
        f"### {emoji} 判定：{judge.get('decision', '未明确')}\n\n"
        f"**原因：**{judge.get('reason', '暂无原因')}\n\n"
        f"**来源：**{judge.get('source', '未检索到明确来源')}\n\n"
        f"**裁判方式：**{judge.get('judge_type', '未知')}"
    )


def render_rules(
    rules: List[Dict[str, Any]],
    debug: bool = False,
    max_rules: int = 5,
) -> None:
    """
    渲染检索到的候选校规。
    """
    if not rules:
        st.info("未检索到相关校规。")
        return

    for index, rule in enumerate(rules[:max_rules], start=1):
        rule_id = rule.get("rule_id", "未知")
        text = rule.get("text", "")

        st.markdown(
            f"""
<div class="rule-item">
<b>{index}. 第{rule_id}条</b><br/>
{text}
</div>
""",
            unsafe_allow_html=True,
        )

        if debug:
            st.code(
                (
                    f"final_score   = {rule.get('final_score', 0.0):.4f}\n"
                    f"semantic_score= {rule.get('semantic_score', 0.0):.4f}\n"
                    f"keyword_score = {rule.get('keyword_score', 0.0):.2f}"
                ),
                language="text",
            )


def initialize_session_state() -> None:
    """
    初始化 Streamlit 会话状态。
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if "last_error" not in st.session_state:
        st.session_state.last_error = None


@st.cache_resource(show_spinner=False)
def load_qa_system(
    rule_path: str,
    embedding_model_name: str,
    ollama_enabled: bool,
    ollama_path: str,
    model_name: str,
    top_k: int,
    max_context_chars: int,
    debug: bool,
) -> SchoolRuleQA:
    """
    缓存加载问答系统。

    说明：
    - Embedding 模型和校规向量生成比较耗时，因此使用 st.cache_resource；
    - 当侧边栏参数变化时，缓存 key 会变化，系统会自动重新初始化；
    - 如果点击“清空系统缓存并重载”，会清除缓存并重新加载。
    """
    return SchoolRuleQA(
        rule_path=rule_path,
        embedding_model_name=embedding_model_name,
        ollama_enabled=ollama_enabled,
        ollama_path=ollama_path,
        model_name=model_name,
        top_k=top_k,
        max_context_chars=max_context_chars,
        debug=debug,
    )


def ask_question(
    qa_system: SchoolRuleQA,
    query: str,
    debug: bool,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    调用核心问答系统，并将结果转成可展示、可保存的字典结构。
    """
    judge_result, retrieved_rules = qa_system.ask(query)

    judge_dict = judge_to_dict(judge_result)
    rule_dicts = [rule_to_dict(rule) for rule in retrieved_rules]

    return judge_dict, rule_dicts


def render_chat_history(debug: bool) -> None:
    """
    渲染历史对话。
    """
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                rules = message.get("rules", [])
                judge = message.get("judge", {})

                if judge:
                    with st.expander("查看判定依据", expanded=False):
                        st.write(f"判定：{judge.get('decision', '未明确')}")
                        st.write(f"来源：{judge.get('source', '未检索到明确来源')}")
                        st.write(f"裁判方式：{judge.get('judge_type', '未知')}")

                if rules:
                    with st.expander("查看检索到的相关校规", expanded=False):
                        render_rules(rules, debug=debug, max_rules=5)


def submit_query(query: str, qa_system: SchoolRuleQA, debug: bool) -> None:
    """
    处理一次用户提问。
    """
    query = query.strip()

    if not query:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    try:
        judge_dict, rule_dicts = ask_question(
            qa_system=qa_system,
            query=query,
            debug=debug,
        )

        answer = format_answer(judge_dict)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "judge": judge_dict,
                "rules": rule_dicts,
            }
        )

        st.session_state.last_error = None

    except Exception as error:
        error_message = (
            "系统运行时出现错误，请检查校规文件、模型配置或本地环境。"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"❌ {error_message}\n\n错误信息：`{error}`",
                "judge": {
                    "decision": "未明确",
                    "reason": error_message,
                    "source": "系统异常",
                    "judge_type": "异常处理",
                },
                "rules": [],
            }
        )

        st.session_state.last_error = traceback.format_exc()


# =========================
# 4. 初始化
# =========================

initialize_session_state()


# =========================
# 5. 侧边栏配置
# =========================

with st.sidebar:
    st.header("⚙️ 系统设置")

    st.caption("这些配置会影响 RAG 检索和最终裁判方式。首次加载模型可能需要等待。")

    rule_path = st.text_input(
        "校规文件路径",
        value=DEFAULT_RULE_PATH,
        help="默认读取 data/school.txt。文件内容建议采用“1. xxx”“2. xxx”的编号格式。",
    )

    embedding_model_name = st.text_input(
        "Embedding 模型",
        value=DEFAULT_EMBEDDING_MODEL_NAME,
        help="用于把校规文本和用户问题转换成向量。",
    )

    top_k = st.slider(
        "检索候选数量 Top-K",
        min_value=3,
        max_value=50,
        value=DEFAULT_TOP_K,
        step=1,
        help="Top-K 越大，候选校规越多，但上下文也会更长。",
    )

    max_context_chars = st.slider(
        "最大上下文字符数",
        min_value=1000,
        max_value=10000,
        value=DEFAULT_MAX_CONTEXT_CHARS,
        step=500,
        help="限制送入裁判模块的候选校规总长度。",
    )

    st.divider()

    ollama_enabled = st.toggle(
        "启用 Ollama 本地大模型裁判",
        value=True,
        help="开启后会调用本地 Ollama 模型进行最终判定；关闭后使用本地规则引擎。",
    )

    ollama_path = st.text_input(
        "Ollama 路径",
        value=DEFAULT_OLLAMA_PATH,
        help="Windows 下通常是 ollama.exe 的完整路径。",
        disabled=not ollama_enabled,
    )

    model_name = st.text_input(
        "Ollama 模型名称",
        value=DEFAULT_MODEL_NAME,
        help="例如 deepseek-r1:8b、qwen2.5:7b 等。",
        disabled=not ollama_enabled,
    )

    st.divider()

    debug_mode = st.toggle(
        "调试模式",
        value=False,
        help="开启后会显示检索分数，包括 final_score、semantic_score、keyword_score。",
    )

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.session_state.last_error = None
            st.rerun()

    with col_b:
        if st.button("重载系统", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.last_error = None
            st.rerun()

    st.divider()

    st.subheader("🧪 示例问题")

    example_questions = [
        "上课可以喝水吗？",
        "学生可以在食堂吃饭吗？",
        "考试作弊会怎么样？",
        "上课睡觉可以吗？",
        "请病假需要什么手续？",
    ]

    for example in example_questions:
        if st.button(example, use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()


# =========================
# 6. 主页面
# =========================

st.title("🤖 AI 校规问答助手")

st.markdown(
    """
<div class="small-caption">
这是一个基于 RAG 检索的校园规则问答系统。系统会先从校规文本中检索相关条款，
再通过本地规则引擎或 Ollama 本地大模型进行判定，最终输出“允许 / 不允许 / 有条件允许 / 未明确”。
<br/>
当前页面会保留本轮对话记录，但历史消息不会作为检索上下文参与下一次回答。
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")

with st.expander("项目流程说明", expanded=False):
    st.markdown(
        """
这个系统的核心流程是：

1. 读取 `data/school.txt` 中的校规文本；
2. 按照编号将校规切分成独立条款；
3. 使用中文 Embedding 模型生成规则向量；
4. 用户提问后，先进行查询扩展；
5. 使用向量相似度召回候选校规；
6. 使用关键词规则进行二次重排；
7. 将候选校规交给本地规则引擎或 Ollama 模型裁判；
8. 输出结构化判定结果，并展示检索依据。

这个项目的重点不是简单聊天，而是展示一个可解释的 RAG 问答链路。
"""
    )


# =========================
# 7. 加载核心系统
# =========================

try:
    with st.spinner("正在加载校规问答系统，首次启动可能需要等待模型加载..."):
        qa_system = load_qa_system(
            rule_path=rule_path,
            embedding_model_name=embedding_model_name,
            ollama_enabled=ollama_enabled,
            ollama_path=ollama_path,
            model_name=model_name,
            top_k=top_k,
            max_context_chars=max_context_chars,
            debug=debug_mode,
        )

    st.success(
        f"系统已就绪：共加载 {len(qa_system.docs)} 条校规，"
        f"当前裁判方式：{'Ollama 本地大模型 + 失败回退本地规则引擎' if ollama_enabled else '本地规则引擎'}。"
    )

except Exception as error:
    st.error("系统初始化失败，请检查配置。")
    st.exception(error)

    with st.expander("查看错误堆栈", expanded=False):
        st.code(traceback.format_exc(), language="text")

    st.stop()


# =========================
# 8. 渲染历史对话
# =========================

render_chat_history(debug=debug_mode)


# =========================
# 9. 处理示例问题
# =========================

if st.session_state.pending_query:
    pending = st.session_state.pending_query
    st.session_state.pending_query = None

    submit_query(
        query=pending,
        qa_system=qa_system,
        debug=debug_mode,
    )

    st.rerun()


# =========================
# 10. 用户输入
# =========================

user_query = st.chat_input("请输入关于校规的问题，例如：上课可以喝水吗？")

if user_query:
    submit_query(
        query=user_query,
        qa_system=qa_system,
        debug=debug_mode,
    )
    st.rerun()


# =========================
# 11. 错误信息展示
# =========================

if st.session_state.last_error:
    with st.expander("查看最近一次系统错误", expanded=False):
        st.code(st.session_state.last_error, language="text")
