"""
AI 校规问答系统 - 命令行版本

功能说明：
1. 读取 data/school.txt 中的校规文本；
2. 按条款切分校规；
3. 使用 SentenceTransformer 生成向量；
4. 对用户问题进行查询扩展；
5. 通过向量相似度召回候选校规；
6. 使用关键词规则进行二次重排；
7. 使用本地规则引擎或 Ollama + DeepSeek 进行最终判定；
8. 输出“允许 / 不允许 / 有条件允许 / 未明确”的结构化结果。

运行方式：
    python main.py

可选运行方式：
    python main.py --no-ollama
    python main.py --debug
    python main.py --rule-path data/school.txt
    python main.py --model-name deepseek-r1:8b
"""

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# =========================
# 1. 默认配置
# =========================

DEFAULT_OLLAMA_ENABLED = True
DEFAULT_OLLAMA_PATH = "C:/Users/23528/AppData/Local/Programs/Ollama/ollama.exe"
DEFAULT_MODEL_NAME = "deepseek-r1:8b"

DEFAULT_RULE_PATH = "data/school.txt"
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh"

DEFAULT_TOP_K = 20
DEFAULT_MAX_CONTEXT_CHARS = 4000
DEFAULT_CANDIDATE_MULTIPLIER = 3
DEFAULT_OLLAMA_TIMEOUT = 300


# =========================
# 2. 数据结构
# =========================

@dataclass
class RetrievedRule:
    """单条检索结果。"""
    rule_id: str
    text: str
    final_score: float
    semantic_score: float
    keyword_score: float


@dataclass
class JudgeResult:
    """最终判定结果。"""
    decision: str
    reason: str
    source: str
    judge_type: str


# =========================
# 3. 查询扩展
# =========================

def expand_query(query: str) -> str:
    """
    将用户的口语化表达扩展为校规中更可能出现的正式表达。

    注意：
    这里的扩展要克制，不能乱扩展。
    例如“喝水”不能扩展成“饮水机”，否则容易误命中“饮水机洗脸、洗衣物”等无关条款。
    """
    mapping = {
        "翘课": "无故缺课 旷课 缺席 课程",
        "逃课": "无故缺课 旷课 缺席 课程",
        "旷课": "无故缺课 缺席 课程",
        "缺课": "无故缺课 旷课 缺席 课程",

        "迟到": "迟到 上课铃",
        "早退": "早退 下课前离开",

        "打架": "斗殴 动武 身体伤害 校园欺凌",
        "干架": "斗殴 动武 身体伤害 校园欺凌",
        "骂人": "谩骂 诋毁 人身攻击 言语侮辱",
        "欺负": "校园欺凌 言语侮辱 身体伤害 网络霸凌",
        "霸凌": "校园欺凌 言语侮辱 身体伤害 网络霸凌",

        "吃饭": "用餐 吃零食 餐具 食堂 教室",
        "吃东西": "用餐 吃零食 教室 实验室 图书馆 食堂",
        "吃零食": "用餐 吃零食 教室 实验室 图书馆",

        "喝水": "喝水 课堂 秩序",

        "睡觉": "课堂 认真听讲 教学秩序",
        "上课睡觉": "课堂 认真听讲 教学秩序",
        "课上睡觉": "课堂 认真听讲 教学秩序",

        "抽烟": "吸烟 烟头 禁烟区域",
        "吸烟": "吸烟 烟头 禁烟区域",

        "喝酒": "酗酒 酒精 聚餐",
        "酗酒": "酗酒 酒精 聚餐",

        "考试作弊": "作弊 交头接耳 传递纸条 替考",
        "作弊": "考试作弊 交头接耳 传递纸条 替考",
        "替考": "作弊 替考 考试纪律",

        "养宠物": "饲养宠物 猫 狗 爬行动物",
        "养猫": "饲养宠物 猫 狗 爬行动物",
        "养狗": "饲养宠物 猫 狗 爬行动物",

        "夜不归宿": "按时归寝 夜不归宿 报备",
        "外出": "外出 请假 报备 家长签字",
        "请假": "请假手续 证明 报备",
        "病假": "请假手续 病假 证明 报备",
        "事假": "请假手续 事假 证明 报备",

        "补考": "补考 期末后两周",
        "缓考": "缓考 缺考 证明",
    }

    query = query.strip()
    extra_words = []

    for key, value in mapping.items():
        if key in query:
            extra_words.append(value)

    if extra_words:
        return query + " " + " ".join(extra_words)

    return query


# =========================
# 4. 关键词重排
# =========================

def keyword_score(query: str, doc: str) -> float:
    """
    简单关键词加分，用于补足纯向量检索对具体场景词不敏感的问题。

    设计目的：
    1. 行为词匹配：例如“打架”“作弊”“喝水”“吃饭”；
    2. 场景词匹配：例如“食堂”“教室”“宿舍”；
    3. 规则词匹配：例如“严禁”“不得”“应当”；
    4. 局部字面匹配：用户问题里的连续两个字如果出现在校规中，少量加分。
    """
    score = 0.0

    action_words = [
        "缺课", "旷课", "迟到", "早退",
        "斗殴", "动武", "打架", "身体伤害", "欺凌",
        "用餐", "吃饭", "吃零食",
        "喝水",
        "睡觉", "认真听讲",
        "作弊", "替考", "补考", "缓考",
        "吸烟", "酗酒", "宠物", "请假", "归寝",
    ]

    location_words = [
        "食堂", "教室", "课堂", "实验室", "图书馆",
        "宿舍", "宿舍楼", "校园", "教学楼",
    ]

    rule_words = [
        "严禁", "不得", "禁止", "必须", "须", "应", "应当", "可以", "可",
    ]

    for word in action_words:
        if word in query and word in doc:
            score += 4.0

    for word in location_words:
        if word in query and word in doc:
            score += 4.0

    for word in rule_words:
        if word in doc:
            score += 0.5

    # 二字片段轻微加分，避免完全依赖 embedding。
    # 加分较小，避免短词把结果带偏。
    for i in range(len(query) - 1):
        token = query[i:i + 2]
        if token.strip() and token in doc:
            score += 0.3

    return score


# =========================
# 5. 文本加载与切分
# =========================

def load_rules(path: str) -> Tuple[List[str], List[str]]:
    """
    加载校规文本，并按照“1. xxx”这种格式切分。

    school.txt 推荐格式：
        1. 学生应当……
        2. 严禁……
        3. 如需请假，应当……
    """
    rule_path = Path(path)

    if not rule_path.exists():
        raise FileNotFoundError(
            f"未找到校规文件：{rule_path.resolve()}\n"
            f"请确认 {path} 是否存在。"
        )

    text = rule_path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    matches = re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)", text, re.S)

    docs: List[str] = []
    ids: List[str] = []

    for num, content in matches:
        content = content.strip().replace("\n", "")
        if content:
            docs.append(content)
            ids.append(num)

    if not docs:
        raise ValueError(
            "没有成功解析到任何校规条款。\n"
            "请检查 school.txt 是否采用类似“1. xxx”“2. xxx”的编号格式。"
        )

    return docs, ids


# =========================
# 6. 检索器
# =========================

class SchoolRuleRetriever:
    """
    校规检索器。

    负责：
    1. 加载 embedding 模型；
    2. 将校规文本向量化；
    3. 对用户问题进行向量召回；
    4. 结合关键词得分进行二次重排。
    """

    def __init__(
        self,
        docs: List[str],
        ids: List[str],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        self.docs = docs
        self.ids = ids
        self.embedding_model_name = embedding_model_name
        self.max_context_chars = max_context_chars
        self.candidate_multiplier = candidate_multiplier

        print("正在加载 embedding 模型，请稍等...")
        self.embed_model = SentenceTransformer(self.embedding_model_name)

        print("正在生成规则向量，请稍等...")
        self.embeddings = self.embed_model.encode(self.docs, normalize_embeddings=True)

        print("✅ 向量生成完成")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievedRule]:
        expanded_query = expand_query(query)

        q_vec = self.embed_model.encode([expanded_query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_vec.T).squeeze()

        candidate_count = min(
            max(top_k * self.candidate_multiplier, 30),
            len(self.docs),
        )

        candidate_indices = np.argsort(scores)[-candidate_count:][::-1]

        reranked = []

        for i in candidate_indices:
            semantic = float(scores[i])
            key_score = keyword_score(expanded_query, self.docs[i])
            final = semantic + key_score * 0.05

            reranked.append(
                RetrievedRule(
                    rule_id=self.ids[i],
                    text=self.docs[i],
                    final_score=final,
                    semantic_score=semantic,
                    keyword_score=key_score,
                )
            )

        reranked.sort(key=lambda item: item.final_score, reverse=True)

        selected: List[RetrievedRule] = []
        char_count = 0

        for item in reranked:
            if len(selected) >= top_k:
                break

            if char_count + len(item.text) > self.max_context_chars:
                break

            selected.append(item)
            char_count += len(item.text)

        return selected


# =========================
# 7. 本地规则判定辅助函数
# =========================

def contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


ACTION_GROUPS: Dict[str, List[str]] = {
    "eat": ["吃饭", "吃东西", "吃零食", "用餐", "餐具"],
    "drink_water": ["喝水"],
    "skip_class": ["翘课", "逃课", "旷课", "缺课", "无故缺课"],
    "late": ["迟到"],
    "leave_early": ["早退"],
    "fight": ["打架", "干架", "斗殴", "动武", "身体伤害", "校园欺凌"],
    "sleep_class": ["睡觉", "上课睡觉", "课堂睡觉", "课上睡觉"],
    "smoke": ["抽烟", "吸烟"],
    "drink_alcohol": ["喝酒", "酗酒"],
    "cheat": ["作弊", "考试作弊", "替考"],
    "pet": ["养宠物", "饲养宠物", "养猫", "养狗"],
    "ask_leave": ["请假", "请病假", "请事假"],
    "makeup_exam": ["补考"],
    "defer_exam": ["缓考"],
}


LOCATION_GROUPS: Dict[str, List[str]] = {
    "canteen": ["食堂"],
    "classroom": ["教室", "课堂", "上课", "课上"],
    "lab": ["实验室"],
    "library": ["图书馆"],
    "dorm": ["宿舍", "寝室", "宿舍楼"],
    "campus": ["校园", "校内", "学校"],
}


FORBID_WORDS = [
    "严禁", "不得", "禁止", "不准", "不可", "不能",
    "违者", "处分", "处理", "移交公安机关",
]


CONDITION_WORDS = [
    "须", "需", "需要", "应", "应当", "按规定",
    "经批准", "提前", "报备", "办理", "提供证明",
]


ALLOW_WORDS = [
    "可以", "可", "有权",
]


def extract_groups(text: str, group_dict: Dict[str, List[str]]) -> set:
    result = set()

    for group_name, words in group_dict.items():
        if contains_any(text, words):
            result.add(group_name)

    return result


def rule_is_about_query(query: str, rule: str) -> bool:
    """
    判断某条校规是否真的在回答用户问题。

    解决典型误判：
    1. 用户问“食堂吃饭”，不能误命中“教室禁止吃饭”；
    2. 用户问“喝水”，不能误命中“饮水机洗脸、洗衣物”；
    3. 用户问“代签到”，不能误命中“代检”。
    """
    query_actions = extract_groups(query, ACTION_GROUPS)
    query_locations = extract_groups(query, LOCATION_GROUPS)

    rule_actions = extract_groups(rule, ACTION_GROUPS)
    rule_locations = extract_groups(rule, LOCATION_GROUPS)

    if query_actions and not (query_actions & rule_actions):
        return False

    if query_locations and rule_locations and not (query_locations & rule_locations):
        return False

    return True


def forbid_is_same_action(query: str, rule: str) -> bool:
    """
    判断校规里的禁止行为是否与用户问题属于同一行为。

    这个函数是为了降低“字面相似但实际行为不同”的误判。
    """
    # 用户问食堂吃饭，不能因为“不得带出餐具”就判为不允许吃饭。
    if "食堂" in query and any(w in query for w in ["吃饭", "用餐", "吃东西"]):
        if any(w in rule for w in ["餐具私自带出", "带出食堂", "洗脸", "洗涤衣物"]):
            return False

    # 用户问喝水，不能因为“饮水机洗脸/洗衣物”就判为不允许喝水。
    if "喝水" in query:
        if any(w in rule for w in ["洗脸", "洗涤衣物", "游泳", "水域"]):
            return False

    # 用户问食堂吃饭，不能因为教室、实验室、图书馆禁止用餐就误判。
    if any(w in query for w in ["吃饭", "吃东西", "用餐"]):
        if "食堂" in query and any(w in rule for w in ["教室", "实验室", "图书馆"]):
            return False

    return True


def get_allow_reason_from_rules(
    query: str,
    top_rules: List[str],
    top_ids: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    从候选规则中寻找支持“允许 / 有条件允许”的相关规则。
    """
    for rule, rule_id in zip(top_rules, top_ids):
        if not rule_is_about_query(query, rule):
            continue

        if contains_any(rule, ALLOW_WORDS + CONDITION_WORDS):
            return rule_id, rule

    return None, None


def local_judge(
    query: str,
    top_rules: List[str],
    top_ids: List[str],
) -> JudgeResult:
    """
    本地规则判定逻辑。

    判定原则：
    1. 不只看第一条检索结果；
    2. 禁止类规则必须同时匹配行为和场景；
    3. 字面相似但行为不同的规则不能引用；
    4. 没有明确禁止时，不直接判为“不允许”；
    5. 对高频易误判问题进行安全特判。
    """
    if not top_rules:
        return JudgeResult(
            decision="未明确",
            reason="校规中没有检索到与该问题直接对应的条款。",
            source="未检索到明确相关条款",
            judge_type="本地规则引擎",
        )

    # A0. 高频明确场景：食堂吃饭
    if "食堂" in query and any(word in query for word in ["吃饭", "用餐", "吃东西"]):
        for rule, rule_id in zip(top_rules, top_ids):
            if "食堂用餐" in rule or ("食堂" in rule and "文明就餐" in rule):
                return JudgeResult(
                    decision="允许",
                    reason=f"根据第{rule_id}条校规：{rule}",
                    source=f"第{rule_id}条校规",
                    judge_type="本地规则引擎",
                )

        return JudgeResult(
            decision="允许",
            reason="校规中没有明确禁止在食堂吃饭。一般可以，但应排队、文明就餐，并按要求归还餐具。",
            source="食堂用餐相关校规",
            judge_type="本地规则引擎",
        )

    # A1. 上课睡觉
    if "睡觉" in query and any(word in query for word in ["上课", "课堂", "课上"]):
        for rule, rule_id in zip(top_rules, top_ids):
            if "认真听讲" in rule or "教学秩序" in rule:
                return JudgeResult(
                    decision="不允许",
                    reason=f"根据第{rule_id}条校规：{rule}",
                    source=f"第{rule_id}条校规",
                    judge_type="本地规则引擎",
                )

        return JudgeResult(
            decision="不允许",
            reason="校规要求学生上课认真听讲，不应在课堂上睡觉。",
            source="课堂纪律相关校规",
            judge_type="本地规则引擎",
        )

    # A2. 喝水
    if "喝水" in query:
        if any(word in query for word in ["上课", "课堂", "课上"]):
            return JudgeResult(
                decision="允许",
                reason="校规中没有明确禁止上课喝水。一般可以喝水，但不应影响课堂秩序。",
                source="未检索到明确禁止条款",
                judge_type="本地规则引擎",
            )

        return JudgeResult(
            decision="允许",
            reason="校规中没有明确禁止喝水。一般可以，但应遵守公共场所秩序。",
            source="未检索到明确禁止条款",
            judge_type="本地规则引擎",
        )

    # B. 先找真正相关的禁止类规则
    for rule, rule_id in zip(top_rules, top_ids):
        if not rule_is_about_query(query, rule):
            continue

        if contains_any(rule, FORBID_WORDS) and forbid_is_same_action(query, rule):
            return JudgeResult(
                decision="不允许",
                reason=f"根据第{rule_id}条校规：{rule}",
                source=f"第{rule_id}条校规",
                judge_type="本地规则引擎",
            )

    # C. 再找真正相关的允许 / 正向要求类规则
    allow_rule_id, allow_rule = get_allow_reason_from_rules(query, top_rules, top_ids)

    if allow_rule and allow_rule_id:
        if contains_any(allow_rule, ALLOW_WORDS):
            return JudgeResult(
                decision="允许",
                reason=f"根据第{allow_rule_id}条校规：{allow_rule}",
                source=f"第{allow_rule_id}条校规",
                judge_type="本地规则引擎",
            )

        if contains_any(allow_rule, CONDITION_WORDS):
            return JudgeResult(
                decision="有条件允许",
                reason=f"根据第{allow_rule_id}条校规：{allow_rule}",
                source=f"第{allow_rule_id}条校规",
                judge_type="本地规则引擎",
            )

    # D. 单独寻找有条件允许类规则
    for rule, rule_id in zip(top_rules, top_ids):
        if not rule_is_about_query(query, rule):
            continue

        if contains_any(rule, CONDITION_WORDS):
            return JudgeResult(
                decision="有条件允许",
                reason=f"根据第{rule_id}条校规：{rule}",
                source=f"第{rule_id}条校规",
                judge_type="本地规则引擎",
            )

    # E. 没有明确禁止，不直接吓唬用户
    return JudgeResult(
        decision="未明确",
        reason="校规中没有检索到与该问题直接对应的明确禁止或明确允许条款，建议按校园公共秩序和教师要求执行。",
        source="未检索到明确相关条款",
        judge_type="本地规则引擎",
    )


# =========================
# 8. Ollama 裁判
# =========================

def clean_deepseek_output(text: str) -> str:
    """
    deepseek-r1 可能输出 <think>...</think>，这里统一删除。
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def normalize_decision(decision: str) -> str:
    """
    规范化模型输出的判定字段。
    """
    decision = decision.strip().replace("：", "").replace(":", "")

    if "有条件" in decision:
        return "有条件允许"

    if "不允许" in decision or "禁止" in decision or "不能" in decision:
        return "不允许"

    if "允许" in decision or "可以" in decision:
        return "允许"

    if "未明确" in decision or "不明确" in decision or "无法确定" in decision:
        return "未明确"

    return "未明确"


def parse_ollama_output(output: str) -> JudgeResult:
    """
    解析 Ollama 输出。

    期望格式：
        判定：允许/不允许/有条件允许/未明确
        原因：……
        来源：……
    """
    decision = "未明确"
    reason = "候选校规中没有找到与该问题直接对应的条款。"
    source = "未检索到明确相关条款"

    for line in output.splitlines():
        line = line.strip()

        if line.startswith("判定：") or line.startswith("判定:"):
            decision = re.sub(r"^判定[:：]", "", line).strip()

        elif line.startswith("原因：") or line.startswith("原因:"):
            reason = re.sub(r"^原因[:：]", "", line).strip()

        elif line.startswith("来源：") or line.startswith("来源:"):
            source = re.sub(r"^来源[:：]", "", line).strip()

    decision = normalize_decision(decision)

    return JudgeResult(
        decision=decision,
        reason=reason,
        source=source,
        judge_type="Ollama 本地大模型裁判",
    )


def ollama_judge(
    query: str,
    top_rules: List[str],
    top_ids: List[str],
    ollama_path: str,
    model_name: str,
    timeout: int = DEFAULT_OLLAMA_TIMEOUT,
) -> JudgeResult:
    """
    使用 Ollama 调用本地大模型进行最终裁判。

    注意：
    大模型只负责根据候选校规进行判断，不允许编造校规。
    """
    if not top_rules:
        return JudgeResult(
            decision="未明确",
            reason="系统没有检索到候选校规，因此无法交给本地大模型判断。",
            source="未检索到候选校规",
            judge_type="Ollama 本地大模型裁判",
        )

    rules_text = ""

    for rule_id, rule in zip(top_ids, top_rules):
        rules_text += f"第{rule_id}条：{rule}\n"

    prompt = f"""
你是一个严格的学校校规问答裁判。

你的任务是根据【候选校规】回答【用户问题】。

重要规则：
1. 你只能依据候选校规回答，不能编造校规。
2. 如果候选校规和用户问题不是同一个行为，不能引用。
3. 如果候选校规只是字面相似，但行为不同，也不能引用。
   例如：
   - “代检”不等于“代签到”
   - “饮水机洗脸”不等于“上课喝水”
   - “带出餐具”不等于“在食堂吃饭”
   - “教室禁止用餐”不等于“食堂不能吃饭”
4. 如果没有明确相关校规，判定为“未明确”。
5. 如果规则明确禁止，判定为“不允许”。
6. 如果规则明确允许，判定为“允许”。
7. 如果规则要求审批、报备、证明、批准，判定为“有条件允许”。
8. 输出必须严格使用下面格式，不要输出思考过程，不要输出多余解释。

【用户问题】
{query}

【候选校规】
{rules_text}

请严格按下面格式输出：

判定：允许/不允许/有条件允许/未明确
原因：用一句话说明依据。如果没有明确依据，就说明“候选校规中没有找到与该问题直接对应的条款”。
来源：第X条校规/未检索到明确相关条款
"""

    try:
        result = subprocess.run(
            [ollama_path, "run", model_name],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
        )

        if result.returncode != 0:
            error_text = result.stderr.strip() if result.stderr else "未知错误"
            return JudgeResult(
                decision="未明确",
                reason=f"本地大模型调用失败：{error_text}",
                source="模型调用失败",
                judge_type="Ollama 本地大模型裁判",
            )

        output = clean_deepseek_output(result.stdout)

        if not output:
            return JudgeResult(
                decision="未明确",
                reason="本地大模型没有返回有效内容。",
                source="模型输出为空",
                judge_type="Ollama 本地大模型裁判",
            )

        return parse_ollama_output(output)

    except subprocess.TimeoutExpired:
        return JudgeResult(
            decision="未明确",
            reason="本地大模型判断超时，系统未能确认校规中是否有明确对应条款。",
            source="模型判断超时",
            judge_type="Ollama 本地大模型裁判",
        )

    except FileNotFoundError:
        return JudgeResult(
            decision="未明确",
            reason=f"未找到 Ollama 程序：{ollama_path}",
            source="Ollama 路径错误",
            judge_type="Ollama 本地大模型裁判",
        )

    except Exception as e:
        return JudgeResult(
            decision="未明确",
            reason=f"本地大模型调用异常：{e}",
            source="模型调用异常",
            judge_type="Ollama 本地大模型裁判",
        )


# =========================
# 9. 问答系统主体
# =========================

class SchoolRuleQA:
    """
    校规问答系统主体。

    负责：
    1. 调用检索器获得候选校规；
    2. 调用本地规则引擎或 Ollama 裁判；
    3. 缓存相同问题的结果；
    4. 返回最终答案与检索证据。
    """

    def __init__(
        self,
        rule_path: str = DEFAULT_RULE_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        ollama_enabled: bool = DEFAULT_OLLAMA_ENABLED,
        ollama_path: str = DEFAULT_OLLAMA_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = DEFAULT_TOP_K,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        debug: bool = False,
    ) -> None:
        self.rule_path = rule_path
        self.embedding_model_name = embedding_model_name
        self.ollama_enabled = ollama_enabled
        self.ollama_path = ollama_path
        self.model_name = model_name
        self.top_k = top_k
        self.debug = debug

        print("正在加载校规文本...")
        self.docs, self.ids = load_rules(self.rule_path)
        print(f"✅ 共加载 {len(self.docs)} 条规则")

        self.retriever = SchoolRuleRetriever(
            docs=self.docs,
            ids=self.ids,
            embedding_model_name=self.embedding_model_name,
            max_context_chars=max_context_chars,
        )

        self.cache: Dict[str, Tuple[JudgeResult, List[RetrievedRule]]] = {}

    def ask(self, query: str) -> Tuple[JudgeResult, List[RetrievedRule]]:
        query = query.strip()

        if not query:
            return (
                JudgeResult(
                    decision="未明确",
                    reason="请输入有效问题。",
                    source="无有效输入",
                    judge_type="系统提示",
                ),
                [],
            )

        cache_key = query.lower()

        if cache_key in self.cache:
            return self.cache[cache_key]

        retrieved_rules = self.retriever.retrieve(query, self.top_k)

        top_rules = [item.text for item in retrieved_rules]
        top_ids = [item.rule_id for item in retrieved_rules]

        if self.ollama_enabled:
            judge_result = ollama_judge(
                query=query,
                top_rules=top_rules,
                top_ids=top_ids,
                ollama_path=self.ollama_path,
                model_name=self.model_name,
            )

            # 如果 Ollama 调用失败，不让系统直接瘫痪，自动回退到本地规则引擎。
            if judge_result.source in [
                "模型调用失败",
                "模型输出为空",
                "模型判断超时",
                "Ollama 路径错误",
                "模型调用异常",
            ]:
                fallback_result = local_judge(query, top_rules, top_ids)
                fallback_result.reason = (
                    f"{judge_result.reason}；系统已自动回退到本地规则引擎。"
                    f"{fallback_result.reason}"
                )
                judge_result = fallback_result
        else:
            judge_result = local_judge(query, top_rules, top_ids)

        result = (judge_result, retrieved_rules)
        self.cache[cache_key] = result

        return result


# =========================
# 10. 命令行交互
# =========================

def print_answer(
    judge_result: JudgeResult,
    retrieved_rules: List[RetrievedRule],
    debug: bool = False,
) -> None:
    print("\n🤖 回答：")
    print(f"判定：{judge_result.decision}")
    print(f"原因：{judge_result.reason}")
    print(f"来源：{judge_result.source}")
    print(f"裁判：{judge_result.judge_type}")

    print("\n🔎 检索到的相关校规：")

    if not retrieved_rules:
        print("- 未检索到相关校规")
        return

    for item in retrieved_rules[:3]:
        print(f"- 第{item.rule_id}条：{item.text}")

    if debug:
        print("\n🧪 调试信息：")
        for item in retrieved_rules[:10]:
            print(
                f"- 第{item.rule_id}条 | "
                f"final={item.final_score:.4f} | "
                f"semantic={item.semantic_score:.4f} | "
                f"keyword={item.keyword_score:.2f} | "
                f"{item.text}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 校规问答系统：RAG 检索 + 本地规则 / Ollama 裁判")

    parser.add_argument(
        "--rule-path",
        type=str,
        default=DEFAULT_RULE_PATH,
        help=f"校规文本路径，默认：{DEFAULT_RULE_PATH}",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help=f"Embedding 模型名称，默认：{DEFAULT_EMBEDDING_MODEL_NAME}",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"检索候选数量，默认：{DEFAULT_TOP_K}",
    )

    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="关闭 Ollama 本地大模型裁判，仅使用本地规则引擎",
    )

    parser.add_argument(
        "--ollama-path",
        type=str,
        default=DEFAULT_OLLAMA_PATH,
        help="Ollama 可执行文件路径",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Ollama 模型名称，默认：{DEFAULT_MODEL_NAME}",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示检索分数等调试信息",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    qa_system = SchoolRuleQA(
        rule_path=args.rule_path,
        embedding_model_name=args.embedding_model,
        ollama_enabled=not args.no_ollama,
        ollama_path=args.ollama_path,
        model_name=args.model_name,
        top_k=args.top_k,
        debug=args.debug,
    )

    print("\n📚 AI 校规问答系统")
    print("模式：RAG 检索 + 关键词重排 + 本地规则 / Ollama 裁判")
    print("输入 q 退出")

    while True:
        query = input("\n请输入问题：").strip()

        if query.lower() in ["q", "quit", "exit"]:
            print("👋 已退出")
            break

        if not query:
            print("请输入有效问题。")
            continue

        judge_result, retrieved_rules = qa_system.ask(query)
        print_answer(judge_result, retrieved_rules, debug=args.debug)


if __name__ == "__main__":
    main()
