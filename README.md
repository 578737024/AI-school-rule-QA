# AI 校规问答系统（混合模式 RAG + 本地 Ollama 模型）

本项目是一个 **混合 RAG + 本地 Ollama 模型** 的学校行为规范问答系统。  
支持单条终端问答，并保留了缓存机制与上下文长度优化。

## 🚀 功能特点

- **RAG 检索**：根据输入问题检索最相关的校规条款
- **本地大模型判断**：结合 Ollama 本地模型 `deepseek-r1:8b` 做最终判断
- **缓存机制**：同一个问题重复查询时直接使用缓存，提高效率
- **上下文截断优化**：限制检索上下文总长度，避免超长输入
- **终端问答**：无需网络即可直接在 CMD / VSCode 终端运行
- **结构化输出**：
- 判定：不允许
- 原因：无故缺课违反校规第一条
- 来源：第1条校规

## 💡 项目亮点（一定要写）

- 使用 RAG 构建校规问答系统
- 本地部署 LLM（Ollama + deepseek）
- 实现语义检索（embedding + 相似度计算）
- 上下文长度优化（避免超长输入）
- 缓存机制减少重复推理
- 支持来源溯源（第xx条校规）

## 🧠 技术架构

用户问题
→ embedding向量化
→ 相似度检索（Top-K）
→ 拼接上下文（限制长度）
→ LLM推理（Ollama）
→ 输出判定 + 原因 + 来源

## 技术组成：

- SentenceTransformers（向量检索）
- NumPy（相似度计算）
- Ollama（本地大模型推理）
- DeepSeek-R1（推理模型）

## 🤖 Ollama 与模型准备

- **1️⃣ 下载 Ollama 官方程序并安装**:
安装完成后记下路径，例如 Windows 默认路径：
C:/Users/你的用户名/AppData/Local/Programs/Ollama/ollama.exe

- **2️⃣ 下载本地模型 deepseek-r1:8b**：
在终端中执行 ollama pull deepseek-r1:8b

- **3️⃣ 在代码中修改 Ollama 路径和模型名称**：
OLLAMA_PATH = "你的 Ollama 安装路径/ollama.exe"
MODEL_NAME = "deepseek-r1:8b"

## ⚙️ 环境依赖

- **使用 Python 3.11+，安装依赖**：
在终端中执行 pip install -r requirements.txt

## 📂 项目结构

- rag-project/
- │
- ├── main.py              # 主程序
- ├── requirements.txt    # 依赖
- ├── data/
- │   └── school.txt      # 校规文本
- └── README.md

## ▶️ 使用方法

- **运行程序**：
python main.py

- **输入问题**：
我可以翘课吗？

- **输出示例**：
- 判定：不允许
- 原因：无故缺课违反校规第一条
- 来源：第1条校规
