# PaperDive 

**PaperDive** 是一个基于多智能体（Multi-Agent）架构的自动化学术论文分析系统，专为数学和硬核理工科学术论文的深度阅读与推理而设计。
它不仅能自动从 arXiv 检索和下载论文，还能通过双轨解析引擎（LaTeX 源码解析 + 本地 PaddleOCR）实现对数学公式、定理、证明、定义等结构化内容的无损提取。结合大语言模型的逻辑推理能力，系统能够构建符号表、提取证明依赖链，并以极高的精确度回答关于论文内容的深度问题。

---

##  核心特性

*  **多智能体架构 (Multi-Agent System)**
    * 采用 `agno` 框架构建，包含主管 (`Team Leader`) 以及三位各司其职的专家专家：外网检索 (`arXiv Researcher`)、知识库管理 (`Paper Librarian`) 和 深度推理 (`Deep Reader`)。
*  **双轨文档解析引擎**
    * **优先 LaTeX 源码解析**：自动下载 arXiv 的 `.tar.gz` 源码包，合并、清洗 `.tex` 文件，**无损保留**复杂的数学公式，避免 OCR 导致的字符乱码。
    * **本地 OCR 降级回退**：针对无源码仅有 PDF 的论文，使用本地 `PaddleOCR` 进行并行页面渲染与阅读顺序识别提取。
*  **深度结构化提取 (Regex + LLM)**
    * **两阶段提取**：先利用正则快速扫描章节、定理、证明、定义、关键公式，再通过 LLM 修复和补充元数据（如归属章节、证明指向）。
    * **符号表抽取 (Notation Map)**：遇到看不懂的符号？系统自动抽取全篇数学符号含义，并在推理时自动带入上下文。
    * **证明依赖链提取**：理解"A定理是如何证明的"时，系统能自动梳理其底层依赖了哪些引理和定义。
*  **混合检索增强 (RAG)**
    * 结合 LanceDB 向量检索（语义分块）与 SQLite 结构化数据库精确查询。
    * 支持按结构化标签（如 `element_type="theorem"`, `element_id="3.1"`）精准定位内容。
*  **双端交互界面**
    * **CLI 命令行交互**：支持流式输出和清爽的终端控制体验。
    * **Gradio Web UI**：提供浏览器可视化界面，支持 LaTeX 公式（`$$` 与 `$`）及 Markdown 实时渲染。

---

##  系统架构与核心模块

### 多智能体协作机制
系统核心由 4 个 Agent 组成，由 `Team Leader` 根据用户的自然语言指令动态路由：
1. **Team Leader**: 掌控对话上下文，将请求精确分发给对应专员，并负责跨论文综合对比。
2. **arXiv Researcher**: 专门负责调用 `search_arxiv_papers` 查询学术前沿，以及 `load_paper_for_deep_analysis` 从外网下载源文件/PDF并触发索引。
3. **Paper Librarian**: 本地数据大管家。负责扫描本地 `papers/` 目录的新增文件、删除旧数据以及重建索引。
4. **Deep Reader**: 核心大脑。具备数学推理能力，遇到公式先查符号表，遇到证明先理依赖链，能精准引用原文（如 `[Theorem 3.1, p.5]`）进行回答。

### 目录与数据结构
项目运行后会自动在根目录生成 `arxiv_test/` 文件夹用于存储状态：
```text
PaperDive/
├── arxiv_test/
│   ├── papers/          # 本地缓存的 PDF 文件目录
│   ├── tex_sources/     # 自动下载解压的 LaTeX 源码缓存
│   ├── lancedb/         # 向量数据库，存储语义分块
│   └── state.db         # SQLite 数据库（存储会话、结构化数据、摘要、符号表等）
├── notes/               # 用户通过 `save_note` 工具保存的 Markdown 笔记
├── paperdive_pro.py     # 主程序入口 & Agent 定义
├── arxiv_source_reader.py # LaTeX 源码下载与解析器
├── structure_extractor.py # 结构/符号/依赖图提取器
├── ocr_pdf_reader.py    # 基于 PaddleOCR 的 PDF 解析器
├── web_ui.py            # Gradio Web 端界面
├── .env
└── requirements.txt
```

---

##  安装与环境配置

### 第一步：基础环境准备

你需要确保本机已安装以下基础工具：
1. **Python 3.10 或 3.11**：请勿使用过高版本（如 3.12+），以避免部分底层科学计算库（如 NumPy、PaddlePaddle）出现兼容性问题。
2. **Git**：用于克隆项目代码。

打开终端（Windows 的 PowerShell/CMD，或 Mac/Linux 的 Terminal），验证 Python 版本：
```bash
python --version
```

### 第二步：获取代码与创建虚拟环境

强烈建议使用独立的虚拟环境，避免与你本机的其他 Python 项目发生依赖冲突。

**1. 克隆项目**
```bash
git clone https://github.com/IsRivulet/PaperDive.git
cd PaperDive
```

**2. 创建虚拟环境**
```bash
python -m venv venv
```

**3. 激活虚拟环境**
* **Windows**:
    ```cmd
    .venv\Scripts\activate
    ```
* **macOS / Linux**:
    ```bash
    source .venv/bin/activate
    ```
*(激活成功后，你的命令行提示符前会多出一个 `(venv)` 的标记。)*

### 第三步：安装核心依赖包

在激活的虚拟环境中，我们需要分两批安装依赖：通用包和深度学习视觉包。

**1. 安装通用核心依赖**
```bash
pip install -r requirements.txt
```

**2. 安装 PaddleOCR 与 PaddlePaddle**
系统依赖 PaddleOCR 来处理没有 LaTeX 源码的纯扫描版或纯 PDF 论文。它的安装取决于你的电脑是否有 NVIDIA 显卡：

* **如果你没有独立显卡（或使用的是 macOS / 普通轻薄本）**：
    直接安装 CPU 版本的 PaddlePaddle：
    ```bash
    pip install paddlepaddle paddleocr
    ```
    *(注意：初次运行时系统会提示自动下载 OCR 模型权重，请保持网络畅通。)*

* **如果你有 NVIDIA 显卡（Windows/Linux 且已配置好 CUDA）**：
    建议安装 GPU 版本以获得极速的 PDF 渲染和识别体验。先安装对应 CUDA 版本的 PaddlePaddle，再安装 OCR：
    ```bash
    # 请根据你的 CUDA 版本前往 PaddlePaddle 官网获取准确指令
    # 示例（CUDA 11.8）：
    python -m pip install paddlepaddle-gpu==2.6.1.post118 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
    pip install paddleocr
    ```
    *配置好 GPU 版本后，请打开代码中的 `paperdive_pro.py`，找到 `pdf_reader` 的初始化代码，将 `use_gpu=False` 改为 `use_gpu=True`。*

### 第四步：配置本地向量嵌入模型 (Ollama)

系统使用本地的 `bge-m3` 模型将论文文本转化为 1024 维的向量，用于语义分块和 LanceDB 数据库的精准检索。这部分完全在本地运行，免费且保护隐私。

1. **下载并安装 Ollama**：前往 [Ollama 官网](https://ollama.com/) 下载对应系统的客户端并安装。
2. **启动 Ollama 服务**：安装完成后，确保 Ollama 应用程序正在后台运行。
3. **拉取指定的 Embedding 模型**：打开终端，运行以下命令下载 `bge-m3` 模型（模型大小约 1GB+）：
    ```bash
    ollama pull bge-m3
    ```

### 第五步：配置大语言模型 (LLM) API
```bash
cp .env.example .env
```
编辑 .env，填写 LLM 相关配置（二选一或按需）：

* OpenAI 兼容接口： `OPENAI_API_KEY=your_key`
* 自定义接口（如 ChatECNU）：
```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-api-base/v1
LLM_MODEL_ID=your-model-id
```
*(提示：推理复杂的数学论文，强烈建议配置上下文窗口大于 32K、且具备较强逻辑推理能力的模型，如 DeepSeek V3/R1 或 Claude 3.5 Sonnet。)*

### 第六步：首次运行与初始化

配置完成后，你可以尝试启动系统。首次启动时，代码会自动创建所需的工作目录（如 `arxiv_test/` 文件夹和内置的 SQLite 数据库文件）。

**启动 Web 交互界面（推荐）：**

提供交互式界面和出色的数学公式渲染。
```bash
python web_ui.py
```
当终端输出 `Running on local URL:  http://0.0.0.0:7860` 时，说明一切就绪。在浏览器中打开 `http://localhost:7860` 即可开始你的学术探索之旅！

**启动 CLI 交互模式：**

如果你更喜欢极客范儿的终端体验：
```bash
python paperdive_pro.py
```

###  典型提问示例

你可以直接向助手发送以下指令体验多 Agent 协同：

* **探索与发现**
    * *"帮我找几篇关于 Graph Neural Networks 匹配算法的最新论文"*
    * *(系统列出后)* *"加载第 2 篇"* 或 *"加载 2301.12345"*
* **全局与结构**
    * *"知识库里现在有哪些论文？"*
    * *"介绍一下 2301.12345 这篇文章的核心贡献和主要技巧"*
    * *"这篇论文的结构是什么？列出所有定理"*
* **深度精读与推理**
    * *"定理 3.1 的内容是什么？解释一下它背后的物理/数学直觉"*
    * *"定理 3.1 的证明过程是怎样的？它依赖了哪些前提条件？"*
    * *"在这篇论文里，$\mathcal{F}$ 这个符号是什么意思？"*
* **笔记与管理**
    * *"把刚才我们讨论的定理 3.1 的证明思路总结一下，保存为笔记"*
    * *"删除 2301.12345 的索引数据"*

---

##  核心数据库表结构说明 (SQLite)
系统会在 `arxiv_test/state.db` 中维护结构化知识：
* `paper_structures`: 论文的总体树形结构 JSON。
* `paper_pages`: 逐页的 OCR / Text 原始内容，用于"深读原文"工具。
* `paper_summaries`: LLM 生成的高层摘要、打标（领域/内容/技巧）、证明思路。
* `paper_structural_elements`: 打碎的定理、引理、证明、公式，包含 `start_page` 和 `depends_on` 字段，实现精准的结构检索。
* `paper_notations`: 抽取的全篇数学符号表。

---

##  注意事项与常见问题

1. **LaTeX 下载失败**：某些老论文或仅投递 PDF 的 arXiv 论文无法获取 `.tex` 源码，系统会自动回退到 OCR 模式。
2. **处理速度较慢**：论文的解析、向量化、两阶段结构提取（需要多次请求 LLM）较为耗时。一篇 20 页的论文完整入库可能需要 1-3 分钟，请耐心等待。
3. **安全清理机制**：系统内置 `_cleanup_polluted_session`，如果 LLM 发生了工具调用幻觉（如输出 `<function=>`），系统会自动清理损坏的 session 以防止对话死循环。
4. 使用Web 交互时需要关闭魔法。
---
## 改进对比
相比基础版本（`PaperDive.py`），`paperdive_pro.py` 在以下方面做了显著改进：

### 1. **多源输入与高精度内容提取**
- **LaTeX 源码优先**：支持下载并解析 arXiv 论文的 LaTeX 源码（通过 `download_and_parse_arxiv_source`），保留原始公式和结构，避免 OCR 误差。
- **OCR 增强**：引入 `OcrPDFReader`（基于 PaddleOCR），可处理扫描版 PDF，支持中英文混合、数学公式，并调整 DPI 优化识别。
- **原文分页存储**：将每页 OCR/源码内容存入 `paper_pages` 表，支持按页或按章节精确读取。

### 2. **结构化知识抽取**
- **章节/定理/定义提取**：通过 `extract_paper_structure` 抽取章节层级、定理、引理、证明、定义和关键公式，存入 `paper_structures` 和 `paper_structural_elements` 表。
- **高层摘要生成**：利用 `extract_paper_summary` 生成论文标题、摘要、证明思路、核心技巧及三维标签（领域/内容/技巧），存入 `paper_summaries` 表。
- **符号表提取**：通过 `extract_notation_map` 抽取论文中定义的数学符号及其含义，存入 `paper_notations` 表，支持后续查询（`get_paper_notation`）。
- **依赖图（证明链）**：提取定理与证明之间的依赖关系（`extract_dependency_graph`），实现 `get_proof_chain` 工具，可追溯完整证明依赖树。

### 3. **深度检索与查询能力**
- **结构单元检索**：新增 `search_structural_elements`，可在定理、证明、定义等元素中精确查找（支持按 ID 或关键词）。
- **论文目录浏览**：通过 `browse_paper_catalog` 展示所有论文的标题和三维标签，便于快速筛选。
- **章节/页原文阅读**：`read_paper_pages` 和 `read_paper_section` 直接读取存储的原文，无需重新 OCR。
- **符号表查询**：`get_paper_notation` 支持按 LaTeX 或含义模糊查询符号定义。

### 4. **索引与数据管理**
- **增量扫描与清理**：`_cleanup_stuck_processing` 自动移除卡住的索引记录，`_cleanup_polluted_session` 清除工具调用污染。
- **删除与重索引**：新增 `delete_paper_data`（可选删除向量、结构、摘要、原文）和 `reindex_paper`（全量重建索引）。
- **诊断工具**：`diagnose_paper` 可快速检查论文的本地状态（文件、向量块、结构、摘要等），便于调试。

### 5. **Agent 架构与任务分工**
- **三专家架构**：将原双 Agent（arXiv Researcher + Local RAG Expert）扩展为三专家：
  - **arXiv Researcher**：仅负责外网检索和下载（严格遵守两步走：先搜索，再根据指令下载）。
  - **Paper Librarian**：管理本地索引（扫描、删除、重索引、列表）。
  - **Deep Reader**：专注精读与数学推理（拥有所有深度检索工具，并增加了证明链、符号表等）。
- **主管增强**：Team Leader 强化了 Focus Paper 状态管理、跨论文综合、指令路由，并内置笔记工具（`save_note`/`list_notes`）。

### 6. **数学推理与回答质量**
- **推理规范**：Deep Reader 的 instructions 要求先查符号表、按依赖链从底层向上解释证明、提供直觉性回答+严谨细节。
- **引用格式**：回答必须带引用（如 `[Theorem 3.1, p.5]`），公式保留 LaTeX。
- **对比与综合**：主管可调用 `browse_paper_catalog` 和 `get_paper_overview` 实现多篇论文对比。

### 7. **工程健壮性与可维护性**
- **错误处理**：两阶段结构提取（LLM → 正则降级），避免单点失败；异常捕获更细致。
- **日志与警告抑制**：屏蔽 numpy 除零警告，降低 agno 日志级别为 ERROR。
- **环境变量与配置**：统一使用 `.env` 管理 LLM 参数，支持自定义模型端点。
- **SQLite 表结构**：新增多个表（`paper_structures`、`paper_pages`、`paper_summaries`、`paper_structural_elements`、`paper_notations`），数据结构更完备。

### 8. **用户体验**
- **启动扫描**：交互前自动执行 `_perform_scan`，告知用户当前论文状态。
- **详细反馈**：索引过程打印进度，返回结果包含结构/摘要/标签统计。
- **笔记保存**：允许将问答总结保存为本地 Markdown 笔记。


---

## 技术栈
- [Agno](https://github.com/agno-agi/agno) - 智能体框架
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR 引擎
- [LanceDB](https://lancedb.com/) - 向量数据库
- [Gradio](https://gradio.app/) - Web 界面库
- [arXiv](https://arxiv.org/) - 论文数据源
