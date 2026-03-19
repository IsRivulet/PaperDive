import sys
import io
import re
import time
import threading
import gradio as gr
from paperdive_pro import arxiv_team, _perform_scan

print("正在扫描论文文件夹...", flush=True)
try:
    print(_perform_scan(), flush=True)
except Exception as e:
    print(f"扫描失败（不影响使用）: {e}", flush=True)


def _extract_response(message: str) -> str:
    """
    直接调用 run() 拿返回对象，避免捕获 Rich 终端输出。
    如果 run() 拿不到内容，降级到 stdout 捕获 + ANSI 清洗。
    """
    # ── 方案一：直接拿返回值（最干净）──────────────────────
    try:
        resp = arxiv_team.run(message, stream=False)
        if resp and hasattr(resp, "content") and resp.content:
            return str(resp.content).strip()
    except Exception:
        pass

    # ── 方案二：捕获 stdout + 清洗（兜底）──────────────────
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        arxiv_team.print_response(message, stream=False)
    except Exception as e:
        sys.stdout = old_stdout
        return f"出错了：{e}"
    finally:
        sys.stdout = old_stdout

    raw = buf.getvalue()
    return _clean_rich_output(raw)


def _clean_rich_output(text: str) -> str:
    """清洗 Rich/agno 终端输出，提取可读的纯文本。"""
    # 1. 去掉所有 ANSI 转义码（颜色、加粗、下划线等）
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)

    # 2. 逐行过滤 Rich UI 外壳
    ui_prefixes = ("┏", "┗", "┃", "┠", "┨", "├", "└", "─", "━")
    ui_keywords = (
        "Working...", "▰", "▱", "INFO ", "Found ",
        " documents", "Tool Calls", "Team Tool Calls",
        "Member Responses", "Thinking (", "Response (",
        "Message ━", "Local RAG",
    )

    clean_lines = []
    for line in text.splitlines():
        stripped = line.strip()

        # 跳过纯 UI 框线
        if stripped.startswith(ui_prefixes):
            # ┃ 开头但带实际内容的行，剥掉边框后保留
            if stripped.startswith("┃"):
                inner = stripped[1:].strip()
                # 跳过空内容行和 UI 关键词行
                if inner and not any(k in inner for k in ui_keywords):
                    clean_lines.append(inner)
            continue

        # 跳过含 UI 关键词的行
        if any(k in stripped for k in ui_keywords):
            continue

        clean_lines.append(line)

    # 3. 合并连续空行（最多保留一个空行）
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(clean_lines))
    return result.strip()


def ask_agent(message: str, history: list):
    if not message.strip():
        yield "", history
        return

    # 先显示用户消息 + 等待占位
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": "思考中 ▌"})
    yield "", history

    # 后台线程跑 agent
    result_box: dict = {"text": None}

    def run():
        result_box["text"] = _extract_response(message)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    # 等待期间刷新光标动画
    cursors = ["▌", "▍", "▎", "▏"]
    i = 0
    while t.is_alive():
        history[-1]["content"] = f"思考中 {cursors[i % len(cursors)]}"
        yield "", history
        time.sleep(0.4)
        i += 1

    t.join()

    history[-1]["content"] = result_box["text"] or "（未收到回复，请重试）"
    yield "", history


# ── Gradio 界面 ────────────────────────────────────────────
with gr.Blocks(title="arXiv 论文助手") as demo:
    gr.Markdown("# arXiv 论文精读助手")
    gr.Markdown("支持公式渲染 · 定理检索 · 证明解析")

    chatbot = gr.Chatbot(
        height=600,
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$",  "right": "$",  "display": False},
        ],
        render_markdown=True,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="输入问题，例如：研究2603.12232v1",
            show_label=False,
            scale=9,
        )
        send_btn = gr.Button("发送", scale=1, variant="primary")

    clear_btn = gr.Button("清空对话")

    msg_box.submit(ask_agent, [msg_box, chatbot], [msg_box, chatbot])
    send_btn.click(ask_agent,  [msg_box, chatbot], [msg_box, chatbot])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )