"""PaperQA — 个人论文知识库（聊天 + 论文原文预览面板）。

侧边栏：论文库管理（批量上传/列表/删除/范围切换）+ 会话历史；
左栏：流式聊天（思考过程 + 论文级引用，引用可点击定位）；
右栏：论文原文预览面板（引用点击后自动定位到对应页，可滚动浏览全文）。
"""
import hashlib
import re
import tempfile
import time
from pathlib import Path

import pymupdf as fitz
import streamlit as st
import streamlit.components.v1 as components

from paperqa import history as history_store
from paperqa.config import Settings
from paperqa.library import STATIC_PAPERS_DIR
from paperqa.pipeline import Pipeline

st.set_page_config(
    page_title="PaperQA · 个人论文知识库",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_FILES = 10
MAX_TOTAL_MB = 200
STATIC_PAPERS_URL = "app/static/papers"  # Streamlit 静态服务挂载在 /app/static/

# ---------------- 会话状态与历史（token 标识，URL ?t= 可恢复/分享）----------------
if "token" not in st.session_state:
    qp_token = dict(st.query_params).get("t")
    st.session_state.token = qp_token if history_store.valid_token(qp_token) else history_store.new_token()
    conv = history_store.load_conversation(st.session_state.token)
    if conv:
        st.session_state.messages = conv.get("messages", [])
        st.session_state.scope = conv.get("scope")
    else:
        st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "scope" not in st.session_state:
    st.session_state.scope = None  # None=全库；paper_id=单篇
if "upload_key" not in st.session_state:
    st.session_state.upload_key = "up_0"  # 换 key 即清空上传区
if "preview" not in st.session_state:
    st.session_state.preview = None  # {"paper_id", "page"} | None

# URL 同步：只保留会话 token；清理历史遗留的 paper/page 参数（预览仅由页内 📍 按钮触发）
qp = dict(st.query_params)
if qp.get("t") != st.session_state.token or "paper" in qp or "page" in qp:
    st.query_params.clear()
    st.query_params["t"] = st.session_state.token


def _persist() -> None:
    """当前会话写入历史文件（空会话不落盘）。"""
    if st.session_state.get("messages"):
        history_store.save_conversation(
            st.session_state.token, st.session_state.messages, st.session_state.get("scope")
        )


# ---------------- 侧边栏：论文库管理 ----------------
with st.sidebar:
    st.title("📚 PaperQA")
    st.caption("个人论文知识库：多论文入库，跨论文提问，回答带论文级引用")

    try:
        pipeline = Pipeline(Settings())
    except Exception as e:
        st.error(f"初始化失败（请确认 Qdrant 已启动：{Settings().qdrant_url}）\n\n{type(e).__name__}: {str(e)[:200]}")
        st.stop()

    papers = pipeline.list_papers()
    paper_ids = [p.paper_id for p in papers]

    # 上传结果一次性提示
    flash_result = st.session_state.pop("last_upload_result", None)
    if flash_result:
        st.success(flash_result)
    flash_fail = st.session_state.pop("last_upload_fail", None)
    if flash_fail:
        st.error("以下文件入库失败：\n" + "\n".join(flash_fail))

    uploaded_files = st.file_uploader(
        "批量上传论文 PDF（加入知识库）",
        type=["pdf"],
        accept_multiple_files=True,
        key=st.session_state.upload_key,
        help=f"一次最多 {MAX_FILES} 个文件，总计不超过 {MAX_TOTAL_MB}MB；同名论文重传自动覆盖更新",
    )
    if uploaded_files:
        total_mb = sum(f.size for f in uploaded_files) / 1024 / 1024
        if len(uploaded_files) > MAX_FILES:
            st.error(f"一次最多上传 {MAX_FILES} 个文件，当前选择了 {len(uploaded_files)} 个")
        elif total_mb > MAX_TOTAL_MB:
            st.error(f"所选文件总计 {total_mb:.0f}MB，超过 {MAX_TOTAL_MB}MB 上限")
        else:
            # 去掉界面层 MD5 预筛选：所有文件都交入库函数判定（未变化→跳过且补建 PDF 存档）
            to_index = list(uploaded_files)

            if to_index:
                ok, updated, fail, last_paper_id, unchanged = 0, 0, [], None, 0
                progress = st.progress(0.0, text="准备入库…")
                for i, f in enumerate(to_index):
                    progress.progress(i / len(to_index), text=f"({i + 1}/{len(to_index)}) {f.name}")
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(f.getbuffer())
                        tmp_path = tmp.name
                    try:
                        info = pipeline.index_paper(tmp_path, filename=f.name)  # 传原始文件名
                        if info.get("skipped"):
                            unchanged += 1  # 库层判定：同标题同 MD5，跳过向量化
                        else:
                            ok += 1
                            if info.get("existed"):
                                updated += 1  # 覆盖更新（同题不同版本 / 同名内容变化）
                            last_paper_id = info["paper_id"]
                    except Exception as e:
                        fail.append(f"{f.name}: {type(e).__name__} {str(e)[:80]}")
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                done_text = f"入库完成：新入库 {ok - updated} / 覆盖更新 {updated} / 内容未变化 {unchanged} / 失败 {len(fail)}"
                progress.progress(1.0, text=done_text)
                # 结果转为一次性提示，并更换上传区 key → 清空已选文件
                st.session_state.last_upload_result = done_text
                st.session_state.last_upload_fail = fail
                st.session_state.upload_key = f"up_{time.time()}"
                if ok > 0:
                    st.session_state.messages = []  # 语料变更，清空旧对话
                    st.session_state.preview = None
                    if ok == 1 and last_paper_id:
                        st.session_state.scope = last_paper_id  # 单篇 → 聚焦该篇
                    elif ok > 1:
                        st.session_state.scope = None  # 多篇 → 全库
                st.rerun()
            elif unchanged:
                st.session_state.last_upload_result = f"所选 {unchanged} 个文件与知识库内容一致，未重复入库"
                st.session_state.upload_key = f"up_{time.time()}"
                st.rerun()

    # 检索范围
    labels = ["🌍 全库检索"] + [f"📄 {p.title[:36]}" for p in papers]
    ids = [None] + paper_ids
    current = st.session_state.scope
    idx = ids.index(current) if current in ids else 0
    choice = st.radio("检索范围", labels, index=idx, label_visibility="collapsed")
    new_scope = ids[labels.index(choice)]
    if new_scope != st.session_state.scope:
        st.session_state.scope = new_scope
        _persist()

    # 论文列表 + 删除
    st.markdown(f"**知识库**（{len(papers)} 篇）")
    for p in papers:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"📄 {p.title[:34]}")
        c1.caption(f"{p.pages}页 / {p.chunks}片段")
        if c2.button("删", key=f"del::{p.paper_id}"):
            pipeline.delete_paper(p.paper_id)
            if st.session_state.scope == p.paper_id:
                st.session_state.scope = None
            st.session_state.messages = []
            st.session_state.preview = None
            st.rerun()

    st.divider()
    if st.button("🆕 新对话", use_container_width=True):
        st.session_state.token = history_store.new_token()
        st.session_state.messages = []
        st.session_state.scope = None
        st.session_state.preview = None
        st.query_params["t"] = st.session_state.token
        st.rerun()
    convs = history_store.list_conversations()
    if convs:
        conv_labels = [f"💬 {c['title'][:28]} · {c['updated_at'][5:16]}" for c in convs]
        sel = st.selectbox("历史会话", conv_labels, index=None, placeholder="切换到历史会话…")
        if sel is not None:
            token_sel = convs[conv_labels.index(sel)]["token"]
            conv = history_store.load_conversation(token_sel) or {"messages": [], "scope": None}
            st.session_state.token = token_sel
            st.session_state.messages = conv.get("messages", [])
            st.session_state.scope = conv.get("scope")
            st.session_state.preview = None
            st.query_params["t"] = token_sel
            st.rerun()

    st.divider()
    st.caption(f"模型：`{Settings().llm_model}`")
    st.caption("回答基于检索到的论文原文并附引用；点击引用可定位原文预览。")

# ---------------- 主区：聊天（左） + 论文预览（右） ----------------
pv = st.session_state.get("preview")
if pv:
    chat_col, preview_col = st.columns([0.6, 0.4])
else:
    chat_col = st.container()
    preview_col = None

with chat_col:
    scope = st.session_state.scope
    scope_label = "🌍 全库" if scope is None else next(
        (f"📄 {p.title[:36]}" for p in papers if p.paper_id == scope), "未知论文"
    )
    st.caption(f"检索范围：{scope_label} · 提问会自动改写并检索相关原文，回答附论文级引用")

user_input = st.chat_input(
    "针对知识库提问，例如：哪篇论文是对知识追踪的改进？",
    disabled=len(papers) == 0,
)

with chat_col:
    # 历史消息
    for mi, msg in enumerate(st.session_state.messages):
        with chat_col.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
            if msg.get("reasoning"):
                with st.status("💭 深度思考", expanded=False):
                    st.markdown(msg["reasoning"])
            if msg.get("intent"):
                st.caption("🧭 已按分析模式作答（允许有据推断）" if msg["intent"] == "analytical" else "📎 提取模式 · 严格摘抄")
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 引用来源（点击 📍 在右侧预览原文）"):
                    for si, c in enumerate(msg["sources"]):
                        cc1, cc2 = st.columns([5, 1])
                        cc1.markdown(f"**[《{c['title']}》 第{c['page']}页 / 片段{c['chunk_id']}]**")
                        cc1.text(c["text"][:300])
                        if cc2.button("📍", key=f"loc::{mi}::{si}", help="定位到原文"):
                            st.session_state.preview = {"paper_id": c["paper_id"], "page": c["page"], "title": c["title"]}
                            st.rerun()

    # 空状态欢迎 + 示例问题
    if not st.session_state.messages:
        with chat_col.chat_message("assistant", avatar="🤖"):
            st.markdown(
                f"你好！知识库现有 **{len(papers)} 篇论文**。\n\n"
                "直接提问——可以问单篇内容，也可以问「哪篇论文做了 X」这类跨论文问题；"
                "回答基于检索到的原文并附引用，点击引用可定位到论文原文预览。"
            )
        st.markdown("##### 试试这些问题")
        suggestions = [
            "哪篇论文是对知识追踪的改进？",
            "这篇论文的核心贡献是什么？",
            "各篇论文分别用了什么方法？",
            "这些研究有哪些共同的局限性？",
        ]
        cols = (st.columns(2), st.columns(2))
        for col, q in zip((cols[0][0], cols[0][1], cols[1][0], cols[1][1]), suggestions):
            if col.button(q, key=f"sug::{q}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

# 预览面板（右栏）：PyMuPDF 渲染成图片，保证显示、不依赖浏览器 PDF 插件
if preview_col is not None and pv:
    with preview_col:
        paper_id, page, title = pv.get("paper_id"), pv.get("page", 1), pv.get("title", "")
        paper = next((pp for pp in papers if pp.paper_id == paper_id), None)
        # 旧会话引用可能指向已合并/删除的条目 → 按标题重映射到现存论文
        if paper is None and title:
            paper = next((pp for pp in papers if pp.title[:24] == title[:24]), None)
            if paper:
                paper_id = paper.paper_id
        pdf_path = STATIC_PAPERS_DIR / f"{paper_id}.pdf"
        st.markdown("##### 📄 论文原文预览")
        if st.button("✕ 关闭预览", use_container_width=True):
            st.session_state.preview = None
            st.rerun()
        if paper is None:
            st.info("该引用对应的论文已从知识库移除，请重新提问获取最新引用。")
        elif not pdf_path.exists():
            st.warning("该论文未存档原始 PDF，重新上传一次即可获得预览。")
        else:
            st.markdown(f"**{paper.title[:60]}**")
            total = paper.pages or 0
            n1, n2, n3 = st.columns([1, 1, 4])
            if n1.button("◀ 上一页", disabled=page <= 1, use_container_width=True):
                st.session_state.preview["page"] = page - 1
                st.rerun()
            if n2.button("下一页 ▶", disabled=bool(total and page >= total), use_container_width=True):
                st.session_state.preview["page"] = page + 1
                st.rerun()
            n3.caption(f"📍 第 {page} / {total or '?'} 页")
            try:
                doc = fitz.open(str(pdf_path))
                idx = min(max(page - 1, 0), len(doc) - 1)
                pix = doc[idx].get_pixmap(dpi=130)
                st.image(pix.tobytes("png"), use_container_width=True)
                doc.close()
            except Exception as e:
                st.error(f"渲染失败：{type(e).__name__}: {str(e)[:150]}")

# 处理输入
prompt = st.session_state.pending or user_input
st.session_state.pending = None
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_col.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    hits, full, reasoning, intent = None, None, "", "extractive"
    with chat_col.chat_message("assistant", avatar="🤖"):
        try:
            retrieved = pipeline.retrieve(prompt, top_k=4, paper_id=scope)
            hits, intent = retrieved.hits, retrieved.intent
            st.caption("🧭 已按分析模式作答（允许有据推断）" if intent == "analytical" else "📎 提取模式 · 严格摘抄")
            status = st.status("💭 思考中…", expanded=True)
            with status:
                reasoning_ph = st.empty()
                reasoning_ph.caption("连接模型中…")
            answer_ph = st.empty()

            t0 = time.time()
            rendered_len = 0
            for kind, delta in pipeline.answer_stream(prompt, hits, intent=intent):
                if kind == "reasoning":
                    reasoning += delta
                    if len(reasoning) - rendered_len >= 24:
                        reasoning_ph.markdown(reasoning)
                        rendered_len = len(reasoning)
                else:
                    if full is None:
                        reasoning_ph.markdown(reasoning)
                        secs = max(1, round(time.time() - t0))
                        status.update(label=f"💭 已深度思考 {secs} 秒", state="complete", expanded=False)
                    full = (full or "") + delta
                    answer_ph.markdown(full)
            if full is None:
                status.update(label="💭 模型未返回内容", state="complete", expanded=True)
        except Exception as e:
            st.error(f"调用失败：{type(e).__name__}: {str(e)[:300]}")
        if full:
            answer_ph.markdown(full)  # 引用预览通过下方 📍 按钮页内唤起，避免新标签跳转
            with st.expander("📎 引用来源（点击 📍 在右侧预览原文）", expanded=True):
                for si, c in enumerate(hits):
                    cc1, cc2 = st.columns([5, 1])
                    cc1.markdown(f"**[《{c['title']}》 第{c['page']}页 / 片段{c['chunk_id']}]**")
                    cc1.text(c["text"][:300])
                    if cc2.button("📍", key=f"locnew::{si}", help="定位到原文"):
                        st.session_state.preview = {"paper_id": c["paper_id"], "page": c["page"], "title": c["title"]}
                        st.rerun()
            st.session_state.messages.append(
                {
                    "role": "assistant", "content": full, "sources": hits,
                    "reasoning": reasoning or None, "intent": intent,
                }
            )
            _persist()
