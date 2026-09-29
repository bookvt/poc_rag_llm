"""Local Streamlit UI for exploring the steps of a CV RAG pipeline."""

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from rag import TokenUsage, answer_question, index_chunks, load_document, make_chunks


load_dotenv()
st.set_page_config(page_title="CV RAG Lab", page_icon="📄", layout="wide")

st.title("CV RAG Lab")
st.caption("ทดลอง RAG กับ CV ของคุณ: อัปโหลด → แบ่งข้อความ → ค้นส่วนที่เกี่ยวข้อง → ตอบพร้อมอ้างอิง")

with st.sidebar:
    st.header("เริ่มต้น")
    uploaded = st.file_uploader("อัปโหลด CV", type=["pdf", "docx"])
    st.caption("รองรับ PDF ที่มีข้อความและ DOCX ขนาดไม่เกิน 10 MB")
    st.divider()
    st.markdown("**ขั้นตอนที่ทดลอง**")
    st.markdown("1. อ่านข้อความจาก CV\n2. แบ่งเป็นช่วงสั้น ๆ\n3. สร้าง embedding และค้นสูงสุด 3 ช่วงที่เกี่ยวข้อง\n4. ส่งช่วงเหล่านั้นให้ GPT-6 Luna ตอบ")
    st.caption("ข้อมูล CV และ embedding อยู่ในหน่วยความจำของ session นี้")

if uploaded is None:
    for key in ("document_id", "blocks", "chunks", "index", "upload_usage", "messages"):
        st.session_state.pop(key, None)
    st.info("อัปโหลด CV เพื่อเริ่มทดลอง")
    st.stop()

data = uploaded.getvalue()
if len(data) > 10 * 1024 * 1024:
    st.error("CV ต้องมีขนาดไม่เกิน 10 MB")
    st.stop()

document_id = hashlib.sha256(uploaded.name.encode("utf-8") + data).hexdigest()
if st.session_state.get("document_id") != document_id:
    for key in ("document_id", "blocks", "chunks", "index", "upload_usage", "messages"):
        st.session_state.pop(key, None)
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("กรุณาตั้งค่า OPENAI_API_KEY ใน .env หรือ environment ก่อนอัปโหลด CV")
        st.stop()
    try:
        with st.spinner("กำลังอ่าน CV และสร้าง embedding..."):
            blocks = load_document(uploaded.name, data)
            chunks = make_chunks(blocks)
            upload_usage = TokenUsage()
            index = index_chunks(OpenAI(), chunks, usage=upload_usage)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    except OpenAIError as error:
        st.error("OpenAI API error: {}".format(error))
        st.stop()
    except Exception as error:
        st.error("อ่าน CV ไม่สำเร็จ: {}".format(type(error).__name__))
        st.stop()
    st.session_state.update(
        document_id=document_id,
        blocks=blocks,
        chunks=chunks,
        index=index,
        upload_usage=upload_usage,
        messages=[],
    )

st.success("พร้อมถามคำถามจาก {}".format(uploaded.name))
count_a, count_b, count_c = st.columns(3)
count_a.metric("ตำแหน่งข้อความที่อ่านได้", len(st.session_state.blocks))
count_b.metric("ช่วงข้อความที่สร้าง embedding", len(st.session_state.chunks))
count_c.metric("ขนาด embedding", len(st.session_state.index[0].embedding))
upload_usage = st.session_state.get("upload_usage")
if upload_usage is not None:
    st.caption(
        "สร้าง index: embedding {} tokens · ค่า API โดยประมาณ ${:.6f}".format(
            upload_usage.embedding_tokens, upload_usage.estimated_usd
        )
    )
else:
    st.caption("อัปโหลด CV อีกครั้งเพื่อดูจำนวน tokens ตอนสร้าง index")
st.caption("ค่าใช้จ่ายประมาณจากราคา Standard วันที่ 29 ก.ย. 2026; ยอดจริงให้ดูที่ OpenAI Usage")

with st.expander("ดูข้อความทั้งหมดที่อ่านจาก CV"):
    st.caption("ถ้าข้อมูลที่มองเห็นใน PDF ไม่ปรากฏตรงนี้ ระบบจะค้นข้อมูลนั้นไม่ได้")
    for block in st.session_state.blocks:
        st.markdown("**{}**".format(block.source))
        st.text(block.text)

with st.expander("ดูตัวอย่าง chunk และ embedding"):
    for item in st.session_state.index[:3]:
        st.markdown("**Chunk {} · {}**".format(item.chunk.id, item.chunk.source))
        st.write(item.chunk.text)
        st.caption("5 ค่าแรกของ embedding: {}".format(
            ", ".join("{:.4f}".format(value) for value in item.embedding[:5])
        ))

st.subheader("ถามเกี่ยวกับ CV")
st.caption("แต่ละคำถามค้น CV ใหม่โดยอิสระ; คะแนน similarity แสดงความใกล้ของ embedding ไม่ใช่ความมั่นใจของคำตอบ")

for message in st.session_state.messages:
    with st.chat_message("user"):
        st.write(message["question"])
    with st.chat_message("assistant"):
        st.write(message["answer"])
        usage = message.get("usage")
        if usage is not None:
            st.caption(
                "ใช้ API: embedding {} tokens · LLM รับ {} tokens · LLM สร้าง {} tokens "
                "(รวม reasoning) · ค่า API โดยประมาณ ${:.6f}".format(
                    usage.embedding_tokens,
                    usage.llm_input_tokens,
                    usage.llm_output_tokens,
                    usage.estimated_usd,
                )
            )
        with st.expander("ดูข้อความที่ RAG ค้นมา"):
            for number, hit in enumerate(message["hits"], start=1):
                reason = {
                    "phone pattern": " · พบรูปแบบเบอร์โทร",
                    "email pattern": " · พบรูปแบบอีเมล",
                    "keyword match": " · พบคำตรง",
                }.get(hit.match_reason, "")
                st.markdown("**[{}] {} · similarity {:.3f}{}**".format(
                    number, hit.chunk.source, hit.score, reason
                ))
                st.write(hit.chunk.text)

question = st.chat_input("เช่น ฉันมีประสบการณ์ด้าน Python อะไรบ้าง?")
if question:
    try:
        with st.spinner("กำลังค้น CV และสร้างคำตอบ..."):
            usage = TokenUsage()
            answer, hits = answer_question(
                OpenAI(), st.session_state.index, question, usage=usage
            )
    except (ValueError, OpenAIError) as error:
        st.error("ตอบคำถามไม่สำเร็จ: {}".format(error))
    else:
        st.session_state.messages.append(
            {"question": question, "answer": answer, "hits": hits, "usage": usage}
        )
        st.rerun()
