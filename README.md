# CV RAG Lab

POC สำหรับเรียนรู้ RAG ด้วย CV ของตัวเอง รองรับ PDF ที่ดึงข้อความได้และ DOCX รวมข้อความในตาราง ใช้ `text-embedding-3-small` สำหรับค้นข้อความที่เกี่ยวข้อง และ `gpt-6-luna` สำหรับตอบคำถาม

## เริ่มใช้งาน

ต้องมี Python 3.9 ขึ้นไป และ OpenAI API key ที่เรียกโมเดลข้างต้นได้

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

ใส่ API key ใน `.env` เป็น `OPENAI_API_KEY=...` แล้วรัน:

```bash
streamlit run app.py
```

เปิด URL ที่ Streamlit แสดงใน terminal แล้วอัปโหลด CV เพื่อเริ่มถามคำถาม แอปตั้งค่าให้รับการเชื่อมต่อจากเครื่องนี้เท่านั้น ห้าม commit `.env` หรือไฟล์ CV; `.gitignore` ได้กันไฟล์เหล่านี้ไว้แล้ว

## การทำงาน

1. อ่านข้อความจาก PDF/DOCX และเก็บตำแหน่งหน้า/ย่อหน้า/ตาราง
2. แบ่งข้อความเป็นช่วงที่ซ้อนทับกัน แล้วสร้าง embedding ของแต่ละช่วงครั้งเดียวต่อการอัปโหลด
3. สร้าง embedding ของคำถาม ค้นด้วย cosine similarity แล้วช่วยจัดอันดับด้วยคำภาษาอังกฤษ/ตัวเลขที่ตรงกัน; คำถามเรื่องเบอร์โทรหรืออีเมลจะเลือกช่วงที่พบข้อมูลนั้นโดยตรง
4. ส่งคำถามกับข้อความต้นฉบับสูงสุด 3 ช่วงให้ `gpt-6-luna` ตอบ และแสดงข้อความต้นทางเพื่อตรวจสอบ

การตรวจคำตรง เบอร์โทร และอีเมลทำในเครื่อง ไม่เรียก LLM เพิ่ม แอปแสดงจำนวน tokens ที่ API รายงานสำหรับการสร้าง index และแต่ละคำตอบ พร้อมค่าใช้จ่ายประมาณด้วย [ราคา GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) และ [ราคา text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small) แบบ Standard ณ วันที่ 29 ก.ย. 2026 (รวมอัตรา cached input และ cache write) ราคาอาจเปลี่ยนได้ ยอดเรียกเก็บจริงให้ดูที่ OpenAI Usage

Embedding และประวัติคำถามอยู่ในหน่วยความจำของ Streamlit session; ปิด session แล้วต้องอัปโหลดและสร้าง embedding ใหม่ ไม่มี vector database ใน POC นี้ PDF ที่เป็นภาพสแกนล้วนต้องผ่าน OCR ก่อนจึงจะอ่านได้ หากมีข้อความที่เห็นใน PDF แต่ไม่มีในแผง “ดูข้อความทั้งหมดที่อ่านจาก CV” การค้นจะหาไม่เจอ DOCX ใน text box อาจไม่ถูกดึงข้อความออกมา

## ทดสอบ

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

ชุดทดสอบใช้ข้อมูลตัวอย่าง ไม่ใช้ CV จริง การทดสอบกับ OpenAI API ต้องมี key และมีค่าใช้จ่ายตามการใช้งาน
