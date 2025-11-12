import os, sys, time, random
from pathlib import Path
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP
import json, re
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


# Initialize FastMCP server
#mcp = FastMCP("research", host = "localhost", port=50001)
mcp = FastMCP("research", host="0.0.0.0", port=50001)

# —— 强制走 Gemini API（避免误走 Vertex）——
for k in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
    os.environ.pop(k, None)

# ★ 将你的 Gemini API Key 填在这里（测试期可硬编码；生产请用环境变量/密钥管理）
API_KEY = "AIzaSyDN3eK8LaEPQYBc6MCAzYhZyrmT7yAmQxs"

MODEL_ID    = "gemini-2.5-pro"
PDF_PATH    = Path("homework.pdf")              # 文件 1：学生作业影像
REF_MD_PATH = Path("习题与解答_selected.md")     # 文件 2：教师用书（Markdown）
OUT_MD      = Path("作业_OCR与批改报告.md")
DEFAULT_MD = "邹思瑞-42404455-作业_OCR与批改报告.md"

# —— 系统指令：作为普通文本放进 contents（更通用，避免 systemInstruction 字段差异）——
SYSTEM_TEXT_hmrg = (
    "【系统指令】\n"
    "你将收到两个文件：\n"
    "1) homework.pdf（学生作业影像，PDF）；\n"
    "2) 习题与解答_selected.md（教师用书：题目与参考答案，Markdown）。\n\n"
    "请输出一个 Markdown 报告，包含两部分：\n"
    "## 一、学生作业 OCR 结果\n"
    "逐字逐行转写 PDF 内容，尽量保持结构与可读性（标题/段落/列表/图片占位/公式均保留；"
    "数学公式用 $...$ 或 $$...$$）。不要加入解释或评语。\n\n"
    "## 二、逐题批改简报\n"
    "按题号列出：完成情况（已作答/未作答）、判断（正确/部分正确/错误）、2~4条评分要点（可给建议分值）、"
    "常见错误/漏步、给学生的简短建议。仅引用题号或关键词，严禁复写或改写教师用书中的原题与答案文字。\n"
)
SYSTEM_TEXT_qb = (
    "【系统指令】\n"
    "你将收到两个文件：\n"
    "1) homework.pdf（学生作业影像，PDF）；\n"
    "2) 习题与解答_selected.md（教师用书：题目与参考答案，Markdown）。\n\n"
    "请输出一个 Markdown 报告，包含两部分：\n"
    "## 一、学生作业 OCR 结果\n"
    "逐字逐行转写 PDF 内容，尽量保持结构与可读性（标题/段落/列表/图片占位/公式均保留；"
    "数学公式用 $...$ 或 $$...$$）。不要加入解释或评语。\n\n"
    "## 二、逐题批改简报\n"
    "按题号列出：完成情况（已作答/未作答）、判断（正确/部分正确/错误）、2~4条评分要点（可给建议分值）、"
    "常见错误/漏步、给学生的简短建议。仅引用题号或关键词，严禁复写或改写教师用书中的原题与答案文字。\n"
)
SYSTEM_TEXT_rtj = (
    "你将收到一份 Markdown 报告，第二部分是“逐题批改简报”。\n"
    "请从该部分中为每一题抽取：章节（如“§2.5”）、题号（原样，如“T6”）、正确性标签。\n"
    "正确性标签只允许四选一：『正确』『过程部分正确』『答案正确结果错误』『错误』。\n"
    "若出现“未作答”或等价表述，请标为『错误』；若无法定位章节，章节置空字符串。\n"
    "### 输出要求（严格 JSON）\n"
    "{\n"
    '  "questions": [\n'
    '    {"section": "§2.5", "id": "T6", "status": "正确"}\n'
    "  ]\n"
    "}\n"
)

@mcp.tool()
def homework_result_generate():
    # 基础校验
    if not PDF_PATH.exists() or PDF_PATH.suffix.lower() != ".pdf":
        print(f"未找到 PDF：{PDF_PATH}（或不是 .pdf）"); sys.exit(1)
    if not REF_MD_PATH.exists() or REF_MD_PATH.suffix.lower() not in (".md", ".markdown"):
        print(f"未找到 Markdown：{REF_MD_PATH}（需 .md/.markdown）"); sys.exit(1)

    # v1 稳定端点 + 10 分钟超时（毫秒）——更稳的网络层设置
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(api_version="v1", timeout=600_000),
    )  # 生成接口：models.generate_content :contentReference[oaicite:2]{index=2}

    # —— 两个“文件 part” ——（一次请求可同时传多文件 + 文本）
    pdf_part = types.Part.from_bytes(              # PDF 文件（视觉+文本理解）
        data=PDF_PATH.read_bytes(),
        mime_type="application/pdf",
    )
    md_part = types.Part.from_bytes(               # Markdown 文件（按文本理解）
        data=REF_MD_PATH.read_bytes(),
        mime_type="text/markdown",
    )
    # 说明：非 PDF（如 Markdown/TXT/HTML）会作为“文本内容”理解，适合承载参考答案；PDF 则走文档视觉能力。:contentReference[oaicite:3]{index=3}

    # 轻量重试（仅对短暂性错误）：503/UNAVAILABLE/overloaded/Server disconnected
    def generate_once():
        return client.models.generate_content(
            model=MODEL_ID,
            contents=[SYSTEM_TEXT_hmrg, pdf_part, md_part],  # ← 本次真正上传 2 个“文件 part”
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=16000,  # 适度降低上限以减小过载概率
            ),
        )

    resp = None
    for attempt in range(5):  # 最多 5 次：1s→2s→4s→8s→16s
        try:
            resp = generate_once()
            break
        except Exception as e:
            msg = str(e)
            transient = (
                "503" in msg
                or "UNAVAILABLE" in msg.upper()
                or "overloaded" in msg.lower()
                or "Server disconnected without sending a response" in msg
            )
            if transient and attempt < 4:
                time.sleep((2 ** attempt) + random.random())
                continue
            raise

    md = (resp.text or "").strip() if resp else ""
    if not md:
        print("模型未返回内容，请检查 2 个文件是否正常。"); sys.exit(2)

    OUT_MD.write_text(md, encoding="utf-8")
    print(f"已生成：{OUT_MD}")


def parse_name_id_from_filename(path: Path):
    stem = path.stem
    parts = stem.split("-")
    name = parts[0].strip() if len(parts) >= 1 else ""
    sid  = parts[1].strip() if len(parts) >= 2 else ""
    sid = re.sub(r"[^0-9A-Za-z]", "", sid)
    return name, sid

def load_one_json(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    name = obj.get("student_name") or parse_name_id_from_filename(path)[0]
    sid  = obj.get("student_id")   or parse_name_id_from_filename(path)[1]
    total = obj.get("total_questions", 0)
    counts = obj.get("counts", {}) or {}
    correct = counts.get("correct", 0)
    partial = counts.get("partial", 0)
    result_wrong = counts.get("result_wrong", 0)
    wrong = counts.get("wrong", 0)
    grade = obj.get("grade", "")

    # 兼容两种结构：新（key+status）和旧（section+id+status）
    qmap: Dict[str, str] = {}
    for q in obj.get("questions", []):
        if "key" in q and q["key"]:
            key = str(q["key"]).strip()
        else:
            raw_id = str(q.get("id", "")).strip()
            qid = re.sub(r"[^0-9A-Za-z\.\-]", "", raw_id) or raw_id
            section_raw = str(q.get("section", "")).strip()
            sec_num = re.sub(r"[^0-9\.]", "", section_raw)
            section = f"§{sec_num}" if sec_num else ""
            key = f"{section} {qid}".strip()
        status = str(q.get("status", "")).strip()
        if key:
            qmap[key] = status  # 同一 key 只保留最后一次

    row = {
        "student_name": name,
        "student_id": sid,
        "total_questions": total,
        "correct": correct,
        "partial": partial,
        "result_wrong": result_wrong,
        "wrong": wrong,
        "grade": grade,  # 新增 grade 列
    }
    # 逐题展开列，列名为 Q:§2.5 T6
    for key, status in qmap.items():
        row[f"Q:{key}"] = status
    return row

def collect_rows(in_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for p in sorted(in_dir.glob("*.json")):
        try:
            rows.append(load_one_json(p))
        except Exception as e:
            print(f"[跳过] 无法解析 {p.name}: {e}")
    return rows

def get_all_qcols(rows: List[Dict]) -> List[str]:
    qcols = set()
    for r in rows:
        for k in r.keys():
            if k.startswith("Q:"):
                qcols.add(k)
    # 直接按字符串排序（如需按章节数字/题号数值排序，可自行扩展解析器）
    return sorted(qcols, key=lambda c: c[2:])

@mcp.tool()
def json_to_excel():
    in_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(".")
    out_xlsx = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("作业分析汇总.xlsx")
    if not in_dir.exists():
        print(f"输入目录不存在：{in_dir}"); sys.exit(1)

    rows = collect_rows(in_dir)
    if not rows:
        print("未找到任何 JSON。"); sys.exit(2)

    base_cols = [
        "student_name","student_id","total_questions",
        "correct","partial","result_wrong","wrong","grade"  # 包含 grade
    ]
    qcols = get_all_qcols(rows)

    for r in rows:
        for c in qcols:
            r.setdefault(c, "")

    df_wide = pd.DataFrame(rows, columns=base_cols + qcols)

    long_records = []
    for r in rows:
        for qc in qcols:
            long_records.append({
                "student_name": r["student_name"],
                "student_id": r["student_id"],
                "key": qc[2:],            # 去掉 'Q:' 前缀 → '§2.5 T6'
                "status": r.get(qc, ""),
                "grade": r.get("grade","")
            })
    df_long = pd.DataFrame(long_records)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_wide.to_excel(writer, sheet_name="汇总(宽表)", index=False)
        df_long.to_excel(writer, sheet_name="明细(长表)", index=False)

    print(f"已生成：{out_xlsx.resolve()}")

@mcp.tool()
def question_abstract():
    # 基础校验
    if not PDF_PATH.exists() or PDF_PATH.suffix.lower() != ".pdf":
        print(f"未找到 PDF：{PDF_PATH}（或不是 .pdf）"); sys.exit(1)
    if not REF_MD_PATH.exists() or REF_MD_PATH.suffix.lower() not in (".md", ".markdown"):
        print(f"未找到 Markdown：{REF_MD_PATH}（需 .md/.markdown）"); sys.exit(1)

    # v1 稳定端点 + 10 分钟超时（毫秒）——更稳的网络层设置
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(api_version="v1", timeout=600_000),
    )  # 生成接口：models.generate_content :contentReference[oaicite:2]{index=2}

    # —— 两个“文件 part” ——（一次请求可同时传多文件 + 文本）
    pdf_part = types.Part.from_bytes(              # PDF 文件（视觉+文本理解）
        data=PDF_PATH.read_bytes(),
        mime_type="application/pdf",
    )
    md_part = types.Part.from_bytes(               # Markdown 文件（按文本理解）
        data=REF_MD_PATH.read_bytes(),
        mime_type="text/markdown",
    )
    # 说明：非 PDF（如 Markdown/TXT/HTML）会作为“文本内容”理解，适合承载参考答案；PDF 则走文档视觉能力。:contentReference[oaicite:3]{index=3}

    # 轻量重试（仅对短暂性错误）：503/UNAVAILABLE/overloaded/Server disconnected
    def generate_once():
        return client.models.generate_content(
            model=MODEL_ID,
            contents=[SYSTEM_TEXT_qb, pdf_part, md_part],  # ← 本次真正上传 2 个“文件 part”
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=16000,  # 适度降低上限以减小过载概率
            ),
        )

    resp = None
    for attempt in range(5):  # 最多 5 次：1s→2s→4s→8s→16s
        try:
            resp = generate_once()
            break
        except Exception as e:
            msg = str(e)
            transient = (
                "503" in msg
                or "UNAVAILABLE" in msg.upper()
                or "overloaded" in msg.lower()
                or "Server disconnected without sending a response" in msg
            )
            if transient and attempt < 4:
                time.sleep((2 ** attempt) + random.random())
                continue
            raise

    md = (resp.text or "").strip() if resp else ""
    if not md:
        print("模型未返回内容，请检查 2 个文件是否正常。"); sys.exit(2)

    OUT_MD.write_text(md, encoding="utf-8")
    print(f"已生成：{OUT_MD}")


def parse_name_id_from_filename(path: Path):
    stem = path.stem
    parts = stem.split("-")
    if len(parts) >= 2:
        name = parts[0].strip()
        sid  = re.sub(r"[^\dA-Za-z]", "", parts[1])
        return name, sid
    m = re.search(r"([0-9A-Za-z]{6,})", stem)
    sid = m.group(1) if m else ""
    return stem, sid

def normalize_status(s: str) -> str:
    s = s.strip().lower()
    mapping = {
        "正确": "正确", "完全正确": "正确", "对": "正确",
        "部分正确": "过程部分正确", "过程部分正确": "过程部分正确",
        "步骤部分正确": "过程部分正确", "思路正确但有疏漏": "过程部分正确",
        "答案正确结果错误": "答案正确结果错误",
        "结果错误": "答案正确结果错误", "计算错误": "答案正确结果错误",
        "错误": "错误", "完全错误": "错误", "未作答": "错误", "空白": "错误",
    }
    for key in list(mapping.keys()):
        if key.lower() in s:
            return mapping[key]
    return "错误"

def grade_from_statuses(statuses: List[str]) -> str:
    steps = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
    full_errors = sum(1 for s in statuses if s in ("错误", "答案正确结果错误"))
    partial_errors = sum(1 for s in statuses if s == "过程部分正确")
    if full_errors == 0:
        if partial_errors <= 1:
            idx = 0
        else:
            idx = min(1 + (partial_errors - 2), len(steps) - 1)
    else:
        idx = min(full_errors, len(steps) - 1)
    return steps[idx]

def extract_questions_with_genai(md_text: str) -> List[Dict[str, Any]]:
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(api_version="v1", timeout=300_000),
    )
    def _once():
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=[SYSTEM_TEXT_rtj, md_text],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=8192),
        )
        return (resp.text or "").strip()

    text = ""
    for attempt in range(4):
        try:
            text = _once()
            break
        except Exception as e:
            msg = str(e)
            transient = "503" in msg or "UNAVAILABLE" in msg.upper() \
                        or "Server disconnected without sending a response" in msg
            if transient and attempt < 3:
                time.sleep((2 ** attempt) + random.random())
                continue
            raise

    if not text:
        return []

    try:
        return json.loads(text).get("questions", [])
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return []
        return json.loads(m.group(0)).get("questions", [])

@mcp.tool()
def report_to_json():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_MD)
    if not in_path.exists():
        print(f"未找到报告：{in_path}"); sys.exit(1)

    student_name, student_id = parse_name_id_from_filename(in_path)
    md_text = in_path.read_text(encoding="utf-8")

    raw = extract_questions_with_genai(md_text)

    # 规范化并合并为 key（例如 "§2.5 T6"）
    questions = []
    for q in raw:
        raw_id = str(q.get("id", "")).strip()
        qid = re.sub(r"[^0-9A-Za-z\.\-]", "", raw_id) or raw_id  # 保留 T 前缀
        section_raw = str(q.get("section", "")).strip()
        sec_num = re.sub(r"[^0-9\.]", "", section_raw)
        section = f"§{sec_num}" if sec_num else ""
        key = f"{section} {qid}".strip()  # 若无章节则仅用题号
        status = normalize_status(str(q.get("status", "")))
        if qid:
            questions.append({"key": key, "status": status})

    statuses = [q["status"] for q in questions]
    counts = {
        "correct": sum(1 for s in statuses if s == "正确"),
        "partial": sum(1 for s in statuses if s == "过程部分正确"),
        "result_wrong": sum(1 for s in statuses if s == "答案正确结果错误"),
        "wrong": sum(1 for s in statuses if s == "错误"),
    }
    grade = grade_from_statuses(statuses)

    result = {
        "student_name": student_name,
        "student_id": student_id,
        "total_questions": len(questions),
        "counts": counts,
        "grade": grade,
        "questions": questions,  # [{"key":"§2.5 T6","status":"正确"}, ...]
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    out_path = in_path.with_suffix(".json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # Initialize and run the server
    #mcp.run(transport='streamable-http')
    mcp.run(transport='sse')