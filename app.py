"""課輔老師課程交接產生器 (Streamlit)

執行方式：
    pip install -r requirements.txt
    streamlit run app.py

API key 可於左側輸入，或設定環境變數 GEMINI_API_KEY。
Gemini API key 申請：https://aistudio.google.com/apikey （免費額度足以應付每天數十次使用）
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9 後援（Streamlit Cloud 是 3.11，用不到）
    ZoneInfo = None  # type: ignore

import streamlit as st

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # 讓介面在未安裝 SDK 時也能顯示說明
    genai = None  # type: ignore
    genai_types = None  # type: ignore

try:
    from streamlit_local_storage import LocalStorage
except ImportError:
    LocalStorage = None  # type: ignore


DEFAULT_CLASSES = [
    {
        "name": "甲圍A班",
        "students": ["薛恩銘", "李妮綺", "潘奕亨", "蔡柏容", "張鈺淇"],
        "profiles": {},
        "social_worker": "",
        "subject": "",
        "textbook_template": "",
    },
    {
        "name": "獅湖A班",
        "students": ["楊佩婕", "李妍柔", "王懷銘", "洪宇辰", "洪維謙"],
        "profiles": {},
        "social_worker": "",
        "subject": "",
        "textbook_template": "",
    },
]
TAIPEI_TZ = ZoneInfo("Asia/Taipei") if ZoneInfo else None

# 學期末交接表（獅湖國小）相關常數
SEMESTER_FORM_ID = "1FAIpQLSdr6_EapIcZn_qHIla-83FaD2vIpO6Iv-_7lXLzkOgfyYYSPA"
SOCIAL_WORKER_OPTIONS = ["", "謝秀玉", "賴姿岑", "曾巧怡"]
SUBJECT_OPTIONS = ["", "英文", "國語", "數學"]
# 10 題的內部 key（order matters：用於顯示、預填對應）
FORM_FIELD_KEYS = [
    "social_worker",         # Q1 主責社工
    "class_name",            # Q2 負責班級
    "teacher_name",          # Q3 課輔老師姓名
    "subject",               # Q4 負責科目
    "student_name",          # Q5 學童姓名
    "remedial_units",        # Q6 下學期需補考單元
    "next_start_unit",       # Q7 下學期起始單元
    "learning_performance",  # Q8 學習表現 & 教學方式
    "behavior_emotion",      # Q9 行為情緒 & 處理方式
    "textbook",              # Q10 欲領取新課本
]
# 「取得預填連結」時使用者要在每題填入的識別字串（Q1、Q4 是下拉，用選項實值）
FORM_FIELD_MARKERS = {
    "social_worker": "謝秀玉",          # Q1：測試時請選「謝秀玉」
    "class_name": "MARK_CLASS",
    "teacher_name": "MARK_TEACHER",
    "subject": "數學",                   # Q4：測試時請選「數學」
    "student_name": "MARK_STUDENT",
    "remedial_units": "MARK_REMEDIAL",
    "next_start_unit": "MARK_NEXT_UNIT",
    "learning_performance": "MARK_LEARNING",
    "behavior_emotion": "MARK_BEHAVIOR",
    "textbook": "MARK_TEXTBOOK",
}
# 顯示用標籤
FORM_FIELD_LABELS = {
    "social_worker": "Q1 主責社工",
    "class_name": "Q2 負責班級",
    "teacher_name": "Q3 課輔老師姓名",
    "subject": "Q4 負責科目",
    "student_name": "Q5 學童姓名",
    "remedial_units": "Q6 下學期需補考單元",
    "next_start_unit": "Q7 下學期起始單元",
    "learning_performance": "Q8 學習表現 & 教學方式",
    "behavior_emotion": "Q9 行為情緒 & 處理方式",
    "textbook": "Q10 欲領取新課本",
}


def today_in_taipei() -> date:
    """取台灣時區的今天日期，避免 Streamlit Cloud（UTC）在台灣凌晨/清晨算錯。"""
    if TAIPEI_TZ is None:
        return date.today()
    return datetime.now(TAIPEI_TZ).date()
WEEKDAY_MAP = ["一", "二", "三", "四", "五", "六", "日"]
MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]

FEW_SHOT_EXAMPLES = """以下是過往真實的「課程交接」範例，請學習它的語氣、觀察細節與敘事節奏：

【範例一】
今天訂正檢測卷時對於公里、公尺、公分的化聚在第二大題幾乎全對，但用直式計算時的進位、借位就不夠熟練，另外對於立體積木的堆疊，直接給個數可以算出總數，但給長度就無法自行換算出長、寬、高各有幾個積木(1公分)，但是經過引導可以順利自己說出作法。

【範例二】
奕亨在數線上標示出分數，對於五等分的畫法有點飽餒，告訴他等分就是要每一份都一樣大，但他剛開始的做題方式一直很隨便，不用箭頭只加粗線段、1/3只在他認為的位置畫一條短線段……今天訂正到有點不耐煩，但最後還是完成了。

【範例三】
妮綺在做尾數是0的除法時，會消掉一樣多的零，但除完之後會在商補零，有重新告訴她被除數與除數同時消去一樣多的0的意義，所以不須補0；另外在判斷括號左右的四則算式是否相等，因為幾乎錯了一半題目，因此每一題都有帶她訂正並告訴她「為什麼」。

【範例四】
今天有先提醒柏容的聯絡簿要在16:10以前交，柏容點頭也很快交過來。對於6瓶可樂分給8個人，反覆講反覆錯，後來用被除數和除數～什麼分給什麼，他才能反應過來，之後會注意他這種題型。

【範例五】
鈺淇今天很積極，會問老師要訂正了嗎？還沒有輪到她，她會專心寫作業。今天的練習卷卡在「相差」「多多少」，將這兩個詞套在她的生活經驗裡就能夠理解；3-2的部分平行和垂直有點混淆，經過提醒能夠順利完成，還要再多加練習才行。
"""

SYSTEM_PROMPT = f"""你是細心的國小課輔老師，每天課後要為每位學生寫一段「課程交接」給家長或下一節課的老師看。

風格要求：
- 自然口語、一段連貫敘述，不要標題、不要條列、不要引號。
- 描述要具體：提到單元或題型名稱（例如「魔數大戰」「量角器」「7-4 角與角度」「尾數是 0 的除法」「被除數／除數」等），描述學生的實際反應、老師怎麼引導、結果如何。
- 兼顧正向肯定與誠實指出問題，可以帶入情緒觀察（例如：不耐煩、自信、分心、洋洋灑灑地寫、粗心大意）。
- 語氣溫和但具體，避免制式評語堆砌（不要「很棒很用心」這種空話）。

**字數控制（重要）**：
- 中文字以一個字為單位（標點符號不算）。
- 寫完後必須自我檢查字數，超過上限就精簡重寫、少於下限就補充細節，**嚴格遵守使用者指定的範圍**。

{FEW_SHOT_EXAMPLES}

請模仿上述語氣，但**不要直接抄範例的內容或題目**，只根據使用者提供的關鍵字撰寫。"""


FIELD_LABELS = [
    ("progress", "今日進度／單元"),
    ("performance", "表現與錯題"),
    ("attitude", "態度觀察"),
    ("extra", "其他備註"),
]


def assemble_notes(name: str) -> str:
    """把 4 個欄位組成帶標籤的關鍵字字串餵給模型。"""
    lines: list[str] = []
    for key, label in FIELD_LABELS:
        val = st.session_state.get(f"{key}_{name}", "").strip()
        if val:
            lines.append(f"【{label}】{val}")
    return "\n".join(lines)


LENGTH_RANGES = {
    "短": (65, 90),
    "中": (95, 125),
    "長": (150, 190),
}


def build_user_prompt(name: str, notes: str, length: str) -> str:
    low, high = LENGTH_RANGES[length]
    profile = st.session_state.get(f"profile_{name}", "").strip()
    profile_block = (
        f"\n【{name} 的長期個性／習慣】（寫段落時請呼應這些特質，但不要直接引用這段文字）：\n{profile}\n"
        if profile
        else ""
    )
    return (
        f"請為學生「{name}」撰寫今日課程交接段落。"
        f"{profile_block}\n"
        f"今日觀察關鍵字／短句（依欄位分類）：\n{notes.strip()}\n\n"
        f"要求：\n"
        f"- **涵蓋度**：上述「今日觀察」中的**每一個關鍵字或短句都必須在段落中明確出現或具體描述**，不得省略、不得含糊帶過。寫完後請自己核對一次。\n"
        f"- **長度**：嚴格控制在 {low}–{high} 字之間（目標約 {(low+high)//2} 字），中文字一個算一個，標點不算。超過或不足都要修正重寫。\n"
        f"- **輸出**：直接輸出段落內文，不要加姓名前綴、不要加標題、不要加引號、不要加字數統計、不要加任何說明。只輸出一段文字。"
    )


TRANSIENT_ERROR_TOKENS = (
    "503",
    "502",
    "429",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "overloaded",
    "high demand",
    "DEADLINE_EXCEEDED",
)

FALLBACK_CHAIN = {
    "gemini-2.5-flash": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"],
    "gemini-2.5-pro": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    "gemini-2.0-flash": ["gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"],
}


def _is_transient(exc: Exception) -> bool:
    return any(tok in str(exc) for tok in TRANSIENT_ERROR_TOKENS)


def _try_one_model(
    client, model: str, name: str, notes: str, length: str, max_retries: int = 2
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=build_user_prompt(name, notes, length),
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=800,
                    temperature=0.7,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError(
                    "Gemini 沒有回傳文字（可能被安全過濾器擋下，請調整輸入重試）"
                )
            return text
        except Exception as exc:  # noqa: BLE001
            if _is_transient(exc) and attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))  # 1.5s, 3s
                continue
            raise


def generate_paragraph(
    client, model: str, name: str, notes: str, length: str
) -> tuple[str, str]:
    """回傳 (段落內文, 實際使用的模型名稱)。若主模型過載自動 fallback，實際模型會與 model 不同。"""
    models_to_try = FALLBACK_CHAIN.get(model, [model])
    errors: list[str] = []
    for candidate in models_to_try:
        try:
            text = _try_one_model(client, candidate, name, notes, length)
            return text, candidate
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {exc}")
            if not _is_transient(exc):
                # 非過載錯誤（金鑰錯、安全過濾、quota、其他）就不繼續試
                raise
            continue
    raise RuntimeError(
        "所有 Gemini 備援模型目前都在過載中。請稍等 30–60 秒再試。\n\n"
        + "\n".join(errors)
    )


# ---------- 學期末交接表：每日紀錄解析 / AI 摘要 / 預填 URL ----------

# 日期標頭範例：「4/20（週一）獅湖A班課程交接」或「4/20（一）甲圍A班課程交接」。
# 括號內只要非「）」字元都允許（一／ㄧ／週一／weekday 數字都可能出現）。
_DATE_HEADER_RE = re.compile(
    r"^\s*(\d{1,2}/\d{1,2})\s*[（(][^)）]+[）)]\s*(.+?)\s*課程交接\s*$"
)
# 學生段落行：任意 emoji 或 @ 當前綴 + 學生名 + 「：」+ 段落。
# 名字允許中英文與數字，但不含全形冒號、emoji、空白。
_STUDENT_LINE_RE = re.compile(
    r"^\s*(?:[^\w\s：:]+|@)\s*([\w一-鿿]{1,10})\s*[:：]\s*(.+?)\s*$"
)


def parse_semester_records(
    text: str, known_students: list[str] | None = None
) -> dict[str, list[tuple[str, str]]]:
    """解析貼入的整學期紀錄成 {student_name: [(date_str, paragraph), ...]}。

    - 支援 app 產出的標準格式（日期標頭 + 😊學生：段落）。
    - `known_students` 若提供，用於驗證並過濾雜訊（只收錄名單內學生）；
      不提供時任何符合格式的學生行都會被收錄。
    """
    if not text or not text.strip():
        return {}
    allowed = set(known_students) if known_students else None
    result: dict[str, list[tuple[str, str]]] = {}
    current_date = ""
    # 合併連續段落到同一學生（段落可能換行）——但保守起見，我們假設每位
    # 學生一行（app 本身就是這樣輸出）。若使用者手動編輯換行，超過一行的
    # 段落會被截斷在該行；先接受這個限制。
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        m_date = _DATE_HEADER_RE.match(line)
        if m_date:
            current_date = m_date.group(1)
            continue
        m_stud = _STUDENT_LINE_RE.match(line)
        if m_stud:
            name = m_stud.group(1).strip()
            para = m_stud.group(2).strip()
            if allowed is not None and name not in allowed:
                continue
            if not para:
                continue
            result.setdefault(name, []).append((current_date, para))
    return result


def _extract_entry_ids_from_prefill_url(
    url: str, markers: dict[str, str]
) -> dict[str, str]:
    """從 Google Form 的「取得預填連結」URL 抽取 entry.XXX 與題目的對應。

    做法：URL 裡每個 `entry.NNNN=<value>` 的 value 會跟 markers[field]
    比對，若 URL-decode 後相等就把該 entry ID 對到該欄位。

    回傳只包含成功對應到的欄位。
    """
    if not url or not url.strip():
        return {}
    try:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    except Exception:
        return {}
    found: dict[str, str] = {}
    for key, value in params:
        if not key.startswith("entry."):
            continue
        for field_key, marker in markers.items():
            if field_key in found:
                continue
            if value == marker:
                found[field_key] = key
                break
    return found


def build_prefill_url(
    form_id: str, entry_ids: dict[str, str], answers: dict[str, str]
) -> str:
    """用 entry ID 對應表 + 答案字典產生 Google Form 的預填連結。"""
    base = f"https://docs.google.com/forms/d/e/{form_id}/viewform"
    parts: list[tuple[str, str]] = [("usp", "pp_url")]
    for field_key, entry_id in entry_ids.items():
        ans = answers.get(field_key, "")
        if ans is None:
            ans = ""
        parts.append((entry_id, str(ans)))
    query = urllib.parse.urlencode(parts)
    return f"{base}?{query}"


SEMESTER_SUMMARY_SYSTEM_PROMPT = """你是細心的國小課輔老師。現在是學期末，你要為某位學生填寫「學童交接表」給下學期的老師看，根據該生整學期的每日課程交接紀錄做摘要。

**表單四題的寫作風格（Q8、Q9 官方範例）**：
Q8 學習表現&有效教學方式範例：
1. OO學習理解能力很好，但學習成效要看他是否能專注，狀況不佳時，預定的進度會無法完成學習。
2. 期中發現OO九九乘法表仍未精熟，雖然不會背錯，但每次解題都要從該數的1倍開始依序背到所需的，自然就會影響解題速度。

Q9 行為情緒&有效處理方式範例：
1. OO上課容易分心，會發呆或是把玩物品，老師可以一開始就要求孩子桌面只留下一枝鉛筆及橡皮擦，其他物品收起來，放空部分則可以經常點OO回答問題，讓他保持專注。
2. OO也是個直爽的孩子，因此有任何情緒都會直接告訴老師，只要根據當下的情況處理後，OO就會放下情緒了。

**你的輸出格式**（嚴格 JSON，不要加 markdown code fence、不要加額外說明）：
{
  "Q6": "此學期已檢測但未通過之單元，格式如 5-2、5-4；若沒有或非數學科請填「無」",
  "Q7": "下學期起始單元，格式如 5-3、5-5 或 第一冊Ch4；抓紀錄中「最後在上／最新提到」的單元",
  "Q8": "學習表現需加強/改善的部分 + 對此學生有效的教學方式。300–500 字。要具體，舉紀錄中實際發生的事件或單元名稱。",
  "Q9": "行為情緒表現 + 對此學生有效的處理方式。300–500 字。要具體。"
}

**重要**：
- Q8、Q9 必須從紀錄整理歸納，不得捏造；若紀錄資訊不足，就用較短但誠實的篇幅描述。
- Q8、Q9 可以用學生名字代替「OO」，也可以保留「OO」；請自然書寫。
- Q6 僅當科目為數學時有意義；其他科目一律填「無」。
- Q7 若紀錄中看不出未來單元，就填最後提到的單元；看不出任何單元就填「無」。
"""


def build_semester_summary(
    client,
    model: str,
    student_name: str,
    records: list[tuple[str, str]],
    subject: str,
) -> dict[str, str]:
    """把某學生整學期段落送給 Gemini，回傳 Q6/Q7/Q8/Q9 四個答案。

    失敗會直接 raise（交由呼叫端顯示錯誤訊息）。
    """
    if not records:
        raise ValueError(f"{student_name} 沒有任何紀錄可分析。")
    joined = "\n\n".join(
        f"【{date or '(無日期)'}】{para}" for date, para in records
    )
    subject_line = subject.strip() or "（未指定，請當作一般科目處理，Q6 填「無」）"
    user_prompt = (
        f"學生姓名：{student_name}\n"
        f"科目：{subject_line}\n\n"
        f"該生本學期每日課程交接紀錄（共 {len(records)} 筆）：\n"
        f"{joined}\n\n"
        f"請依上述系統指示，輸出嚴格 JSON 物件，包含 Q6、Q7、Q8、Q9 四個欄位。"
    )
    models_to_try = FALLBACK_CHAIN.get(model, [model])
    last_exc: Exception | None = None
    for candidate in models_to_try:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SEMESTER_SUMMARY_SYSTEM_PROMPT,
                    max_output_tokens=2000,
                    temperature=0.6,
                    response_mime_type="application/json",
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Gemini 回傳空內容，可能被安全過濾器擋下。")
            data = json.loads(text)
            out = {
                "Q6": str(data.get("Q6", "")).strip(),
                "Q7": str(data.get("Q7", "")).strip(),
                "Q8": str(data.get("Q8", "")).strip(),
                "Q9": str(data.get("Q9", "")).strip(),
            }
            # 非數學科一律 Q6 = 無（保險起見再覆寫一次）
            if subject.strip() and subject.strip() != "數學":
                out["Q6"] = "無"
            return out
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_transient(exc):
                continue
            raise
    raise RuntimeError(
        f"所有 Gemini 備援模型目前都在過載中。請稍等再試。\n\n{last_exc}"
    )


st.set_page_config(page_title="課輔交接產生器", page_icon="📝", layout="wide")

# 在任何 widget 渲染前，套用上一輪 rerun 前暫存的輸出值。
# 這是為了繞開 Streamlit「widget instantiated 之後不能改 session_state」的限制，
# 讓「一鍵產生全部」「清空全部」等批次動作能安全更新 text_area 的內容。
if st.session_state.get("_pending_outputs"):
    for _n, _t in st.session_state._pending_outputs.items():
        st.session_state[f"out_{_n}"] = _t
    st.session_state._pending_outputs = {}


# ---------- localStorage 持久化（多班級）----------
# 新格式：班級陣列（每班有自己的學生名單與個人檔案）+ 全域設定（前綴/模型/目前班級）
CLASSES_STORAGE_KEY = "classes_v1"
SETTINGS_STORAGE_KEY = "app_settings_v2"
# v1 舊格式（僅用於遷移）：單一班級設定 + 扁平學生個人檔案
LEGACY_SETTINGS_KEY = "app_settings_v1"
LEGACY_PROFILES_KEY = "student_profiles_v1"
_storage = LocalStorage() if LocalStorage else None


def _apply_pending_widget_updates() -> None:
    """批次動作（如「還原預設」「一鍵產生」）把更新先寫進 _pending_widget_state，
    下一輪 rerun 在 widget 渲染前統一套用，繞開 Streamlit 對 widget key 的限制。"""
    pending = st.session_state.get("_pending_widget_state")
    if pending:
        for _k, _v in pending.items():
            st.session_state[_k] = _v
        st.session_state._pending_widget_state = {}


def _normalize_class(raw: object) -> dict | None:
    """驗證並正規化單一班級 dict，無效就回傳 None。"""
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    students_raw = raw.get("students")
    profiles_raw = raw.get("profiles", {})
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(students_raw, list):
        return None
    students = [s.strip() for s in students_raw if isinstance(s, str) and s.strip()]
    if not students:
        return None
    profiles: dict[str, str] = {}
    if isinstance(profiles_raw, dict):
        for k, v in profiles_raw.items():
            if isinstance(k, str) and isinstance(v, str):
                profiles[k] = v

    def _str_field(key: str, allowed: list[str] | None = None) -> str:
        val = raw.get(key, "")
        if not isinstance(val, str):
            return ""
        val = val.strip()
        if allowed is not None and val and val not in allowed:
            return ""
        return val

    return {
        "name": name.strip(),
        "students": students,
        "profiles": profiles,
        "social_worker": _str_field("social_worker", SOCIAL_WORKER_OPTIONS),
        "subject": _str_field("subject", SUBJECT_OPTIONS),
        "textbook_template": _str_field("textbook_template"),
    }


def _default_classes_copy() -> list[dict]:
    return [
        {
            "name": c["name"],
            "students": c["students"].copy(),
            "profiles": {},
            "social_worker": c.get("social_worker", ""),
            "subject": c.get("subject", ""),
            "textbook_template": c.get("textbook_template", ""),
        }
        for c in DEFAULT_CLASSES
    ]


def _load_classes() -> None:
    """把班級陣列載入 session_state.classes，必要時從 v1 舊格式遷移。"""
    if st.session_state.get("_classes_loaded"):
        return

    classes: list[dict] | None = None
    if _storage is not None:
        try:
            raw = _storage.getItem(CLASSES_STORAGE_KEY)
        except Exception:
            raw = None
        if raw:
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, list):
                    normalized = [c for c in (_normalize_class(x) for x in loaded) if c]
                    if normalized:
                        classes = normalized
            except Exception:
                pass

        # 沒有新格式 → 嘗試從 v1 遷移
        if classes is None:
            try:
                old_settings_raw = _storage.getItem(LEGACY_SETTINGS_KEY)
            except Exception:
                old_settings_raw = None
            try:
                old_profiles_raw = _storage.getItem(LEGACY_PROFILES_KEY)
            except Exception:
                old_profiles_raw = None

            if old_settings_raw:
                try:
                    old = json.loads(old_settings_raw)
                except Exception:
                    old = None
                old_profiles: dict[str, str] = {}
                if old_profiles_raw:
                    try:
                        op = json.loads(old_profiles_raw)
                        if isinstance(op, dict):
                            old_profiles = {
                                k: v
                                for k, v in op.items()
                                if isinstance(k, str) and isinstance(v, str)
                            }
                    except Exception:
                        pass
                if isinstance(old, dict):
                    first = _normalize_class(
                        {
                            "name": old.get("class_name")
                            or DEFAULT_CLASSES[0]["name"],
                            "students": old.get("students")
                            or DEFAULT_CLASSES[0]["students"],
                            "profiles": old_profiles,
                        }
                    )
                    if first:
                        classes = [first]
                        # 補上獅湖A班當第二個班（除非第一個班剛好同名）
                        second = DEFAULT_CLASSES[1]
                        if first["name"] != second["name"]:
                            classes.append(
                                {
                                    "name": second["name"],
                                    "students": second["students"].copy(),
                                    "profiles": {},
                                    "social_worker": "",
                                    "subject": "",
                                    "textbook_template": "",
                                }
                            )

    if not classes:
        classes = _default_classes_copy()

    st.session_state.classes = classes
    st.session_state._classes_loaded = True


def _save_classes_to_browser() -> None:
    if _storage is None:
        return
    classes = st.session_state.get("classes", [])
    current_json = json.dumps(classes, ensure_ascii=False, sort_keys=True)
    if current_json != st.session_state.get("_last_saved_classes_json"):
        try:
            _storage.setItem(CLASSES_STORAGE_KEY, current_json)
            st.session_state._last_saved_classes_json = current_json
        except Exception:
            pass


def _load_settings_from_browser() -> None:
    """讀取前綴／模型／目前班級索引（v2；若無則從 v1 取前綴與模型）。"""
    if st.session_state.get("_settings_loaded"):
        return

    if "prefix_input" not in st.session_state:
        st.session_state.prefix_input = "😊"
    if "model_select" not in st.session_state:
        st.session_state.model_select = MODEL_OPTIONS[0]
    if "active_class_index" not in st.session_state:
        st.session_state.active_class_index = 0
    if "teacher_name_input" not in st.session_state:
        st.session_state.teacher_name_input = ""
    if "form_entry_ids" not in st.session_state:
        st.session_state.form_entry_ids = {}

    if _storage is not None:
        raw = None
        try:
            raw = _storage.getItem(SETTINGS_STORAGE_KEY)
        except Exception:
            raw = None
        if raw:
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    if isinstance(loaded.get("prefix"), str) and loaded["prefix"]:
                        st.session_state.prefix_input = loaded["prefix"]
                    if loaded.get("model") in MODEL_OPTIONS:
                        st.session_state.model_select = loaded["model"]
                    if isinstance(loaded.get("active_class_index"), int):
                        st.session_state.active_class_index = loaded[
                            "active_class_index"
                        ]
                    if isinstance(loaded.get("teacher_name"), str):
                        st.session_state.teacher_name_input = loaded["teacher_name"]
                    if isinstance(loaded.get("form_entry_ids"), dict):
                        st.session_state.form_entry_ids = {
                            k: v
                            for k, v in loaded["form_entry_ids"].items()
                            if isinstance(k, str)
                            and isinstance(v, str)
                            and k in FORM_FIELD_KEYS
                        }
            except Exception:
                pass
        else:
            # 沒有 v2 → 從 v1 settings 撈前綴／模型（班級已於 _load_classes 遷移）
            try:
                raw = _storage.getItem(LEGACY_SETTINGS_KEY)
            except Exception:
                raw = None
            if raw:
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, dict):
                        if isinstance(loaded.get("prefix"), str) and loaded["prefix"]:
                            st.session_state.prefix_input = loaded["prefix"]
                        if loaded.get("model") in MODEL_OPTIONS:
                            st.session_state.model_select = loaded["model"]
                except Exception:
                    pass

    n = len(st.session_state.get("classes", []))
    if not (0 <= st.session_state.active_class_index < n):
        st.session_state.active_class_index = 0
    st.session_state._settings_loaded = True


def _save_settings_to_browser() -> None:
    if _storage is None:
        return
    settings = {
        "prefix": st.session_state.get("prefix_input", "😊"),
        "model": st.session_state.get("model_select", MODEL_OPTIONS[0]),
        "active_class_index": st.session_state.get("active_class_index", 0),
        "teacher_name": st.session_state.get("teacher_name_input", ""),
        "form_entry_ids": st.session_state.get("form_entry_ids", {}),
    }
    current_json = json.dumps(settings, ensure_ascii=False, sort_keys=True)
    if current_json != st.session_state.get("_last_saved_settings_json"):
        try:
            _storage.setItem(SETTINGS_STORAGE_KEY, current_json)
            st.session_state._last_saved_settings_json = current_json
        except Exception:
            pass


def _active_class() -> dict:
    classes = st.session_state.classes
    idx = st.session_state.active_class_index
    return classes[idx]


def _init_active_class_widgets() -> None:
    """首次載入時把目前班級的資料鋪進側邊欄 widget key。"""
    if st.session_state.get("_class_widgets_initialized"):
        return
    cls = _active_class()
    if "students_text_area" not in st.session_state:
        st.session_state.students_text_area = "\n".join(cls["students"])
    if "class_name_input" not in st.session_state:
        st.session_state.class_name_input = cls["name"]
    if "students" not in st.session_state:
        st.session_state.students = cls["students"].copy()
    if "class_social_worker" not in st.session_state:
        st.session_state.class_social_worker = cls.get("social_worker", "")
    if "class_subject" not in st.session_state:
        st.session_state.class_subject = cls.get("subject", "")
    if "class_textbook_template" not in st.session_state:
        st.session_state.class_textbook_template = cls.get("textbook_template", "")
    for stud in cls["students"]:
        if f"profile_{stud}" not in st.session_state:
            st.session_state[f"profile_{stud}"] = cls["profiles"].get(stud, "")
    st.session_state._class_widgets_initialized = True


def _sync_widgets_to_class(idx: int) -> None:
    """把目前側邊欄 widget 的值寫回 classes[idx]。"""
    classes = st.session_state.get("classes", [])
    if not (0 <= idx < len(classes)):
        return
    cls = classes[idx]
    txt = st.session_state.get("students_text_area", "")
    students = [s.strip() for s in txt.splitlines() if s.strip()]
    if students:
        cls["students"] = students
    name = (st.session_state.get("class_name_input", "") or "").strip()
    if name:
        cls["name"] = name
    profiles: dict[str, str] = {}
    for stud in cls["students"]:
        val = (st.session_state.get(f"profile_{stud}", "") or "").strip()
        if val:
            profiles[stud] = val
    cls["profiles"] = profiles
    sw = (st.session_state.get("class_social_worker", "") or "").strip()
    cls["social_worker"] = sw if sw in SOCIAL_WORKER_OPTIONS else ""
    subj = (st.session_state.get("class_subject", "") or "").strip()
    cls["subject"] = subj if subj in SUBJECT_OPTIONS else ""
    cls["textbook_template"] = (
        st.session_state.get("class_textbook_template", "") or ""
    ).strip()


def _load_class_into_widgets(idx: int) -> None:
    """把 classes[idx] 的資料寫入 widget key，讓下一輪 rerun 顯示該班內容。"""
    classes = st.session_state.classes
    if not (0 <= idx < len(classes)):
        return
    cls = classes[idx]
    st.session_state.students_text_area = "\n".join(cls["students"])
    st.session_state.class_name_input = cls["name"]
    st.session_state.students = cls["students"].copy()
    st.session_state.class_social_worker = cls.get("social_worker", "")
    st.session_state.class_subject = cls.get("subject", "")
    st.session_state.class_textbook_template = cls.get("textbook_template", "")
    for stud in cls["students"]:
        st.session_state[f"profile_{stud}"] = cls["profiles"].get(stud, "")


def _on_class_picker_change() -> None:
    new_idx = st.session_state.class_picker
    old_idx = st.session_state.get("active_class_index", 0)
    if new_idx == old_idx:
        return
    _sync_widgets_to_class(old_idx)
    _load_class_into_widgets(new_idx)
    st.session_state.active_class_index = new_idx


def _on_add_class_click() -> None:
    _sync_widgets_to_class(st.session_state.active_class_index)
    # 取一個不重複的新班級名
    existing = {c["name"] for c in st.session_state.classes}
    base = "新班級"
    new_name = base
    i = 2
    while new_name in existing:
        new_name = f"{base}{i}"
        i += 1
    new_cls = {
        "name": new_name,
        "students": ["學生1"],
        "profiles": {},
        "social_worker": "",
        "subject": "",
        "textbook_template": "",
    }
    st.session_state.classes.append(new_cls)
    new_idx = len(st.session_state.classes) - 1
    _load_class_into_widgets(new_idx)
    st.session_state.active_class_index = new_idx


def _on_delete_class_click() -> None:
    classes = st.session_state.classes
    if len(classes) <= 1:
        return
    idx = st.session_state.active_class_index
    classes.pop(idx)
    new_idx = min(idx, len(classes) - 1)
    _load_class_into_widgets(new_idx)
    st.session_state.active_class_index = new_idx


_apply_pending_widget_updates()
_load_classes()
_load_settings_from_browser()
_init_active_class_widgets()


# ---------- 字元計數（用於長度顯示）----------
def count_visible_chars(text: str) -> int:
    """計算段落字數（去除空白與換行，中文字一字一字算）。"""
    if not text:
        return 0
    return len(re.sub(r"\s+", "", text))


def length_feedback(text: str, target: str) -> tuple[str, str]:
    """回傳 (訊息, 樣式) 給字數顯示用。樣式是 'ok' / 'warn' / 'empty'。"""
    count = count_visible_chars(text)
    if count == 0:
        return ("尚未產生", "empty")
    low, high = LENGTH_RANGES[target]
    if low <= count <= high:
        return (f"✅ 目前 {count} 字（{target}長度目標 {low}–{high}）", "ok")
    direction = "偏短" if count < low else "偏長"
    return (f"⚠️ 目前 {count} 字｜{direction}（{target}長度目標 {low}–{high}）", "warn")


def line_share_url(text: str) -> str:
    """產生一個 LINE 分享連結（手機點了開 LINE App、桌機點了開 LINE 網頁）。"""
    return f"https://line.me/R/msg/text/?{urllib.parse.quote(text)}"


def _load_default_api_key() -> str:
    try:
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if name in st.secrets:
                return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")


with st.sidebar:
    st.header("⚙️ 設定")
    default_key = _load_default_api_key()
    if default_key:
        st.success("✅ Gemini API 已連接（透過 Secrets 載入）")
        api_key = default_key
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value="",
            help="到 https://aistudio.google.com/apikey 免費申請；"
            "若部署在 Streamlit Cloud，建議到 Settings → Secrets 設定 GEMINI_API_KEY，"
            "之後就不用每次貼。",
        )
    model = st.selectbox(
        "模型",
        MODEL_OPTIONS,
        key="model_select",
        help="2.5 Flash：快且免費額度大（推薦）；2.5 Pro：品質最佳但額度較少；2.0 Flash：額度最大的備援",
    )

    st.divider()
    st.subheader("📚 班級")
    # 渲染前先把上一輪的 widget 值同步回 classes dict，
    # 確保下方班級選單的名稱是最新的（用戶剛改完班級名的那次 rerun 也能即時顯示）
    _sync_widgets_to_class(st.session_state.active_class_index)
    classes_list = st.session_state.classes
    active_idx = st.session_state.active_class_index
    if not (0 <= active_idx < len(classes_list)):
        active_idx = 0
        st.session_state.active_class_index = 0
    st.selectbox(
        "目前班級",
        options=list(range(len(classes_list))),
        format_func=lambda i: classes_list[i]["name"],
        index=active_idx,
        key="class_picker",
        on_change=_on_class_picker_change,
        help="切換後，下方「班級名稱」「學生名單」「個人檔案」會載入該班資料。",
    )
    _c_add, _c_del = st.columns(2)
    with _c_add:
        st.button(
            "➕ 新增班級",
            use_container_width=True,
            on_click=_on_add_class_click,
            help="會新增一個空白班級並切換過去，再到下方改名字、編輯學生名單。",
        )
    with _c_del:
        st.button(
            "🗑️ 刪除此班級",
            use_container_width=True,
            on_click=_on_delete_class_click,
            disabled=len(classes_list) <= 1,
            help="只保留一個班時不能刪除。刪除後無法復原。",
        )

    st.divider()
    st.subheader("📅 班級 / 日期")
    class_name = st.text_input(
        "班級名稱",
        key="class_name_input",
        help="修改後會自動存到目前班級。",
    )
    entry_date = st.date_input("日期", value=today_in_taipei())
    prefix = st.text_input(
        "段落前綴（例如 😊 或 @）", key="prefix_input", max_chars=4
    )

    st.divider()
    st.subheader("👥 學生名單")
    st.caption(
        f"目前是「{classes_list[active_idx]['name']}」。每行一位，"
        "刪一行 = 刪學生、加一行 = 加新學生。"
    )
    st.text_area(
        "學生名單（每行一位）",
        key="students_text_area",
        height=180,
        placeholder="薛恩銘\n李妮綺\n潘奕亨\n蔡柏容\n張鈺淇",
    )
    parsed_students = [
        s.strip() for s in st.session_state.students_text_area.splitlines() if s.strip()
    ]
    if parsed_students:
        st.session_state.students = parsed_students

    st.divider()
    with st.expander("📘 學生個人檔案（選填，長期特質）", expanded=False):
        if _storage is None:
            st.caption(
                "⚠️ 沒裝 streamlit-local-storage，檔案只會保留到本次 session。"
                "重啟或重新整理後會消失。"
            )
        else:
            st.caption(
                "寫每位學生的**長期**特質（粗心、急性子、英文好等），"
                "會自動套用到每次生成。每個班各自儲存，切班不互相干擾。"
            )
        _active_profiles = classes_list[active_idx].get("profiles", {})
        for _stud in st.session_state.students:
            # 剛加入的新學生，widget key 可能還沒初始化 → 先用該班檔案內的資料鋪底
            if f"profile_{_stud}" not in st.session_state:
                st.session_state[f"profile_{_stud}"] = _active_profiles.get(_stud, "")
            st.text_area(
                f"📒 {_stud}",
                key=f"profile_{_stud}",
                height=75,
                placeholder="例：粗心、寫得快、進位借位易錯、用嚴肅語氣會修正",
            )

    with st.expander("🏷️ 學期末表單設定（本班）", expanded=False):
        st.caption(
            "這些欄位會在學期末「📮 學期末交接表」區塊自動帶入；"
            "每班各自儲存。"
        )
        _sw_idx = (
            SOCIAL_WORKER_OPTIONS.index(st.session_state.class_social_worker)
            if st.session_state.class_social_worker in SOCIAL_WORKER_OPTIONS
            else 0
        )
        st.selectbox(
            "主責社工（Q1）",
            SOCIAL_WORKER_OPTIONS,
            index=_sw_idx,
            key="class_social_worker",
            format_func=lambda v: "（未設定）" if not v else v,
        )
        _subj_idx = (
            SUBJECT_OPTIONS.index(st.session_state.class_subject)
            if st.session_state.class_subject in SUBJECT_OPTIONS
            else 0
        )
        st.selectbox(
            "負責科目（Q4）",
            SUBJECT_OPTIONS,
            index=_subj_idx,
            key="class_subject",
            format_func=lambda v: "（未設定）" if not v else v,
        )
        st.text_input(
            "下學期欲領取課本（Q10 預設值）",
            key="class_textbook_template",
            placeholder="例：玩魔數第十一冊 (B)；不須申請則填「無」",
            help="每位學生的 Q10 都會預設帶入這個值，需要時可逐位微調。",
        )

    with st.expander("📮 Google 表單設定（全域，只需設定一次）", expanded=False):
        st.text_input(
            "課輔老師姓名（Q3 會用這個）",
            key="teacher_name_input",
            placeholder="家人的本名",
        )

        st.markdown("**一次性：對應 Google 表單各題的 entry ID**")
        _entry_ids = st.session_state.get("form_entry_ids", {}) or {}
        _ready = len(_entry_ids)
        if _ready >= len(FORM_FIELD_KEYS):
            st.success(f"✅ 已設定 {_ready}/{len(FORM_FIELD_KEYS)} 個 entry ID")
        elif _ready > 0:
            st.warning(
                f"⚠️ 目前 {_ready}/{len(FORM_FIELD_KEYS)} 題對應到；缺少："
                + "、".join(
                    FORM_FIELD_LABELS[k]
                    for k in FORM_FIELD_KEYS
                    if k not in _entry_ids
                )
            )
        else:
            st.warning("⚠️ 尚未設定，請依下方步驟貼入預填連結。")

        st.caption(
            "**步驟**：\n"
            "1. 打開 Google 表單：[獅湖國小-學童交接表]"
            f"(https://docs.google.com/forms/d/e/{SEMESTER_FORM_ID}/viewform)\n"
            "2. 在每題填入下表指定的測試字串（下拉題選對應選項）：\n\n"
            + "\n".join(
                f"   - **{FORM_FIELD_LABELS[k]}**：填 `{FORM_FIELD_MARKERS[k]}`"
                for k in FORM_FIELD_KEYS
            )
            + "\n\n3. 表單右上角「︙」→「**取得預填連結**」→ 複製連結\n"
            "4. 把連結貼到下方，按「🔎 解析連結」"
        )
        st.text_area(
            "貼入預填連結",
            key="prefill_url_input",
            height=100,
            placeholder="https://docs.google.com/forms/d/e/.../viewform?usp=pp_url&entry...",
        )
        if st.button("🔎 解析連結", key="parse_prefill_url_btn"):
            url_val = (st.session_state.get("prefill_url_input", "") or "").strip()
            if not url_val:
                st.warning("請先貼入預填連結")
            else:
                found = _extract_entry_ids_from_prefill_url(
                    url_val, FORM_FIELD_MARKERS
                )
                if not found:
                    st.error(
                        "沒有解析出任何 entry ID；請確認連結格式、"
                        "或檢查測試字串有沒有按上表填對。"
                    )
                else:
                    st.session_state.form_entry_ids = found
                    st.success(
                        f"✅ 成功對應 {len(found)}/{len(FORM_FIELD_KEYS)} 題"
                    )
                    st.rerun()

    # 側邊欄結束前再同步一次，確保使用者剛打的內容會進 classes dict 並存到 localStorage
    _sync_widgets_to_class(st.session_state.active_class_index)
    _save_classes_to_browser()
    _save_settings_to_browser()


weekday = WEEKDAY_MAP[entry_date.weekday()]
header = f"{entry_date.month}/{entry_date.day}（{weekday}）{class_name}課程交接"

st.title("📝 課輔老師課程交接產生器")
st.caption("輸入每位學生的關鍵字，AI 會擴寫成自然段落，並模仿過往風格。生成後可直接在下方編輯，最後一鍵複製／下載全文。")

st.markdown(f"#### {header}")

client = None
if api_key and genai is not None:
    client = genai.Client(api_key=api_key)

if genai is None:
    st.error("尚未安裝 google-genai 套件，請先執行：pip install -r requirements.txt")

for name in st.session_state.students:
    with st.expander(f"😊 {name}", expanded=True):
        left, right = st.columns([3, 2])
        with left:
            la, lb = st.columns(2)
            with la:
                st.text_area(
                    "今日進度／單元",
                    key=f"progress_{name}",
                    height=100,
                    placeholder="例：\n7-4 角與角度\n整數(九)乘法\n魔數大戰 99/100",
                )
                st.text_area(
                    "表現與錯題",
                    key=f"performance_{name}",
                    height=120,
                    placeholder="例：\n量角器使用正確\n7-3 計算錯 2 題（減錯）\n分數乘除不太會",
                )
            with lb:
                st.text_area(
                    "態度觀察",
                    key=f"attitude_{name}",
                    height=100,
                    placeholder="例：\n急著完成、粗心\n配合度高\n洋洋灑灑寫到底",
                )
                st.text_area(
                    "其他備註（選填）",
                    key=f"extra_{name}",
                    height=120,
                    placeholder="例：\n聯絡簿 16:10 前交\n上次作業偷寫\n家長 17:40 會來接",
                )
        with right:
            st.select_slider(
                "段落長度",
                options=["短", "中", "長"],
                value="中",
                key=f"len_{name}",
            )
            gen_clicked = st.button(
                "✨ 產生段落", key=f"gen_{name}", use_container_width=True
            )
            regen_clicked = st.button(
                "🔁 重新生成（換一版）",
                key=f"regen_{name}",
                use_container_width=True,
            )

        should_generate = gen_clicked or regen_clicked
        if should_generate:
            notes_val = assemble_notes(name)
            length_val = st.session_state.get(f"len_{name}", "中")
            if not client:
                st.error("請先在左側填入 Gemini API Key")
            elif not notes_val.strip():
                st.warning("請至少填寫一個欄位")
            else:
                with st.spinner(f"為 {name} 產生段落中..."):
                    try:
                        para, used = generate_paragraph(
                            client, model, name, notes_val, length_val
                        )
                        st.session_state[f"out_{name}"] = para
                        if used != model:
                            st.info(f"⚠️ 主模型 {model} 過載，自動改用 {used} 產生。")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"產生失敗：{exc}")

        st.text_area(
            "段落（可直接編輯）",
            key=f"out_{name}",
            height=170,
        )
        _current = st.session_state.get(f"out_{name}", "")
        _target = st.session_state.get(f"len_{name}", "中")
        _msg, _ = length_feedback(_current, _target)
        _caption_col, _share_col = st.columns([3, 1])
        with _caption_col:
            st.caption(_msg)
        with _share_col:
            if _current.strip():
                _share_text = f"{prefix}{name}：{_current.strip()}"
                st.link_button(
                    "📲 分享到 LINE",
                    line_share_url(_share_text),
                    use_container_width=True,
                )


st.divider()
col_bulk, col_clear, _ = st.columns([1, 1, 3])
with col_bulk:
    bulk_clicked = st.button(
        "🚀 一鍵產生全部", type="primary", use_container_width=True
    )
with col_clear:
    clear_clicked = st.button("🧹 清空全部段落", use_container_width=True)

if bulk_clicked:
    if not client:
        st.error("請先在左側填入 Gemini API Key")
    else:
        # 1. 在主執行緒先搜集所有學生的輸入，工作執行緒不要碰 st.session_state
        tasks: list[tuple[str, str, str]] = []
        for name in st.session_state.students:
            notes_val = assemble_notes(name)
            if not notes_val.strip():
                continue
            length_val = st.session_state.get(f"len_{name}", "中")
            tasks.append((name, notes_val, length_val))

        if not tasks:
            st.info("沒有填寫任何學生的重點，略過產生。")
        else:
            start = time.time()
            prog = st.progress(0.0, text=f"同時產生 {len(tasks)} 位學生...")
            pending: dict[str, str] = {}
            failures: list[str] = []
            fallback_notices: list[str] = []

            # 2. 平行發 API 請求。每位學生一個 worker，Gemini 免費 tier 每分鐘 10–15 RPM 綽綽有餘。
            with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
                future_to_name = {
                    pool.submit(
                        generate_paragraph, client, model, n, nt, ln
                    ): n
                    for n, nt, ln in tasks
                }
                done = 0
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    done += 1
                    try:
                        text, used = future.result()
                        pending[name] = text
                        if used != model:
                            fallback_notices.append(
                                f"{name}：主模型過載，改用 {used}"
                            )
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"{name}：{exc}")
                    prog.progress(
                        done / len(tasks),
                        text=f"已完成 {done}/{len(tasks)}",
                    )

            elapsed = time.time() - start
            prog.progress(1.0, text=f"完成（共 {elapsed:.1f} 秒）")
            st.session_state._pending_outputs = pending
            if failures:
                st.session_state._pending_failures = failures
            if fallback_notices:
                st.session_state._pending_fallbacks = fallback_notices
            st.rerun()

if clear_clicked:
    st.session_state._pending_outputs = {
        name: "" for name in st.session_state.students
    }
    st.rerun()

# 顯示上一輪批次產生的訊息（失敗、fallback）
if st.session_state.get("_pending_failures"):
    for msg in st.session_state._pending_failures:
        st.warning(msg)
    st.session_state._pending_failures = []

if st.session_state.get("_pending_fallbacks"):
    for msg in st.session_state._pending_fallbacks:
        st.info(f"⚠️ {msg}")
    st.session_state._pending_fallbacks = []


st.divider()
st.subheader("📋 最終輸出")

lines: list[str] = [header]
for name in st.session_state.students:
    para = st.session_state.get(f"out_{name}", "").strip()
    if para:
        lines.append(f"{prefix}{name}：{para}")

final_text = "\n".join(lines)

st.code(final_text or header, language="text")
_total_chars = count_visible_chars(final_text) - count_visible_chars(header)
_student_count = len(lines) - 1  # 扣掉標題行
if _student_count > 0:
    st.caption(f"📊 共 {_student_count} 位學生段落 ｜ 內文總字數 {_total_chars}")

_has_content = bool(final_text.strip()) and final_text.strip() != header.strip()

_dl_col, _share_all_col, _ = st.columns([1, 1, 2])
with _dl_col:
    st.download_button(
        "💾 下載 .txt",
        data=final_text,
        file_name=f"{entry_date.isoformat()}_{class_name}.txt",
        mime="text/plain",
        disabled=not _has_content,
        use_container_width=True,
    )
with _share_all_col:
    if _has_content:
        st.link_button(
            "📲 分享全部到 LINE",
            line_share_url(final_text),
            use_container_width=True,
        )
    else:
        st.button(
            "📲 分享全部到 LINE",
            disabled=True,
            use_container_width=True,
            help="產生段落後才能分享",
        )


# ================================================================
# 📮 學期末交接表（獅湖國小）
# ================================================================
st.divider()
st.subheader("📮 學期末交接表（獅湖國小）")
st.caption(
    "貼入整學期每日紀錄 → AI 依每位學生摘要 Q6–Q9 → 一鍵開啟預填 Google 表單，"
    "家人檢查後按送出。"
)

_semester_class = _active_class()
_sem_sw = _semester_class.get("social_worker", "")
_sem_subj = _semester_class.get("subject", "")
_sem_textbook = _semester_class.get("textbook_template", "")
_sem_teacher = (st.session_state.get("teacher_name_input", "") or "").strip()
_sem_entry_ids = st.session_state.get("form_entry_ids", {}) or {}

# 目前班級資訊摘要＋前置檢查
_cols_info = st.columns([2, 1, 1, 1])
with _cols_info[0]:
    st.markdown(
        f"**目前班級**：{_semester_class['name']}　"
        f"｜ 學生 {len(_semester_class['students'])} 位"
    )
with _cols_info[1]:
    st.markdown(f"**社工**：{_sem_sw or '⚠️未設定'}")
with _cols_info[2]:
    st.markdown(f"**科目**：{_sem_subj or '⚠️未設定'}")
with _cols_info[3]:
    st.markdown(
        f"**老師**：{_sem_teacher or '⚠️未設定'}　"
        f"｜ entry {len(_sem_entry_ids)}/{len(FORM_FIELD_KEYS)}"
    )

_missing_setup: list[str] = []
if not _sem_sw:
    _missing_setup.append("班級的主責社工")
if not _sem_subj:
    _missing_setup.append("班級的負責科目")
if not _sem_teacher:
    _missing_setup.append("課輔老師姓名（全域）")
if len(_sem_entry_ids) < len(FORM_FIELD_KEYS):
    _missing_setup.append(
        f"Google 表單 entry ID（目前 {len(_sem_entry_ids)}/{len(FORM_FIELD_KEYS)}）"
    )

if _missing_setup:
    st.info(
        "ℹ️ 還沒設定：" + "、".join(_missing_setup) + "。"
        "沒設也可以試用解析功能，但「打開預填表單」會受影響。"
        "到左側「🏷️ 學期末表單設定」和「📮 Google 表單設定」補完。"
    )

st.markdown("#### Step 1：貼入本學期每日紀錄")
st.text_area(
    "整學期紀錄",
    key="semester_records_input",
    height=250,
    placeholder=(
        "貼入格式範例（app 每日輸出的原格式）：\n"
        "4/20（一）獅湖A班課程交接\n"
        "🙂楊佩婕：今天已經可以注意到分數和整數的不同……\n"
        "🙂李妍柔：今天的練習卷寫得又進步了一點……\n"
        "\n"
        "4/22（三）獅湖A班課程交接\n"
        "🙂楊佩婕：……\n"
        "……"
    ),
    label_visibility="collapsed",
)

_parse_col, _ = st.columns([1, 3])
with _parse_col:
    _parse_clicked = st.button(
        "🔎 解析紀錄", key="parse_semester_btn", use_container_width=True
    )

if _parse_clicked:
    _records_text = st.session_state.get("semester_records_input", "") or ""
    _parsed = parse_semester_records(
        _records_text, known_students=_semester_class["students"]
    )
    st.session_state.semester_parsed = _parsed
    # 清掉上一輪的 AI 草稿，避免錯亂
    for _stud in _semester_class["students"]:
        for _q in ("Q6", "Q7", "Q8", "Q9", "Q10"):
            st.session_state.pop(f"sem_{_q}_{_stud}", None)

_parsed_records: dict[str, list[tuple[str, str]]] = st.session_state.get(
    "semester_parsed", {}
)

if _parsed_records:
    _total = sum(len(v) for v in _parsed_records.values())
    _students_with = [
        s for s in _semester_class["students"] if _parsed_records.get(s)
    ]
    _students_without = [
        s for s in _semester_class["students"] if not _parsed_records.get(s)
    ]
    st.success(
        f"✅ 解析出 {len(_students_with)} 位學生、共 {_total} 筆段落。"
        + (
            "　⚠️ 這些學生沒有紀錄：" + "、".join(_students_without)
            if _students_without
            else ""
        )
    )

    st.markdown("#### Step 2：為每位學生產生 Q6–Q9 草稿 → 打開預填表單")

    if st.button(
        "🚀 一次分析全部學生（有紀錄的才會打）",
        key="bulk_analyze_btn",
        type="primary",
    ):
        if not client:
            st.error("請先在左側填入 Gemini API Key")
        else:
            _bulk_tasks = [
                (s, _parsed_records[s])
                for s in _semester_class["students"]
                if _parsed_records.get(s)
            ]
            _start = time.time()
            _prog = st.progress(0.0, text=f"開始分析 {len(_bulk_tasks)} 位學生...")
            _bulk_failures: list[str] = []
            _bulk_results: dict[str, dict[str, str]] = {}
            with ThreadPoolExecutor(max_workers=min(len(_bulk_tasks), 5)) as pool:
                _futs = {
                    pool.submit(
                        build_semester_summary,
                        client,
                        model,
                        _s,
                        _recs,
                        _sem_subj,
                    ): _s
                    for _s, _recs in _bulk_tasks
                }
                _done = 0
                for _fut in as_completed(_futs):
                    _s = _futs[_fut]
                    _done += 1
                    try:
                        _bulk_results[_s] = _fut.result()
                    except Exception as exc:  # noqa: BLE001
                        _bulk_failures.append(f"{_s}：{exc}")
                    _prog.progress(
                        _done / len(_bulk_tasks),
                        text=f"已完成 {_done}/{len(_bulk_tasks)}",
                    )
            _prog.progress(1.0, text=f"完成（共 {time.time()-_start:.1f} 秒）")
            # 把結果寫進 widget 用的 session_state，下一輪 rerun 會顯示
            _pending_widget = st.session_state.get("_pending_widget_state", {})
            for _s, _r in _bulk_results.items():
                _pending_widget[f"sem_Q6_{_s}"] = _r.get("Q6", "")
                _pending_widget[f"sem_Q7_{_s}"] = _r.get("Q7", "")
                _pending_widget[f"sem_Q8_{_s}"] = _r.get("Q8", "")
                _pending_widget[f"sem_Q9_{_s}"] = _r.get("Q9", "")
                # Q10 還是用班級模板（使用者可再編輯）
                if f"sem_Q10_{_s}" not in st.session_state:
                    _pending_widget[f"sem_Q10_{_s}"] = _sem_textbook
            st.session_state._pending_widget_state = _pending_widget
            if _bulk_failures:
                st.session_state._pending_failures = _bulk_failures
            st.rerun()

    for _stud in _semester_class["students"]:
        _recs = _parsed_records.get(_stud, [])
        _label = f"😊 {_stud}（{len(_recs)} 筆紀錄）"
        with st.expander(_label, expanded=bool(_recs)):
            if not _recs:
                st.caption("此學生沒有解析到紀錄，請回 Step 1 檢查格式或學生名單。")
                continue

            _ai_clicked = st.button(
                "🤖 AI 分析本學期", key=f"sem_analyze_{_stud}"
            )
            if _ai_clicked:
                if not client:
                    st.error("請先在左側填入 Gemini API Key")
                else:
                    try:
                        with st.spinner(f"分析 {_stud} 的整學期紀錄..."):
                            _result = build_semester_summary(
                                client, model, _stud, _recs, _sem_subj
                            )
                        # 透過 pending 機制更新 widget
                        _pending_widget = st.session_state.get(
                            "_pending_widget_state", {}
                        )
                        _pending_widget[f"sem_Q6_{_stud}"] = _result.get("Q6", "")
                        _pending_widget[f"sem_Q7_{_stud}"] = _result.get("Q7", "")
                        _pending_widget[f"sem_Q8_{_stud}"] = _result.get("Q8", "")
                        _pending_widget[f"sem_Q9_{_stud}"] = _result.get("Q9", "")
                        if f"sem_Q10_{_stud}" not in st.session_state:
                            _pending_widget[f"sem_Q10_{_stud}"] = _sem_textbook
                        st.session_state._pending_widget_state = _pending_widget
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"AI 分析失敗：{exc}")

            # 初始化 Q10 預設值（第一次看到此學生時填入班級模板）
            if f"sem_Q10_{_stud}" not in st.session_state:
                st.session_state[f"sem_Q10_{_stud}"] = _sem_textbook

            st.text_input(
                "Q6 下學期需補考單元（英/國請填「無」）",
                key=f"sem_Q6_{_stud}",
                placeholder="例：5-2、5-4；若都通過或非數學科請填「無」",
            )
            st.text_input(
                "Q7 下學期起始單元",
                key=f"sem_Q7_{_stud}",
                placeholder="例：5-3、5-5；（英）第一冊Ch4",
            )
            st.text_area(
                "Q8 學習表現 & 有效教學方式",
                key=f"sem_Q8_{_stud}",
                height=180,
                placeholder="AI 草稿會出現在這裡，可直接編輯。",
            )
            st.text_area(
                "Q9 行為情緒 & 有效處理方式",
                key=f"sem_Q9_{_stud}",
                height=180,
                placeholder="AI 草稿會出現在這裡，可直接編輯。",
            )
            st.text_input(
                "Q10 欲領取新課本",
                key=f"sem_Q10_{_stud}",
                placeholder="例：玩魔數第十一冊 (B)；不須申請則填「無」",
            )

            _answers = {
                "social_worker": _sem_sw,
                "class_name": _semester_class["name"],
                "teacher_name": _sem_teacher,
                "subject": _sem_subj,
                "student_name": _stud,
                "remedial_units": st.session_state.get(f"sem_Q6_{_stud}", ""),
                "next_start_unit": st.session_state.get(f"sem_Q7_{_stud}", ""),
                "learning_performance": st.session_state.get(f"sem_Q8_{_stud}", ""),
                "behavior_emotion": st.session_state.get(f"sem_Q9_{_stud}", ""),
                "textbook": st.session_state.get(f"sem_Q10_{_stud}", ""),
            }
            _ready_to_submit = (
                len(_sem_entry_ids) == len(FORM_FIELD_KEYS)
                and _sem_sw
                and _sem_subj
                and _sem_teacher
                and any(_answers.values())
            )
            if _ready_to_submit:
                _prefill = build_prefill_url(
                    SEMESTER_FORM_ID, _sem_entry_ids, _answers
                )
                st.link_button(
                    f"📮 打開 {_stud} 的預填表單（新分頁）",
                    _prefill,
                    use_container_width=True,
                )
            else:
                st.button(
                    f"📮 打開 {_stud} 的預填表單",
                    disabled=True,
                    use_container_width=True,
                    help=(
                        "需要先補完：Q1 社工、Q4 科目、Q3 老師姓名、"
                        "以及完整 10 題的 entry ID 對應。"
                    ),
                )
else:
    st.caption(
        "（貼入紀錄並按「🔎 解析紀錄」後，這裡會出現每位學生的設定卡片）"
    )
