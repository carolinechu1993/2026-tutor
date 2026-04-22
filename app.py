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
    },
    {
        "name": "獅湖A班",
        "students": ["楊佩婕", "李妍柔", "王懷銘", "洪宇辰", "洪維謙"],
        "profiles": {},
    },
]
TAIPEI_TZ = ZoneInfo("Asia/Taipei") if ZoneInfo else None


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
    return {"name": name.strip(), "students": students, "profiles": profiles}


def _default_classes_copy() -> list[dict]:
    return [
        {"name": c["name"], "students": c["students"].copy(), "profiles": {}}
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


def _load_class_into_widgets(idx: int) -> None:
    """把 classes[idx] 的資料寫入 widget key，讓下一輪 rerun 顯示該班內容。"""
    classes = st.session_state.classes
    if not (0 <= idx < len(classes)):
        return
    cls = classes[idx]
    st.session_state.students_text_area = "\n".join(cls["students"])
    st.session_state.class_name_input = cls["name"]
    st.session_state.students = cls["students"].copy()
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
    new_cls = {"name": new_name, "students": ["學生1"], "profiles": {}}
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
