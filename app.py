"""課輔老師課程交接產生器 (Streamlit)

執行方式：
    pip install -r requirements.txt
    streamlit run app.py

API key 可於左側輸入，或設定環境變數 GEMINI_API_KEY。
Gemini API key 申請：https://aistudio.google.com/apikey （免費額度足以應付每天數十次使用）
"""

from __future__ import annotations

import os
from datetime import date

import streamlit as st

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # 讓介面在未安裝 SDK 時也能顯示說明
    genai = None  # type: ignore
    genai_types = None  # type: ignore


DEFAULT_STUDENTS = ["薛恩銘", "李妮綺", "潘奕亨", "蔡柏容", "張鈺淇"]
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
- 段落長度依使用者指定。

{FEW_SHOT_EXAMPLES}

請模仿上述語氣，但**不要直接抄範例的內容或題目**，只根據使用者提供的關鍵字撰寫。"""


def build_user_prompt(name: str, notes: str, length: str) -> str:
    length_hint = {
        "短": "約 50–80 字",
        "中": "約 80–140 字",
        "長": "約 140–220 字",
    }[length]
    return (
        f"請為學生「{name}」撰寫今日課程交接段落。\n\n"
        f"今日觀察關鍵字／短句：\n{notes.strip()}\n\n"
        f"要求：\n"
        f"- 長度：{length_hint}\n"
        f"- 直接輸出段落內文，不要加姓名前綴、不要加標題、不要加引號\n"
        f"- 只輸出一段文字"
    )


def generate_paragraph(client, model: str, name: str, notes: str, length: str) -> str:
    response = client.models.generate_content(
        model=model,
        contents=build_user_prompt(name, notes, length),
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=800,
            temperature=0.85,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini 沒有回傳文字（可能被安全過濾器擋下，請調整輸入重試）")
    return text


st.set_page_config(page_title="課輔交接產生器", page_icon="📝", layout="wide")

if "students" not in st.session_state:
    st.session_state.students = DEFAULT_STUDENTS.copy()
if "students_text_area" not in st.session_state:
    st.session_state.students_text_area = "\n".join(DEFAULT_STUDENTS)


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
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=default_key,
        help="到 https://aistudio.google.com/apikey 免費申請；也可設環境變數 GEMINI_API_KEY",
    )
    model = st.selectbox(
        "模型",
        MODEL_OPTIONS,
        index=0,
        help="2.5 Flash：快且免費額度大（推薦）；2.5 Pro：品質最佳但額度較少；2.0 Flash：額度最大的備援",
    )

    st.divider()
    st.subheader("📅 班級 / 日期")
    class_name = st.text_input("班級名稱", value="甲圍A班")
    entry_date = st.date_input("日期", value=date.today())
    prefix = st.text_input("段落前綴（例如 😊 或 @）", value="😊", max_chars=4)

    st.divider()
    st.subheader("👥 學生名單")
    st.caption("每行一位，可直接在下面新增／刪除")
    st.text_area(
        "學生名單",
        key="students_text_area",
        height=180,
        label_visibility="collapsed",
    )
    parsed_students = [
        s.strip() for s in st.session_state.students_text_area.splitlines() if s.strip()
    ]
    if parsed_students:
        st.session_state.students = parsed_students


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
            st.text_area(
                "今日重點（關鍵字或短句，逗號／換行分隔皆可）",
                key=f"notes_{name}",
                height=140,
                placeholder=(
                    "例：\n"
                    "・7-4 角與角度完成\n"
                    "・量角器使用正確\n"
                    "・7-3 寫太快錯兩題（減錯）\n"
                    "・觀察是急著完成，需要加強"
                ),
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
            notes_val = st.session_state.get(f"notes_{name}", "")
            length_val = st.session_state.get(f"len_{name}", "中")
            if not client:
                st.error("請先在左側填入 Gemini API Key")
            elif not notes_val.strip():
                st.warning("請先輸入今日重點")
            else:
                with st.spinner(f"為 {name} 產生段落中..."):
                    try:
                        para = generate_paragraph(
                            client, model, name, notes_val, length_val
                        )
                        st.session_state[f"out_{name}"] = para
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"產生失敗：{exc}")

        st.text_area(
            "段落（可直接編輯）",
            key=f"out_{name}",
            height=170,
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
        prog = st.progress(0.0, text="準備中...")
        total = len(st.session_state.students)
        for i, name in enumerate(st.session_state.students):
            notes_val = st.session_state.get(f"notes_{name}", "")
            length_val = st.session_state.get(f"len_{name}", "中")
            prog.progress(i / max(total, 1), text=f"產生 {name} 中...")
            if not notes_val.strip():
                continue
            try:
                para = generate_paragraph(client, model, name, notes_val, length_val)
                st.session_state[f"out_{name}"] = para
            except Exception as exc:  # noqa: BLE001
                st.warning(f"{name} 產生失敗：{exc}")
        prog.progress(1.0, text="完成")
        st.rerun()

if clear_clicked:
    for name in st.session_state.students:
        st.session_state[f"out_{name}"] = ""
    st.rerun()


st.divider()
st.subheader("📋 最終輸出")

lines: list[str] = [header]
for name in st.session_state.students:
    para = st.session_state.get(f"out_{name}", "").strip()
    if para:
        lines.append(f"{prefix}{name}：{para}")

final_text = "\n".join(lines)

st.code(final_text or header, language="text")

st.download_button(
    "💾 下載 .txt",
    data=final_text,
    file_name=f"{entry_date.isoformat()}_{class_name}.txt",
    mime="text/plain",
    disabled=not final_text.strip() or final_text.strip() == header.strip(),
)
