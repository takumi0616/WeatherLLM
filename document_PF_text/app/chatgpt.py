import os
from pathlib import Path
import requests


def _find_env_path() -> Path:
    var = os.environ.get("OPENAI_ENV_FILE") or os.environ.get("ENV_PATH")
    if var:
        p = Path(var)
        if p.exists():
            return p
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        cand = p / ".env"
        if cand.exists():
            return cand
    return here.parent / ".env"


def _load_api_key() -> str:
    # Try common env var casings first
    key = os.environ.get("OpenAI_KEY_TOKEN") or os.environ.get("OPENAI_KEY_TOKEN")
    if key:
        return key

    env_path = _find_env_path()

    # Try python-dotenv if available
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=env_path)
        key = os.environ.get("OpenAI_KEY_TOKEN") or os.environ.get("OPENAI_KEY_TOKEN")
        if key:
            return key
    except Exception:
        pass

    # Fallback: manual .env parsing
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "OpenAI_KEY_TOKEN":
                        return v.strip().strip("\"'")
    except Exception:
        pass

    raise RuntimeError("OpenAI API key not found. Set OpenAI_KEY_TOKEN in environment or in a .env file.")


def clean_comment(content):
    markdown_symbols = [
        "*",
        "_",
        "`",
        "~",
        "#",
        "+",
        "-",
        ">",
        "[",
        "]",
        "(",
        ")",
        "{",
        "}",
        "!",
        "|",
    ]
    translation_table = str.maketrans("", "", "".join(markdown_symbols))
    cleaned_content = content.translate(translation_table)
    cleaned_content = cleaned_content.strip().replace("\n", "")
    return cleaned_content


def generate_weather_comment(
    weather_text: str, date_label: str, date: str, point: str, point_place: str
) -> str:
    prompt = f"""

# 命令文
- あなたは気象学にとても詳しく、気象予報士の資格を持っており、気象数値データから「天気コメント」を作成することができます。
- そしてあなたは以下に書いてあるルールを必ず守ります。
- あなたには、以下の情報を使って、「ポイント番号: {point},地名: {point_place}」に対する「天気コメント」を作成して欲しいです。

# 作成の手順
天気コメントを作成する際は以下のstepにそって、step by stepで作成してください。
step1.天気コメントのルールを守って、「天気」に関する天気コメントを1文作成してください。
step2.天気コメントのルールを守って、「気温」に関する天気コメントを1文作成してください。
step3.天気コメントのルールを守って、「雨」に関する天気コメントを1文作成してください。
step4.天気コメントのルールを守って、「風速」に関する天気コメントを1文作成してください。
step5.天気コメントのルールを守って、「降雪」に関する天気コメントを1文作成してください。
step6.重要度の比較的小さい文章を削除してください。
step7.出力のルールを守って、出力フォーマットに従って文章をそのまま出力してください。

# 天気コメントのルール
- 外部のデータや情報を一切使用しないで、「地名: {point_place}の、{date_label}: {date}の気象データ」のみを参考に天気コメントを作成してください。
- 天気コメントとは、パッと読むことでその地域の天気をすぐに把握できる文章のことであり、「地名: {point_place}の{date_label}: {date}の気象データ」の中でも重要度という数値が比較的大きいものを選択してから使用して文章を作成してください。
- 重要度が小さいデータに関しては文章にしなくて良いです。
- 気象データの数値はコメント内に必ず含めてください。
- 重要度に関する数値は絶対に天気コメント内に含めないでください。
- 時系列の言葉の使い方を間違わないでください。特に「あした」のコメントを作成する際に「きのう」という言葉は「きょう」という言葉を使ってください。
- 最低気温と最高気温は重要度が高い方どちらかに言及してください。また、平年と前日に関することもどちらかだけでいいです。
- 日降水量と最大1時間降水量は重要度が高い方どちらかに言及してください。
- 日降水量は特別高い数値でない場合は〜mmを超えたみたいにわかりやすくしてください。
- 「{date}の天気予報」という補足情報は、コメント内に入れることによって気象の理解の解像度があがるなど、必要があれば使用してください。
- 「{date}の天気予報」はあくまで予報です。もっとも正しい情報は「観測」と書かれたデータです。
- 1つの文章の中に「気温」「雨」「風速」「降雪」の要素は1つだけです。絶対に混ぜないでください。

# 出力のルール
- 「~に関して入れて欲しい言葉」の文字は必ず使って文章を作成してください。
- 天気コメントの中身の文章のみを出力してください。
- 地名や現在の年月日時刻に関するデータは他で付け足される予定なので、それらの情報は絶対に天気コメントの文章の中に入れないでください。
- 日付に関する情報は「{date_label}」にしてください。
- 必ずマークダウン形式で出力してください。
- 必ず自然な日本語で出力してください。
- 出力フォーマットを必ず守って出力してください。
- 最初の文章の文字だけは「{date_label}は、」にしてください。それ以降の文章にはいりません。

# 出力フォーマット
- 文章

# 地名: {point_place}の、{date_label}: {date}の気象データ
```
{weather_text}
```

    """

    def call_api():
        try:
            print("コメント作成開始")
            messages = [{"role": "user", "content": prompt}]
            # body = {"model": "gpt-4o-2024-11-20", "messages": messages, "temperature": 0.8}
            body = {"model": "o1-preview", "messages": messages}
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_load_api_key()}",
            }

            url = "https://api.openai.com/v1/chat/completions"
            response = requests.post(url, json=body, headers=headers)

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                comment = clean_comment(content)
                return comment
            else:
                return f"API Error: {response.status_code}"

        except Exception as e:
            return f"Error generating comment: {str(e)}"

    # 再生成のロジック
    for attempt in range(10000):
        comment = call_api()
        if comment.startswith(("きのうは", "きょうは", "あしたは")):
            if date_label == "あした" and "きのう" in comment:
                print(
                    f"Attempt {attempt + 1}: 'きのう' found in comment for 'あした'. Regenerating..."
                )
                continue
            return comment
        else:
            print(f"Attempt {attempt + 1}: Regenerating comment...")

    return "Error: Failed to generate a valid comment after 3 attempts."
