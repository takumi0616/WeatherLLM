"""
気象現象キーワード検出スクリプト

`pdf-to-text/` ディレクトリ内の月別テキストファイル（YYMM.txt 形式、例: 0201.txt は 2002年1月）
すべてに含まれる気象現象に関するキーワードの出現件数を集計し、
`word_detect.log` に詳細を日本語で出力する。

集計の前提:
  - 入力テキストは 2002年1月〜2025年12月（24年間）の日々の天気図解説を月単位で分割したもの。
  - 単純な部分文字列マッチで件数を数える（同じ行に複数回出現すれば複数回カウント）。
  - 親キーワード（例: 「高気圧」）と子キーワード（例: 「太平洋高気圧」）は
    重複してカウントされる点に注意。
  - 同様に「雷雨」は「雷」にも、「暴風域」は「暴風」にも、
    「最高気温」は「気温」にも、「梅雨前線」は「前線」と「梅雨」の両方にカウントされる。

使い方:
  $ python3 word_detect.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "pdf-to-text"
LOG_FILE = SCRIPT_DIR / "word_detect.log"

YEAR_START = 2002
YEAR_END = 2025
YEAR_SPAN = YEAR_END - YEAR_START + 1


@dataclass
class Keyword:
    """1 つのキーワード（および解説と子キーワード）。"""

    name: str
    note: str = ""
    children: list["Keyword"] = field(default_factory=list)


@dataclass
class Section:
    """関連キーワードをまとめるセクション。"""

    title: str
    description: str
    keywords: list[Keyword]


SECTIONS: list[Section] = [
    Section(
        title="1. 気圧配置（総観場のパターン）",
        description="季節を象徴する典型的な気圧配置の表現。",
        keywords=[
            Keyword(
                name="冬型",
                note="冬季(12〜2月、約90日/年)の「西高東低の冬型気圧配置」として頻出。",
                children=[
                    Keyword(name="西高東低", note="典型的な冬型気圧配置の表現。"),
                    Keyword(name="冬型の気圧配置", note="冬型を明示する定型句（「の」あり）。"),
                    Keyword(name="冬型気圧配置", note="冬型を明示する定型句（「の」なし）。"),
                    Keyword(name="強い冬型", note="冬型が強化された状態の表現。"),
                    Keyword(name="冬型が強まる", note="冬型強化の動的表現。"),
                    Keyword(name="冬型強まる", note="冬型強化の表現（「が」省略形）。"),
                    Keyword(name="冬型が緩む", note="冬型が弱まる動的表現。"),
                    Keyword(name="冬型緩む", note="冬型が弱まる表現（「が」省略形）。"),
                    Keyword(name="冬型続く", note="冬型が継続する表現。"),
                    Keyword(name="冬型が続く", note="冬型継続の標準表現。"),
                ],
            ),
            Keyword(name="南高北低", note="夏型気圧配置。太平洋高気圧が日本を覆う形。"),
            Keyword(name="北高南低", note="北日本に高気圧、南に低気圧の配置。"),
            Keyword(name="北高型", note="オホーツク海高気圧優勢で北日本が冷涼となる配置。"),
            Keyword(name="帯状高気圧", note="高気圧が東西に帯状に連なる配置。春・秋に多い。"),
            Keyword(name="鯨の尾型", note="太平洋高気圧の張り出しが鯨の尾に似た夏の配置。"),
            Keyword(name="気圧配置", note="総観場の配置全般を示す総称表現。"),
        ],
    ),
    Section(
        title="2. 低気圧",
        description="日本付近で発生・通過する各種の低気圧。",
        keywords=[
            Keyword(
                name="低気圧",
                note="温帯低気圧・南岸低気圧など多様な低気圧の総称として通年出現。",
                children=[
                    Keyword(name="温帯低気圧", note="中緯度で前線を伴って発達する低気圧。"),
                    Keyword(name="南岸低気圧", note="本州南岸を進む低気圧。冬の関東降雪要因。"),
                    Keyword(name="日本海低気圧", note="春に日本海で発達。南風で気温上昇を伴う。"),
                    Keyword(name="二つ玉低気圧", note="日本海と太平洋岸を同時に進む二つの低気圧。"),
                    Keyword(name="寒冷低気圧", note="上空の寒気に伴う背の低い低気圧。"),
                    Keyword(name="寒冷渦", note="上空の寒冷低気圧。大気不安定の主因。"),
                    Keyword(name="爆弾低気圧", note="24時間で急発達する低気圧。気象庁公式語ではない俗称。"),
                    Keyword(name="低気圧の影響", note="低気圧通過に伴う天候悪化の最頻出定型句。"),
                    Keyword(name="低気圧や前線", note="低気圧と前線の複合作用を示す頻出フレーズ。"),
                    Keyword(name="低気圧と前線", note="低気圧と前線の併存表現。"),
                    Keyword(name="低気圧は", note="低気圧を主題化する表現の頻出冒頭。"),
                    Keyword(name="低気圧へ", note="低気圧へ向かう・変わる表現。"),
                    Keyword(name="低気圧から", note="低気圧から伸びる前線等の表現。"),
                    Keyword(name="低気圧の接近", note="低気圧が近づく表現。"),
                    Keyword(name="低気圧の通過", note="低気圧通過の名詞句。"),
                    Keyword(name="低気圧が発達", note="低気圧の勢力強化を示す動的表現。"),
                    Keyword(name="低気圧が東進", note="低気圧の東進を示す動的表現。"),
                    Keyword(name="低気圧が日本", note="低気圧が日本付近に到達した状態の表現。"),
                    Keyword(name="低気圧により", note="低気圧を原因とする表現の定型句。"),
                    Keyword(name="低気圧が通過", note="低気圧通過の動的表現。"),
                    Keyword(name="低気圧が発生", note="低気圧の発生を示す表現。"),
                    Keyword(name="低気圧が北", note="北上・北東進する低気圧の動的表現。"),
                    Keyword(name="低気圧が南", note="南下する低気圧の動的表現。"),
                    Keyword(name="海の低気圧", note="日本海・オホーツク海・東シナ海等の低気圧の総称。"),
                    Keyword(name="千島近海", note="千島近海で発達する低気圧の頻出位置表現。"),
                    Keyword(name="伴った低気圧", note="前線を伴う低気圧の頻出表現。"),
                    Keyword(name="発達した低気圧", note="勢力を増した低気圧の定型表現。"),
                    Keyword(name="急速に発達", note="低気圧の急発達を示す定型表現。"),
                    Keyword(name="低気圧の中心", note="低気圧の中心位置を指す表現。"),
                    Keyword(name="アリューシャン", note="アリューシャン低気圧。冬の北太平洋の半永久的低気圧。"),
                    Keyword(name="大陸の低気圧", note="大陸上の低気圧の総称表現。"),
                ],
            ),
        ],
    ),
    Section(
        title="3. 高気圧",
        description="季節ごとに支配する各種高気圧。",
        keywords=[
            Keyword(
                name="高気圧",
                note="季節ごとに異なる高気圧（太平洋・オホーツク海・移動性・シベリア等）の総称。",
                children=[
                    Keyword(name="太平洋高気圧", note="夏季の主役。高温多湿な南東風をもたらす。"),
                    Keyword(name="オホーツク海高気圧", note="初夏〜梅雨期に「やませ」を吹かせる冷湿な高気圧。"),
                    Keyword(name="移動性高気圧", note="春・秋に大陸から日本付近を通過。短い晴天をもたらす。"),
                    Keyword(name="シベリア高気圧", note="冬の寒気を蓄える大陸性高気圧。多くは「冬型」表現に吸収される。"),
                    Keyword(name="チベット高気圧", note="夏季上層に現れる高気圧。猛暑要因として言及される。"),
                    Keyword(name="北太平洋高気圧", note="太平洋高気圧の正式名称的表現。"),
                    Keyword(name="大陸の高気圧", note="春・秋に大陸から張り出す高気圧の総称表現。"),
                    Keyword(name="高気圧に覆われ", note="高気圧支配下の天気の最頻出定型表現。"),
                    Keyword(name="高気圧が張り出", note="高気圧が日本へ伸びる動的表現。"),
                    Keyword(name="高気圧の張り出し", note="高気圧の伸長を示す名詞句。"),
                    Keyword(name="張り出し", note="高気圧の勢力拡大の汎用表現。"),
                    Keyword(name="高気圧に広く", note="広範囲を覆う高気圧の表現。"),
                    Keyword(name="高気圧の縁", note="高気圧の縁辺で湿った空気が流入する文脈。"),
                    Keyword(name="高気圧の圏内", note="高気圧の影響下にある状態の表現。"),
                    Keyword(name="高気圧が日本", note="高気圧が日本に位置する状態の表現。"),
                    Keyword(name="高気圧が東", note="高気圧の東進・東側位置を示す表現。"),
                ],
            ),
        ],
    ),
    Section(
        title="4. 前線",
        description="日本付近で活動する各種の前線。",
        keywords=[
            Keyword(
                name="前線",
                note="日本は寒帯前線帯に位置し、いずれかの前線が通年存在。",
                children=[
                    Keyword(name="梅雨前線", note="6〜7月に本州付近で停滞。大雨の主要因。"),
                    Keyword(name="秋雨前線", note="9〜10月に日本列島で停滞。台風と相互作用しやすい。"),
                    Keyword(name="温暖前線", note="低気圧の東側、暖気が寒気の上に乗り上げる前線。"),
                    Keyword(name="寒冷前線", note="低気圧の西側、寒気が暖気を押し上げる前線。雷雨や突風を伴う。"),
                    Keyword(name="停滞前線", note="動きが遅い前線の総称。梅雨前線・秋雨前線を含む。"),
                    Keyword(name="閉塞前線", note="寒冷前線が温暖前線に追いついて発生する前線。"),
                    Keyword(name="前線の影響", note="前線通過に伴う天候悪化の最頻出定型句。"),
                    Keyword(name="前線を伴", note="低気圧が前線を伴う表現（「前線を伴った」「前線を伴う」を捕捉）。"),
                    Keyword(name="前線が停滞", note="前線停滞の動的表現。"),
                    Keyword(name="前線停滞", note="前線停滞の名詞句的表現。"),
                    Keyword(name="前線通過", note="前線が観測点を通り過ぎる現象。"),
                    Keyword(name="前線が通過", note="前線通過の動的表現。"),
                    Keyword(name="前線により", note="前線を原因とする表現の定型句。"),
                    Keyword(name="前線に近い", note="前線付近の表現。"),
                    Keyword(name="前線や低気圧", note="前線と低気圧の複合作用を示す頻出フレーズ。"),
                    Keyword(name="のびる前線", note="伸長する前線の動的表現。"),
                    Keyword(name="前線が日本", note="前線が日本付近に達した状態の表現。"),
                    Keyword(name="前線の活動", note="前線の活発化を示す表現。"),
                    Keyword(name="活発な前線", note="降水の強い活動的な前線。"),
                ],
            ),
        ],
    ),
    Section(
        title="5. 台風・熱帯擾乱",
        description="台風および関連する熱帯系現象。",
        keywords=[
            Keyword(
                name="台風",
                note="年間平均25〜30個発生。発生段階・北上・温帯低気圧化まで広く言及される。",
                children=[
                    Keyword(name="台風第", note="「台風第N号」の番号表記。最頻出の台風表現。"),
                    Keyword(name="熱帯低気圧", note="台風になる前後の段階。台風から弱まった場合も含む。"),
                    Keyword(name="暴風域", note="平均風速25m/s以上の領域。"),
                    Keyword(name="強風域", note="平均風速15m/s以上の領域。"),
                    Keyword(name="上陸", note="台風中心が国内に到達した状態。"),
                    Keyword(name="接近", note="台風が日本に近づく状態の表現。"),
                    Keyword(name="温帯低気圧化", note="台風が中緯度で温帯低気圧へ変質する過程。"),
                    Keyword(name="勢力", note="台風の強さを表す表現で頻出。"),
                    Keyword(name="台風の眼", note="台風中心部の風の弱い領域。"),
                    Keyword(name="台風一過", note="台風通過後の青空・好天の定型表現。"),
                    Keyword(name="台風の影響", note="台風による天候への影響を示す表現。"),
                    Keyword(name="台風が", note="台風を主語とする動的表現の頻出冒頭。"),
                    Keyword(name="進路", note="台風の進む方向の表現。"),
                    Keyword(name="最大風速", note="平均風速の最大値。"),
                    Keyword(name="最大瞬間風速", note="瞬間風速の最大値。"),
                    Keyword(name="中心気圧", note="台風・低気圧の強度指標。"),
                ],
            ),
        ],
    ),
    Section(
        title="6. 空気の性質（暖気・寒気・湿り）",
        description="空気塊の温度・湿度を表す定型表現。",
        keywords=[
            Keyword(name="暖かい空気", note="南からの暖気移流。気温上昇・大気不安定の文脈で出現。"),
            Keyword(name="湿った空気", note="梅雨期や前線・台風接近時に降水を強める要因として頻出。"),
            Keyword(name="湿った空気の影響", note="湿った空気を原因とする降水の最頻出定型句。"),
            Keyword(name="湿った空気が流入", note="暖湿気が流入する動的表現。"),
            Keyword(name="湿った空気が流れ込", note="暖湿気が流れ込む動的表現。"),
            Keyword(name="暖かく湿った空気", note="梅雨〜盛夏の豪雨要因の頻出表現。"),
            Keyword(name="冷たい空気", note="寒気移流。冬型・上空寒気・大気不安定の文脈で出現。"),
            Keyword(name="乾いた空気", note="冬の大陸由来の乾燥した気団に関連する表現。"),
            Keyword(name="暖気", note="暖気移流。低気圧の東側や春先の急激な昇温で出現。"),
            Keyword(name="暖気が入", note="暖気が流入する動的表現。"),
            Keyword(name="寒気", note="上空寒気・寒気の南下など、冬型強化や大気不安定の文脈で頻出。"),
            Keyword(name="寒気の影響", note="寒気を原因とする最頻出定型句。"),
            Keyword(name="寒気を伴", note="低気圧が寒気を伴う表現（「寒気を伴った」「伴う」を捕捉）。"),
            Keyword(name="寒気が入", note="寒気の流入の動的表現。"),
            Keyword(name="寒気が流入", note="寒気の流入の標準表現。"),
            Keyword(name="寒気流入", note="寒気流入の名詞句的表現。"),
            Keyword(name="寒気により", note="寒気を原因とする表現。"),
            Keyword(name="上空の寒気", note="500hPaなど上層に流入する寒気。雷雨・大気不安定の主因。"),
            Keyword(name="上空寒気", note="上空の寒気（「の」省略形）。"),
            Keyword(name="寒気の南下", note="寒気移流の動的表現。"),
            Keyword(name="寒気の流入", note="寒気が日本付近に入る動的表現。"),
            Keyword(name="暖気の流入", note="暖気移流の動的表現。"),
            Keyword(name="暖湿気", note="暖かく湿った空気の略称的表現。"),
            Keyword(name="湿舌", note="低気圧南東側に伸びる暖湿気の流入帯。"),
            Keyword(name="暖湿流", note="暖かく湿った空気の流れ。豪雨の要因。"),
            Keyword(name="暖気移流", note="暖気が水平に流入する現象。"),
            Keyword(name="寒気移流", note="寒気が水平に流入する現象。"),
        ],
    ),
    Section(
        title="7. 上層の流れ・大気層",
        description="上空のジェット気流・トラフ・リッジに関する表現。",
        keywords=[
            Keyword(name="気圧の谷", note="上空のトラフ。天気を崩す主要因の一つ。"),
            Keyword(name="気圧の谷の影響", note="気圧の谷を原因とする最頻出定型句。"),
            Keyword(name="上空の気圧の谷", note="上層トラフの明示表現。"),
            Keyword(name="気圧の尾根", note="上空のリッジ。晴天をもたらす要因。"),
            Keyword(name="トラフ", note="気圧の谷の英語由来表現。"),
            Keyword(name="リッジ", note="気圧の尾根の英語由来表現。"),
            Keyword(name="ジェット気流", note="対流圏上層の強い偏西風。低気圧発達と関連。"),
            Keyword(name="偏西風", note="中緯度を西から東へ吹く卓越風。"),
            Keyword(name="蛇行", note="偏西風の南北蛇行。異常気象の要因。"),
            Keyword(name="上空", note="上層大気を指す一般的表現。"),
            Keyword(name="上層", note="対流圏上層を指す表現。"),
            Keyword(name="下層", note="対流圏下層を指す表現。"),
            Keyword(name="高層", note="高層大気・高層天気図に関連。"),
            Keyword(name="ブロッキング", note="偏西風蛇行で停滞性の高気圧／低気圧が居座る状態。"),
            Keyword(name="北極振動", note="北極を中心とした気圧変動パターン。"),
        ],
    ),
    Section(
        title="8. 季節風・局地風",
        description="季節を象徴する風や地形性の局地風。",
        keywords=[
            Keyword(name="季節風", note="モンスーン。冬は北西、夏は南東が卓越。"),
            Keyword(name="モンスーン", note="季節風の英語由来表現。"),
            Keyword(name="木枯らし", note="晩秋〜初冬に吹く強い北寄りの風。"),
            Keyword(name="春一番", note="立春後に最初に吹く強い南風。"),
            Keyword(name="やませ", note="初夏〜盛夏に東北太平洋側へ吹く冷湿な北東風。冷害要因。"),
            Keyword(name="フェーン", note="山越えで高温乾燥となる風。記録的高温の要因。"),
            Keyword(name="フェーン現象", note="フェーンの完全表記。"),
            Keyword(name="海風", note="昼間に海から陸へ吹く風。"),
            Keyword(name="陸風", note="夜間に陸から海へ吹く風。"),
            Keyword(name="南風", note="温暖前線通過時や春一番などで頻出。"),
            Keyword(name="北風", note="寒気流入や冬型で頻出。"),
            Keyword(name="東風", note="オホーツク海高気圧などに伴う風。"),
            Keyword(name="西風", note="冬型や偏西風の卓越方向。"),
            Keyword(name="北東風", note="やませなど冷涼な風の方向。"),
            Keyword(name="北西風", note="冬型気圧配置で卓越する風。"),
            Keyword(name="南東風", note="太平洋高気圧の縁辺で卓越。"),
            Keyword(name="南西風", note="低気圧南側や梅雨期に卓越。"),
            Keyword(name="山谷風", note="日中は谷風、夜は山風となる地形性局地風。"),
        ],
    ),
    Section(
        title="9. 大気の状態",
        description="大気安定度・対流に関わる表現。",
        keywords=[
            Keyword(name="大気の状態が不安定", note="積乱雲・雷雨・突風の発生を示唆する定型表現。"),
            Keyword(name="不安定", note="大気不安定全般。"),
            Keyword(name="安定", note="大気が安定している状態の表現。"),
            Keyword(name="対流", note="鉛直方向の大気混合。雷雨発生の要因。"),
            Keyword(name="対流活動", note="対流の活発さを示す表現。"),
            Keyword(name="逆転層", note="上空ほど気温が高い層。大気汚染・霧の要因。"),
            Keyword(name="飽和", note="水蒸気が飽和して凝結する状態。"),
        ],
    ),
    Section(
        title="10. 視程・大気現象",
        description="視程を悪化させる現象や大気光学現象。",
        keywords=[
            Keyword(name="霧", note="水滴が空中に浮遊して視程1km未満となる現象。"),
            Keyword(name="濃霧", note="特に視程が悪い霧。"),
            Keyword(name="もや", note="視程1km以上10km未満。"),
            Keyword(name="煙霧", note="乾いた微粒子による視程悪化。"),
            Keyword(name="黄砂", note="大陸の砂塵が偏西風で運ばれる現象。春に多い。"),
            Keyword(name="視程", note="水平に見通せる距離。"),
            Keyword(name="PM2.5", note="微小粒子状物質。視程・健康に影響。"),
        ],
    ),
    Section(
        title="11. 季節・梅雨",
        description="梅雨入り・梅雨明け等の季節推移表現。",
        keywords=[
            Keyword(name="梅雨", note="6〜7月の雨季。"),
            Keyword(name="梅雨入り", note="梅雨期の始まり。"),
            Keyword(name="梅雨明け", note="梅雨期の終わり。"),
            Keyword(name="秋雨", note="秋の長雨。"),
            Keyword(name="入梅", note="暦の上での梅雨入り。"),
            Keyword(name="五月晴れ", note="梅雨の合間の晴天、または5月の好天。"),
            Keyword(name="小春日和", note="晩秋〜初冬の穏やかな晴天。"),
            Keyword(name="立春", note="二十四節気。冬と春の境。"),
            Keyword(name="立夏", note="二十四節気。春と夏の境。"),
            Keyword(name="立秋", note="二十四節気。夏と秋の境。"),
            Keyword(name="立冬", note="二十四節気。秋と冬の境。"),
            Keyword(name="春分", note="二十四節気。昼夜の長さがほぼ等しい。"),
            Keyword(name="夏至", note="二十四節気。一年で最も昼が長い。"),
            Keyword(name="秋分", note="二十四節気。昼夜の長さがほぼ等しい。"),
            Keyword(name="冬至", note="二十四節気。一年で最も昼が短い。"),
        ],
    ),
    Section(
        title="12. 異常気象・気候",
        description="気候変動・異常気象に関する表現。",
        keywords=[
            Keyword(name="異常気象", note="平年からの大きな逸脱。"),
            Keyword(name="異常高温", note="顕著な高温現象。"),
            Keyword(name="異常低温", note="顕著な低温現象。"),
            Keyword(name="エルニーニョ", note="東太平洋赤道域の海面水温が高い状態。"),
            Keyword(name="ラニーニャ", note="東太平洋赤道域の海面水温が低い状態。"),
            Keyword(name="ヒートアイランド", note="都市部が周辺より高温になる現象。"),
            Keyword(name="気候変動", note="長期的な気候の変化。"),
            Keyword(name="温暖化", note="地球規模の気温上昇。"),
            Keyword(name="記録的", note="観測史上に残るレベルの現象。"),
        ],
    ),
    Section(
        title="13. 動き・強度変化",
        description="気象擾乱の動きや強さの変化表現。",
        keywords=[
            Keyword(name="発達", note="低気圧・台風等の勢力強化。"),
            Keyword(name="発達しながら", note="勢力を増しつつ進む表現。"),
            Keyword(name="衰弱", note="勢力が弱まる表現。"),
            Keyword(name="弱まる", note="勢力低下の動的表現。"),
            Keyword(name="強まる", note="勢力強化の動的表現。"),
            Keyword(name="北上", note="北方向への移動。"),
            Keyword(name="南下", note="南方向への移動。寒気南下等。"),
            Keyword(name="東進", note="東方向への移動。"),
            Keyword(name="西進", note="西方向への移動。"),
            Keyword(name="進む", note="気象擾乱の移動の基本表現。"),
            Keyword(name="通過", note="擾乱が観測点を過ぎる現象。"),
            Keyword(name="停滞", note="擾乱が動かない状態。"),
            Keyword(name="急発達", note="短時間で勢力を強める表現。"),
            Keyword(name="続く", note="気圧配置・天候等の状態継続を示す汎用表現。"),
        ],
    ),
]


def setup_logger() -> logging.Logger:
    """ファイル出力用のロガーをセットアップ。"""
    logger = logging.getLogger("word_detect")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger


def load_text_from_dir(directory: Path) -> tuple[str, list[Path]]:
    """ディレクトリ内の *.txt をファイル名昇順で連結して 1 つの文字列にまとめる。"""
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {directory}")
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"入力ディレクトリにテキストファイルがありません: {directory}")
    parts = [path.read_text(encoding="utf-8", errors="replace") for path in files]
    return "\n".join(parts), files


def count_keyword(text: str, keyword: str) -> int:
    """部分文字列としての出現回数を数える（重複を含む）。"""
    return text.count(keyword)


def annual_average(total: int) -> float:
    return total / YEAR_SPAN if YEAR_SPAN > 0 else 0.0


def main() -> None:
    logger = setup_logger()

    logger.info("=" * 78)
    logger.info("気象現象キーワード検出結果")
    logger.info("=" * 78)
    logger.info("入力ディレクトリ : %s", INPUT_DIR)
    logger.info("出力ログ         : %s", LOG_FILE)
    logger.info("対象期間         : %d年〜%d年（%d年間）", YEAR_START, YEAR_END, YEAR_SPAN)
    logger.info("集計方法         : 単純な部分文字列マッチによる出現回数カウント")
    logger.info(
        "注意             : 親⇔子や類似語（例: 雷⊃雷雨、暴風⊃暴風域、気温⊃最高気温）は重複してカウントされる。"
    )
    logger.info("")

    text, files = load_text_from_dir(INPUT_DIR)
    logger.info("入力ファイル数   : %d 個（%s 〜 %s）", len(files), files[0].name, files[-1].name)
    logger.info("テキスト総文字数 : %s 文字", f"{len(text):,}")
    logger.info("")

    all_results: list[tuple[str, str, int, list[tuple[Keyword, int]]]] = []
    section_totals: list[tuple[str, int, int]] = []

    for section in SECTIONS:
        logger.info("=" * 78)
        logger.info("【%s】", section.title)
        if section.description:
            logger.info("    %s", section.description)
        logger.info("=" * 78)

        section_main_total = 0
        section_keyword_count = 0

        # まず全キーワードをカウントしてからランキング順に並び替える
        scored: list[tuple[Keyword, int, list[tuple[Keyword, int]]]] = []
        for kw in section.keywords:
            main_count = count_keyword(text, kw.name)
            section_main_total += main_count
            section_keyword_count += 1

            child_results: list[tuple[Keyword, int]] = []
            for child in kw.children:
                child_count = count_keyword(text, child.name)
                child_results.append((child, child_count))
                section_keyword_count += 1

            # 子キーワードは出現回数の降順に並べる
            child_results.sort(key=lambda pair: pair[1], reverse=True)
            scored.append((kw, main_count, child_results))

        # 親キーワードも出現回数の降順に並べる
        scored.sort(key=lambda triple: triple[1], reverse=True)

        for kw, main_count, child_results in scored:
            all_results.append((section.title, kw.name, main_count, child_results))

            children_sum = sum(c for _, c in child_results)

            logger.info("")
            if kw.children:
                coverage = (children_sum / main_count * 100) if main_count > 0 else 0.0
                logger.info(
                    "%s：%d件（年平均：%.1f） [子キーワード合計: %d件 / カバー率: %.1f%%]",
                    kw.name,
                    main_count,
                    annual_average(main_count),
                    children_sum,
                    coverage,
                )
            else:
                logger.info(
                    "%s：%d件（年平均：%.1f）",
                    kw.name,
                    main_count,
                    annual_average(main_count),
                )
            if kw.note:
                logger.info("    解説: %s", kw.note)
            for child, child_count in child_results:
                logger.info(
                    "    └ %s：%d件（年平均：%.1f）",
                    child.name,
                    child_count,
                    annual_average(child_count),
                )
                if child.note:
                    logger.info("         解説: %s", child.note)

        section_totals.append((section.title, section_main_total, section_keyword_count))
        logger.info("")

    logger.info("=" * 78)
    logger.info("セクション別サマリ（親キーワードのみの単純合計）")
    logger.info("=" * 78)
    for title, total, count in section_totals:
        logger.info(
            "  %-40s  親計 %7d 件 / キーワード %3d 個",
            title,
            total,
            count,
        )

    flat_ranking: list[tuple[str, int]] = []
    for _, name, total, children in all_results:
        flat_ranking.append((name, total))
        for child, ctotal in children:
            flat_ranking.append((child.name, ctotal))

    flat_ranking.sort(key=lambda x: x[1], reverse=True)

    logger.info("")
    logger.info("=" * 78)
    logger.info("出現回数ランキング（全キーワード・親子区別なし）")
    logger.info("=" * 78)
    for rank, (name, total) in enumerate(flat_ranking, start=1):
        logger.info(
            "  %3d位  %-22s  %7d 件  （年平均 %7.1f 件）",
            rank,
            name,
            total,
            annual_average(total),
        )

    logger.info("")
    logger.info("=" * 78)
    logger.info("出現が0件のキーワード（テキスト中に登場しなかったもの）")
    logger.info("=" * 78)
    zeros = [name for name, total in flat_ranking if total == 0]
    if zeros:
        for name in zeros:
            logger.info("  - %s", name)
    else:
        logger.info("  （該当なし）")

    logger.info("")
    logger.info("=" * 78)
    logger.info("総キーワード数: %d 個", len(flat_ranking))
    logger.info(
        "総出現件数（重複あり、単純合算）: %d 件",
        sum(t for _, t in flat_ranking),
    )
    logger.info("0件キーワード数: %d 個", len(zeros))
    logger.info("ヒットしたユニークキーワード数: %d 個", len(flat_ranking) - len(zeros))
    logger.info("集計が完了しました。詳細は %s を参照してください。", LOG_FILE.name)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
