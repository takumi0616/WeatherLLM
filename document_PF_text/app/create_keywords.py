def generate_keywords(weather_data):
    mintemp_keyword = []
    maxtemp_keyword = []
    prcp_day_keyword = []
    maxprcp_1h_keyword = []
    wind_keyword = []
    snow_keyword = []
    pressure_pattern_keyword = []

    # mintemp_anom_normal（最低気温、平年値との差）
    mintemp_anom_normal = weather_data.get("mintemp_anom_normal")
    if mintemp_anom_normal and "value" in mintemp_anom_normal:
        mintemp_value = mintemp_anom_normal["value"]
        if mintemp_value < -5:
            mintemp_keyword.append("平年よりも大幅に低い")
        elif -5 <= mintemp_value < -1.5:
            mintemp_keyword.append("平年よりも低い")
        elif -1.5 <= mintemp_value <= 1.5:
            mintemp_keyword.append("平年並み")
        elif 1.5 < mintemp_value < 5:
            mintemp_keyword.append("平年よりも高い")
        elif mintemp_value >= 5:
            mintemp_keyword.append("平年よりも大幅に高い")

    # mintemp_anom_prev（最低気温、前日値との差）
    mintemp_anom_prev = weather_data.get("mintemp_anom_prev")
    if mintemp_anom_prev and "value" in mintemp_anom_prev:
        mintemp_prev_value = mintemp_anom_prev["value"]
        if mintemp_prev_value < -5:
            mintemp_keyword.append("きのうよりも大幅に低い")
        elif -5 <= mintemp_prev_value < -1.5:
            mintemp_keyword.append("きのうよりも低い")
        elif -1.5 <= mintemp_prev_value <= 1.5:
            mintemp_keyword.append("きのうと同じくらい")
        elif 1.5 < mintemp_prev_value < 5:
            mintemp_keyword.append("きのうよりも高い")
        elif mintemp_prev_value >= 5:
            mintemp_keyword.append("きのうよりも大幅に高い")

    # maxtemp_anom_normal（最高気温、平年値との差）
    maxtemp_anom_normal = weather_data.get("maxtemp_anom_normal")
    if maxtemp_anom_normal and "value" in maxtemp_anom_normal:
        maxtemp_value = maxtemp_anom_normal["value"]
        if maxtemp_value < -5:
            maxtemp_keyword.append("平年よりも大幅に低い")
        elif -5 <= maxtemp_value < -1.5:
            maxtemp_keyword.append("平年よりも低い")
        elif -1.5 <= maxtemp_value <= 1.5:
            maxtemp_keyword.append("平年並み")
        elif 1.5 < maxtemp_value < 5:
            maxtemp_keyword.append("平年よりも高い")
        elif maxtemp_value >= 5:
            maxtemp_keyword.append("平年よりも大幅に高い")

    # maxtemp_anom_prev（最高気温、前日値との差）
    maxtemp_anom_prev = weather_data.get("maxtemp_anom_prev")
    if maxtemp_anom_prev and "value" in maxtemp_anom_prev:
        maxtemp_prev_value = maxtemp_anom_prev["value"]
        if maxtemp_prev_value < -5:
            maxtemp_keyword.append("きのうよりも大幅に低い")
        elif -5 <= maxtemp_prev_value < -1.5:
            maxtemp_keyword.append("きのうよりも低い")
        elif -1.5 <= maxtemp_prev_value <= 1.5:
            maxtemp_keyword.append("きのうと同じくらい")
        elif 1.5 < maxtemp_prev_value < 5:
            maxtemp_keyword.append("きのうよりも高い")
        elif maxtemp_prev_value >= 5:
            maxtemp_keyword.append("きのうよりも大幅に高い")

    # prcp_day（1日の降水量）
    prcp_day = weather_data.get("prcp_day")
    if prcp_day and "value" in prcp_day:
        prcp_day_value = prcp_day["value"]
        if prcp_day_value >= 30:
            prcp_day_keyword.append("大雨")
        elif prcp_day_value <= 1:
            prcp_day_keyword.append("小雨")

    # maxprcp_1h（最大1時間降水量）
    maxprcp_1h = weather_data.get("maxprcp_1h")
    if maxprcp_1h and "value" in maxprcp_1h:
        maxprcp_1h_value = maxprcp_1h["value"]
        maxprcp_1h_time = maxprcp_1h.get("time", "none")

        # 時間の部分だけ取り出す
        if maxprcp_1h_time != "none" and 0 < maxprcp_1h_value:
            hour = int(maxprcp_1h_time.split()[1].split(":")[0])

            # 時間に基づいて適切な言葉を追加
            if 0 <= hour <= 2:
                maxprcp_1h_keyword.append("未明")
            elif 3 <= hour <= 5:
                maxprcp_1h_keyword.append("明け方")
            elif 6 <= hour <= 8:
                maxprcp_1h_keyword.append("朝")
            elif 9 <= hour <= 11:
                maxprcp_1h_keyword.append("昼前")
            elif 12 <= hour <= 14:
                maxprcp_1h_keyword.append("昼過ぎ")
            elif 15 <= hour <= 17:
                maxprcp_1h_keyword.append("夕方")
            elif 18 <= hour <= 20:
                maxprcp_1h_keyword.append("夜のはじめ頃")
            elif 21 <= hour <= 24:
                maxprcp_1h_keyword.append("夜遅く")

        # 降水量の強さに基づく言葉を追加
        if maxprcp_1h_value >= 80:
            maxprcp_1h_keyword.append("猛烈な雨")
        elif 50 <= maxprcp_1h_value < 80:
            maxprcp_1h_keyword.append("非常に激しい雨")
        elif 30 <= maxprcp_1h_value < 50:
            maxprcp_1h_keyword.append("激しい雨")
        elif 20 <= maxprcp_1h_value < 30:
            maxprcp_1h_keyword.append("強い雨")
        elif 10 <= maxprcp_1h_value < 20:
            maxprcp_1h_keyword.append("やや強い雨")
        elif 0 < maxprcp_1h_value < 3:
            maxprcp_1h_keyword.append("弱い雨")

    # wind（1日の平均風速）
    wind = weather_data.get("wind")
    if wind and "value" in wind:
        wind_value = wind["value"]
        if 5 <= wind_value < 10:
            wind_keyword.append("穏やかな風")
        elif 10 <= wind_value < 15:
            wind_keyword.append("やや強い風")
        elif 15 <= wind_value < 20:
            wind_keyword.append("強い風")
        elif 20 <= wind_value < 30:
            wind_keyword.append("非常に強い風")
        elif wind_value >= 30:
            wind_keyword.append("猛烈な風")

        # 風向を日本語に変換して wind_keyword に追加
        wind_dir = wind.get("dir", "")
        if wind_dir:
            # 英語の風向を日本語に変換するマッピング
            direction_mapping = {
                'N': '北',
                'NNE': '北',
                'NE': '北東',
                'ENE': '北東',
                'E': '東',
                'ESE': '南東',
                'SE': '南東',
                'SSE': '南',
                'S': '南',
                'SSW': '南',
                'SW': '南西',
                'WSW': '南西',
                'W': '西',
                'WNW': '北西',
                'NW': '北西',
                'NNW': '北',
            }
            # 風向コードをそのまま使用してマッピング
            japanese_direction = direction_mapping.get(wind_dir)
            if japanese_direction:
                wind_keyword.append(japanese_direction)

    # snow（1日の降雪量）
    snow = weather_data.get("snow")
    if snow and "value" in snow:
        snow_value = snow["value"]
        if snow_value >= 20:
            snow_keyword.append("大雪")
        elif snow_value >= 3:
            snow_keyword.append("強い雪")
        elif 0 < snow_value <= 1:
            snow_keyword.append("弱い雪")

    # pressuer_pattern の処理を追加
    pressuer_pattern = weather_data.get("pressuer_pattern")
    if pressuer_pattern is not None:
        # pressuer_pattern は整数値
        if pressuer_pattern == 1:
            pressure_pattern_keyword.append("冬型の気圧配置")
        elif pressuer_pattern == 2:
            pressure_pattern_keyword.append("夏型の気圧配置")
        elif pressuer_pattern == 3:
            pressure_pattern_keyword.append("梅雨型の気圧配置")
        elif pressuer_pattern == 4:
            pressure_pattern_keyword.append("台風")

    return {
        "mintemp_keyword": mintemp_keyword,
        "maxtemp_keyword": maxtemp_keyword,
        "prcp_day_keyword": prcp_day_keyword,
        "maxprcp_1h_keyword": maxprcp_1h_keyword,
        "wind_keyword": wind_keyword,
        "snow_keyword": snow_keyword,
        "pressure_pattern_keyword": pressure_pattern_keyword
    }
