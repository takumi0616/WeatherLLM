def create_weather_text(weather_data: dict) -> str:
    text = ""
    date = weather_data["info"]["date"]
    weather_info = weather_data.get("weather", {})

    # 天気予報情報の追加
    if weather_info:
        text += f"{date} の天気の予報\\n"
        if "com" in weather_info:
            text += (
                f"{weather_info['com']} (重要度: {weather_info.get('importance', 0)})\\n"
            )
        if "list" in weather_info:
            for key, value in weather_info["list"].items():
                text += f"{value}\\n"
        # Ws_textの追加
        if "Ws_text" in weather_info:
            text += "時間帯ごとの天気の予報:\\n"
            for ws in weather_info["Ws_text"]:
                text += f"{ws['time']}: {ws['weather']}\\n"
            text += "\\n"

    # 気圧配置のキーワードを追加
    pressure_pattern_keyword = weather_data.get("pressure_pattern_keyword", [])
    if pressure_pattern_keyword:
        text += f"「天気」に関する天気コメントに必要な言葉: {', '.join(pressure_pattern_keyword)} (重要度: 2.00)\\n\\n"

    # Minimum Temperature
    text += f"{date} 最低気温\n"
    if "mintemp_keyword" in weather_data and weather_data["mintemp_keyword"]:
        text += f"最低気温のコメント作成に必要な言葉: {', '.join(weather_data['mintemp_keyword'])}\n"
    text += f"最低気温{weather_data['mintemp']['value']}℃を"
    text += "観測\n" if weather_data["mintemp"]["kind"] == "observation" else "予想\n"
    text += f"平年差{weather_data['mintemp_anom_normal']['value']}℃ (重要度: {weather_data['mintemp_anom_normal']['importance']:.2f})\n"
    if weather_data.get("mintemp_anom_prev"):
        text += f"前日差{weather_data['mintemp_anom_prev']['value']}℃ (重要度: {weather_data['mintemp_anom_prev']['importance']:.2f})\n"
    text += "\n"

    # Maximum Temperature
    text += f"{date} 最高気温\n"
    if "maxtemp_keyword" in weather_data and weather_data["maxtemp_keyword"]:
        text += f"最高気温のコメント作成に必要な言葉: {', '.join(weather_data['maxtemp_keyword'])}\n"
    text += f"最高気温{weather_data['maxtemp']['value']}℃を"
    text += "観測\n" if weather_data["maxtemp"]["kind"] == "observation" else "予想\n"
    text += f"平年差{weather_data['maxtemp_anom_normal']['value']}℃ (重要度: {weather_data['maxtemp_anom_normal']['importance']:.2f})\n"
    if weather_data.get("maxtemp_anom_prev"):
        text += f"前日差{weather_data['maxtemp_anom_prev']['value']}℃ (重要度: {weather_data['maxtemp_anom_prev']['importance']:.2f})\n"
    text += "\n"

    # Prcp
    text += f"{date} 1日の降水量\n"
    if "prcp_day_keyword" in weather_data and weather_data["prcp_day_keyword"]:
        text += f"降水量のコメント作成に必要な言葉: {', '.join(weather_data['prcp_day_keyword'])}\n"
    text += f"降水量{weather_data['prcp_day']['value']}mmを"
    text += "観測" if weather_data["prcp_day"]["kind"] == "observation" else "予想"
    text += f" (重要度: {weather_data['prcp_day']['importance']:.2f})\n"
    text += "\n"

    # Maximum 1-Hour Precipitation
    text += f"{date} 最大1時間降水量\n"
    if "maxprcp_1h_keyword" in weather_data and weather_data["maxprcp_1h_keyword"]:
        text += f"最大1時間降水量のコメント作成に必要な言葉: {', '.join(weather_data['maxprcp_1h_keyword'])}\n"
    text += f"最大1時間降水量{weather_data['maxprcp_1h']['value']}mmを"
    text += "観測" if weather_data["maxprcp_1h"]["kind"] == "observation" else "予想"
    text += f" (重要度: {weather_data['maxprcp_1h']['importance']:.2f})\n"
    if (
        "time" in weather_data["maxprcp_1h"]
        and weather_data["maxprcp_1h"]["time"] != "none"
    ):
        text += f"発生時刻: {weather_data['maxprcp_1h']['time']}\n"
    text += "\n"

    # Wind
    text += f"{date} 平均風速\n"
    if "wind_keyword" in weather_data and weather_data["wind_keyword"]:
        text += f"平均風速のコメント作成に必要な言葉: {', '.join(weather_data['wind_keyword'])}\n"
    text += f"平均風速{weather_data['wind']['value']}m/sを"
    text += "観測" if weather_data["wind"]["kind"] == "observation" else "予想"
    text += f" (重要度: {weather_data['wind']['importance']:.2f})\n"
    text += "\n"

    # Snow
    text += f"{date} 積雪量\n"
    if "snow_keyword" in weather_data and weather_data["snow_keyword"]:
        text += f"積雪量キーワード: {', '.join(weather_data['snow_keyword'])}\n"
    text += f"積雪量{weather_data['snow']['value']}cmを"
    text += "観測" if weather_data["snow"]["kind"] == "observation" else "予想"
    text += f" (重要度: {weather_data['snow']['importance']:.2f})\n"

    print(f"{date} \n\n{text}")
    return text
