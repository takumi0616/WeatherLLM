#!/home/devel/miniconda3/envs/weather310/bin/python

import json

from app import tokyocabinet


def main():
    TC = tokyocabinet.TT("192.168.110.34", 17851)  # stb-sub
    info = json.loads(TC.get("info"))
    dic = {"ini_time": info["ini_time"]}
    print("Content-type: application/json\n\n")
    print(json.dumps(dic))


main()
