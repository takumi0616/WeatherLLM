import psycopg2


def connect_amedas_tsdb():
    username = "read_only"
    password = "read_only"
    hostname = "192.168.110.74"
    port = "5432"
    database = "amedas_tsdb"
    dsn = f"postgresql://{username}:{password}@{hostname}:{port}/{database}"
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    return conn, cur
