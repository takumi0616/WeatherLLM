import ctypes
import sys

LC = ctypes.cdll.LoadLibrary("libc.so.6")

TC = ctypes.cdll.LoadLibrary("/usr/local/lib/libtokyocabinet.so")

TC.tchdbnew.restype = ctypes.c_ulong

TC.tchdbopen.argtypes = [ctypes.c_ulong, ctypes.c_char_p, ctypes.c_uint]
TC.tchdbopen.restype = ctypes.c_bool

TC.tchdbget2.argtypes = [ctypes.c_ulong, ctypes.c_char_p]
TC.tchdbget2.restype = ctypes.POINTER(ctypes.c_char_p)
TC.tchdbput2.argtypes = [ctypes.c_ulong, ctypes.c_char_p, ctypes.c_char_p]
TC.tchdbput2.restype = ctypes.c_bool

TC.tchdbvanish.argtypes = [ctypes.c_ulong]
TC.tchdbvanish.restype = ctypes.c_bool

TC.tchdbclose.argtypes = [ctypes.c_ulong]
TC.tchdbclose.restype = ctypes.c_bool
TC.tchdbdel.argtypes = [ctypes.c_ulong]

TC.tchdbecode.argtypes = [ctypes.c_ulong]
TC.tchdbecode.restype = ctypes.c_uint
TC.tchdberrmsg.argtypes = [ctypes.c_uint]
TC.tchdberrmsg.restype = ctypes.c_char_p


class TCH:
    def __init__(self, tchfile="", opt=""):
        self.hdb = TC.tchdbnew()
        if tchfile != "":
            self.open(tchfile, opt)

    def open(self, tchfile, opt=""):
        if opt == "w":
            flg = 6  # write:2 + create:4
        else:
            flg = 1  # read:1
        ret = TC.tchdbopen(self.hdb, tchfile.encode("utf-8"), flg)
        self.errck(ret)
        return ret

    def get(self, key):
        res_p = TC.tchdbget2(self.hdb, key.encode("utf-8"))
        res = ctypes.cast(res_p, ctypes.c_char_p).value
        if res is None:
            return None
        LC.free(res_p)
        return res.decode("utf-8")

    def put(self, key, val):
        ret = TC.tchdbput2(self.hdb, key.encode("utf-8"), val.encode("utf-8"))
        self.errck(ret)
        return ret

    def vanish(self):
        ret = TC.tchdbvanish(self.hdb)
        self.errck(ret)
        return ret

    def close(self):
        ret = TC.tchdbclose(self.hdb)
        self.errck(ret)
        TC.tchdbdel(self.hdb)
        return ret

    def errck(self, ret):
        if not ret:
            ecode = TC.tchdbecode(self.hdb)
            # print(TC.tchdberrmsg(ecode).decode("utf-8"), file=sys.stderr)


TC.tctdbnew.restype = ctypes.c_ulong

TC.tctdbopen.argtypes = [ctypes.c_ulong, ctypes.c_char_p, ctypes.c_uint]
TC.tctdbopen.restype = ctypes.c_bool

TC.tctdbget3.argtypes = [ctypes.c_ulong, ctypes.c_char_p]
TC.tctdbget3.restype = ctypes.POINTER(ctypes.c_char_p)
TC.tctdbput3.argtypes = [ctypes.c_ulong, ctypes.c_char_p, ctypes.c_char_p]
TC.tctdbput3.restype = ctypes.c_bool

TC.tctdbvanish.argtypes = [ctypes.c_ulong]
TC.tctdbvanish.restype = ctypes.c_bool

TC.tctdbclose.argtypes = [ctypes.c_ulong]
TC.tctdbclose.restype = ctypes.c_bool
TC.tctdbdel.argtypes = [ctypes.c_ulong]

TC.tctdbecode.argtypes = [ctypes.c_ulong]
TC.tctdbecode.restype = ctypes.c_uint
TC.tctdberrmsg.argtypes = [ctypes.c_uint]
TC.tctdberrmsg.restype = ctypes.c_char_p


class TCT:
    def __init__(self, tctfile="", opt=""):
        self.tdb = TC.tctdbnew()
        if tctfile != "":
            self.open(tctfile, opt)

    def open(self, tctfile, opt=""):
        if opt == "w":
            flg = 6  # write:2 + create:4
        else:
            flg = 1  # read:1
        ret = TC.tctdbopen(self.tdb, tctfile.encode("utf-8"), flg)
        self.errck(ret)
        return ret

    def get(self, key, col="all"):
        res_p = TC.tctdbget3(self.tdb, key.encode("utf-8"))
        res_t = ctypes.cast(res_p, ctypes.c_char_p).value
        if res_t is None:
            return None
        LC.free(res_p)
        tmp = res_t.decode("utf-8").split("\t")
        res = {}
        for i in range(0, len(tmp), 2):
            res[tmp[i]] = tmp[i + 1]
        if col == "all":
            return res
        else:
            return res[col]

    def put(self, key, vals):
        val = ""
        for k, v in vals.items():
            val += f"\t{k}\t{v}"
        val = val[1:]
        ret = TC.tctdbput3(self.tdb, key.encode("utf-8"), val.encode("utf-8"))
        self.errck(ret)
        return ret

    def vanish(self):
        ret = TC.tctdbvanish(self.tdb)
        self.errck(ret)
        return ret

    def close(self):
        ret = TC.tctdbclose(self.tdb)
        self.errck(ret)
        TC.tctdbdel(self.tdb)
        return ret

    def errck(self, ret):
        if not ret:
            ecode = TC.tctdbecode(self.tdb)
            # print(TC.tctdberrmsg(ecode).decode("utf-8"), file=sys.stderr)


TR = ctypes.cdll.LoadLibrary("/usr/local/lib/libtokyotyrant.so")

TR.tcrdbnew.restype = ctypes.c_ulong

TR.tcrdbopen.argtypes = [ctypes.c_ulong, ctypes.c_char_p, ctypes.c_uint]
TR.tcrdbopen.restype = ctypes.c_bool

TR.tcrdbget2.argtypes = [ctypes.c_ulong, ctypes.c_char_p]
TR.tcrdbget2.restype = ctypes.POINTER(ctypes.c_char_p)
TR.tcrdbput2.argtypes = [ctypes.c_ulong, ctypes.c_char_p, ctypes.c_char_p]
TR.tcrdbput2.restype = ctypes.c_bool

TR.tcrdbclose.argtypes = [ctypes.c_ulong]
TR.tcrdbclose.restype = ctypes.c_bool
TR.tcrdbdel.argtypes = [ctypes.c_ulong]

TR.tcrdbecode.argtypes = [ctypes.c_ulong]
TR.tcrdbecode.restype = ctypes.c_uint
TR.tcrdberrmsg.argtypes = [ctypes.c_uint]
TR.tcrdberrmsg.restype = ctypes.c_char_p


class TT:
    def __init__(self, ip="", port: int = 17851):
        if port != "":
            self.open(ip, port)

    def open(self, ip, port):
        self.rdb = TR.tcrdbnew()
        ret = TR.tcrdbopen(self.rdb, ip.encode("utf-8"), port)
        self.errck(ret)
        return ret

    def get(self, key):
        res_p = TR.tcrdbget2(self.rdb, key.encode("utf-8"))
        res = ctypes.cast(res_p, ctypes.c_char_p).value
        if res is None:
            return None
        LC.free(res_p)
        return res.decode("utf-8")

    def put(self, key, val):
        ret = TR.tcrdbput2(self.rdb, key.encode("utf-8"), val.encode("utf-8"))
        self.errck(ret)
        return ret

    def close(self):
        ret = TR.tcrdbclose(self.rdb)
        self.errck(ret)
        TR.tcrdbdel(self.rdb)
        return ret

    def errck(self, ret):
        if not ret:
            ecode = TR.tcrdbecode(self.rdb)
            # print(TR.tcrdberrmsg(ecode).decode("utf-8"), file=sys.stderr)
