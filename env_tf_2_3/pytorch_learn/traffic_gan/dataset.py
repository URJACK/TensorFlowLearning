import dpkt
import numpy as np

# 从pcap包抓取时间间隔的数组
SPAN_ALPHA = 0.4


def main():
    f = open('D:\\Storage\\wiresharkSave\\demo1.pcap', 'rb')
    readBuffer = dpkt.pcap.Reader(f)
    spanMap = {}
    timeMap = {}
    for ts, buf in readBuffer:
        eth = dpkt.ethernet.Ethernet(buf)
        ip = eth.data
        if not hasattr(ip, 'tcp'):
            continue
        bufferStrA = ""
        bufferStrB = ""
        keyStr = ""
        for data in ip.src:
            bufferStrA += str(data)
        for data in ip.dst:
            bufferStrA += str(data)
        for data in ip.dst:
            bufferStrB += str(data)
        for data in ip.src:
            bufferStrB += str(data)
        # strA 源IP 到 目的IP
        # strB 目的IP 到 源IP

        if bufferStrB > bufferStrA:
            keyStr = bufferStrB
        else:
            keyStr = bufferStrA

        if timeMap.get(keyStr) is None:
            timeMap[keyStr] = ts
            # 第一次，默认产生一个流量
            getSpanList(keyStr, spanMap, True)
        else:
            # 两个 具有相同 源IP地址 和 目的IP地址的 TCP包，的时间差
            span = ts - timeMap.get(keyStr)
            if span > SPAN_ALPHA:
                # 时间差太大了 本次span不作为计算 并且应当产生一个新的流量
                timeMap[keyStr] = ts
                getSpanList(keyStr, spanMap, True)
            else:
                # 时间差在SPAN_ALPHA内 当前“流量集”的最新“流量”需要添加一个新的“时间差”
                spanList = getSpanList(keyStr, spanMap)
                spanList.append(span)
    print(spanMap)


# 通过keyStr从spanMap中取得“流量集”
# 如果adder为True，那么意味着“流量集”需要添加一个新的“流量”
#         为False，那么意味着当前，从“流量集”中取得当前最新的“流量”
def getSpanList(keyStr, spanMap, adder=False):
    spanListList = spanMap.get(keyStr)
    if spanListList is None:
        spanListList = []
        spanMap[keyStr] = spanListList
    spanList: list
    if adder:
        spanListList.append([])
    spanList = spanListList[len(spanListList) - 1]
    return spanList


if __name__ == '__main__':
    main()
