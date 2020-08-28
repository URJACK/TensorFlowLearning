import numpy as np
import pandas
from keras.utils import to_categorical

# 浮点数转整数通用误差
e = 0.000000001


class DataGenerator:
    steps: int
    protocol_dict: dict
    service_dict: dict
    flag_dict: dict
    label_dict: dict
    protocol_list: list
    service_list: list
    flag_list: list
    label_list: list

    protocol_types: int
    service_types: int
    flag_types: int
    label_types: int
    features_num: int  # 特征个数(不包括标签)
    onehot_features_num: int  # 需要被onehot化的对象个数()

    # 给定地址加载数据
    def __init__(self, source_file: str, batch_size: int, noise_dim: int, label_class_num: int = 23):
        self.features_num = 41
        self.onehot_features_num = 3
        self.protocol_types = 3
        self.service_types = 70
        self.flag_types = 11
        self.label_types = label_class_num

        self.MAX_DURATION: int = 58329
        self.MAX_SRC_BYTES: int = 1379963888
        self.MAX_DST_BYTES: int = 1309937401
        self.MAX_WRONG_FRAGMENT: int = 3
        self.MAX_URGENT: int = 14

        self.MAX_HOT: int = 101
        self.MAX_NUM_FAILED_LOGINS: int = 5
        self.MAX_NUM_COMPROMISED: int = 7479
        self.MAX_NUM_ROOT: int = 7468
        self.MAX_NUM_FILE_CREATIONS: int = 100
        self.MAX_NUM_SHELLS: int = 5
        self.MAX_NUM_ACCESS_FILES: int = 9
        self.MAX_NUM_OUTBOUND_CMDS: int = 10

        self.MAX_COUNT: int = 511
        self.MAX_SRV_COUNT: int = 511
        self.MAX_SERROR_RATE: int = 1
        self.MAX_SRV_SERROR_RATE: int = 1
        self.MAX_RERROR_RATE: int = 1
        self.MAX_SRV_RERROR_RATE: int = 1
        self.MAX_SAME_SRV_RATE: int = 1
        self.MAX_DIFF_SRV_RATE: int = 1
        self.MAX_SRV_DIFF_HOST_RATE: int = 1

        self.MAX_DST_HOST_COUNT: int = 255
        self.MAX_DST_HOST_SRV_COUNT: int = 255
        self.MAX_DST_HOST_SAME_SRV_RATE: int = 1
        self.MAX_DST_HOST_DIFF_SRV_RATE: int = 1
        self.MAX_DST_HOST_SAME_SRC_PORT_RATE: int = 1
        self.MAX_DST_HOST_SRV_DIFF_HOST_RATE: int = 1
        self.MAX_DST_HOST_SERROR_RATE: int = 1
        self.MAX_DST_HOST_SRV_SERROR_RATE: int = 1
        self.MAX_DST_HOST_RERROR_RATE: int = 1
        self.MAX_DST_HOST_SRV_RERROR_RATE: int = 1

        datasets = pandas.read_csv(source_file)
        self.datasets = np.array(datasets)
        dataLen = self.datasets.shape[0]
        self.dataLen = dataLen
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.label_class_num = label_class_num
        self.steps = dataLen // batch_size
        self.feature_dim = self.features_num - self.onehot_features_num
        self.feature_dim = self.feature_dim + self.protocol_types + self.service_types + self.flag_types
        self.onehot_offset = 0  # 对部分属性进行one-hot编码后，整体偏差的列数
        self.preprocess()
        if dataLen % batch_size != 0:
            self.steps = self.steps + 1

    def __len__(self):
        return self.steps

    def gan_iter(self):
        while True:
            begin: int = 0
            end: int = self.batch_size
            while end <= self.dataLen:
                data = self.datasets[begin:end]
                x, y = self.analysis(data)
                yield x, y
            if begin < self.dataLen:
                data = self.datasets[begin:self.dataLen]
                x, y = self.analysis(data)
                yield x, y

    def classifier_iter(self):
        analysis_type: int = 2
        while True:
            begin: int = 0
            end: int = self.batch_size
            while end <= self.dataLen:
                data = self.datasets[begin:end]
                x, y = self.analysis(data, analysis_type)
                yield x, y
            if begin < self.dataLen:
                data = self.datasets[begin:self.dataLen]
                x, y = self.analysis(data, analysis_type)
                yield x, y

    def analysis(self, data, analysis_type: int = 1):
        # index from 0 to 41
        stopColBeginIndex: int = 1
        stopColumnEndIndex: int = 3
        labelIndex: int = self.features_num  # labelIndex 同时也是输入特征的个数 features_num
        # 抛弃protocol : 1,  service : 2  ,  flag : 3
        x = np.concatenate([data[:, 0:stopColBeginIndex], data[:, stopColumnEndIndex + 1:labelIndex]], axis=1)
        protocol_onehot = to_categorical(data[:, 1], num_classes=self.protocol_types)
        service_onehot = to_categorical(data[:, 2], num_classes=self.service_types)
        flag_onehot = to_categorical(data[:, 3], num_classes=self.flag_types)

        x = np.concatenate([x, protocol_onehot, service_onehot, flag_onehot], axis=1)
        x = x.astype(np.float)
        if analysis_type == 1:
            z = np.random.randn(x.shape[0], self.noise_dim)
            # x 的维度等于 (41 - 3) + (3 + 70 + 11) == 122
            return [x, z], None
        elif analysis_type == 2:
            label_onehot = to_categorical(data[:, labelIndex], num_classes=self.label_class_num)
            return x, label_onehot
        else:
            return None, None

    def preprocess(self):
        self.protocol_dict = {'tcp': 0, 'udp': 1, 'icmp': 2}
        self.service_dict = {'http': 0, 'smtp': 1, 'finger': 2, 'domain_u': 3, 'auth': 4, 'telnet': 5, 'ftp': 6,
                             'eco_i': 7, 'ntp_u': 8, 'ecr_i': 9, 'other': 10, 'private': 11, 'pop_3': 12,
                             'ftp_data': 13, 'rje': 14, 'time': 15, 'mtp': 16, 'link': 17, 'remote_job': 18,
                             'gopher': 19, 'ssh': 20, 'name': 21, 'whois': 22, 'domain': 23, 'login': 24,
                             'imap4': 25, 'daytime': 26, 'ctf': 27, 'nntp': 28, 'shell': 29, 'IRC': 30,
                             'nnsp': 31, 'http_443': 32, 'exec': 33, 'printer': 34, 'efs': 35, 'courier': 36,
                             'uucp': 37, 'klogin': 38, 'kshell': 39, 'echo': 40, 'discard': 41, 'systat': 42,
                             'supdup': 43, 'iso_tsap': 44, 'hostnames': 45, 'csnet_ns': 46, 'pop_2': 47,
                             'uucp_path': 48,
                             'netbios_ns': 49, 'netbios_ssn': 50, 'netbios_dgm': 51, 'sql_net': 52, 'vmnet': 53,
                             'bgp': 54,
                             'Z39_50': 55, 'ldap': 56, 'netstat': 57, 'urh_i': 58, 'X11': 59, 'urp_i': 60,
                             'pm_dump': 61, 'tftp_u': 62, 'tim_i': 63, 'red_i': 64, 'sunrpc': 65, 'aol': 66,
                             'harvest': 67, 'http_2784': 68, 'http_8001': 69}
        self.flag_dict = {'SF': 0, 'S1': 1, 'REJ': 2, 'S2': 3, 'S0': 4, 'S3': 5,
                          'RSTO': 6, 'RSTR': 7, 'RSTOS0': 8, 'OTH': 9, 'SH': 10}
        self.label_dict = {
            'normal.': 0, 'buffer_overflow.': 1, 'loadmodule.': 2, 'perl.': 3, 'neptune.': 4,
            'smurf.': 5, 'guess_passwd.': 6, 'pod.': 7, 'teardrop.': 8, 'portsweep.': 9,
            'ipsweep.': 10, 'land.': 11, 'ftp_write.': 12, 'back.': 13, 'imap.': 14,
            'satan.': 15, 'phf.': 16, 'nmap.': 17, 'multihop.': 18, 'warezmaster.': 19,
            'warezclient.': 20, 'spy.': 21, 'rootkit.': 22
        }
        self.protocol_list = ['tcp', 'udp', 'icmp']
        self.service_list = ['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp', 'eco_i', 'ntp_u',
                             'ecr_i', 'other', 'private', 'pop_3', 'ftp_data', 'rje', 'time', 'mtp', 'link',
                             'remote_job', 'gopher', 'ssh', 'name', 'whois', 'domain', 'login', 'imap4',
                             'daytime', 'ctf', 'nntp', 'shell', 'IRC', 'nnsp', 'http_443', 'exec', 'printer',
                             'efs', 'courier', 'uucp', 'klogin', 'kshell', 'echo', 'discard', 'systat',
                             'supdup', 'iso_tsap', 'hostnames', 'csnet_ns', 'pop_2', 'sunrpc', 'uucp_path',
                             'netbios_ns', 'netbios_ssn', 'netbios_dgm', 'sql_net', 'vmnet', 'bgp', 'Z39_50',
                             'ldap', 'netstat', 'urh_i', 'X11', 'urp_i', 'pm_dump', 'tftp_u', 'tim_i', 'red_i',
                             'aol', 'harvest', 'http_2784', 'http_8001']
        self.flag_list = ['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'OTH', 'SH']
        self.label_list = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.', 'guess_passwd.',
                           'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.',
                           'satan.', 'phf.', 'nmap.', 'multihop.''warezmaster.', 'warezclient.', 'spy.', 'rootkit.']
        self.datasets[:, 1] = self.preprocess_protocol(self.datasets[:, 1], False)
        self.datasets[:, 2] = self.preprocess_service(self.datasets[:, 2], False)
        self.datasets[:, 3] = self.preprocess_flag(self.datasets[:, 3], False)
        self.datasets[:, self.features_num] = self.preprocess_label(self.datasets[:, self.features_num], False)
        self.preprocess_normalize_feature()

    # protocol_type ['tcp' 'udp' 'icmp']
    def preprocess_protocol(self, data: np.ndarray, toStr: bool):
        datalist = data.tolist()
        index = 0
        for ele in datalist:
            if not toStr:
                datalist[index] = self.protocol_dict[ele]
            else:
                datalist[index] = self.protocol_list[ele]
            index = index + 1
        data = np.array(datalist)
        return data

    # service ['http' 'smtp' 'finger' 'domain_u' 'auth' 'telnet' 'ftp' 'eco_i' 'ntp_u'
    #  'ecr_i' 'other' 'private' 'pop_3' 'ftp_data' 'rje' 'time' 'mtp' 'link'
    #  'remote_job' 'gopher' 'ssh' 'name' 'whois' 'domain' 'login' 'imap4'
    #  'daytime' 'ctf' 'nntp' 'shell' 'IRC' 'nnsp' 'http_443' 'exec' 'printer'
    #  'efs' 'courier' 'uucp' 'klogin' 'kshell' 'echo' 'discard' 'systat'
    #  'supdup' 'iso_tsap' 'hostnames' 'csnet_ns' 'pop_2' 'sunrpc' 'uucp_path'
    #  'netbios_ns' 'netbios_ssn' 'netbios_dgm' 'sql_net' 'vmnet' 'bgp' 'Z39_50'
    #  'ldap' 'netstat' 'urh_i' 'X11' 'urp_i' 'pm_dump' 'tftp_u' 'tim_i' 'red_i']
    def preprocess_service(self, data: np.ndarray, toStr: bool):
        datalist = data.tolist()
        index = 0
        for ele in datalist:
            if not toStr:
                # 转换成整数
                datalist[index] = self.service_dict[ele]
            else:
                datalist[index] = self.service_list[ele]
            index = index + 1
        data = np.array(datalist)
        return data

    # flag ['SF' 'S1' 'REJ' 'S2' 'S0' 'S3' 'RSTO' 'RSTR' 'RSTOS0' 'OTH' 'SH']
    def preprocess_flag(self, data: np.ndarray, toStr: bool):
        datalist = data.tolist()
        index = 0
        for ele in datalist:
            if not toStr:
                datalist[index] = self.flag_dict[ele]
            else:
                datalist[index] = self.flag_list[ele]
            index = index + 1
        data = np.array(datalist)
        return data

    # label ['normal.' 'buffer_overflow.' 'loadmodule.' 'perl.' 'neptune.' 'smurf.'
    #  'guess_passwd.' 'pod.' 'teardrop.' 'portsweep.' 'ipsweep.' 'land.'
    #  'ftp_write.' 'back.' 'imap.' 'satan.' 'phf.' 'nmap.' 'multihop.'
    #  'warezmaster.' 'warezclient.' 'spy.' 'rootkit.']
    def preprocess_label(self, data: np.ndarray, toStr: bool) -> np.ndarray:
        datalist = data.tolist()
        index = 0
        for ele in datalist:
            if not toStr:
                datalist[index] = self.label_dict[ele]
            else:
                datalist[index] = self.label_list[ele]
            index = index + 1
        data = np.array(datalist)
        return data

    @staticmethod
    def normalize(data: np.ndarray, value, method, toInt: bool = True):
        if method:
            data = data / value * 2 - 1
        else:
            data = (data + 1) * value / 2
            data = data + e
            if toInt:
                data = data.astype(np.int)
        return data

    def preprocess_normalize_feature(self):
        self.datasets[:, 0] = self.normalize(self.datasets[:, 0], self.MAX_DURATION, True)
        self.datasets[:, 4] = self.normalize(self.datasets[:, 4], self.MAX_SRC_BYTES, True)
        self.datasets[:, 5] = self.normalize(self.datasets[:, 5], self.MAX_DST_BYTES, True)
        self.datasets[:, 7] = self.normalize(self.datasets[:, 7], self.MAX_WRONG_FRAGMENT, True)
        self.datasets[:, 8] = self.normalize(self.datasets[:, 8], self.MAX_URGENT, True)

        self.datasets[:, 9] = self.normalize(self.datasets[:, 9], self.MAX_HOT, True)
        self.datasets[:, 10] = self.normalize(self.datasets[:, 10], self.MAX_NUM_FAILED_LOGINS, True)
        self.datasets[:, 12] = self.normalize(self.datasets[:, 12], self.MAX_NUM_COMPROMISED, True)
        self.datasets[:, 15] = self.normalize(self.datasets[:, 15], self.MAX_NUM_ROOT, True)
        self.datasets[:, 16] = self.normalize(self.datasets[:, 16], self.MAX_NUM_FILE_CREATIONS, True)
        self.datasets[:, 17] = self.normalize(self.datasets[:, 17], self.MAX_NUM_SHELLS, True)
        self.datasets[:, 18] = self.normalize(self.datasets[:, 18], self.MAX_NUM_ACCESS_FILES, True)
        self.datasets[:, 19] = self.normalize(self.datasets[:, 19], self.MAX_NUM_OUTBOUND_CMDS, True)

        self.datasets[:, 22] = self.normalize(self.datasets[:, 22], self.MAX_COUNT, True)
        self.datasets[:, 23] = self.normalize(self.datasets[:, 23], self.MAX_SRV_COUNT, True)
        self.datasets[:, 24] = self.normalize(self.datasets[:, 24], self.MAX_SERROR_RATE, True)
        self.datasets[:, 25] = self.normalize(self.datasets[:, 25], self.MAX_SRV_SERROR_RATE, True)
        self.datasets[:, 26] = self.normalize(self.datasets[:, 26], self.MAX_RERROR_RATE, True)
        self.datasets[:, 27] = self.normalize(self.datasets[:, 27], self.MAX_SRV_SERROR_RATE, True)
        self.datasets[:, 28] = self.normalize(self.datasets[:, 28], self.MAX_SAME_SRV_RATE, True)
        self.datasets[:, 29] = self.normalize(self.datasets[:, 29], self.MAX_DIFF_SRV_RATE, True)
        self.datasets[:, 30] = self.normalize(self.datasets[:, 30], self.MAX_SRV_DIFF_HOST_RATE, True)

        self.datasets[:, 31] = self.normalize(self.datasets[:, 31], self.MAX_DST_HOST_COUNT, True)
        self.datasets[:, 32] = self.normalize(self.datasets[:, 32], self.MAX_DST_HOST_SRV_COUNT, True)
        self.datasets[:, 33] = self.normalize(self.datasets[:, 33], self.MAX_DST_HOST_SAME_SRV_RATE, True)
        self.datasets[:, 34] = self.normalize(self.datasets[:, 34], self.MAX_DST_HOST_DIFF_SRV_RATE, True)
        self.datasets[:, 35] = self.normalize(self.datasets[:, 35], self.MAX_DST_HOST_SAME_SRC_PORT_RATE, True)
        self.datasets[:, 36] = self.normalize(self.datasets[:, 36], self.MAX_DST_HOST_SRV_DIFF_HOST_RATE, True)
        self.datasets[:, 37] = self.normalize(self.datasets[:, 37], self.MAX_DST_HOST_SERROR_RATE, True)
        self.datasets[:, 38] = self.normalize(self.datasets[:, 38], self.MAX_DST_HOST_SRV_SERROR_RATE, True)
        self.datasets[:, 39] = self.normalize(self.datasets[:, 39], self.MAX_DST_HOST_RERROR_RATE, True)
        self.datasets[:, 40] = self.normalize(self.datasets[:, 40], self.MAX_DST_HOST_SRV_RERROR_RATE, True)

    def deAnalysis(self, data):
        # sample_num = data.shape[0]
        protocolBeginIndex = self.features_num - self.onehot_features_num
        serviceBeginIndex = protocolBeginIndex + self.protocol_types
        flagBeginIndex = serviceBeginIndex + self.service_types
        flagEndBorder = flagBeginIndex + self.flag_types
        protocolIndexArr = np.argmax(data[:, protocolBeginIndex:serviceBeginIndex], axis=1)
        serviceIndexArr = np.argmax(data[:, serviceBeginIndex:flagBeginIndex], axis=1)
        flagIndexArr = np.argmax(data[:, flagBeginIndex:flagEndBorder], axis=1)

        protocolIndexArr = protocolIndexArr.squeeze()
        data = np.insert(data, 1, values=protocolIndexArr, axis=1)

        serviceIndexArr = serviceIndexArr.squeeze()
        data = np.insert(data, 2, values=serviceIndexArr, axis=1)

        flagIndexArr = flagIndexArr.squeeze()
        data = np.insert(data, 3, values=flagIndexArr, axis=1)

        self.postprocess_classify_feature(data)
        self.postprocess_normalize_feature(data)
        # 清除掉多余的onehot部分
        return data[:, :self.features_num]

    def postprocess_classify_feature(self, data):
        data[:, 1] = self.preprocess_protocol(data[:, 1], True)
        data[:, 2] = self.preprocess_service(data[:, 2], True)
        data[:, 3] = self.preprocess_flag(data[:, 3], True)

    def postprocess_normalize_feature(self, data):
        data[:, 0] = self.normalize(data[:, 0], self.MAX_DURATION, False)
        data[:, 4] = self.normalize(data[:, 4], self.MAX_SRC_BYTES, False)
        data[:, 5] = self.normalize(data[:, 5], self.MAX_DST_BYTES, False)
        data[:, 7] = self.normalize(data[:, 7], self.MAX_WRONG_FRAGMENT, False)
        data[:, 8] = self.normalize(data[:, 8], self.MAX_URGENT, False)

        data[:, 9] = self.normalize(data[:, 9], self.MAX_HOT, False)
        data[:, 10] = self.normalize(data[:, 10], self.MAX_NUM_FAILED_LOGINS, False)
        data[:, 12] = self.normalize(data[:, 12], self.MAX_NUM_COMPROMISED, False)
        data[:, 15] = self.normalize(data[:, 15], self.MAX_NUM_ROOT, False)
        data[:, 16] = self.normalize(data[:, 16], self.MAX_NUM_FILE_CREATIONS, False)
        data[:, 17] = self.normalize(data[:, 17], self.MAX_NUM_SHELLS, False)
        data[:, 18] = self.normalize(data[:, 18], self.MAX_NUM_ACCESS_FILES, False)
        data[:, 19] = self.normalize(data[:, 19], self.MAX_NUM_OUTBOUND_CMDS, False)

        data[:, 22] = self.normalize(data[:, 22], self.MAX_COUNT, False)
        data[:, 23] = self.normalize(data[:, 23], self.MAX_SRV_COUNT, False)
        data[:, 24] = self.normalize(data[:, 24], self.MAX_SERROR_RATE, False, False)
        data[:, 25] = self.normalize(data[:, 25], self.MAX_SRV_SERROR_RATE, False, False)
        data[:, 26] = self.normalize(data[:, 26], self.MAX_RERROR_RATE, False, False)
        data[:, 27] = self.normalize(data[:, 27], self.MAX_SRV_SERROR_RATE, False, False)
        data[:, 28] = self.normalize(data[:, 28], self.MAX_SAME_SRV_RATE, False, False)
        data[:, 29] = self.normalize(data[:, 29], self.MAX_DIFF_SRV_RATE, False, False)
        data[:, 30] = self.normalize(data[:, 30], self.MAX_SRV_DIFF_HOST_RATE, False, False)

        data[:, 31] = self.normalize(data[:, 31], self.MAX_DST_HOST_COUNT, False)
        data[:, 32] = self.normalize(data[:, 32], self.MAX_DST_HOST_SRV_COUNT, False)
        data[:, 33] = self.normalize(data[:, 33], self.MAX_DST_HOST_SAME_SRV_RATE, False, False)
        data[:, 34] = self.normalize(data[:, 34], self.MAX_DST_HOST_DIFF_SRV_RATE, False, False)
        data[:, 35] = self.normalize(data[:, 35], self.MAX_DST_HOST_SAME_SRC_PORT_RATE, False, False)
        data[:, 36] = self.normalize(data[:, 36], self.MAX_DST_HOST_SRV_DIFF_HOST_RATE, False, False)
        data[:, 37] = self.normalize(data[:, 37], self.MAX_DST_HOST_SERROR_RATE, False, False)
        data[:, 38] = self.normalize(data[:, 38], self.MAX_DST_HOST_SRV_SERROR_RATE, False, False)
        data[:, 39] = self.normalize(data[:, 39], self.MAX_DST_HOST_RERROR_RATE, False, False)
        data[:, 40] = self.normalize(data[:, 40], self.MAX_DST_HOST_SRV_RERROR_RATE, False, False)

        data[:, 6] = self.logicialize(data[:, 6])
        data[:, 11] = self.logicialize(data[:, 11])
        data[:, 13] = self.logicialize(data[:, 13])
        data[:, 14] = self.logicialize(data[:, 14])
        data[:, 20] = self.logicialize(data[:, 20])
        data[:, 21] = self.logicialize(data[:, 21])

    @staticmethod
    def logicialize(data: np.ndarray):
        data = np.where(data > 0, True, False)
        return data
