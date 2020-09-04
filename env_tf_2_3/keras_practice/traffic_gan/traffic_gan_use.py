import numpy as np
from env_tf_2_3.keras_practice.traffic_gan.data_generator import DataGenerator
import env_tf_2_3.keras_practice.traffic_gan.traffic_gan as tgan
import env_tf_2_3.keras_practice.traffic_gan.config as tganconfig
import matplotlib.pyplot as plt


# 展示数据的指定位置的元素
def display(data: np.ndarray, index: int):
    print("====TCP连接基本特征====")
    print("连接持续时间:", data[index, 0])
    print("协议类型:", data[index, 1])
    print("服务类型:", data[index, 2])
    print("连接状态:", data[index, 3])
    print("从源主机到目标主机的数据的字节数:", data[index, 4])
    print("从目标主机到源主机的数据的字节数:", data[index, 5])
    print("若连接来自/送达同一个主机/端口:", data[index, 6])
    print("错误分段的数量:", data[index, 7])
    print("加急包的个数:", data[index, 8])

    print("====TCP连接的内容特征====")
    print("访问系统敏感文件和目录的次数:", data[index, 9])
    print("登录尝试失败的次数:", data[index, 10])
    print("是否成功登录:", data[index, 11])
    print("compromised条件（**）出现的次数:", data[index, 12])
    print("获得root shell", data[index, 13])
    print("出现su root", data[index, 14])
    print("root用户访问次数:", data[index, 15])
    print("文件创建操作的次数:", data[index, 16])
    print("使用shell命令的次数:", data[index, 17])
    print("访问控制文件的次数:", data[index, 18])
    print("一个FTP会话中出站连接的次数:", data[index, 19])
    print("登录是否属于“hot”列表:", data[index, 20])
    print("是否是guest登录:", data[index, 21])

    print("====基于时间的网路流量统计特征====")
    print("过去两秒内，与当前连接具有相同的目标主机的连接数:", data[index, 22])
    print("过去两秒内，与当前连接具有相同服务的连接数:", data[index, 23])
    print("过去两秒内，在与当前连接具有相同目标主机的连接中，出现“SYN” 错误的连接的百分比:", data[index, 24])
    print("过去两秒内，在与当前连接具有相同服务的连接中，出现“SYN” 错误的连接的百分比:", data[index, 25])
    print("过去两秒内，在与当前连接具有相同目标主机的连接中，出现“REJ” 错误的连接的百分比:", data[index, 26])
    print("过去两秒内，在与当前连接具有相同服务的连接中，出现“REJ” 错误的连接的百分比:", data[index, 27])
    print("过去两秒内，在与当前连接具有相同目标主机的连接中，与当前连接具有相同服务的连接的百分比:", data[index, 28])
    print("过去两秒内，在与当前连接具有相同目标主机的连接中，与当前连接具有不同服务的连接的百分比:", data[index, 29])
    print("过去两秒内，在与当前连接具有相同服务的连接中，与当前连接具有不同目标主机的连接的百分比:", data[index, 30])

    print("====基于主机的网络流量统计特征====")
    print("前100个连接中，与当前连接具有相同目标主机的连接数:", data[index, 31])
    print("前100个连接中，与当前连接具有相同目标主机相同服务的连接数:", data[index, 32])
    print("前100个连接中，与当前连接具有相同目标主机相同服务的连接所占的百分比:", data[index, 33])
    print("前100个连接中，与当前连接具有相同目标主机不同服务的连接所占的百分比，连续:", data[index, 34])
    print("前100个连接中，与当前连接具有相同目标主机相同源端口的连接所占的百分比:", data[index, 35])
    print("前100个连接中，与当前连接具有相同目标主机相同服务的连接中，与当前连接具有不同源主机的连接所占的百分比:", data[index, 36])
    print("前100个连接中，与当前连接具有相同目标主机的连接中，出现SYN错误的连接所占的百分比:", data[index, 37])
    print("前100个连接中，与当前连接具有相同目标主机相同服务的连接中，出现SYN错误的连接所占的百分比:", data[index, 38])
    print("前100个连接中，与当前连接具有相同目标主机的连接中，出现REJ错误的连接所占的百分比:", data[index, 39])
    print("前100个连接中，与当前连接具有相同目标主机相同服务的连接中，出现REJ错误的连接所占的百分比:", data[index, 40])


def use():
    noise_dim: int = tganconfig.NOISE_DIM
    batch_size: int = tganconfig.BATCH_SIZE
    data_g = DataGenerator(tganconfig.FILEPATH, batch_size, noise_dim)
    traffic_dim: int = data_g.feature_dim
    g_model = tgan.createTrafficGANGenerator(traffic_dim, noise_dim)
    e_model = tgan.createTrafficGANEncoder(traffic_dim, noise_dim)
    trafficGAN = tgan.createTrafficGAN(g_model, e_model, traffic_dim, noise_dim)
    trafficGAN.load_weights(tgan.MODELPATH)
    # 正态分布噪声
    # noise = np.random.randn(1, noise_dim)
    # 随机分布噪声
    noise = np.random.uniform(size=(1, noise_dim), low=-1, high=1)
    # 一半0.5 一半0.6的噪声
    # dz = np.ones((1, noise_dim // 2), dtype=np.float)
    # dz2 = np.ones((1, noise_dim // 2), dtype=np.float)
    # dz = dz * 0.5
    # dz2 = dz2 * 0.6
    # noise = np.concatenate([dz, dz2], axis=1)

    data: np.ndarray = g_model.predict(noise)
    code: np.ndarray = e_model.predict(data)
    # 对data进行预处理，从而可以展示
    data = data.astype(np.object)
    data = data_g.deAnalysis(data)
    display(data, 0)
    # 对code进行一次预处理
    max_code = np.max(code)
    min_code = np.min(code)
    code = (code - min_code) / (max_code - min_code)
    # 对noise进行一次预处理
    max_noise = np.max(noise)
    min_noise = np.min(noise)
    noise = (noise - min_noise) / (max_noise - min_noise)
    # 让code与噪声进行对比
    plot_code = code.squeeze()
    plot_noise: np.ndarray = noise

    # min_noise = np.min(plot_noise)
    # max_noise = np.max(plot_noise)
    # plot_noise = (code - min_noise) / (max_noise - min_noise)

    plot_noise = plot_noise.squeeze()

    plt.plot(plot_noise, 'b', label='origin_noise', linewidth=1)
    plt.plot(plot_code, 'r', label='restores_noise', linewidth=1)
    plt.title("还原噪声对比")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    use()
