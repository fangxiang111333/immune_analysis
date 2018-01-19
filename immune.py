# -*-coding: utf-8-*-
######免疫算法求解 TSP 问题##########
###作者:徐芳###
###创建时间:2017-12-27###
###代码语言:python3###
###运行编辑器:pycharm3.6.3###
import math#调用数学库里面的相关函数
import random#生成随机数
import matplotlib.pyplot as plt #python 绘图模块
import numpy as np


'''
    旅行商问题：构造城市
'''
#以下三个参数分别为:城市数目,权值最大值,存储各城市之间路径权值的数组
num_of_city, max_value, val_list = 30, 100, []

#生成城市的邻接矩阵
#进行两层循环,首先遍历所有城市,如果行等于列的时候,就把0加入进去,如果不等于,就随机生成一个1到maxValue的值
#这样子做的效果是生成一个numOfCity*numOfCityde的下三角矩阵,这个下三角矩阵的主对角线全为0,
for row in range(num_of_city):
    val_list.append([ random.randint(1, max_value) for _ in range(num_of_city) ])
for row in range(num_of_city):
    val_list[row][row] = 0

size_of_antibody_list = 20 # 抗体数大小
iterate_time = 1000        # 迭代次数
avg_affinity_list = []
max_affinity_list = []

#取编号为 x，y 城市之间路径的权值
#这里分三种情况,当x>y时,返回生成列表 ;当x<y时交换x与y的位置;当x=y时,我们就返回0.
#这样可以求出最小的距离值,下面打印权值图需要调用这个函数
def val(x, y):
    if x > y:
        return val_list[x - 1][y - 1]
    elif x == y:
        return 0
    elif x < y:
        return val(y, x)

## 随机生成抗体群，抗体编码从 1 开始
def shuffle_list():
    # 辅助函数，生成一个洗牌之后的数组，且第一个元素为 1
    res_list = list(range(2, num_of_city + 1))
    random.shuffle(res_list)
    return [1] + res_list


def produce_antibody_list():
    # 产生抗体列表
    return [ shuffle_list() for i in range(size_of_antibody_list) ]

# 根据抗体群产生抗体信息群
# affinity 相似度
# density 浓度

## 三个辅助函数

# 计算抗体间的相似度（此处采用两向量的欧几里得距离），返回一个布尔值，表示是否低于门限值
def simularity(antibody1, antibody2, threshold):
    return np.linalg.norm(np.array(antibody1) - np.array(antibody2)) <= threshold

# 亲和力计算：亲和力越高，说明该抗体找到的路径越小，结果越好
#计算抗体群中每一个解的亲和力,看抗体与抗原之间的匹配程度,为后面的记忆细胞的产生提供依据
def affinity(antibody):
    res = 0
    for i in range(num_of_city - 1):
        res += val(antibody[i], antibody[i + 1])
    res += val(antibody[-1], antibody[0])
    # res为一个抗体代表的路径长
    return round(num_of_city / float(res), 3)

# 浓度计算
#抗体浓度是指抗体群中,相似抗体的数量占整个抗体群的比例,
def density(antibody, antibody_list):
    res = 0
    for antibody2 in antibody_list:
        res += simularity(antibody, antibody2, 15.0)
    # res为antibody与整个抗体种群的相似度，相似度越高，浓度越大，相似度越低，浓度越低
    return float(res) / size_of_antibody_list

#根据抗体群产生抗体信息群
def produce_antibody_info_list(antibody_list):
    # 关键字key为affinity,关键值value为抗体的亲和力
    # 把抗体在抗体群中的浓度赋值给抗体信息群中的浓度
    antibody_info_list = [{
            'affinity': affinity(antibody),
            'antibody': antibody,
            'density': density(antibody, antibody_list)
        } for antibody in antibody_list]
    # 这是一个自定义的key和reverse,根据亲和力对抗体信息群进行降序排列
    antibody_info_list.sort(reverse = True, key = lambda antibody: antibody['affinity'])
    return antibody_info_list

# 随机互换免疫算子
def pattern_change(antibody):
    i, j = 0, 0
    # 如果i等于j,那么就进入while循环,然后随机生成i和j.
    while i == j:
        i, j = random.randint(1, num_of_city - 1), random.randint(1, num_of_city - 1)
    # 这是一个简单的交换操作,让tempAnitbody[j]等于tempAnitbody[i].
    temp_antibody = antibody
    temp_antibody[i], temp_antibody[j] = temp_antibody[j], temp_antibody[i]
    return temp_antibody


# 抗体促进与抑制机制中计算抗体增长个数
def num_of_antibody_inc(antibody, antibody_list):
    para_a, para_b, para_c = 0.9, 0.5, 0.4
    return int(math.exp(-1 * para_a * affinity(antibody)) / (para_b * density(antibody, antibody_list) + para_c))

#更新抗体群
#在群体更新过程中,我们希望适应度高的抗体被保留下来,但如果此抗体过于集中,则多样性受损,因此我们基于浓度来维持多样性
def update_antibody_list(antibody_list, antibody_info_list):
    # 初始化一个增长抗体列表
    increase_antibody_list = []
    # 初始化一个增长抗体信息列表
    increase_antibody_info_list = []
    for antibody_info in antibody_info_list:
        increase_num = num_of_antibody_inc(antibody_info['antibody'], antibody_list)
        # 抗体增长过程
        for i in range(increase_num):
            increase_antibody_list.append(antibody_info['antibody'])
        # 抗体信息表同时更新
        increase_antibody_info_list.append({
            'antibody': antibody_info['antibody'],
            'affinity': antibody_info['affinity']
        })
    for antibody in increase_antibody_list:
        antibody_list.append(antibody)
    for antibody_info in increase_antibody_info_list:
        antibody_info_list.append(antibody_info)

#绘制城市之间的最短路径
def draw_path(x, y, path):
    for i in range(num_of_city):
        for j in range(i, num_of_city):
            #用黄色表示城市之间的所有路径的连线
            plt.plot((x[path[i] - 1], x[path[j] - 1]), (y[path[i] - 1], y[path[j] - 1]), 'y-')
    for i in range(num_of_city - 1):
        #用红色表示两城市之间的最短路径的连线
        plt.plot((x[path[i] - 1], x[path[i + 1] - 1]), (y[path[i] - 1], y[path[i + 1] - 1]), 'r-')
    plt.plot(x, y, 'bo')

#程序入口
if __name__ == '__main__':

    # 初始化抗体群
    antibody_list = produce_antibody_list()
    # 根据抗体群生成抗体信息表，并按亲和力大小降序排列
    antibody_info_list = produce_antibody_info_list(antibody_list)

    # 迭代
    for i in range(iterate_time):

        # 抗体群促进抑制过程
        update_antibody_list(antibody_list, antibody_info_list)
        # 变异算子执行过程
        mutate_antibody_info_list = []

        # 计算亲和力平均值
        avg_affinity = 0
        for antibody_info in antibody_info_list:
            avg_affinity += antibody_info['affinity']
        avg_affinity /= len(antibody_info_list)
        avg_affinity_list.append(avg_affinity)

        # 遍历抗体群，按照一定概率进行变异
        for antibody_info in antibody_info_list:
            # 根据亲和力得出变异概率，变异概率的基值为 0.02
            p_mutate = (0.185 - (antibody_info['affinity'] - avg_affinity) / avg_affinity)**2
            if random.uniform(0, 1) < p_mutate:
                mutate_antibody = pattern_change(antibody_info['antibody']) # 随机换一个路径
                antibody_info_list.remove(antibody_info)
                mutate_antibody_info_list.append({
                    'antibody': mutate_antibody,
                    'affinity': affinity(mutate_antibody)
                })

        for antibody_info in mutate_antibody_info_list:
            antibody_info_list.append(antibody_info)
        # 对变异后的抗体群按照亲和力大小降序排列
        antibody_info_list.sort(reverse = True, key = lambda antibody: antibody['affinity'])

        # 形成新的抗体群
        # 剔除重复抗体，留下亲和力最高的一个抗体群
        antibody_list = []
        [antibody_list.append(item['antibody']) for item in antibody_info_list if item['antibody'] not in antibody_list]
        antibody_info_list = produce_antibody_info_list(antibody_list)

    # 绘制
    x = np.random.randn(num_of_city)
    y = np.random.randn(num_of_city)
    plt.figure(1)
    plt.plot(range(1, iterate_time + 1), avg_affinity_list)
    plt.title('Average Affinity')
    plt.figure(2)
    draw_path(x, y, antibody_list[0])
    plt.show()