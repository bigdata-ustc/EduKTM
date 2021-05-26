# coding: utf-8
# 2021/5/26 @ tongshiwei

def pseudo_data_generation(ku_num, record_num=10, max_length=20):
    # 在这里定义测试用伪数据流
    import random
    random.seed(10)

    raw_data = [
        [
            (random.randint(0, ku_num - 1), random.randint(-1, 1))
            for _ in range(random.randint(2, max_length))
        ] for _ in range(record_num)
    ]

    return raw_data
