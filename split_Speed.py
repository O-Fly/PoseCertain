import json
import os
import random


def split_speed_stratified(dataset_dir, train_ratio=0.7, val_ratio=0.2):
    '''
    将 SPEED 数据集按指定比例（默认 7:2:1）划分为 train_no_val, val, my_test。
    采用分层抽样策略：确保纯黑背景（前6000）和复杂背景（后6000）在各个子集中均匀分布。
    '''

    # 1. 读取原始的 train.json
    json_path = os.path.join(dataset_dir, 'train.json')
    with open(json_path, 'r') as f:
        dataset = json.load(f)  # 使用 json.load 更高效

    print(f"成功读取数据集，总图片数: {len(dataset)}")

    # 2. 按照你的观察，将数据集分为两个域 (Domains)
    # 前 6000 张为纯黑背景，后 6000 张为复杂背景
    simple_bg_data = dataset[:6000]
    complex_bg_data = dataset[6000:]

    # 设置随机种子以保证每次运行划分结果一致（可选，但推荐）
    random.seed(42)

    # 3. 分别对两个域的数据进行独立打乱
    random.shuffle(simple_bg_data)
    random.shuffle(complex_bg_data)

    # 定义一个内部函数用来按比例切分列表
    def get_splits(data_list, t_ratio, v_ratio):
        total = len(data_list)
        train_end = int(total * t_ratio)
        val_end = train_end + int(total * v_ratio)

        t_split = data_list[:train_end]
        v_split = data_list[train_end:val_end]
        test_split = data_list[val_end:]
        return t_split, v_split, test_split

    # 4. 对纯黑背景数据进行 7:2:1 切分 (4200 : 1200 : 600)
    simple_train, simple_val, simple_test = get_splits(simple_bg_data, train_ratio, val_ratio)

    # 5. 对复杂背景数据进行 7:2:1 切分 (4200 : 1200 : 600)
    complex_train, complex_val, complex_test = get_splits(complex_bg_data, train_ratio, val_ratio)

    # 6. 将两部分数据合并，构成最终的 train, val, test 集合
    train_set = simple_train + complex_train
    val_set = simple_val + complex_val
    test_set = simple_test + complex_test

    # 7. 再次打乱合并后的集合
    # (极其重要！这保证了在训练时，纯黑和复杂背景的图片是交替出现的，让 Loss 下降更平滑)
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    # 8. 写入 JSON 文件，按照要求的命名
    train_out_path = os.path.join(dataset_dir, 'train_no_val.json')
    val_out_path = os.path.join(dataset_dir, 'val.json')
    test_out_path = os.path.join(dataset_dir, 'my_test.json')

    with open(train_out_path, 'w') as f:
        json.dump(train_set, f)

    with open(val_out_path, 'w') as f:
        json.dump(val_set, f)

    with open(test_out_path, 'w') as f:
        json.dump(test_set, f)

    # 打印统计信息，让你心里有数
    print("\n划分完成！各子集统计如下：")
    print(f"训练集 (train_no_val.json): {len(train_set)} 张 (预期 8400)")
    print(f"验证集 (val.json):          {len(val_set)} 张 (预期 2400)")
    print(f"测试集 (my_test.json):      {len(test_set)} 张 (预期 1200)")
    print("\n数据分布检查：")
    print(f"训练集中包含纯黑背景 {len(simple_train)} 张，复杂背景 {len(complex_train)} 张")


# ==========================================
# 运行示例
# ==========================================
if __name__ == '__main__':
    # 请将这里的路径替换为你电脑上 SPEED 数据集的真实路径
    # 例如：'C:/datasets/speed' 或者 '/home/user/datasets/speed'
    dataset_directory = 'datasets/speed'

    # 调用函数进行划分
    split_speed_stratified(dataset_directory)   #可以设置后两个参数train_ratio, val_ratio
