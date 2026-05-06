import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_training_losses_from_tensorboard(log_dir, train_tag='loss', val_tag='val_loss'):
    """
    从TensorBoard事件文件中提取训练和验证损失

    Args:
        log_dir: 包含TensorBoard事件文件的目录
        train_tag: 训练损失的标签名
        val_tag: 验证损失的标签名

    Returns:
        tuple: (train_df, val_df) 包含训练和验证损失的DataFrame
    """
    # 查找所有事件文件
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))

    if not event_files:
        raise FileNotFoundError(f"在目录 {log_dir} 中未找到TensorBoard事件文件")

    print(f"找到 {len(event_files)} 个事件文件")
    for file in event_files:
        size_kb = os.path.getsize(file) / 1024
        print(f"  - {os.path.basename(file)} ({size_kb:.1f} KB)")

    # 存储训练和验证损失
    train_losses = []
    val_losses = []

    for event_file in event_files:
        try:
            # 创建EventAccumulator
            ea = EventAccumulator(event_file)
            ea.Reload()  # 加载事件文件

            # 获取所有标量标签
            tags = ea.Tags().get('scalars', [])

            # 提取训练损失
            if train_tag in tags:
                train_events = ea.Scalars(train_tag)
                for event in train_events:
                    train_losses.append({
                        'step': event.step,
                        'value': event.value,
                        'wall_time': event.wall_time,
                        'file': os.path.basename(event_file)
                    })

            # 提取验证损失
            if val_tag in tags:
                val_events = ea.Scalars(val_tag)
                for event in val_events:
                    val_losses.append({
                        'step': event.step,
                        'value': event.value,
                        'wall_time': event.wall_time,
                        'file': os.path.basename(event_file)
                    })

        except Exception as e:
            print(f"处理文件 {os.path.basename(event_file)} 时出错: {e}")
            continue

    # 转换为DataFrame
    train_df = pd.DataFrame(train_losses) if train_losses else pd.DataFrame()
    val_df = pd.DataFrame(val_losses) if val_losses else pd.DataFrame()

    # 按步数排序
    if not train_df.empty:
        train_df = train_df.sort_values('step')
    if not val_df.empty:
        val_df = val_df.sort_values('step')

    return train_df, val_df


def plot_training_curves(train_df, val_df, output_path=None):
    """
    绘制训练和验证损失曲线

    Args:
        train_df: 训练损失DataFrame
        val_df: 验证损失DataFrame
        output_path: 图片保存路径
    """
    if train_df.empty and val_df.empty:
        print("没有训练或验证损失数据可绘制")
        return

    plt.figure(figsize=(12, 6))

    # 绘制训练损失
    if not train_df.empty:
        plt.plot(train_df['step'], train_df['value'],
                 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')
        # 标记最小训练损失
        min_train_idx = train_df['value'].idxmin()
        min_train_step = train_df.loc[min_train_idx, 'step']
        min_train_value = train_df.loc[min_train_idx, 'value']
        plt.scatter(min_train_step, min_train_value,
                    color='blue', s=100, zorder=5,
                    label=f'Min Train Loss: {min_train_value:.4f}')

    # 绘制验证损失
    if not val_df.empty:
        plt.plot(val_df['step'], val_df['value'],
                 'r-', linewidth=1.5, alpha=0.7, label='Validation Loss')
        # 标记最小验证损失
        min_val_idx = val_df['value'].idxmin()
        min_val_step = val_df.loc[min_val_idx, 'step']
        min_val_value = val_df.loc[min_val_idx, 'value']
        plt.scatter(min_val_step, min_val_value,
                    color='red', s=100, zorder=5,
                    label=f'Min Val Loss: {min_val_value:.4f}')

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 设置x轴格式
    if not train_df.empty:
        max_step = train_df['step'].max()
        plt.xticks(np.arange(0, max_step + 1, max(1, max_step // 10)))

    # 自动调整y轴范围
    all_values = []
    if not train_df.empty:
        all_values.extend(train_df['value'].values)
    if not val_df.empty:
        all_values.extend(val_df['value'].values)

    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_range = y_max - y_min
        plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"曲线图已保存到: {output_path}")

    plt.show()


def analyze_training_curves(train_df, val_df):
    """
    分析训练和验证损失曲线

    Args:
        train_df: 训练损失DataFrame
        val_df: 验证损失DataFrame
    """
    print("=" * 60)
    print("训练损失分析")
    print("=" * 60)

    if not train_df.empty:
        print(f"训练损失:")
        print(f"  数据点数: {len(train_df)}")
        print(f"  Step范围: [{train_df['step'].min()}, {train_df['step'].max()}]")
        print(f"  损失范围: [{train_df['value'].min():.6f}, {train_df['value'].max():.6f}]")
        print(f"  最终训练损失: {train_df['value'].iloc[-1]:.6f}")

        # 找到最小训练损失
        min_train_idx = train_df['value'].idxmin()
        min_train_step = train_df.loc[min_train_idx, 'step']
        min_train_value = train_df.loc[min_train_idx, 'value']
        print(f"  最小训练损失: {min_train_value:.6f} (step={min_train_step})")

    if not val_df.empty:
        print(f"\n验证损失:")
        print(f"  数据点数: {len(val_df)}")
        print(f"  Step范围: [{val_df['step'].min()}, {val_df['step'].max()}]")
        print(f"  损失范围: [{val_df['value'].min():.6f}, {val_df['value'].max():.6f}]")
        print(f"  最终验证损失: {val_df['value'].iloc[-1]:.6f}")

        # 找到最小验证损失
        min_val_idx = val_df['value'].idxmin()
        min_val_step = val_df.loc[min_val_idx, 'step']
        min_val_value = val_df.loc[min_val_idx, 'value']
        print(f"  最小验证损失: {min_val_value:.6f} (step={min_val_step})")

    # 分析过拟合
    if not train_df.empty and not val_df.empty:
        train_final = train_df['value'].iloc[-1]
        val_final = val_df['value'].iloc[-1]
        gap = val_final - train_final

        print(f"\n过拟合分析:")
        print(f"  最终训练损失: {train_final:.6f}")
        print(f"  最终验证损失: {val_final:.6f}")
        print(f"  差距(验证-训练): {gap:.6f}")

        if gap > 0:
            if gap > train_final * 0.3:
                print(f"  ⚠️ 警告: 可能过拟合 (验证损失明显高于训练损失)")
            else:
                print(f"  ✓ 训练正常 (验证损失略高于训练损失)")
        elif gap < 0:
            print(f"  ✓ 验证损失低于训练损失，训练效果良好")
        else:
            print(f"  ✓ 训练和验证损失相近")


def save_losses_to_csv(train_df, val_df, output_dir):
    """
    将损失数据保存为CSV文件

    Args:
        train_df: 训练损失DataFrame
        val_df: 验证损失DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    if not train_df.empty:
        train_path = os.path.join(output_dir, "training_loss.csv")
        train_df.to_csv(train_path, index=False)
        print(f"训练损失已保存到: {train_path}")

    if not val_df.empty:
        val_path = os.path.join(output_dir, "validation_loss.csv")
        val_df.to_csv(val_path, index=False)
        print(f"验证损失已保存到: {val_path}")


def main():
    # 设置日志目录路径（请根据实际情况修改）
    # 根据您的图片，应该是包含events文件的目录
    log_dir = "./models/logs/speed20260421T1251"  # 修改为您的实际路径

    if not os.path.exists(log_dir):
        print(f"错误: 目录不存在: {log_dir}")
        print("请修改log_dir变量为您的实际路径")
        return

    print(f"正在读取目录: {log_dir}")

    try:
        # 1. 从TensorBoard事件文件提取损失数据
        # 注意：标签名可能需要根据实际情况调整
        # 如果您的TensorBoard中训练损失标签不是'loss'，请修改train_tag参数
        # 如果验证损失标签不是'val_loss'，请修改val_tag参数
        train_df, val_df = extract_training_losses_from_tensorboard(
            log_dir,
            train_tag='epoch_loss',  # 训练损失标签
            val_tag='epoch_val_loss'  # 验证损失标签
        )

        if train_df.empty and val_df.empty:
            print("没有找到训练或验证损失数据")
            print("可用的标签可能是:")
            # 尝试列出所有可用的标签
            event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
            if event_files:
                try:
                    ea = EventAccumulator(event_files[0])
                    ea.Reload()
                    tags = ea.Tags().get('scalars', [])
                    print(tags)
                except:
                    pass
            return

        # 2. 分析训练曲线
        analyze_training_curves(train_df, val_df)

        # 3. 绘制训练曲线
        output_plot_path = os.path.join(log_dir, "training_curves.png")
        plot_training_curves(train_df, val_df, output_plot_path)

        # 4. 保存为CSV文件
        csv_dir = os.path.join(log_dir, "loss_data")
        save_losses_to_csv(train_df, val_df, csv_dir)

        # 5. 显示前几行数据
        print(f"\n{'=' * 60}")
        print("数据预览")
        print(f"{'=' * 60}")

        if not train_df.empty:
            print(f"\n训练损失前5行:")
            print(train_df[['step', 'value']].head().to_string())

        if not val_df.empty:
            print(f"\n验证损失前5行:")
            print(val_df[['step', 'value']].head().to_string())

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()