import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载日志数据
log_path = "logs/train_logs.csv"  # 替换为你的实际路径
df = pd.read_csv(log_path)

# 添加 step 列，避免 batch 重置问题
df["step"] = range(len(df))

# 设置 seaborn 样式
sns.set(style="whitegrid")

# 创建上下两个子图（共享 x 轴）
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 全局设置字体大小、线条粗细
plt.rcParams["axes.linewidth"] = 1.5  # 边框线粗细
line_width = 2.5  # 曲线粗细

# -------- 上图：Loss --------
sns.lineplot(ax=axes[0], data=df, x="step", y="avgLoss", color="red", label="Loss", linewidth=line_width)
axes[0].set_title("Training Loss", fontsize=14, weight="bold")
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].legend()
axes[0].grid(True)

# -------- 下图：Accuracy --------
sns.lineplot(ax=axes[1], data=df, x="step", y="charAcc", label="Char Accuracy", linewidth=line_width)
sns.lineplot(ax=axes[1], data=df, x="step", y="wordAcc", label="Word Accuracy", linewidth=line_width)
sns.lineplot(ax=axes[1], data=df, x="step", y="sentAcc", label="Sent Accuracy", linewidth=line_width)
axes[1].set_title("Training Accuracy", fontsize=14, weight="bold")
axes[1].set_xlabel("Step", fontsize=12)
axes[1].set_ylabel("Accuracy", fontsize=12)
axes[1].legend()
axes[1].grid(True)

# 设置边框线可见 & 加粗（四边都显示）
for ax in axes:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

# 调整布局
plt.tight_layout()
plt.show()