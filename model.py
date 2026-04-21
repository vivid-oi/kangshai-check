import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, accuracy_score, f1_score,
    roc_curve, confusion_matrix
)

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['lines.linewidth'] = 2.5

# 1. 读取数据
data = pd.read_csv("C:/Users/user/Desktop/数据集/提取字段后数据集/三表合并_含高血压标签.csv")

# 三分类标签
def make_risk_level(sbp, dbp):
    if sbp < 120 and dbp < 80:
        return 0
    elif (120 <= sbp < 140) or (80 <= dbp < 90):
        return 1
    else:
        return 2

data['risk_level'] = data.apply(lambda x: make_risk_level(x['SBP'], x['DBP']), axis=1)

# 2. 特征分组
feat_base = ['RIDAGEYR', 'RIAGENDR', 'DMDEDUC2']
feat_phys = ['BMXBMI', 'BMXWAIST', '血糖', '血脂']
feat_life = ['吸烟', '饮酒', '运动天数_周', '睡眠时长']

data['年龄_BMI交互'] = data['RIDAGEYR'] * data['BMXBMI']
feat_phys = feat_phys + ['年龄_BMI交互']

all_feats = feat_base + feat_phys + feat_life
X = data[all_feats]
y = data['risk_level']

# 3. 8:2 划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 预处理
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_base_scaled = scaler.fit_transform(X_train_smote[feat_base])
X_test_base_scaled = scaler.transform(X_test[feat_base])

# 5. 分组建模
model_base = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, solver='lbfgs')
model_base.fit(X_train_base_scaled, y_train_smote)
base_proba = model_base.predict_proba(X_test_base_scaled)

model_phys = RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced_subsample', random_state=42, oob_score=True)
model_phys.fit(X_train_smote[feat_phys], y_train_smote)
phys_proba = model_phys.predict_proba(X_test[feat_phys])

model_life = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, eval_metric='mlogloss', random_state=42, use_label_encoder=False, subsample=0.8, colsample_bytree=0.8)
model_life.fit(X_train_smote[feat_life], y_train_smote)
life_proba = model_life.predict_proba(X_test[feat_life])

# 权重：0.4 + 0.5 + 0.1
fusion_proba = 0.4 * base_proba + 0.5 * phys_proba + 0.1 * life_proba
fusion_pred = np.argmax(fusion_proba, axis=1)

y_test_binary = (y_test >= 1).astype(int)
fusion_proba_binary = fusion_proba[:,1] + fusion_proba[:,2]
fusion_pred_binary = (fusion_pred >= 1).astype(int)

def get_metrics(y_true, prob, pred):
    return {
        "AUC": round(roc_auc_score(y_true, prob),3),
        "召回率": round(recall_score(y_true, pred),3),
        "精确率": round(precision_score(y_true, pred),3),
        "准确率": round(accuracy_score(y_true, pred),3),
        "F1": round(f1_score(y_true, pred),3)
    }

m1 = get_metrics((y_test>=1).astype(int), base_proba[:,1]+base_proba[:,2], np.argmax(base_proba, axis=1)>=1)
m2 = get_metrics((y_test>=1).astype(int), phys_proba[:,1]+phys_proba[:,2], np.argmax(phys_proba, axis=1)>=1)
m3 = get_metrics((y_test>=1).astype(int), life_proba[:,1]+life_proba[:,2], np.argmax(life_proba, axis=1)>=1)
m4 = get_metrics((y_test>=1).astype(int), fusion_proba_binary, fusion_pred_binary)

df_metrics = pd.DataFrame([m1, m2, m3, m4],
                  index=["基础组(逻辑回归)", "体检组(随机森林)", "生活方式(XGBoost)", "加权融合模型"])

print("           高血压早筛模型      ")
print(df_metrics)

# 图1 ROC 对比图
plt.figure(num=1, figsize=(8, 4))
for name, p in [("基础组", base_proba), ("体检组", phys_proba), ("生活方式组", life_proba), ("融合模型", fusion_proba)]:
    fpr, tpr, _ = roc_curve(y_test, p[:, 1], pos_label=1)
    plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test==1, p[:,1]):.2f}")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC 模型对比")
plt.legend()
plt.show()

# 图2 混淆矩阵
plt.figure(num=2, figsize=(6, 4))
cm = confusion_matrix(y_test, fusion_pred)

# 三分类
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=['正常', '前期', '确诊'],
    yticklabels=['正常', '前期', '确诊']
)
plt.title("融合模型 混淆矩阵（三分类）")
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.tight_layout()
plt.show()

# 图3 特征重要性
plt.figure(num=3, figsize=(8, 5))
imp = pd.DataFrame({
    "特征": ["BMI", "腰围", "血糖", "血脂", "年龄×BMI交互"],
    "重要性": model_phys.feature_importances_
}).sort_values("重要性", ascending=True)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
plt.barh(imp["特征"], imp["重要性"], color=colors)

norm = plt.Normalize(imp["重要性"].min(), imp["重要性"].max())
sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
sm.set_array([])
plt.colorbar(sm, label='重要性', ax=plt.gca())

plt.xlabel("重要性")
plt.ylabel("特征")
plt.title("特征重要性")
plt.tight_layout()
plt.show()

# 图4 指标柱状图
plt.figure(num=4, figsize=(9, 4))
df_metrics.plot(
    kind='bar',
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    ax=plt.gca()
)
plt.title("各模型指标对比")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# 图5 雷达图
categories = ['AUC', '召回率', '精确率', '准确率', 'F1']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # 闭合角度

# 定义4个模型的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = ['基础组(LR)', '体检组(LGBM)', '生活方式组(XGB)', '融合模型(Stacking)']

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)

# 循环画4个模型的雷达图
for i, model_name in enumerate(df_metrics.index):
    values = df_metrics.loc[model_name][categories].values.tolist()
    values += values[:1]  # 闭合数值
    ax.plot(angles, values, 'o-', linewidth=2, label=labels[i], color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

# 设置标签和标题
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title("模型性能对比（测试集）", pad=20, fontsize=16)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.show()
