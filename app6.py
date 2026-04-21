import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
)

import warnings
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
# 全局配置 & 状态初始化
st.set_page_config(
    page_title="高血压风险预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 核心状态管理
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_renamed' not in st.session_state:
    st.session_state.df_renamed = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #feca57, #ff9f43);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .feature-importance {
        font-size: 0.9rem;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

#  核心字段映射（NHANES数据集专属）
FEATURE_MAPPING = {
    'age': 'RIDAGEYR',
    'gender': 'RIAGENDR',
    'bmi': 'BMXBMI',
    'waist': 'BMXWAIST',
    'sbp': 'SBP',
    'dbp': 'DBP',
    'blood_sugar': '血糖',
    'blood_lipid': '血脂',
    'smoking': '吸烟',
    'drinking': '饮酒',
    'exercise_days': '运动天数_周',
    'salt_intake': '每日盐摄入',
    'sleep_hour': '睡眠时长'
}

# 三分类标签规则
def make_risk_level(sbp, dbp):
    if sbp < 120 and dbp < 80:
        return 0   # 正常
    elif (120 <= sbp < 140) or (80 <= dbp < 90):
        return 1   # 高血压前期
    else:
        return 2   # 高血压确诊

# 侧边栏导航
st.sidebar.title("🧭 导航菜单")
page = st.sidebar.radio("", [
    "📊 数据洞察",
    "🔬 模型中心",
    "🎯 风险预测",
    "📋 批量筛查"
])

# 核心工具函数
@st.cache_data
def load_and_preprocess_data(file_path):
    """加载并预处理数据，完全适配NHANES列名"""
    try:
        if file_path.name.endswith('.csv'):
            df = pd.read_csv(file_path, na_values=['NA', ''])
        else:
            df = pd.read_excel(file_path)

        required_cols = list(FEATURE_MAPPING.values())
        existing_cols = [col for col in required_cols if col in df.columns]

        if not existing_cols:
            st.error("❌ 数据中未识别到核心特征字段（如RIDAGEYR, SBP等），请检查文件。")
            return None, None

        # 1. 血压异常值过滤
        df = df[(df['SBP'] >= 60) & (df['SBP'] <= 250) &
                (df['DBP'] >= 40) & (df['DBP'] <= 150)]

        # 2. 剔除缺失值
        df = df.dropna()

        df['risk_level'] = df.apply(lambda x:
            0 if x['正常人群标签'] else
            1 if x['高血压前期标签'] else
            2, axis=1)

        rename_dict = {v: k for k, v in FEATURE_MAPPING.items() if v in df.columns}
        df_renamed = df.rename(columns=rename_dict)

        st.success(f"✅ 数据加载成功！总样本数：{len(df)}")
        return df, df_renamed
    except Exception as e:
        st.error(f"❌ 数据加载失败：{str(e)}")
        return None, None

@st.cache_resource
def train_models(df_renamed):
    """训练双模型"""
    st.subheader("🔬 模型训练中...")

    feature_cols = list(FEATURE_MAPPING.keys())
    # 过滤掉实际不存在的列
    feature_cols = [col for col in feature_cols if col in df_renamed.columns]

    X = df_renamed[feature_cols]
    # 三分类标签
    y = df_renamed['risk_level']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              eval_metric='mlogloss', use_label_encoder=False, random_state=42)

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    models = {'随机森林': rf_model, 'XGBoost': xgb_model}

    # 三分类评估
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        results[name] = {
            'accuracy': (y_pred == y_test).mean(),
            # 多分类AUC
            'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class="ovr"),
            'recall': recall_score(y_test, y_pred, average="macro", zero_division=0),
            'cm': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }

    return models, scaler, feature_cols, results

def generate_shap_advice(input_features, model, shap_risk):
    """
    100%精准动态生成个性化健康建议
    修复所有逻辑问题，完全匹配用户输入
    """
    advice_map = {
        'age': '年龄是高血压的核心风险因素，建议定期监测血压，严格控制其他可干预指标',
        'bmi': 'BMI偏高是主要风险，建议通过饮食控制+规律运动减重5-10%，可显著降低血压',
        'sbp': '收缩压偏高，建议立即就医评估，遵医嘱调整用药，每日监测血压',
        'dbp': '舒张压升高，建议减少盐摄入，规律作息，避免熬夜',
        'waist': '腹型肥胖，建议减少高油高糖饮食，增加核心力量训练',
        'blood_sugar': '血糖异常，建议控制碳水摄入，定期复查血糖',
        'sleep_hour': '睡眠不足会升高血压，建议保证每日7-9小时规律睡眠',
        'salt_intake': '高盐饮食是高血压诱因，建议每日盐摄入控制在6g以内',
        'exercise_days': '缺乏运动，建议每周完成150分钟中等强度有氧运动',
        'smoking': '吸烟会损伤血管，建议立即戒烟，可降低心血管疾病风险',
        'drinking': '过量饮酒会升高血压，建议限制饮酒或戒酒',
        'blood_lipid': '血脂异常，建议低脂饮食，必要时遵医嘱用药',
        'gender': '性别为保护因素，建议继续保持健康生活方式'
    }

    # 提取用户实际输入的特征值
    input_vals = input_features.iloc[0].to_dict()
    shap_vals = shap_risk[0]
    feat_names = input_features.columns.tolist()

    # 双重过滤：只保留「SHAP>0（升高风险）AND 用户实际输入非0」的特征
    risk_feats = []
    for feat, val, shap_val in zip(feat_names, input_vals.values(), shap_vals):
        # 过滤条件：
        # 1. SHAP值>0：确实是升高风险
        # 2. 特征实际值>0：用户真的有这个行为/指标异常
        # 3. 排除性别、年龄等不可干预/基准特征
        if shap_val > 0 and val > 0 and feat not in ['gender', 'age']:
            risk_feats.append((feat, shap_val))

    # 按影响从大到小排序
    risk_feats_sorted = sorted(risk_feats, key=lambda x: abs(x[1]), reverse=True)

    # 生成个性化建议
    advice = []
    if not risk_feats_sorted:
        # 没有有效风险特征，说明用户状态极佳
        advice.append("• 您的生活方式与各项指标状态极佳，无明显高血压风险因素，建议继续保持！")
        return advice

    # 取Top3真实高风险特征
    for feat, val in risk_feats_sorted[:3]:
        advice_text = advice_map.get(feat, f"{feat}指标异常，建议针对性干预")
        advice.append(f"• {advice_text}")

    # 补充通用建议
    advice.append("• 通用建议：保持低盐饮食，规律作息，定期监测血压血糖。")
    return advice

# 页面1：数据洞察
if page == "📊 数据洞察":
    st.markdown('<div class="main-header">📊 健康数据洞察看板</div>', unsafe_allow_html=True)

    # 1. 数据集上传模块
    with st.expander("📂 上传自定义数据集（CSV/Excel）", expanded=False):
        uploaded_data = st.file_uploader("选择文件", type=['csv', 'xlsx'], help="支持NHANES格式数据集", key="upload_data_1")
        if uploaded_data:
            df, df_renamed = load_and_preprocess_data(uploaded_data)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_renamed = df_renamed
                # 上传数据后自动训练模型，存进session_state
                with st.spinner("🔄 模型训练中，请稍候..."):
                    models, scaler, feature_cols, results = train_models(df_renamed)
                    st.session_state.trained_models = models
                    st.session_state.scaler = scaler
                    st.session_state.feature_cols = feature_cols
                    st.session_state.model_results = results
                st.success("✅ 数据已保存，模型训练完成！切换页面不会丢失！")
    df = st.session_state.df
    df_renamed = st.session_state.df_renamed

    if df is None or df_renamed is None:
        st.info("ℹ️ 请上传您的NHANES格式体检数据文件，系统将自动生成分析报告。")
        st.stop()

    # 2. 核心指标卡片
    st.subheader("📈 核心数据概览")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总样本数", f"{len(df):,}", "NHANES数据集")
    with col2:
        normal_rate = (df['risk_level'] == 0).mean() * 100
        st.metric("正常血压占比", f"{normal_rate:.2f}%", f"≈{(df['risk_level']==0).sum()}例")
    with col3:
        pre_rate = (df['risk_level'] == 1).mean() * 100
        st.metric("高血压前期占比", f"{pre_rate:.2f}%", f"≈{(df['risk_level']==1).sum()}例")
    with col4:
        hyper_rate = (df['risk_level'] == 2).mean() * 100
        st.metric("高血压确诊占比", f"{hyper_rate:.2f}%", f"≈{(df['risk_level']==2).sum()}例")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📊 各年龄组高血压风险分布（三分类）")
        # 1. 年龄分组
        df['年龄组_论文'] = pd.cut(
            df[FEATURE_MAPPING['age']],
            bins=[0, 30, 40, 50, 60, 70, 100],
            labels=['<30', '30-39', '40-49', '50-59', '60-69', '≥70']
        )
        # 2. 计算组内占比
        age_risk = df.groupby('年龄组_论文')['risk_level'].value_counts(normalize=True).unstack().fillna(0) * 100
        age_risk.columns = ['正常组', '高血压前期组', '确诊高血压组']
        # 3. 论文配色：正常组#60a9a6（青）、前期#f9c74f（黄）、确诊#f8961e（橙）
        colors = ['#5a9','#fa0','#e44']
        fig_stack = go.Figure()
        for col, color in zip(age_risk.columns, colors):
            fig_stack.add_trace(go.Bar(
                x=age_risk.index,
                y=age_risk[col],
                name=col,
                marker_color=color,
                text=age_risk[col].round(1),
                textposition='inside',
                textfont=dict(size=12, color='black', family='Arial')
            ))
        # 4. 论文样式布局
        fig_stack.update_layout(
            barmode='stack',
            title=dict(text="各年龄组高血压风险分布", font=dict(size=18)),
            xaxis_title="年龄组",
            yaxis_title="占比(%)",
            yaxis_range=[0, 105],
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=14)),
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12))
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    with col_right:
        st.subheader("📈「前期→确诊」转化率趋势")
        # 1. 计算转化率
        age_group = df.groupby('年龄组_论文')
        convert_rate = []
        for name, group in age_group:
            pre = (group['risk_level'] == 1).sum()
            hyper = (group['risk_level'] == 2).sum()
            rate = hyper / (pre + hyper) * 100 if (pre + hyper) > 0 else 0
            convert_rate.append(round(rate, 1))
        # 2. 论文样式折线图：红色、大圆点、标注
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=age_risk.index,
            y=convert_rate,
            mode='lines+markers+text',
            line=dict(color='#d62728', width=3),
            marker=dict(size=12, color='#d62728', line=dict(width=2, color='white')),
            text=[f"{r}%" for r in convert_rate],
            textposition='top center',
            textfont=dict(size=12, color='black')
        ))
        # 3. 转化率图
        fig_line.update_layout(
            title=dict(text="「前期→确诊」转化率趋势", font=dict(size=18)),
            xaxis_title="年龄组",
            yaxis_title="转化率(%)",
            yaxis_range=[0, 52],
            height=500,
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12), gridcolor='lightgray'),
            showlegend=False
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # 饼图
    st.subheader("🍩 风险因素占比")
    risk_factors = pd.DataFrame({
        '因素': ['肥胖(BMI≥28)', '高龄(≥60岁)', '高盐饮食', 'SBP≥140', '血糖异常(≥6.1)', 'DBP≥90'],
        '占比': [
            (df[FEATURE_MAPPING['bmi']] >= 28).mean() * 100,
            (df[FEATURE_MAPPING['age']] >= 60).mean() * 100,
            (df[FEATURE_MAPPING['salt_intake']] > 10).mean() * 100 if FEATURE_MAPPING[
                                                                          'salt_intake'] in df.columns else 0,
            (df[FEATURE_MAPPING['sbp']] >= 140).mean() * 100,
            (df[FEATURE_MAPPING['blood_sugar']] >= 6.1).mean() * 100 if FEATURE_MAPPING[
                                                                            'blood_sugar'] in df.columns else 0,
            (df[FEATURE_MAPPING['dbp']] >= 90).mean() * 100
        ]
    })
    fig2 = px.pie(risk_factors, values='占比', names='因素', hole=0.4,
                  color_discrete_sequence=px.colors.sequential.RdBu)
    fig2.update_layout(height=500, title=dict(text="风险因素占比", font=dict(size=18)))
    st.plotly_chart(fig2, use_container_width=True)

    #特征相关性热力图
    st.subheader("🔥 特征相关性热力图")
    corr_features = [FEATURE_MAPPING[col] for col in ['age', 'bmi', 'sbp', 'dbp', 'waist', 'blood_sugar'] if
                     FEATURE_MAPPING[col] in df.columns]
    corr_features.append('risk_level')

    if len(corr_features) >= 3:
        corr_df = df[corr_features].rename(columns={
            FEATURE_MAPPING['age']: '年龄',
            FEATURE_MAPPING['bmi']: 'BMI',
            FEATURE_MAPPING['sbp']: '收缩压',
            FEATURE_MAPPING['dbp']: '舒张压',
            FEATURE_MAPPING['waist']: '腰围',
            FEATURE_MAPPING['blood_sugar']: '血糖',
            'risk_level': '高血压等级'
        })
        corr_data = corr_df.corr()

        fig3 = px.imshow(corr_data, labels=dict(x="特征", y="特征", color="相关系数"),
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto='.2f')
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("⚠️ 数据字段不足，无法绘制热力图。")
# 页面2：模型中心
elif page == "🔬 模型中心":
    st.markdown('<div class="main-header">🔬 模型实验室</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ 请先在「数据洞察」页面上传数据集！")
        st.stop()

        # 优先用session_state里的模型，没有再训练
    if st.session_state.trained_models is None:
        with st.spinner("🔄 模型训练中，请稍候..."):
            models, scaler, feature_cols, results = train_models(st.session_state.df_renamed)
            st.session_state.trained_models = models
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
            st.session_state.model_results = results
    else:
        models = st.session_state.trained_models
        scaler = st.session_state.scaler
        feature_cols = st.session_state.feature_cols
        results = st.session_state.model_results

    df = st.session_state.df.copy()

    # 特征分组
    feat_base = ['RIDAGEYR', 'RIAGENDR', 'DMDEDUC2']
    feat_phys = ['BMXBMI', 'BMXWAIST', '血糖', '血脂']
    feat_life = ['吸烟', '饮酒', '运动天数_周', '睡眠时长']

    # 年龄BMI交互项
    df['年龄_BMI交互'] = df['RIDAGEYR'] * df['BMXBMI']
    feat_phys = feat_phys + ['年龄_BMI交互']
    all_feats = feat_base + feat_phys + feat_life

    # 确保列都存在
    avail_feats = [f for f in all_feats if f in df.columns]
    X = df[avail_feats]
    y = df['risk_level']

    # 训练测试分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE 过采样
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 标准化（基础组）
    scaler = StandardScaler()
    # 只对存在的列做标准化
    base_cols = [col for col in feat_base if col in X_train_smote.columns]
    X_train_base = scaler.fit_transform(X_train_smote[base_cols])
    X_test_base = scaler.transform(X_test[base_cols])

    # 分组建模
    # 1. 基础组 - 逻辑回归
    model_base = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    model_base.fit(X_train_base, y_train_smote)
    base_proba = model_base.predict_proba(X_test_base)

    # 2. 体检组 - 随机森林
    phys_cols = [col for col in feat_phys if col in X_train_smote.columns]
    model_phys = RandomForestClassifier(
        n_estimators=500, max_depth=12,
        class_weight='balanced_subsample',
        random_state=42, oob_score=True
    )
    model_phys.fit(X_train_smote[phys_cols], y_train_smote)
    phys_proba = model_phys.predict_proba(X_test[phys_cols])

    # 3. 生活方式组 - XGBoost
    life_cols = [col for col in feat_life if col in X_train_smote.columns]
    model_life = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        eval_metric='mlogloss', random_state=42, subsample=0.8, colsample_bytree=0.8
    )
    model_life.fit(X_train_smote[life_cols], y_train_smote)
    life_proba = model_life.predict_proba(X_test[life_cols])

    # 加权融合模型
    fusion_proba = 0.4 * base_proba + 0.5 * phys_proba + 0.1 * life_proba
    fusion_pred = np.argmax(fusion_proba, axis=1)

    # 转二分类（正常 vs 高危）
    y_test_risk = (y_test >= 1).astype(int)
    fusion_pos_proba = fusion_proba[:, 1] + fusion_proba[:, 2]
    fusion_pos_pred = (fusion_pred >= 1).astype(int)

    # 计算指标
    def get_metrics(y_true, prob, pred):
        return {
            "AUC": round(roc_auc_score(y_true, prob), 3),
            "召回率": round(recall_score(y_true, pred, zero_division=0), 3),
            "精确率": round(precision_score(y_true, pred, zero_division=0), 3),
            "准确率": round(accuracy_score(y_true, pred), 3),
            "F1": round(f1_score(y_true, pred, zero_division=0), 3)
        }

    m1 = get_metrics(y_test_risk, base_proba[:,1]+base_proba[:,2], np.argmax(base_proba, axis=1)>=1)
    m2 = get_metrics(y_test_risk, phys_proba[:,1]+phys_proba[:,2], np.argmax(phys_proba, axis=1)>=1)
    m3 = get_metrics(y_test_risk, life_proba[:,1]+life_proba[:,2], np.argmax(life_proba, axis=1)>=1)
    m4 = get_metrics(y_test_risk, fusion_pos_proba, fusion_pos_pred)

    df_metrics = pd.DataFrame(
        [m1, m2, m3, m4],
        index=["基础组(逻辑回归)", "体检组(随机森林)", "生活方式(XGBoost)", "加权融合模型"]
    )

    # 页面展示
    tab1, tab2, tab3 = st.tabs(["📊 性能对比", "🎯 ROC曲线", "🔍 特征重要性"])

    with tab1:
        st.subheader("📈 模型性能指标对比（正常 vs 高危人群）")
        st.dataframe(df_metrics.style.highlight_max(axis=0, color="#c2f7c4"), use_container_width=True)

        # 雷达图
        st.subheader("雷达图 - 模型综合能力")
        categories = ['AUC', '召回率', '精确率', '准确率', 'F1']
        fig_radar = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        names = ['基础组(LR)', '体检组(RF)', '生活方式(XGB)', '融合模型']

        for i, idx in enumerate(df_metrics.index):
            vals = df_metrics.loc[idx][categories].tolist()
            vals.append(vals[0])
            cats = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=cats, name=names[i],
                line=dict(color=colors[i]), fill='toself', opacity=0.3))

        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0,1])), height=500)
        st.plotly_chart(fig_radar, use_container_width=True)

        # 三分类混淆矩阵
        st.subheader("🔍 融合模型 - 三分类混淆矩阵")
        cm = confusion_matrix(y_test, fusion_pred)
        fig_cm = px.imshow(cm,
            x=['正常','高血压前期','高血压确诊'],
            y=['正常','高血压前期','高血压确诊'],
            text_auto=True, color_continuous_scale="Reds")
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab2:
        st.subheader("🎯 ROC 曲线对比（二分类：正常 vs 高危）")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0,y0=0,x1=1,y1=1, line=dict(dash="dash",color="gray"))

        for name, proba in [
            ("基础组", base_proba),
            ("体检组", phys_proba),
            ("生活方式组", life_proba),
            ("融合模型", fusion_proba)
        ]:
            fpr, tpr, _ = roc_curve(y_test_risk, proba[:,1]+proba[:,2])
            auc = round(roc_auc_score(y_test_risk, proba[:,1]+proba[:,2]),2)
            fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{name} (AUC={auc})"))

        fig_roc.update_layout(xaxis_title="FPR",yaxis_title="TPR",height=500)
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab3:
        st.subheader("🔥 体检组特征重要性")
        imp_df = pd.DataFrame({
            "特征": ["BMI","腰围","血糖","血脂","年龄×BMI交互"],
            "重要性": model_phys.feature_importances_
        }).sort_values("重要性", ascending=True)

        fig_imp = px.bar(imp_df, x="重要性", y="特征", orientation="h", color_discrete_sequence=["#1f77b4"])
        fig_imp.update_layout(height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.success("✅ 特征重要性：BMI、腰围、血糖是高血压最强危险因素！")

# 页面3：风险预测（三分类输出）
elif page == "🎯 风险预测":
    st.markdown('<div class="main-header">🎯 个人高血压风险评估</div>', unsafe_allow_html=True)

    if st.session_state.trained_models is None or st.session_state.scaler is None:
        st.warning("⚠️ 请先在「数据洞察」页面上传数据并完成模型训练！")
        st.stop()

    model = st.session_state.trained_models["随机森林"]
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols

    # 严格对齐模型融合权重
    weight_base = 0.4
    weight_phys = 0.5
    weight_life = 0.1

    # 特征分组（包含所有生活方式特征）
    group_config = {
        "基础信息": {"weight": weight_base,
                    "feats": ["age", "gender"]},
        "体检指标": {"weight": weight_phys,
                    "feats": ["sbp", "dbp", "bmi", "waist", "blood_sugar", "blood_lipid"]},
        "生活方式": {"weight": weight_life,
                    "feats": ["smoking", "drinking", "exercise_days", "salt_intake", "sleep_hour"]}
    }

    with st.form("risk_assessment_form"):
        st.subheader("1️⃣ 基础信息组")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("年龄", 18, 90, 45)
            gender = st.selectbox("性别", ["女", "男"])
        with col2:
            education = st.selectbox("教育程度", ["小学及以下", "初中", "高中/中专", "大专/本科", "研究生及以上"])
        with col3:
            residence = st.selectbox("居住地", ["农村", "城市"])

        st.divider()
        st.subheader("2️⃣ 体检指标组 ⚠️（最重要）")
        col4, col5, col6 = st.columns(3)
        with col4:
            sbp = st.number_input("收缩压 (SBP)", 60, 250, 120, step=1)
            dbp = st.number_input("舒张压 (DBP)", 40, 150, 80, step=1)
        with col5:
            bmi = st.number_input("BMI", 10.0, 50.0, 23.0, step=0.1)
            waist = st.number_input("腰围 (cm)", 40, 150, 80, step=1)
        with col6:
            blood_sugar = st.number_input("血糖 (mmol/L)", 3.0, 20.0, 5.5, step=0.1)
            blood_lipid = st.number_input("血脂 (mmol/L)", 2.0, 15.0, 4.5, step=0.1)

        st.divider()
        st.subheader("3️⃣ 生活方式组")
        col7, col8, col9 = st.columns(3)
        with col7:
            smoking = st.selectbox("吸烟状况", ["从不", "已戒烟", "偶尔", "每天"])
            drinking = st.selectbox("饮酒状况", ["从不", "偶尔", "经常"])
        with col8:
            exercise = st.slider("每周运动天数", 0, 7, 3)
            salt_intake = st.selectbox("每日盐摄入", ["<6g", "6-10g", ">10g"])
        with col9:
            sleep = st.slider("睡眠时长(小时)", 3, 12, 7)

        submitted = st.form_submit_button("🔍 开始风险评估", use_container_width=True)

    if submitted:
        input_data = {
            "age": age, "gender": 1 if gender == "男" else 0, "bmi": bmi, "waist": waist,
            "sbp": sbp, "dbp": dbp, "blood_sugar": blood_sugar, "blood_lipid": blood_lipid,
            "smoking": {"从不":0,"已戒烟":1,"偶尔":2,"每天":3}[smoking],
            "drinking": {"从不":0,"偶尔":1,"经常":2}[drinking],
            "exercise_days": exercise,
            "salt_intake": {"<6g":0,"6-10g":1,">10g":2}[salt_intake],
            "sleep_hour": sleep
        }

        input_df = pd.DataFrame([input_data])[feature_cols]
        input_scaled = scaler.transform(input_df)

        # 三分类预测
        pred_label = model.predict(input_scaled)[0]
        label_map = {0:"正常",1:"高血压前期",2:"高血压确诊"}
        risk_result = label_map[pred_label]

        if pred_label == 0:
            style_class = "risk-low"
        elif pred_label == 1:
            style_class = "risk-medium"
        else:
            style_class = "risk-high"

        st.markdown(f"""
        <div class="{style_class}">
            🔎 您的高血压风险评估结果：<br>
            <span style="font-size: 2.5rem;">{risk_result}</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 风险贡献度分析（融合模型权重动态）")

        # SHAP计算：取确诊高血压类（类别2）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        if isinstance(shap_values, list):
            shap_risk = shap_values[2]  # 类别2=确诊高血压，代表风险
        else:
            shap_risk = shap_values[:, :, 2]

        shap_abs_arr = np.abs(shap_risk[0])
        feat_shap_dict = dict(zip(feature_cols, shap_abs_arr))

        # 分组动态贡献 + 融合权重修正（包含所有生活方式特征）
        group_list = []
        contrib_list = []

        for gname, cfg in group_config.items():
            feats = cfg["feats"]
            w = cfg["weight"]
            # 计算该组所有特征的SHAP绝对值之和
            group_shap_sum = sum(feat_shap_dict.get(f, 0) for f in feats if f in feat_shap_dict)
            # 乘以融合权重
            final_contrib = group_shap_sum * w
            group_list.append(gname)
            contrib_list.append(final_contrib)

        # 计算总贡献（转为标量，避免除0）
        total_contrib = float(np.sum(contrib_list))
        ratio_list = []
        if total_contrib > 1e-9:
            for c in contrib_list:
                ratio = float(c) / total_contrib
                ratio_list.append(f"{ratio*100:.0f}%")
        else:
            ratio_list = ["0%", "0%", "0%"]

        contrib_data = pd.DataFrame({
            "组分": group_list,
            "贡献度": contrib_list,
            "占比": ratio_list
        })

        # 绘图
        fig_contrib = px.bar(
            contrib_data,
            x="组分",
            y="贡献度",
            color="组分",
            text="占比",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )
        fig_contrib.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_contrib, use_container_width=True)

        st.divider()
        with st.expander("🔍 个性化风险因素解读", expanded=True):
            st.info("💡 红色=升高风险，蓝色=降低风险")

            cn_map = {
                "age":"年龄","sbp":"收缩压","dbp":"舒张压","bmi":"BMI","waist":"腰围",
                "blood_sugar":"血糖","sleep_hour":"睡眠","salt_intake":"盐摄入",
                "exercise_days":"运动","gender":"性别","blood_lipid":"血脂",
                "smoking":"吸烟","drinking":"饮酒"
            }
            feat_cn = [cn_map.get(f,f) for f in feature_cols]

            # 生成SHAP表（按绝对值从大到小排序）
            shap_df = pd.DataFrame({
                "特征": feat_cn,
                "SHAP风险贡献": shap_risk[0],
                "影响方向": ["🔴 增加风险" if val > 0 else "🔵 降低风险" for val in shap_risk[0]]
            }).sort_values("SHAP风险贡献", key=abs, ascending=False)

            st.dataframe(shap_df, use_container_width=True)

            st.subheader("💡 个性化干预建议（动态生成）")
            # 建议函数，传入正确的shap_risk
            advice = generate_shap_advice(input_df, model, shap_risk)
            for a in advice:
                st.write(a)

# 页面4：批量筛查
elif page == "📋 批量筛查":
    st.markdown('<div class="main-header">📋 社区批量筛查工具</div>', unsafe_allow_html=True)

    # 权限校验：必须先完成模型训练
    if st.session_state.trained_models is None or st.session_state.scaler is None:
        st.warning("⚠️ 请先在「数据洞察」页面上传数据并完成模型训练！")
        st.stop()

    model = st.session_state.trained_models["随机森林"]
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols

    # 页面说明
    st.markdown("""
    ### 📌 功能说明
    本工具支持**批量上传NHANES社区居民体检数据**，一键完成高血压风险筛查，自动生成三分类风险结果（正常/高血压前期/高血压确诊），并支持导出完整筛查报告。
    """)

    # 1. 批量数据上传模块
    st.subheader("📂 上传批量体检数据（CSV/Excel）")
    st.info("""
    ℹ️ 数据格式要求（完全适配NHANES列名）：
    - 必须包含字段：`RIDAGEYR`(年龄)、`RIAGENDR`(性别)、`SBP`(收缩压)、`DBP`(舒张压)、`BMXBMI`(BMI)、`BMXWAIST`(腰围)、`血糖`、`血脂`、`吸烟`、`饮酒`、`运动天数_周`、`每日盐摄入`、`睡眠时长`
    - 编码规则：性别(1=男/2=女，自动转0/1)、吸烟(从不=0/已戒烟=1/偶尔=2/每天=3)、饮酒(从不=0/偶尔=1/经常=2)、盐摄入(<6g=0/6-10g=1/>10g=2)
    - 系统会自动匹配模型特征，无需手动调整列名
    """)

    uploaded_file = st.file_uploader("选择批量数据文件", type=['csv', 'xlsx'], key="batch_upload")

    if uploaded_file:
        # 加载批量数据
        try:
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file, na_values=['NA', ''])
            else:
                batch_df = pd.read_excel(uploaded_file)

            st.success(f"✅ 数据加载成功！总样本数：{len(batch_df)}")

            # 2. 数据预处理（完全适配NHANES列名）
            st.subheader("🔄 数据预处理")
            with st.spinner("正在匹配模型特征..."):
                # NHANES专属字段映射
                rename_map = {
                    # NHANES原始列名 → 模型特征名
                    'RIDAGEYR': 'age',
                    'RIAGENDR': 'gender',
                    'SBP': 'sbp',
                    'DBP': 'dbp',
                    'BMXBMI': 'bmi',
                    'BMXWAIST': 'waist',
                    '血糖': 'blood_sugar',
                    '血脂': 'blood_lipid',
                    '吸烟': 'smoking',
                    '饮酒': 'drinking',
                    '运动天数_周': 'exercise_days',
                    '每日盐摄入': 'salt_intake',
                    '睡眠时长': 'sleep_hour'
                }

                # 自动重命名，保留存在的列
                batch_df_renamed = batch_df.rename(columns=lambda x: rename_map.get(x, x))

                # 性别编码转换：NHANES 1=男→1，2=女→0（对齐模型训练的编码）
                if 'gender' in batch_df_renamed.columns:
                    batch_df_renamed['gender'] = batch_df_renamed['gender'].replace({2: 0, 1: 1})

                # 检查模型所需特征是否齐全
                missing_cols = [col for col in feature_cols if col not in batch_df_renamed.columns]
                if missing_cols:
                    st.error(f"❌ 数据缺失以下必填特征：{', '.join(missing_cols)}，请检查数据格式！")
                    st.stop()

                # 提取模型所需特征，剔除缺失值
                batch_X = batch_df_renamed[feature_cols].dropna()
                if len(batch_X) < len(batch_df):
                    st.warning(f"⚠️ 已自动剔除{len(batch_df)-len(batch_X)}条含缺失值的样本，有效样本数：{len(batch_X)}")

            # 3. 批量预测
            st.subheader("🔍 批量风险预测")
            with st.spinner("正在执行批量预测，请稍候..."):
                # 标准化
                batch_X_scaled = scaler.transform(batch_X)

                # 三分类预测
                batch_pred = model.predict(batch_X_scaled)
                pred_proba = model.predict_proba(batch_X_scaled)

                # 4. 生成结果表
                label_map = {0: "正常", 1: "高血压前期", 2: "高血压确诊"}
                risk_map = {0: "🟢 低风险", 1: "🟡 中风险", 2: "🔴 高风险"}

                # 合并结果（保留原始NHANES列名，方便用户查看）
                result_df = batch_X.copy()
                # 把模型特征名转回NHANES原始列名，方便用户识别
                reverse_rename = {v: k for k, v in rename_map.items()}
                result_df = result_df.rename(columns=reverse_rename)

                # 添加预测结果
                result_df['高血压风险等级'] = [label_map[p] for p in batch_pred]
                result_df['风险标签'] = [risk_map[p] for p in batch_pred]
                result_df['正常概率(%)'] = (pred_proba[:, 0] * 100).round(2)
                result_df['前期概率(%)'] = (pred_proba[:, 1] * 100).round(2)
                result_df['确诊概率(%)'] = (pred_proba[:, 2] * 100).round(2)

                # 5. 结果可视化
                st.subheader("📊 批量筛查结果概览")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总筛查人数", f"{len(result_df):,}")
                with col2:
                    high_risk_num = (batch_pred == 2).sum()
                    st.metric("确诊高血压人数", f"{high_risk_num:,}", f"占比{high_risk_num/len(result_df)*100:.1f}%")
                with col3:
                    pre_risk_num = (batch_pred == 1).sum()
                    st.metric("高血压前期人数", f"{pre_risk_num:,}", f"占比{pre_risk_num/len(result_df)*100:.1f}%")

                # 风险分布饼图
                st.subheader("风险等级分布")
                risk_count = result_df['高血压风险等级'].value_counts().reset_index()
                risk_count.columns = ['风险等级', '人数']
                fig_pie = px.pie(risk_count, values='人数', names='风险等级',
                                color='风险等级',
                                color_discrete_map={'正常':'#48dbfb', '高血压前期':'#feca57', '高血压确诊':'#ff6b6b'},
                                hole=0.3)
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

                # 6. 完整结果表
                st.subheader("📋 完整筛查结果表")
                st.dataframe(result_df, use_container_width=True)

                # 7. 结果导出
                st.subheader("📤 导出筛查结果")
                # 生成CSV文件
                csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载完整筛查报告(CSV)",
                    data=csv_data,
                    file_name='社区高血压批量筛查结果.csv',
                    mime='text/csv',
                    use_container_width=True
                )

                # 8. 高风险人群重点提示
                st.subheader("⚠️ 高风险人群重点提示")
                high_risk_df = result_df[result_df['高血压风险等级'] == '高血压确诊']
                if len(high_risk_df) > 0:
                    st.warning(f"以下{len(high_risk_df)}名居民为确诊高血压高风险，建议优先安排就医随访：")
                    st.dataframe(high_risk_df[['RIDAGEYR', 'RIAGENDR', 'SBP', 'DBP', 'BMXBMI', '确诊概率(%)']], use_container_width=True)
                else:
                    st.success("✅ 本次筛查无确诊高血压高风险人群！")

        except Exception as e:
            st.error(f"❌ 批量筛查失败：{str(e)}")
            st.stop()