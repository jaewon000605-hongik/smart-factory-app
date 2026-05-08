import streamlit as st
from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt

# 1. 페이지 설정
st.set_page_config(page_title="원예장비 APP 진단 시스템", layout="wide")
st.title("🚜 원예장비 제조업체 총괄생산계획(APP) 종합 진단 시스템")

# 2. 사이드바 설정 (비용 파라미터)
st.sidebar.header("📋 비용 설정 (단위: 천원)")
reg_wage_hour = st.sidebar.number_input("정규 임금 (천원/시간)", value=4, key="reg_wage")
over_wage = st.sidebar.number_input("초과 임금 (천원/시간)", value=6, key="over_wage")
hiring_cost = st.sidebar.number_input("고용 비용 (천원/인)", value=300, key="hiring")
firing_cost = st.sidebar.number_input("해고 비용 (천원/인)", value=500, key="firing")
inv_cost = st.sidebar.number_input("재고 유지비 (천원/개/월)", value=2, key="inv_cost")
backlog_cost = st.sidebar.number_input("부재고 비용 (천원/개/월)", value=5, key="backlog")
mat_cost = st.sidebar.number_input("재료비 (천원/개)", value=10, key="mat_cost")
sub_cost = st.sidebar.number_input("하청 비용 (천원/개)", value=30, key="sub_cost")

st.sidebar.header("⚙️ 생산 제약 설정")
init_w = st.sidebar.number_input("초기 근로자 수", value=80, key="init_w")
init_i = st.sidebar.number_input("초기 재고", value=1000, key="init_i")
target_i = st.sidebar.number_input("6월말 최종 목표 재고", value=500, key="target_i")

# 3. 수요 입력
st.subheader("📊 월별 예상 수요 설정")
demand_input = st.text_input("1월~6월 수요 (쉼표 구분)", "1600, 3000, 3200, 3800, 2200, 2200")
D = [int(x.strip()) for x in demand_input.split(",")]

# 4. 최적화 엔진
def solve_app(D_list):
    m = ConcreteModel()
    T = range(1, len(D_list) + 1)
    TIME = range(0, len(D_list) + 1)
    m.W = Var(TIME, domain=NonNegativeIntegers); m.H = Var(TIME, domain=NonNegativeIntegers)
    m.L = Var(TIME, domain=NonNegativeIntegers); m.P = Var(TIME, domain=NonNegativeIntegers)
    m.I = Var(TIME, domain=NonNegativeIntegers); m.S = Var(TIME, domain=NonNegativeIntegers)
    m.C = Var(TIME, domain=NonNegativeIntegers); m.O = Var(TIME, domain=NonNegativeIntegers)

    # 목적함수 (시간당 임금 4 -> 월급 640 자동 변환 계산)
    m.Cost = Objective(expr=sum(
        (reg_wage_hour * 160) * m.W[t] + over_wage * m.O[t] + 
        hiring_cost * m.H[t] + firing_cost * m.L[t] +
        inv_cost * m.I[t] + backlog_cost * m.S[t] + 
        mat_cost * m.P[t] + sub_cost * m.C[t] for t in T), 
        sense=minimize)

    m.c1 = Constraint(T, rule=lambda m, t: m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
    m.c2 = Constraint(T, rule=lambda m, t: m.P[t] <= 40*m.W[t] + 0.25*m.O[t])
    m.c3 = Constraint(T, rule=lambda m, t: m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D_list[t-1] - m.S[t-1] + m.S[t])
    m.c4 = Constraint(T, rule=lambda m, t: m.O[t] <= 10*m.W[t])
    m.W[0].fix(init_w); m.I[0].fix(init_i); m.S[0].fix(0)
    m.last_inv = Constraint(rule=lambda m: m.I[len(D_list)] >= target_i)
    m.last_short = Constraint(rule=lambda m: m.S[len(D_list)] == 0)

    SolverFactory('glpk').solve(m)
    return m

# 5. 실행 및 결과 전시
if st.button("🚀 최적 생산계획 수립 및 진단 시작"):
    model = solve_app(D)
    
    # 상단 요약 지표 (KPI)
    st.write("### 📌 계획 수립 결과 요약")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Cost (총 비용)", f"{model.Cost():,.0f} K-KRW")
    kpi2.metric("Avg Inventory (평균 재고)", f"{sum(model.I[t]() for t in range(1, 7))/6:,.1f} EA")
    kpi3.metric("Max Workers (최대 인원)", f"{max(model.W[t]() for t in range(1, 7)):.0f} Pers")

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📈 생산/재고 분석", "💰 비용 세부 분석", "📋 상세 데이터"])

    res = {
        "Month": [f"M{i}" for i in range(1, 7)],
        "Demand": D,
        "Production": [model.P[t]() for t in range(1, 7)],
        "Inventory": [model.I[t]() for t in range(1, 7)],
        "Workers": [model.W[t]() for t in range(1, 7)],
        "Subcon": [model.C[t]() for t in range(1, 7)]
    }
    df = pd.DataFrame(res)

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Demand vs Production")
            st.line_chart(df.set_index("Month")[["Demand", "Production"]])
        with col2:
            st.write("#### Monthly Inventory Level")
            st.bar_chart(df.set_index("Month")["Inventory"])

    with tab2:
        st.write("#### Cost Breakdown (비용 효율성 판단)")
        c_reg = sum((reg_wage_hour * 160) * model.W[t]() for t in range(1, 7))
        c_inv = sum(inv_cost * model.I[t]() for t in range(1, 7))
        c_sub = sum(sub_cost * model.C[t]() for t in range(1, 7))
        c_others = model.Cost() - (c_reg + c_inv + c_sub)
        
        # 영문 카테고리로 폰트 깨짐 방지
        cost_data = pd.DataFrame({
            'Category': ['Labor', 'Inventory', 'Subcon', 'Others'],
            'Value': [c_reg, c_inv, c_sub, c_others]
        })
        
        # 차트 크기 최적화 및 중앙 정렬
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(cost_data['Value'], labels=cost_data['Category'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 7})
        ax.axis('equal')
        
        col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
        with col_c2:
            st.pyplot(fig)
        st.info("💡 Labor(인건비) 비중을 확인하여 생산 전략의 효율성을 판단하세요.")

    with tab3:
        st.write("#### Monthly Operation Data")
        st.dataframe(df.set_index("Month"), use_container_width=True)
