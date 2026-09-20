import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Palo Alto Networks Attrition Analysis",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Palo Alto Networks_Processed.csv")

# Attrition labels
df["Attrition_Status"] = df["Attrition"].map({
    0: "Retained",
    1: "Exited"
})

# ---------------- TITLE ----------------
st.title("📊 Workforce Attrition Patterns and Risk Hotspot Analysis")
st.subheader("Palo Alto Networks")

st.write(
    "This dashboard analyzes employee attrition patterns across departments, "
    "job roles, demographics, tenure, overtime, business travel, and distance from home."
)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Dashboard Filters")

department_options = ["All"] + sorted(df["Department"].dropna().unique().tolist())
selected_department = st.sidebar.selectbox(
    "Department",
    department_options
)

role_options = ["All"] + sorted(df["JobRole"].dropna().unique().tolist())
selected_role = st.sidebar.selectbox(
    "Job Role",
    role_options
)

min_years = int(df["YearsAtCompany"].min())
max_years = int(df["YearsAtCompany"].max())

selected_years = st.sidebar.slider(
    "Years at Company",
    min_years,
    max_years,
    (min_years, max_years)
)

overtime_options = ["All"] + sorted(df["OverTime"].dropna().unique().tolist())
selected_overtime = st.sidebar.selectbox(
    "Overtime",
    overtime_options
)

travel_options = ["All"] + sorted(df["BusinessTravel"].dropna().unique().tolist())
selected_travel = st.sidebar.selectbox(
    "Business Travel",
    travel_options
)

# ---------------- APPLY FILTERS ----------------
filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[
        filtered_df["Department"] == selected_department
    ]

if selected_role != "All":
    filtered_df = filtered_df[
        filtered_df["JobRole"] == selected_role
    ]

filtered_df = filtered_df[
    (filtered_df["YearsAtCompany"] >= selected_years[0]) &
    (filtered_df["YearsAtCompany"] <= selected_years[1])
]

if selected_overtime != "All":
    filtered_df = filtered_df[
        filtered_df["OverTime"] == selected_overtime
    ]

if selected_travel != "All":
    filtered_df = filtered_df[
        filtered_df["BusinessTravel"] == selected_travel
    ]

# ---------------- ATTRITION OVERVIEW ----------------
st.header("1. Attrition Overview")

total_employees = len(filtered_df)
employees_left = int(filtered_df["Attrition"].sum())
employees_retained = total_employees - employees_left

if total_employees > 0:
    attrition_rate = (employees_left / total_employees) * 100
else:
    attrition_rate = 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Employees", total_employees)
col2.metric("Employees Exited", employees_left)
col3.metric("Employees Retained", employees_retained)
col4.metric("Attrition Rate", f"{attrition_rate:.2f}%")

# Retained vs Exited
status_counts = filtered_df["Attrition_Status"].value_counts()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(status_counts.index, status_counts.values)
ax.set_title("Retained vs Exited Employees")
ax.set_xlabel("Employee Status")
ax.set_ylabel("Number of Employees")
st.pyplot(fig)

# ---------------- DEPARTMENT & ROLE ----------------
st.header("2. Department & Role Analysis")

department_data = filtered_df.groupby("Department")["Attrition"].agg(
    Total_Employees="count",
    Employees_Left="sum"
)

department_data["Attrition_Rate"] = (
    department_data["Employees_Left"] /
    department_data["Total_Employees"] * 100
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Department Attrition Rate")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        department_data.index,
        department_data["Attrition_Rate"]
    )
    ax.set_xlabel("Department")
    ax.set_ylabel("Attrition Rate (%)")
    ax.set_title("Attrition Rate by Department")
    plt.xticks(rotation=20)
    st.pyplot(fig)

with col2:
    st.subheader("Job Role Attrition Rate")

    role_data = filtered_df.groupby("JobRole")["Attrition"].agg(
        Total_Employees="count",
        Employees_Left="sum"
    )

    role_data["Attrition_Rate"] = (
        role_data["Employees_Left"] /
        role_data["Total_Employees"] * 100
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        role_data.index,
        role_data["Attrition_Rate"]
    )
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_ylabel("Job Role")
    ax.set_title("Attrition Rate by Job Role")
    st.pyplot(fig)

# ---------------- HEATMAP ----------------
st.subheader("Department & Job Role Attrition Heatmap")

heatmap_data = pd.crosstab(
    filtered_df["Department"],
    filtered_df["JobRole"],
    values=filtered_df["Attrition"],
    aggfunc="mean"
) * 100

if not heatmap_data.empty:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Attrition Rate Heatmap (%)")
    ax.set_xlabel("Job Role")
    ax.set_ylabel("Department")
    st.pyplot(fig)

# ---------------- DEMOGRAPHIC EXPLORER ----------------
st.header("3. Demographic Explorer")

demographic_option = st.selectbox(
    "Select Demographic Category",
    [
        "Age_Group",
        "Gender",
        "MaritalStatus",
        "Education",
        "EducationField"
    ]
)

if demographic_option == "Age_Group" and "Age_Group" not in filtered_df.columns:
    filtered_df["Age_Group"] = pd.cut(
        filtered_df["Age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"]
    )

if demographic_option == "Age_Group":
    demographic_data = filtered_df.groupby(
        "Age_Group", observed=True
    )["Attrition"].mean() * 100
else:
    demographic_data = filtered_df.groupby(
        demographic_option
    )["Attrition"].mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(
    demographic_data.index.astype(str),
    demographic_data.values
)
ax.set_title(f"Attrition Rate by {demographic_option}")
ax.set_xlabel(demographic_option)
ax.set_ylabel("Attrition Rate (%)")
plt.xticks(rotation=30)
st.pyplot(fig)

# ---------------- TENURE ----------------
st.header("4. Tenure & Career Analysis")

if "Tenure_Group" not in filtered_df.columns:
    filtered_df["Tenure_Group"] = pd.cut(
        filtered_df["YearsAtCompany"],
        bins=[-1, 2, 5, 10, 20, 100],
        labels=[
            "0-2 Years",
            "3-5 Years",
            "6-10 Years",
            "11-20 Years",
            "20+ Years"
        ]
    )

tenure_data = filtered_df.groupby(
    "Tenure_Group", observed=True
)["Attrition"].mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(
    tenure_data.index.astype(str),
    tenure_data.values
)
ax.set_title("Attrition Rate by Years at Company")
ax.set_xlabel("Tenure Group")
ax.set_ylabel("Attrition Rate (%)")
st.pyplot(fig)

# ---------------- OVERTIME & TRAVEL ----------------
st.header("5. Workload & Mobility Analysis")

col1, col2 = st.columns(2)

with col1:
    overtime_data = filtered_df.groupby(
        "OverTime"
    )["Attrition"].mean() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        overtime_data.index,
        overtime_data.values
    )
    ax.set_title("Attrition Rate by Overtime")
    ax.set_xlabel("Overtime")
    ax.set_ylabel("Attrition Rate (%)")
    st.pyplot(fig)

with col2:
    travel_data = filtered_df.groupby(
        "BusinessTravel"
    )["Attrition"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        travel_data.index,
        travel_data.values
    )
    ax.set_title("Attrition Rate by Business Travel")
    ax.set_xlabel("Business Travel")
    ax.set_ylabel("Attrition Rate (%)")
    plt.xticks(rotation=15)
    st.pyplot(fig)

# ---------------- DISTANCE FROM HOME ----------------
st.subheader("Distance from Home")

distance_groups = pd.cut(
    filtered_df["DistanceFromHome"],
    bins=[0, 5, 10, 20, 50],
    labels=["0-5 km", "6-10 km", "11-20 km", "21+ km"]
)

distance_data = filtered_df.groupby(
    distance_groups,
    observed=True
)["Attrition"].mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(
    distance_data.index.astype(str),
    distance_data.values
)
ax.set_title("Attrition Rate by Distance from Home")
ax.set_xlabel("Distance from Home")
ax.set_ylabel("Attrition Rate (%)")
st.pyplot(fig)

# ---------------- EARLY TENURE ----------------
st.header("6. Early-Tenure Attrition")

early_tenure = filtered_df[
    filtered_df["YearsAtCompany"] <= 2
]

if len(early_tenure) > 0:
    early_rate = early_tenure["Attrition"].mean() * 100
else:
    early_rate = 0

st.metric(
    "Early-Tenure Attrition Rate (0-2 Years)",
    f"{early_rate:.2f}%"
)

# ---------------- WORKLOAD RISK ----------------
st.header("7. Workload / Mobility Risk")

filtered_df["Workload_Risk"] = (
    (filtered_df["OverTime"] == "Yes") |
    (filtered_df["BusinessTravel"] == "Travel_Frequently")
)

risk_data = filtered_df.groupby(
    "Workload_Risk"
)["Attrition"].mean() * 100

risk_data.index = risk_data.index.map({
    True: "Higher Workload/Mobility",
    False: "Lower Workload/Mobility"
})

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(
    risk_data.index,
    risk_data.values
)
ax.set_title("Attrition Rate by Workload/Mobility Risk")
ax.set_xlabel("Risk Category")
ax.set_ylabel("Attrition Rate (%)")
st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.write(
    "Workforce Attrition Analysis Dashboard | Data Analyst Internship Project"
)