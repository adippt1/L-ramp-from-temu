# co2_to_co_l_ramp_app.py
# Streamlit L-RAMP Dashboard — CO2->CO Electrolyzer
# Author: Aditya Prajapati
# Run with: streamlit run co2_to_co_l_ramp_app.py

from __future__ import annotations
import json
from datetime import date, timedelta
from typing import Dict, List, Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------- App Config -------------------------
st.set_page_config(
    page_title="L-RAMP — CO2->CO Electrolyzer",
    page_icon="🧪",
    layout="wide"
)

# ------------------------- CRISP Title Block -------------------------
st.markdown("""
# 🧩 **CRISP** — *CO₂ Risk Identification & Scale-up Planning*
### *Because a CRISP plan keeps scale-up from crumbling.*
---
""")
with st.sidebar:
    st.markdown(
        """
        ---
        **Created by**  
        **[Aditya Prajapati (Adi)](https://people.llnl.gov/prajapati3)**
        ---
        """,
        unsafe_allow_html=True
    )

# ------------------------- Helpers -------------------------
def _init_session_state():
    """One-time initialization of session_state with sensible defaults for a CO2->CO system."""
    if "apps_df" not in st.session_state:
        # Candidate applications with default KPIs & weights scaffold
        st.session_state.kpis = [
            "Capex ($/tCO/yr)",
            "Opex ($/tCO)",
            "Energy Intensity (kWh/kg CO)",
            "Single-Pass Conversion (%)",
            "Durability / Lifetime (h)",
            "Safety / HSE",
            "Supply-Chain Risk",
            "Integration Complexity",
        ]

        # Weights 1-5 (editable)
        st.session_state.kpi_weights = {k: v for k, v in zip(st.session_state.kpis, [4,4,5,3,4,5,3,3])}

        # Applications (rows) with per-KPI scores -1..+1 (editable later in UI)
        st.session_state.apps_df = pd.DataFrame({
            "Application": [
                "Flue gas CO2 -> CO (w/ upstream separation)",
                "High-purity CO2 -> CO (chemical feedstock)",
                "DAC CO2 -> CO (pilot/niche)",
            ]
        })
        for k in st.session_state.kpis:
            st.session_state.apps_df[k] = 0  # neutral to start

    if "components_df" not in st.session_state:
        # Core technology + inputs/outputs; TRL 1..9
        st.session_state.components_df = pd.DataFrame([
            # Core stack
            {"Component": "Cathode GDE (Ag/C catalyst + ionomer)", "Category": "Core", "TRL": 3, "Notes": "Selectivity to CO; flooding/drying risk"},
            {"Component": "Anode (OER catalyst + ionomer)", "Category": "Core", "TRL": 5, "Notes": "NiFe or Ir-based; durability & cost trade-off"},
            {"Component": "Anion Exchange Membrane (AEM)", "Category": "Core", "TRL": 4, "Notes": "Carbonate transport; conductivity vs. stability"},
            {"Component": "Flow Fields / Bipolar Plates", "Category": "Core", "TRL": 6, "Notes": "Pressure drop, uniformity, corrosion"},
            {"Component": "Gaskets/Seals", "Category": "Core", "TRL": 7, "Notes": "Chemical compatibility & compression set"},
            # Inputs / Outputs / BOP
            {"Component": "CO2 Feed (spec, humidity, pressure)", "Category": "Input", "TRL": 6, "Notes": "Impurity tolerance (SOx/NOx), water management"},
            {"Component": "Power Electronics (DC, ripple)", "Category": "Input", "TRL": 7, "Notes": "Efficiency, control, safety"},
            {"Component": "Cathode Gas Handling (CO/CO2/H2)", "Category": "Output", "TRL": 5, "Notes": "Gas separation, monitoring, flare/scrub if needed"},
            {"Component": "Anolyte/Water Management", "Category": "Output", "TRL": 5, "Notes": "pH, conductivity, crossover handling"},
            {"Component": "Thermal Management", "Category": "Core", "TRL": 4, "Notes": "Temperature uniformity; hot spots at scale"},
        ])

    if "risks_df" not in st.session_state:
        st.session_state.risks_df = pd.DataFrame([
            {"Risk": "High cell voltage -> elevated energy cost", "Category": "Cost", "Likelihood": 3, "Severity": 4, "Owner": "", "Evidence": ""},
            {"Risk": "Carbonate crossover -> product loss/complex separation", "Category": "Quality", "Likelihood": 4, "Severity": 4, "Owner": "", "Evidence": ""},
            {"Risk": "Flooding/drying cycles -> rapid performance decay", "Category": "Time", "Likelihood": 3, "Severity": 4, "Owner": "", "Evidence": ""},
            {"Risk": "Membrane degradation -> lifetime < target", "Category": "Quality", "Likelihood": 3, "Severity": 5, "Owner": "", "Evidence": ""},
            {"Risk": "CO safety (leaks / monitoring)", "Category": "HSE", "Likelihood": 2, "Severity": 5, "Owner": "", "Evidence": ""},
            {"Risk": "Critical material supply (ionomer/membrane)", "Category": "Supply", "Likelihood": 3, "Severity": 3, "Owner": "", "Evidence": ""},
        ])
        _recompute_rpn()

    if "mitigations_df" not in st.session_state:
        st.session_state.mitigations_df = pd.DataFrame(columns=[
            "Action", "Type", "Linked Risk", "Resources", "Success Criteria", "Owner", "Start", "End"
        ])

    if "project_title" not in st.session_state:
        st.session_state.project_title = "CO2->CO L-RAMP — Pilot-ready Stack & BOP"

def _recompute_rpn():
    df = st.session_state.risks_df
    # Clamp to 1..5
    df["Likelihood"] = df["Likelihood"].clip(1,5).astype(int)
    df["Severity"] = df["Severity"].clip(1,5).astype(int)
    st.session_state.risks_df["RPN"] = st.session_state.risks_df["Likelihood"] * st.session_state.risks_df["Severity"]

def _completion_badge() -> float:
    # Very simple heuristic for "completion": have non-defaults filled in
    score = 0
    total = 4
    # Applications scored?
    if "apps_df" in st.session_state and st.session_state.apps_df[st.session_state.kpis].abs().sum().sum() > 0:
        score += 1
    # Components have TRLs set (presence is enough)
    if "components_df" in st.session_state:
        score += 1
    # Risks have owners/evidence filled in (some)
    if "risks_df" in st.session_state and (st.session_state.risks_df["Owner"].astype(str).str.len() > 0).sum() >= 2:
        score += 1
    # Mitigations exist
    if "mitigations_df" in st.session_state and len(st.session_state.mitigations_df) > 0:
        score += 1
    return 100.0 * score / total

def _download_button(label: str, data: bytes, file_name: str, mime: str = "text/plain"):
    st.download_button(label, data=data, file_name=file_name, mime=mime)

# ------------------------- Sidebar -------------------------
_init_session_state()
st.sidebar.title("L-RAMP — CO2->CO")
st.sidebar.write("A lightweight workflow for **Application -> TRL map -> Risk -> Mitigation -> Plan**.")
st.sidebar.progress(int(_completion_badge()))
st.sidebar.caption("Progress heuristic based on filled artifacts.")

# ------------------------- Main Tabs -------------------------
tabs = st.tabs([
    "0) Overview",
    "1) Application Benchmarking",
    "2) Technology Maturity",
    "3) Risk Assessment",
    "4) Mitigation Planner",
    "5) Project Plan",
    "6) Exergy / Energy Check",
    "7) Export"
])

# ------------------------- 0) Overview -------------------------
with tabs[0]:
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown(f"## {st.session_state.project_title}")
        st.write("""
L-RAMP is a structured, five-step protocol to **identify use-cases**, map **technology maturity**,
run a **risk workshop**, define **mitigations**, and assemble a **project plan**. This app packages those steps
for a CO2->CO electrolyzer.

**Workflow**
1) Application benchmarking -> prioritize use-cases with a KPI matrix  
2) Technology maturity -> component tree with TRL sliders  
3) Risk assessment -> register with likelihood x severity and RPN heat-map  
4) Mitigation planner -> actions linked to risks, owners, and dates  
5) Project plan -> Gantt-style timeline generated from mitigations  
6) Exergy/Energy check -> quick plausibility screen based on inputs
""")
st.markdown("---")
st.subheader("References")
st.markdown("""
1. [Nitopi, Stephanie, et al. "Progress and perspectives of electrochemical CO2 reduction on copper in aqueous electrolyte." 
Chemical reviews 119.12 (2019): 7610-7672.](https://pubs.acs.org/doi/full/10.1021/acs.chemrev.8b00705)

2. [Perry, John H. "Chemical engineers' handbook." (1950): 533.](https://pubs.acs.org/doi/pdf/10.1021/ed027p533.1): 
Link is just an exerpt but a good starting point for one to go out in the wild to find this book.

3. [Data, C. P. T. NIST Chemistry WebBook, NIST Standard Reference Database Number 69, 2005.](https://webbook.nist.gov/chemistry/)
""")

    with col2:
        st.metric("Completion", f"{int(_completion_badge())}%")
        st.text_input("Project Title", key="project_title")

# ------------------------- 1) Application Benchmarking -------------------------
with tabs[1]:
    st.subheader("Define KPIs and score applications")
    # Edit KPI weights
    with st.expander("KPI Weights (1–5)", expanded=False):
        cols = st.columns(4)
        for i, k in enumerate(st.session_state.kpis):
            with cols[i % 4]:
                st.session_state.kpi_weights[k] = st.number_input(
                    k, min_value=1, max_value=5, value=int(st.session_state.kpi_weights[k]), step=1, key=f"w_{k}"
                )

    st.write("### Applications Matrix (scores -1, 0, +1 recommended scale)")
    apps_df = st.session_state.apps_df.copy()
    st.dataframe(apps_df, use_container_width=True, hide_index=True)

    st.write("Edit application rows and KPI scores below:")
    with st.form("apps_edit_form"):
        edited = st.data_editor(apps_df, num_rows="dynamic", use_container_width=True, key="apps_editor")
        submitted = st.form_submit_button("Update Applications")
    if submitted:
        # Persist back
        st.session_state.apps_df = edited

    # Weighted ranking
    weights = np.array([st.session_state.kpi_weights[k] for k in st.session_state.kpis], dtype=float)
    scores = st.session_state.apps_df[st.session_state.kpis].to_numpy(dtype=float)
    weighted = (scores * weights).sum(axis=1)
    ranked = st.session_state.apps_df.assign(WeightedScore=weighted).sort_values("WeightedScore", ascending=False).reset_index(drop=True)

    st.write("### Ranked Applications")
    st.dataframe(ranked, use_container_width=True, hide_index=True)

    # Chart
    chart = alt.Chart(ranked).mark_bar().encode(
        x=alt.X("WeightedScore:Q", title="Weighted Score"),
        y=alt.Y("Application:N", sort="-x", title=""),
        tooltip=list(ranked.columns)
    ).properties(height=200)
    st.altair_chart(chart, use_container_width=True)

# ------------------------- 2) Technology Maturity -------------------------
with tabs[2]:
    st.subheader("Component Map & TRL")
    st.caption("Categorize items as Core / Input / Output and set TRL (1–9). Add notes for constraints or gaps.")
    components_df = st.session_state.components_df.copy()

    with st.form("components_form"):
        edited_components = st.data_editor(
            components_df,
            column_config={
                "TRL": st.column_config.NumberColumn(min_value=1, max_value=9, step=1)
            },
            num_rows="dynamic",
            use_container_width=True,
            key="components_editor"
        )
        submitted_comp = st.form_submit_button("Update Components")
    if submitted_comp:
        st.session_state.components_df = edited_components

    # Summary by category and TRL
    colA, colB = st.columns(2)
    with colA:
        by_cat = st.session_state.components_df.groupby("Category")["TRL"].mean().reset_index()
        st.write("**Average TRL by Category**")
        st.dataframe(by_cat, hide_index=True, use_container_width=True)
        cat_chart = alt.Chart(by_cat).mark_bar().encode(
            x=alt.X("TRL:Q", scale=alt.Scale(domain=[1,9])),
            y=alt.Y("Category:N", sort="-x"),
            tooltip=["Category","TRL"]
        ).properties(height=200)
        st.altair_chart(cat_chart, use_container_width=True)
    with colB:
        hist = alt.Chart(st.session_state.components_df).mark_bar().encode(
            x=alt.X("TRL:Q", bin=alt.Bin(step=1), title="TRL"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=["count()"]
        ).properties(height=200)
        st.write("**TRL Distribution**")
        st.altair_chart(hist, use_container_width=True)

# ------------------------- 3) Risk Assessment -------------------------
with tabs[3]:
    st.subheader("Risk Register")
    st.caption("Likelihood & Severity on 1–5 scales. RPN = LxS.")

    with st.form("risks_form"):
        edited_risks = st.data_editor(
            st.session_state.risks_df,
            column_config={
                "Likelihood": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
                "Severity": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="risks_editor"
        )
        submitted_risks = st.form_submit_button("Update Risks")
    if submitted_risks:
        st.session_state.risks_df = edited_risks
        _recompute_rpn()

    st.write("### Prioritized Risks (by RPN)")
    risks_ranked = st.session_state.risks_df.sort_values("RPN", ascending=False).reset_index(drop=True)
    st.dataframe(risks_ranked, hide_index=True, use_container_width=True)

    # Heat map (Likelihood vs Severity weighted by count)
    heat_df = (st.session_state.risks_df
               .groupby(["Likelihood","Severity"])
               .size()
               .reset_index(name="Count"))
    heat = alt.Chart(heat_df).mark_rect().encode(
        x=alt.X("Likelihood:O"),
        y=alt.Y("Severity:O"),
        color=alt.Color("Count:Q"),
        tooltip=["Likelihood","Severity","Count"]
    ).properties(height=250)
    st.altair_chart(heat, use_container_width=True)

# ------------------------- 4) Mitigation Planner -------------------------
with tabs[4]:
    st.subheader("Define Mitigation Actions")
    st.caption("Link actions to risks; specify resources, criteria, owners, and dates.")

    risks_list = st.session_state.risks_df["Risk"].tolist()
    with st.form("add_mitigation"):
        col1, col2 = st.columns(2)
        with col1:
            action = st.text_input("Action (short imperative)")
            mtype = st.selectbox("Type", ["Lit review / quick calc", "Modeling study", "Bench experiment", "Pilot experiment", "Supply-chain analysis", "Safety/HazOp"], index=2)
            linked = st.selectbox("Linked Risk", risks_list if risks_list else [""])
            owner = st.text_input("Owner")
        with col2:
            resources = st.text_area("Resources (people/equipment/materials)")
            success = st.text_area("Success Criteria (objective)")
            start = st.date_input("Start", value=date.today())
            end = st.date_input("End", value=date.today() + timedelta(days=30))

        add_btn = st.form_submit_button("Add Mitigation")
        if add_btn and action.strip():
            new_row = {
                "Action": action.strip(),
                "Type": mtype,
                "Linked Risk": linked,
                "Resources": resources.strip(),
                "Success Criteria": success.strip(),
                "Owner": owner.strip(),
                "Start": str(start),
                "End": str(end)
            }
            st.session_state.mitigations_df = pd.concat([st.session_state.mitigations_df, pd.DataFrame([new_row])], ignore_index=True)

    st.write("### Mitigations")
    st.dataframe(st.session_state.mitigations_df, hide_index=True, use_container_width=True)

# ------------------------- 5) Project Plan -------------------------
with tabs[5]:
    st.subheader("Roadmap (Gantt-style)")
    plan_df = st.session_state.mitigations_df.copy()
    if len(plan_df) == 0:
        st.info("Add mitigation actions to populate the plan.")
    else:
        plan_df["Start"] = pd.to_datetime(plan_df["Start"])
        plan_df["End"] = pd.to_datetime(plan_df["End"])
        plan_df["Days"] = (plan_df["End"] - plan_df["Start"]).dt.days.clip(lower=1)

        st.dataframe(plan_df[["Action","Owner","Type","Linked Risk","Start","End","Days"]], hide_index=True, use_container_width=True)

        base = alt.Chart(plan_df).encode(
            y=alt.Y("Action:N", sort="-x", title="Tasks"),
            x=alt.X("Start:T", title=""),
        )
        bars = base.mark_bar().encode(
            x2="End:T",
            color="Type:N",
            tooltip=["Action","Owner","Type","Linked Risk","Start","End","Days"]
        ).properties(height=300)
        st.altair_chart(bars, use_container_width=True)

# ------------------------- 6) Exergy / Energy Check -------------------------
with tabs[6]:
    st.subheader("Quick Energy Plausibility (CO2->CO)")
    st.caption("Back-of-envelope: energy per mol of CO based on cell voltage, FE, and electrons per CO (n=2).")

    col1, col2, col3 = st.columns(3)
    with col1:
        V_cell = st.number_input("Cell Voltage (V)", min_value=0.0, value=3.0, step=0.1)
        FE_CO = st.slider("Faradaic Efficiency to CO (%)", min_value=1, max_value=100, value=90, step=1)
    with col2:
        j = st.number_input("Current Density (mA/cm2)", min_value=0.0, value=500.0, step=10.0)
        area_cm2 = st.number_input("Active Area (cm2)", min_value=1.0, value=25.0, step=1.0)
    with col3:
        single_pass = st.slider("Single-Pass CO2 Conversion (%)", min_value=1, max_value=100, value=30, step=1)
        ref_min = st.number_input("Reference Efficiency Lower Bound (%)", min_value=1.0, value=30.0, step=1.0)
        ref_max = st.number_input("Reference Efficiency Upper Bound (%)", min_value=1.0, value=80.0, step=1.0)

    F = 96485.33212  # C/mol
    n_e = 2  # electrons per CO
    FE = FE_CO / 100.0

    # Moles CO per second = (I * FE) / (n F)
    I = (j/1000.0) * area_cm2  # A
    mol_CO_s = (I * FE) / (n_e * F) if FE > 0 else np.nan

    # Electrical power = I * V (W); Energy per mol CO (J/mol)
    power_W = I * V_cell
    J_per_mol = power_W / mol_CO_s if mol_CO_s > 0 else np.nan

    # Convert to kWh per kg CO
    M_CO = 28.01e-3  # kg/mol
    kWh_per_kg = (J_per_mol / 3.6e6) / M_CO if J_per_mol == J_per_mol else np.nan  # NaN guard

    st.write("### Results")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Current (A)", f"{I:.3f}")
        st.metric("Power (W)", f"{power_W:.1f}")
    with colB:
        st.metric("CO rate (mol/s)", f"{mol_CO_s:.6f}")
        st.metric("Energy (kWh/kg CO)", f"{kWh_per_kg:.2f}")
    with colC:
        # A crude 'efficiency' proxy vs. a user-provided reference band: lower kWh/kg implies better
        # We'll invert to pseudo-efficiency by mapping the user band into a 0..100 scale
        # User sets ref_min/ref_max for "typical" second-law efficiencies in their context.
        pseudo_eff = None
        if kWh_per_kg == kWh_per_kg and ref_min < ref_max:
            # Normalize inverse of energy between bounds
            # Assume lower energy is better (higher efficiency). Map kWh_per_kg to [0,100].
            x = np.clip((ref_max - kWh_per_kg) / (ref_max - ref_min), 0, 1)
            pseudo_eff = 100 * x
            st.metric("Plausibility Index (%)", f"{pseudo_eff:.0f}")
        else:
            st.write("Set a valid reference band to compute plausibility.")

    st.caption("Note: This is a fast screen to spot order-of-magnitude issues and track improvements.")

# ------------------------- 7) Export -------------------------
with tabs[7]:
    st.subheader("Export Artifacts")
    # Assemble a JSON bundle
    bundle: Dict[str, Any] = {
        "project_title": st.session_state.project_title,
        "kpi_weights": st.session_state.kpi_weights,
        "applications": st.session_state.apps_df.to_dict(orient="records"),
        "components": st.session_state.components_df.to_dict(orient="records"),
        "risks": st.session_state.risks_df.to_dict(orient="records"),
        "mitigations": st.session_state.mitigations_df.to_dict(orient="records"),
    }
    json_bytes = json.dumps(bundle, indent=2).encode("utf-8")
    _download_button("Download Project JSON", json_bytes, "co2_to_co_l_ramp_bundle.json", "application/json")

    st.write("Download individual CSVs:")
    col1, col2, col3 = st.columns(3)
    with col1:
        _download_button("Applications CSV", st.session_state.apps_df.to_csv(index=False).encode("utf-8"), "applications.csv", "text/csv")
        _download_button("Components CSV", st.session_state.components_df.to_csv(index=False).encode("utf-8"), "components.csv", "text/csv")
    with col2:
        _download_button("Risks CSV", st.session_state.risks_df.to_csv(index=False).encode("utf-8"), "risks.csv", "text/csv")
        _download_button("Mitigations CSV", st.session_state.mitigations_df.to_csv(index=False).encode("utf-8"), "mitigations.csv", "text/csv")

    st.success("Exports ready. Keep JSON under version control after each workshop session.")
