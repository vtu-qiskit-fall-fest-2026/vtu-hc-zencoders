"""Interactive Streamlit Dashboard for Q-Nav.

Provides 2D mission visualization, real-time simulation controls, side-by-side
Classical vs Quantum trajectory rendering, comparative performance analytics,
and training loss curves.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path to enable smooth imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from config import (
    DEFAULT_CONFIG,
    MAPS_DIR,
    SCENARIOS_DIR,
    METRICS_DIR,
    TRAJECTORIES_DIR,
    LOGS_DIR,
    EXPLANATION_LOG_PATH,
    EXPLANATION_SUMMARY_PATH,
    DYNAMIC_TELEMETRY_PATH,
    HARDWARE_METRICS_PATH,
    HARDWARE_CSV_PATH,
    FEATURES_CSV_PATH,
    QNavConfig,
)
from utils.io_utils import load_json, load_csv
from simulation.environment import UAVEnvironment, verify_solvability
from classical.baseline import ClassicalBaselineNavigator
from quantum.quantum_decision import QuantumDecisionMaker
from navigation.controller import NavigationController
from evaluation.metrics import compute_episode_metrics
from dashboard.visualization import plot_trajectory_comparison, plot_dynamic_trajectory_comparison
from dashboard.charts import plot_metric_bar_comparison


# Page configuration
st.set_page_config(
    page_title="Q-Nav — Quantum UAV Navigation",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-val {
        color: #F8FAFC;
        font-size: 24px;
        font-weight: bold;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_all_scenarios_and_maps():
    """Load scenario and map JSON definitions including dynamic scenarios."""
    scenarios_file = SCENARIOS_DIR / "test_scenarios.json"
    scenarios = load_json(scenarios_file).get("scenarios", [])

    for dyn_file in sorted(SCENARIOS_DIR.glob("scenario_dynamic_*.json")):
        d_data = load_json(dyn_file)
        if "id" not in d_data:
            d_data["id"] = dyn_file.stem
        scenarios.append(d_data)

    maps = {}
    for map_file in MAPS_DIR.glob("*.json"):
        m_data = load_json(map_file)
        maps[m_data.get("map_id", map_file.stem)] = m_data

    return scenarios, maps


def run_simulation(scenario, map_data, model_type, config, is_hybrid=False, explain_decisions=False):
    """Execute live simulation episode and return result and metrics."""
    is_dyn = bool(scenario.get("type") == "dynamic" or "obstacles" in scenario)
    if not map_data:
        bounds = scenario.get("bounds", [config.sim.map_width, config.sim.map_height])
        map_data = {
            "map_id": scenario.get("map_id", scenario.get("id", "map_01")),
            "dimensions": {"width": float(bounds[0]), "height": float(bounds[1])},
            "obstacles": [],
        }

    env = UAVEnvironment(
        config=config,
        map_data=map_data,
        scenario_data=scenario,
        dynamic_obstacles=is_dyn,
    )
    start_pos = tuple(scenario.get("start", (5.0, 5.0)))
    target_pos = tuple(scenario.get("target", (45.0, 45.0)))
    max_steps = int(scenario.get("max_steps", config.sim.max_steps))

    m_lower = model_type.lower()
    if m_lower == "classical":
        navigator = ClassicalBaselineNavigator(config=config, obstacles=env.obstacles)
        hybrid_flag = False
    elif m_lower == "hybrid":
        navigator = QuantumDecisionMaker(config=config)
        hybrid_flag = True
    else:
        navigator = QuantumDecisionMaker(config=config)
        hybrid_flag = is_hybrid or getattr(config.navigation, "use_hybrid", False)

    controller = NavigationController(
        environment=env,
        navigator=navigator,
        config=config,
        is_hybrid=hybrid_flag,
        explain_decisions=explain_decisions,
    )
    controller.reset(start_position=start_pos, target_position=target_pos)

    t0 = time.perf_counter()
    ep_result = controller.run_episode(max_steps=max_steps)
    duration = time.perf_counter() - t0

    metrics = compute_episode_metrics(ep_result, start_pos, target_pos, duration)
    return ep_result, metrics



def main():
    st.title("🛸 Q-Nav — Quantum-Assisted UAV Navigation")
    st.caption("Controlled Benchmarking Platform: Classical Potential Fields vs Variational Quantum Circuits")

    # Mandatory Scientific Disclaimer
    st.info(
        "**Scientific Notice:** Quantum-assisted results are produced using a Qiskit simulator. "
        "They do not imply quantum computational advantage over the classical baseline."
    )

    scenarios, maps_by_id = load_all_scenarios_and_maps()

    # Sidebar controls
    st.sidebar.header("Mission Controls")
    model_mode = st.sidebar.radio(
        "Comparison Mode",
        ["Both (Side-by-Side)", "Classical APF Only", "Quantum-Assisted Only"],
        index=0,
    )

    use_hybrid = st.sidebar.checkbox(
        "Enable Hybrid Decision Fusion (70% Quantum + 30% Classical Safety)",
        value=True,
        help="Combines quantum high-level preferences with classical safety and goal-seeking guidance among safe actions.",
    )

    # Scenario Selection (Improvement 4)
    st.sidebar.header("Scenario Selection")
    scenario_choice = st.sidebar.selectbox(
        "Choose Scenario",
        options=[
            "scenario_01", "scenario_02", "scenario_03", "scenario_04", "scenario_05",
            "scenario_dynamic_01", "scenario_dynamic_02", "scenario_dynamic_03",
        ],
        format_func=lambda x: x.replace("_", " ").replace("0", "").title(),
    )

    import config as cfg
    scenario_file = cfg.SCENARIOS_DIR / f"{scenario_choice}.json"
    if scenario_file.exists():
        scenario_data = load_json(scenario_file)
    else:
        scenario_data = next((s for s in scenarios if s["id"] == scenario_choice), scenarios[0])

    selected_scenario = scenario_data
    selected_scn_id = selected_scenario.get("id", scenario_choice)
    map_id = selected_scenario.get("map_id", "map_01")
    selected_map = maps_by_id.get(map_id, {})

    st.sidebar.write(f"**Selected:** {scenario_data.get('description', '')}")
    st.sidebar.write(f"**Difficulty:** {scenario_data.get('difficulty', 'Medium')}")
    st.sidebar.write(f"**Obstacles:** {scenario_data.get('obstacle_type', 'Static')}")
    st.sidebar.markdown(f"**Map:** `{map_id}` — *{selected_map.get('name', '')}*")
    st.sidebar.markdown(f"**Start:** `{selected_scenario.get('start')}` | **Target:** `{selected_scenario.get('target')}`")
    st.sidebar.markdown(f"**Max Steps:** `{selected_scenario.get('max_steps')}`")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Parameters")
    drone_speed = st.sidebar.slider("Drone Speed (units/step)", 0.5, 3.0, 1.5, 0.1)
    k_att = st.sidebar.slider("Classical APF Attractive Coeff", 0.5, 3.0, 1.2, 0.1)
    k_rep = st.sidebar.slider("Classical APF Repulsive Coeff", 50.0, 300.0, 150.0, 10.0)
    n_qubits = st.sidebar.slider("Quantum Circuit Qubits", 4, 8, 4, 1)
    n_layers = st.sidebar.slider("Quantum Ansatz Layers", 1, 3, 2, 1)

    explain_decisions = st.sidebar.checkbox("🧠 Enable Decision Explanations (XAI)", value=True)

    # Build active configuration from overrides
    sim_config = QNavConfig()
    sim_config.sim.drone_speed = drone_speed
    sim_config.classical.attractive_coefficient = k_att
    sim_config.classical.repulsive_coefficient = k_rep
    sim_config.quantum.n_qubits = n_qubits
    sim_config.quantum.n_layers = n_layers
    sim_config.navigation.use_hybrid = use_hybrid
    sim_config.navigation.explain_decisions = explain_decisions

    run_btn = st.sidebar.button("🚀 Run Live Navigation Experiment", type="primary")

    # State persistence for experiment results
    if "experiment_results" not in st.session_state or run_btn:
        with st.spinner("Executing Classical and Quantum simulation episodes..."):
            c_res, c_met = run_simulation(selected_scenario, selected_map, "classical", sim_config)
            q_res, q_met = run_simulation(
                selected_scenario,
                selected_map,
                "quantum",
                sim_config,
                is_hybrid=use_hybrid,
                explain_decisions=explain_decisions,
            )
            st.session_state["experiment_results"] = {
                "scenario": selected_scenario,
                "map": selected_map,
                "classical": {"res": c_res, "met": c_met},
                "quantum": {"res": q_res, "met": q_met},
            }

    data = st.session_state["experiment_results"]
    c_met = data["classical"]["met"]
    q_met = data["quantum"]["met"]

    # Dashboard Tabs
    (
        tab_traj,
        tab_dynamic,
        tab_hardware,
        tab_bench,
        tab_explain,
        tab_why_quantum,
        tab_train,
        tab_circuit,
        tab_telemetry,
        tab_method,
    ) = st.tabs([
        "📍 Trajectory & Mission Map",
        "🛸 Dynamic Obstacles",
        "⚛️ IBM Quantum Hardware",
        "📊 Comparative Metrics",
        "🧠 Explain Decisions",
        "🔍 Why Quantum Lost",
        "📈 Training Loss & VQC Optimization",
        "🔬 Quantum Architecture",
        "📋 Telemetry & Features",
        "📖 Methodology & Honest Disclosure",
    ])


    with tab_traj:
        # Key performance indicators
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Classical Success", "REACHED" if c_met["success"] else "FAILED")
            st.metric("Quantum Success", "REACHED" if q_met["success"] else "FAILED")
        with col2:
            st.metric("Classical Path Length", f"{c_met['path_length']:.1f}")
            st.metric("Quantum Path Length", f"{q_met['path_length']:.1f}", delta=f"{q_met['path_length'] - c_met['path_length']:.1f}", delta_color="inverse")
        with col3:
            st.metric("Classical Steps", f"{c_met['steps']}")
            st.metric("Quantum Steps", f"{q_met['steps']}", delta=f"{q_met['steps'] - c_met['steps']}", delta_color="inverse")
        with col4:
            st.metric("Classical Collisions", f"{c_met['collision_count']}")
            st.metric("Quantum Collisions", f"{q_met['collision_count']}", delta=f"{q_met['collision_count'] - c_met['collision_count']}", delta_color="inverse")
        with col5:
            st.metric("Classical Efficiency", f"{c_met['navigation_efficiency']:.2%}")
            st.metric("Quantum Efficiency", f"{q_met['navigation_efficiency']:.2%}", delta=f"{(q_met['navigation_efficiency'] - c_met['navigation_efficiency']):.2%}")

        st.markdown("---")

        # Select which trajectories to render based on model_mode
        c_traj = data["classical"]["res"]["trajectory"] if "Classical" in model_mode or "Both" in model_mode else []
        q_traj = data["quantum"]["res"]["trajectory"] if "Quantum" in model_mode or "Both" in model_mode else []

        if selected_scenario.get("type") == "dynamic" or "obstacles" in selected_scenario:
            d_telem = data["classical"]["res"].get("dynamic_telemetry_history", [])
            fig = plot_dynamic_trajectory_comparison(
                scenario=data["scenario"],
                classical_trajectory=c_traj,
                quantum_trajectory=q_traj,
                dynamic_telemetry=d_telem,
                title=f"Dynamic Flight Trajectories: {data['scenario']['name']}",
            )
        else:
            fig = plot_trajectory_comparison(
                scenario=data["scenario"],
                map_data=data["map"],
                classical_trajectory=c_traj,
                quantum_trajectory=q_traj,
                title=f"Flight Trajectories: {data['scenario']['name']}",
            )
        st.pyplot(fig)
        plt.close(fig)

    with tab_dynamic:
        st.subheader("🛸 Dynamic & Moving Obstacle Navigation")
        st.caption(
            "Benchmark and visualize autonomous UAV navigation in environments with moving obstacles "
            "(linear crossing, circular orbits, and stochastic random walks)."
        )

        col_d1, col_d2, col_d3 = st.columns([2, 1, 1])
        with col_d1:
            dyn_scenario_choice = st.selectbox(
                "Dynamic Scenario",
                options=[
                    "scenario_dynamic_01",
                    "scenario_dynamic_02",
                    "scenario_dynamic_03",
                    "scenario_dynamic_04",
                    "scenario_dynamic_05",
                ],
                format_func=lambda s: {
                    "scenario_dynamic_01": "Dynamic 01: Linear Crossing Obstacle",
                    "scenario_dynamic_02": "Dynamic 02: Circular Orbiting Obstacle",
                    "scenario_dynamic_03": "Dynamic 03: Deterministic Random Walk",
                    "scenario_dynamic_04": "Dynamic 04: Dual Opposing Linear Patrol",
                    "scenario_dynamic_05": "Dynamic 05: Orbital-Linear Multi-Hazard",
                }.get(s, s),
                key="dyn_scn_select",
            )
        with col_d2:
            dyn_seed = st.number_input("Random Seed", value=42, step=1, key="dyn_seed_input")
        with col_d3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_dyn_btn = st.button("🚀 Run 3-Model Benchmark", type="primary", key="dyn_run_btn")

        # Load dynamic scenario
        dyn_scn_file = SCENARIOS_DIR / f"{dyn_scenario_choice}.json"
        dyn_scn_data = load_json(dyn_scn_file) if dyn_scn_file.exists() else {}

        # Solvability verification badge
        if dyn_scn_data:
            solv_ok, solv_msg = verify_solvability(dyn_scn_data)
            if solv_ok:
                st.success(f"✅ Target Reachability Verified: {solv_msg}")
            else:
                st.error(f"❌ Solvability Verification Failed: {solv_msg}")

        # Session state for dynamic results
        dyn_key = f"dyn_res_{dyn_scenario_choice}_{dyn_seed}"
        if dyn_key not in st.session_state or run_dyn_btn:
            with st.spinner("Executing 3-Model Dynamic Navigation (Classical APF, Pure Quantum, Hybrid)..."):
                from utils.random_utils import set_seed
                set_seed(int(dyn_seed))
                c_d_res, c_d_met = run_simulation(dyn_scn_data, {}, "classical", sim_config)
                set_seed(int(dyn_seed))
                q_d_res, q_d_met = run_simulation(dyn_scn_data, {}, "quantum", sim_config, is_hybrid=False)
                set_seed(int(dyn_seed))
                h_d_res, h_d_met = run_simulation(dyn_scn_data, {}, "hybrid", sim_config, is_hybrid=True)

                st.session_state[dyn_key] = {
                    "classical": {"res": c_d_res, "met": c_d_met},
                    "quantum": {"res": q_d_res, "met": q_d_met},
                    "hybrid": {"res": h_d_res, "met": h_d_met},
                }

        dyn_data = st.session_state[dyn_key]
        cd_met = dyn_data["classical"]["met"]
        qd_met = dyn_data["quantum"]["met"]
        hd_met = dyn_data["hybrid"]["met"]

        # KPI Metrics Cards (Classical vs Pure Quantum vs Hybrid)
        st.markdown("### 3-Model Performance Comparison")
        kcol1, kcol2, kcol3, kcol4, kcol5, kcol6 = st.columns(6)
        with kcol1:
            st.metric("Classical Success", "REACHED" if cd_met.get("success") else "FAILED")
            st.metric("Quantum Success", "REACHED" if qd_met.get("success") else "FAILED")
            st.metric("Hybrid Success", "REACHED" if hd_met.get("success") else "FAILED")
        with kcol2:
            st.metric("Classical Collisions", f"{cd_met.get('collision_count', 0)}")
            st.metric("Quantum Collisions", f"{qd_met.get('collision_count', 0)}")
            st.metric("Hybrid Collisions", f"{hd_met.get('collision_count', 0)}")
        with kcol3:
            st.metric("Classical Min Moving Clr", f"{cd_met.get('min_moving_obstacle_clearance', 0.0):.2f}")
            st.metric("Quantum Min Moving Clr", f"{qd_met.get('min_moving_obstacle_clearance', 0.0):.2f}")
            st.metric("Hybrid Min Moving Clr", f"{hd_met.get('min_moving_obstacle_clearance', 0.0):.2f}")
        with kcol4:
            st.metric("Classical Pred. Colls", f"{cd_met.get('predicted_collision_count', 0)}")
            st.metric("Quantum Pred. Colls", f"{qd_met.get('predicted_collision_count', 0)}")
            st.metric("Hybrid Pred. Colls", f"{hd_met.get('predicted_collision_count', 0)}")
        with kcol5:
            st.metric("Classical Interventions", f"{cd_met.get('safety_interventions', 0)}")
            st.metric("Quantum Interventions", f"{qd_met.get('safety_interventions', 0)}")
            st.metric("Hybrid Interventions", f"{hd_met.get('safety_interventions', 0)}")
        with kcol6:
            st.metric("Classical Decision Time", f"{cd_met.get('mean_decision_time', 0.0)*1000:.2f} ms")
            st.metric("Quantum Decision Time", f"{qd_met.get('mean_decision_time', 0.0)*1000:.2f} ms")
            st.metric("Hybrid Decision Time", f"{hd_met.get('mean_decision_time', 0.0)*1000:.2f} ms")

        st.markdown("---")
        st.subheader("Dynamic Flight Trajectories & Moving Obstacle Vectors")

        # Extract trajectories & dynamic telemetry
        cd_traj = dyn_data["classical"]["res"].get("trajectory", [])
        qd_traj = dyn_data["quantum"]["res"].get("trajectory", [])
        hd_traj = dyn_data["hybrid"]["res"].get("trajectory", [])
        dyn_telem_list = dyn_data["classical"]["res"].get("dynamic_telemetry_history", [])

        fig_dyn = plot_dynamic_trajectory_comparison(
            scenario=dyn_scn_data,
            classical_trajectory=cd_traj,
            quantum_trajectory=qd_traj,
            hybrid_trajectory=hd_traj,
            dynamic_telemetry=dyn_telem_list,
            title=f"Dynamic Airspace: {dyn_scn_data.get('name', dyn_scenario_choice)} (Seed: {dyn_seed})",
        )
        st.pyplot(fig_dyn)
        plt.close(fig_dyn)

        # Dynamic Telemetry Table
        st.markdown("---")
        st.subheader("📋 Dynamic Obstacle Telemetry Log")
        st.caption("Per-step relative kinematics, proactive swept collision predictions, and safety interventions.")

        telemetry_model = st.radio(
            "Show Telemetry For Model:",
            ["Classical APF", "Pure Quantum", "Hybrid Quantum-Assisted"],
            horizontal=True,
            key="telem_model_choice",
        )
        m_map = {
            "Classical APF": "classical",
            "Pure Quantum": "quantum",
            "Hybrid Quantum-Assisted": "hybrid",
        }
        selected_res = dyn_data[m_map[telemetry_model]]["res"]
        selected_telem = selected_res.get("dynamic_telemetry_history", [])

        if selected_telem:
            df_telem = pd.DataFrame(selected_telem)
            st.dataframe(df_telem, width="stretch", height=300)
            csv_data = df_telem.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {telemetry_model} Dynamic Telemetry CSV",
                data=csv_data,
                file_name=f"dynamic_telemetry_{m_map[telemetry_model]}_{dyn_scenario_choice}.csv",
                mime="text/csv",
            )
        else:
            st.info("No dynamic obstacle telemetry records for this episode.")

        # Action Divergence & Telemetry Breakdown (Pure Quantum vs Hybrid)
        st.markdown("---")
        st.subheader("🔬 Action Divergence Analysis: Pure Quantum vs Hybrid")
        st.caption(
            "Empirical side-by-side verification proving whether Pure Quantum and Hybrid models produce "
            "genuinely divergent decisions under identical initial conditions."
        )

        q_hist = dyn_data["quantum"]["res"].get("telemetry_history", [])
        h_hist = dyn_data["hybrid"]["res"].get("telemetry_history", [])

        # Build step-by-step comparison
        comp_steps = []
        div_count = 0
        min_len = min(len(q_hist), len(h_hist))
        for i in range(min_len):
            qs = q_hist[i]
            hs = h_hist[i]
            q_act = qs.get("quantum_action", qs.get("action", "STAY"))
            h_act = hs.get("hybrid_action", hs.get("action", "STAY"))
            c_act = hs.get("classical_action", qs.get("classical_action", "STAY"))
            is_div = (q_act != h_act)
            if is_div:
                div_count += 1
            comp_steps.append({
                "Step": qs.get("step", i + 1),
                "Classical APF Action": c_act,
                "Pure Quantum Action": q_act,
                "Hybrid Action": h_act,
                "Final Executed (Hybrid)": hs.get("final_action", hs.get("action", "STAY")),
                "Divergence": "⚡ DIVERGENT" if is_div else "✅ MATCH",
                "Pure Q Confidence": round(float(qs.get("confidence", 1.0)), 3),
                "Hybrid Confidence": round(float(hs.get("confidence", 1.0)), 3),
                "Safety Override": "YES" if hs.get("safety_override") or hs.get("was_overridden") else "NO",
            })

        col_div1, col_div2, col_div3, col_div4 = st.columns(4)
        div_rate = (div_count / min_len * 100.0) if min_len > 0 else 0.0
        with col_div1:
            st.metric("Total Compared Steps", f"{min_len}")
        with col_div2:
            st.metric("Divergent Decision Steps", f"{div_count}", delta=f"{div_rate:.1f}% Divergence")
        with col_div3:
            st.metric("Pure Quantum Steps", f"{len(q_hist)}")
        with col_div4:
            st.metric("Hybrid Steps", f"{len(h_hist)}")

        if comp_steps:
            df_comp = pd.DataFrame(comp_steps)
            st.dataframe(df_comp, width="stretch", height=280)


    with tab_hardware:
        st.subheader("⚛️ Real IBM Quantum Hardware Validation & Noise Characterization")
        st.markdown(
            """
            Validate the exact Q-Nav parameterized quantum decision circuit on **IBM Quantum hardware**.
            Characterizes finite-shot sampling variance, device noise, transpilation depth expansion,
            and distribution deviation from ideal statevector simulation.
            """
        )

        # Load hardware validation metrics if available
        hw_data = None
        if HARDWARE_METRICS_PATH.exists():
            try:
                hw_data = load_json(HARDWARE_METRICS_PATH)
            except Exception as e:
                logger.warning(f"Failed to load hardware validation metrics: {e}")

        if hw_data and "metadata" in hw_data:
            meta = hw_data.get("metadata", {})
            fid = hw_data.get("fidelity_summary", {})
            samples_list = hw_data.get("samples", [])
            is_real = meta.get("is_real_hardware", False)

            # Execution badge
            col_b1, col_b2 = st.columns([1, 4])
            with col_b1:
                if is_real:
                    st.markdown(
                        '<div style="background-color: #6b21a8; color: white; padding: 6px 16px; '
                        'border-radius: 8px; font-weight: bold; text-align: center; font-size: 14px;">'
                        '⚡ REAL HARDWARE</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="background-color: #1e40af; color: white; padding: 6px 16px; '
                        'border-radius: 8px; font-weight: bold; text-align: center; font-size: 14px;">'
                        '🖥️ SIMULATOR VALIDATION</div>',
                        unsafe_allow_html=True,
                    )
            with col_b2:
                st.caption(f"Backend: **{meta.get('backend_name', 'N/A')}** | Qubits: **{meta.get('num_qubits', 4)}** | Configured Shots: **{meta.get('shots', 1024)}** | Timestamp: `{meta.get('timestamp', 'N/A')}`")

            st.markdown("---")

            # High-Level Fidelity Metric Cards
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            with mcol1:
                st.metric("Action Agreement", f"{fid.get('action_agreement_rate', 0.0):.1%}")
            with mcol2:
                st.metric("Mean JS Divergence", f"{fid.get('mean_js_divergence', 0.0):.4f}")
            with mcol3:
                st.metric("Mean Abs Prob Diff", f"{fid.get('mean_absolute_probability_difference', 0.0):.4f}")
            with mcol4:
                st.metric("Invalid Probability Mass", f"{fid.get('mean_invalid_probability_mass', 0.0):.1%}")
            with mcol5:
                st.metric("Depth (Logical → HW)", f"{fid.get('mean_logical_depth', 0.0):.0f} → {fid.get('mean_transpiled_depth', 0.0):.0f}")

            st.markdown("---")

            # Sample Inspection
            if samples_list:
                sample_ids = [s["sample_id"] for s in samples_list]
                selected_sample_id = st.selectbox("Select Validation Sample for Deep Inspection:", sample_ids, key="hw_sample_select")
                curr_sample = next((s for s in samples_list if s["sample_id"] == selected_sample_id), samples_list[0])

                m_curr = curr_sample.get("metrics", {})
                hw_curr = curr_sample.get("hardware_result", {})
                sim_curr = curr_sample.get("simulator_result", {})

                st.markdown(f"#### Sample Detail: `{selected_sample_id}` (Scenario: `{curr_sample.get('scenario_id')}`)")
                scol1, scol2, scol3, scol4 = st.columns(4)
                with scol1:
                    sim_a = sim_curr.get("action_name", str(m_curr.get("simulator_action")))
                    st.metric("Simulator Action", sim_a)
                with scol2:
                    hw_a = hw_curr.get("action_name", str(m_curr.get("hardware_action")))
                    st.metric("Hardware Action", hw_a, delta="MATCH" if m_curr.get("action_agreement") else "DIVERGENCE")
                with scol3:
                    st.metric("Simulator Confidence", f"{m_curr.get('simulator_confidence', 0.0):.1%}")
                with scol4:
                    st.metric("Hardware Confidence", f"{m_curr.get('hardware_confidence', 0.0):.1%}")

                # Probability Distribution Comparison Chart (Section 35)
                st.markdown("##### Action Probability Distribution: Simulator vs Real Hardware")
                action_names = DEFAULT_CONFIG.navigation.action_names
                sim_p = sim_curr.get("action_probabilities", [0.0]*9)
                hw_p = hw_curr.get("action_probabilities", [0.0]*9)

                fig_prob = go.Figure()
                fig_prob.add_trace(go.Bar(
                    x=action_names,
                    y=sim_p,
                    name="Simulator (Ideal Statevector)",
                    marker_color="#2563eb",
                ))
                fig_prob.add_trace(go.Bar(
                    x=action_names,
                    y=hw_p,
                    name=f"Hardware ({hw_curr.get('backend', 'IBM Backend')})",
                    marker_color="#9333ea",
                ))

                fig_prob.update_layout(
                    barmode="group",
                    title="Action Probability Comparison (9 Valid Navigation Actions)",
                    xaxis_title="Discrete Action (Compass Heading)",
                    yaxis_title="Normalized Action Probability",
                    template="plotly_white",
                    height=380,
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # Invalid Probability Mass callout
                inv_mass = hw_curr.get("invalid_probability_mass", 0.0)
                st.info(
                    f"**Invalid Computational State Tracking:** Unmapped basis states (|1001⟩ through |1111⟩) "
                    f"account for **{inv_mass:.2%}** of total physical measurement shots. "
                    "These invalid states are tracked separately to avoid distorting navigational decisions."
                )

                # Job and Transpilation details
                st.markdown("##### Job & Circuit Transpilation Telemetry")
                jcol1, jcol2, jcol3, jcol4 = st.columns(4)
                with jcol1:
                    st.text_input("Job ID", value=str(hw_curr.get("job_id", "N/A")), disabled=True)
                with jcol2:
                    st.text_input("Circuit Hash (SHA256)", value=str(hw_curr.get("circuit_hash", "N/A")), disabled=True)
                with jcol3:
                    st.text_input("Logical Depth / Gates", value=f"{hw_curr.get('logical_depth')} / {hw_curr.get('logical_gate_count', 'N/A')}", disabled=True)
                with jcol4:
                    st.text_input("Transpiled Depth / Gates", value=f"{hw_curr.get('transpiled_depth')} / {hw_curr.get('transpiled_gate_count', 'N/A')}", disabled=True)

            # Table of All Hardware Records
            if HARDWARE_CSV_PATH.exists():
                st.markdown("---")
                st.markdown("#### Hardware Validation Log (CSV)")
                df_hw = pd.read_csv(HARDWARE_CSV_PATH)
                st.dataframe(df_hw, width="stretch", height=250)
                csv_bytes = df_hw.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hardware Validation CSV",
                    data=csv_bytes,
                    file_name="hardware_validation.csv",
                    mime="text/csv",
                )
        else:
            st.info(
                "No hardware validation results found yet. Run hardware validation via the CLI:\n\n"
                "```bash\n"
                "# 1. List available operational backends:\n"
                "python run.py --list-hardware-backends\n\n"
                "# 2. Run hardware validation on real IBM Quantum hardware:\n"
                "python run.py --hardware-validation --real-hardware --shots 1024 --hardware-samples 3\n\n"
                "# Or run simulated hardware validation (no IBM token required):\n"
                "python run.py --hardware-validation --shots 1024 --hardware-samples 3\n"
                "```"
            )

    with tab_bench:
        st.subheader("Numeric Performance Comparison")

        # Clean dataframe with consistent numeric float values to completely prevent ArrowTypeError
        comparison_df = pd.DataFrame({
            "Metric": [
                "Success Rate",
                "Mean Path Length",
                "Mean Steps",
                "Mean Collisions",
                "Collision Rate",
                "Navigation Efficiency",
                "Safety Interventions",
                "Unsafe Action Attempts",
                "Boundary Violations",
                "Min Obstacle Clearance",
                "Min Boundary Clearance",
                "Heading Smoothness",
                "Action Switches",
                "Mean Decision Time (s)",
                "Mean Episode Time (s)",
            ],
            "Classical Baseline": [
                float(c_met.get("success_rate", 0.0)),
                float(c_met.get("path_length", 0.0)),
                float(c_met.get("steps", 0)),
                float(c_met.get("collision_count", 0)),
                float(c_met.get("collision_rate", 0.0)),
                float(c_met.get("navigation_efficiency", 0.0)),
                float(c_met.get("safety_interventions", 0)),
                float(c_met.get("unsafe_action_attempts", 0)),
                float(c_met.get("boundary_violations", 0)),
                float(c_met.get("min_obstacle_clearance", 0.0)),
                float(c_met.get("min_boundary_clearance", 0.0)),
                float(c_met.get("heading_smoothness", 0.0)),
                float(c_met.get("action_switch_count", 0)),
                float(c_met.get("mean_decision_time", 0.0)),
                float(c_met.get("total_episode_time", c_met.get("duration_seconds", 0.0))),
            ],
            "Quantum-Assisted": [
                float(q_met.get("success_rate", 0.0)),
                float(q_met.get("path_length", 0.0)),
                float(q_met.get("steps", 0)),
                float(q_met.get("collision_count", 0)),
                float(q_met.get("collision_rate", 0.0)),
                float(q_met.get("navigation_efficiency", 0.0)),
                float(q_met.get("safety_interventions", 0)),
                float(q_met.get("unsafe_action_attempts", 0)),
                float(q_met.get("boundary_violations", 0)),
                float(q_met.get("min_obstacle_clearance", 0.0)),
                float(q_met.get("min_boundary_clearance", 0.0)),
                float(q_met.get("heading_smoothness", 0.0)),
                float(q_met.get("action_switch_count", 0)),
                float(q_met.get("mean_decision_time", 0.0)),
                float(q_met.get("total_episode_time", q_met.get("duration_seconds", 0.0))),
            ],
        })

        st.dataframe(comparison_df, width="stretch", hide_index=True)

        st.markdown("### Radar Performance Profile")
        radar_categories = ["Efficiency", "Smoothness Norm", "Safety Compliance", "Progress Ratio", "Clearance Norm"]
        c_radar = [
            float(c_met.get("navigation_efficiency", 0.0)),
            float(max(0.0, 1.0 - min(1.0, c_met.get("heading_smoothness", 0.0) / 50.0))),
            float(max(0.0, 1.0 - min(1.0, c_met.get("safety_interventions", 0) / 30.0))),
            1.0 if c_met.get("success", False) else float(max(0.0, 1.0 - c_met.get("final_distance", 50.0) / 50.0)),
            float(min(1.0, c_met.get("min_obstacle_clearance", 0.0) / 10.0)),
        ]
        q_radar = [
            float(q_met.get("navigation_efficiency", 0.0)),
            float(max(0.0, 1.0 - min(1.0, q_met.get("heading_smoothness", 0.0) / 50.0))),
            float(max(0.0, 1.0 - min(1.0, q_met.get("safety_interventions", 0) / 30.0))),
            1.0 if q_met.get("success", False) else float(max(0.0, 1.0 - q_met.get("final_distance", 50.0) / 50.0)),
            float(min(1.0, q_met.get("min_obstacle_clearance", 0.0) / 10.0)),
        ]

        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(r=c_radar + [c_radar[0]], theta=radar_categories + [radar_categories[0]], fill='toself', name='Classical APF', line=dict(color='#38BDF8')))
        radar_fig.add_trace(go.Scatterpolar(r=q_radar + [q_radar[0]], theta=radar_categories + [radar_categories[0]], fill='toself', name='Quantum-Assisted', line=dict(color='#F97316')))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, paper_bgcolor='#0F172A', plot_bgcolor='#1E293B', font=dict(color='#F8FAFC'))
        st.plotly_chart(radar_fig, width="stretch")

        # Load aggregate benchmark summary if present
        comp_summary_file = METRICS_DIR / "comparison.json"
        if comp_summary_file.exists():
            st.markdown("### Aggregated Benchmark Overview (All Scenarios)")
            saved_comp = load_json(comp_summary_file)
            c_s = saved_comp.get("summary", {}).get("classical", {})
            q_s = saved_comp.get("summary", {}).get("quantum", {})
            if c_s and q_s:
                agg_df = pd.DataFrame({
                    "Metric": [
                        "Overall Success Rate",
                        "Mean Path Length",
                        "Mean Steps",
                        "Mean Collisions",
                        "Mean Navigation Efficiency",
                        "Mean Safety Interventions",
                        "Mean Episode Time (s)",
                    ],
                    "Classical Baseline": [
                        float(c_s.get("success_rate", 0.0)),
                        float(c_s.get("mean_path_length", 0.0)),
                        float(c_s.get("mean_steps", 0.0)),
                        float(c_s.get("mean_collisions", 0.0)),
                        float(c_s.get("mean_navigation_efficiency", c_s.get("mean_efficiency", 0.0))),
                        float(c_s.get("mean_safety_interventions", c_s.get("mean_safety_violations", 0.0))),
                        float(c_s.get("mean_episode_time", c_s.get("mean_duration_seconds", 0.0))),
                    ],
                    "Quantum-Assisted": [
                        float(q_s.get("success_rate", 0.0)),
                        float(q_s.get("mean_path_length", 0.0)),
                        float(q_s.get("mean_steps", 0.0)),
                        float(q_s.get("mean_collisions", 0.0)),
                        float(q_s.get("mean_navigation_efficiency", q_s.get("mean_efficiency", 0.0))),
                        float(q_s.get("mean_safety_interventions", q_s.get("mean_safety_violations", 0.0))),
                        float(q_s.get("mean_episode_time", q_s.get("mean_duration_seconds", 0.0))),
                    ],
                })
                st.dataframe(agg_df, width="stretch", hide_index=True)

    with tab_explain:
        st.subheader("🧠 Quantum Decision Explainability (XAI)")
        st.caption(
            "Local sensitivity-based feature influence estimation via central finite differences. "
            "Estimates which input features locally affected the quantum action probabilities. "
            "Note: Numerical sensitivity estimates local model response, not causal influence."
        )

        # 1. Source resolution: saved logs vs live experiment
        saved_logs = []
        if EXPLANATION_LOG_PATH.exists():
            try:
                saved_logs = load_json(EXPLANATION_LOG_PATH)
            except Exception:
                saved_logs = []

        live_logs = data.get("quantum", {}).get("res", {}).get("explanation_log", [])

        if not saved_logs and not live_logs:
            st.warning(
                "No explanation logs available. Ensure '🧠 Enable Decision Explanations (XAI)' is checked in the sidebar and click "
                "'Run Live Navigation Experiment', or run `python run.py --explain-decisions`."
            )
        else:
            col_src1, col_src2 = st.columns([2, 3])
            with col_src1:
                if saved_logs and live_logs:
                    src_choice = st.radio(
                        "Data Source",
                        ["Saved Benchmark Logs", "Live Simulation Session"],
                        horizontal=True,
                    )
                    active_logs = saved_logs if "Saved" in src_choice else live_logs
                elif saved_logs:
                    st.info("Displaying saved benchmark explanations.")
                    active_logs = saved_logs
                else:
                    st.info("Displaying live simulation session explanations.")
                    active_logs = live_logs

            # Scenario and Seed Selectors
            has_scenarios = any("scenario" in e for e in active_logs)
            if has_scenarios:
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    available_scns = sorted(list(set(e.get("scenario", selected_scn_id) for e in active_logs)))
                    selected_exp_scn = st.selectbox("1. Scenario Selector", available_scns, index=0)
                    logs_for_scn = [e for e in active_logs if e.get("scenario", selected_scn_id) == selected_exp_scn]
                with col_sel2:
                    available_seeds = sorted(list(set(e.get("seed", 42) for e in logs_for_scn)))
                    selected_exp_seed = st.selectbox("2. Seed Selector", available_seeds, index=0)
                    episode_logs = [e for e in logs_for_scn if e.get("seed", 42) == selected_exp_seed]
            else:
                episode_logs = active_logs

            if not episode_logs:
                st.warning("No explanation records found for the selected scenario/seed.")
            else:
                # 3. Step Selector
                step_numbers = [int(e.get("step", i)) for i, e in enumerate(episode_logs)]
                selected_step = st.select_slider(
                    "3. Step Selector",
                    options=step_numbers,
                    value=step_numbers[0],
                )
                curr_entry = next((e for e in episode_logs if int(e.get("step", 0)) == selected_step), episode_logs[0])

                # 14. Action-Conditioned Explanation Target
                col_act1, col_act2 = st.columns([2, 3])
                with col_act1:
                    action_target_mode = st.radio(
                        "Explanation Target",
                        ["Explain Quantum Top Action", "Explain Final Hybrid Action"],
                        horizontal=True,
                    )

                q_top = curr_entry.get("quantum_top_action", "STAY")
                final_act = curr_entry.get("final_action", curr_entry.get("selected_action", "STAY"))
                c_act = curr_entry.get("classical_action", "N/A")
                q_used = curr_entry.get("quantum_action_used", q_top == final_act)

                if "Final" in action_target_mode:
                    exp_data = curr_entry.get("final_action_explanation", curr_entry.get("explanation", {}))
                    target_action_name = final_act
                    if final_act != q_top:
                        st.warning(
                            f"⚠️ **Action Sensitivity Notice:** Estimating quantum model sensitivity to the final hybrid action (`{final_act}`). "
                            f"This does **NOT** mean the quantum model selected this action. The classical APF guidance or safety layer "
                            f"overrode the quantum model's recommendation (`{q_top}`)."
                        )
                else:
                    exp_data = curr_entry.get("explanation", {})
                    target_action_name = q_top

                # Metrics row: Classical, Quantum top, Final hybrid, Was quantum used, Action prob
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                with col_m1:
                    st.metric("4. Classical Action", str(c_act))
                with col_m2:
                    st.metric("5. Quantum Top Action", str(q_top))
                with col_m3:
                    st.metric("6. Final Hybrid Action", str(final_act))
                with col_m4:
                    st.metric("7. Quantum Action Used?", "YES (Retained)" if q_used else "NO (Overridden)")
                with col_m5:
                    act_prob = exp_data.get("action_probability")
                    if act_prob is not None:
                        st.metric("8. Action Probability", f"{float(act_prob):.3f}")
                    else:
                        st.metric("8. Target Action", str(target_action_name))

                # 9. Top 3 influential features
                st.markdown("#### Top 3 Influential Features (Saliency-Based Sensitivity)")
                top_3 = exp_data.get("top_features", [])
                if top_3:
                    col_t1, col_t2, col_t3 = st.columns(3)
                    for idx, c in enumerate([col_t1, col_t2, col_t3]):
                        if idx < len(top_3):
                            item = top_3[idx]
                            feat_name = item.get("feature", f"feature_{idx}")
                            imp_val = item.get("importance", 0.0)
                            grad_val = item.get("signed_gradient", 0.0)
                            with c:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-title">Rank #{idx+1} Feature</div>
                                        <div class="metric-val" style="font-size: 18px;">{feat_name}</div>
                                        <div style="color: #38BDF8; margin-top: 6px; font-weight: 600;">Estimated Influence: {imp_val:.2%}</div>
                                        <div style="color: {'#34D399' if grad_val >= 0 else '#F87171'}; font-size: 13px;">Signed Grad: {grad_val:+.4f}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                st.markdown("---")

                # 10. Full feature importance chart & 11. Signed gradient chart
                col_chart1, col_chart2 = st.columns(2)
                f_imp_dict = exp_data.get("feature_importance", {})
                s_grad_dict = exp_data.get("signed_gradients", {})

                with col_chart1:
                    st.markdown("##### 10. Full Feature Importance Chart")
                    if f_imp_dict:
                        df_imp = pd.DataFrame([
                            {"Feature": k, "Importance": float(v)}
                            for k, v in f_imp_dict.items()
                        ]).sort_values("Importance", ascending=True)

                        fig_imp = px.bar(
                            df_imp,
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title="Saliency-based estimated influence",
                            color="Importance",
                            color_continuous_scale="Blues",
                        )
                        fig_imp.update_layout(
                            xaxis_title="Estimated Feature Influence (Normalized)",
                            yaxis_title="Input Feature",
                            showlegend=False,
                            height=340,
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig_imp, width="stretch")
                    else:
                        st.info("No feature importance data for this step.")

                with col_chart2:
                    st.markdown("##### 11. Signed Gradient Chart")
                    if s_grad_dict:
                        df_grad = pd.DataFrame([
                            {"Feature": k, "Signed Gradient": float(v)}
                            for k, v in s_grad_dict.items()
                        ]).sort_values("Signed Gradient", ascending=True)

                        fig_grad = px.bar(
                            df_grad,
                            x="Signed Gradient",
                            y="Feature",
                            orientation="h",
                            title="Signed Gradient (Directional Local Influence)",
                            color="Signed Gradient",
                            color_continuous_scale="RdBu_r",
                        )
                        fig_grad.update_layout(
                            xaxis_title="Signed Local Sensitivity (dP/dx_i)",
                            yaxis_title="Input Feature",
                            showlegend=False,
                            height=340,
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig_grad, width="stretch")
                    else:
                        st.info("No signed gradient data for this step.")

                st.markdown("---")

                # 12 & 13. Feature Importance Over Time (Mission Aggregate)
                st.markdown("#### 12. Feature Importance Over Time (Mission Aggregate)")
                st.caption(
                    "Aggregated feature sensitivity across all flight steps. "
                    "Estimates which input signals consistently drove quantum preferences."
                )

                step_history = []
                rank1_tally = {}
                for ep in episode_logs:
                    s_num = int(ep.get("step", 0))
                    e_exp = ep.get("explanation", {})
                    e_imps = e_exp.get("feature_importance", {})
                    for fn, iv in e_imps.items():
                        step_history.append({"step": s_num, "feature": fn, "importance": float(iv)})
                    top_list = e_exp.get("top_features", [])
                    if top_list:
                        top1_name = top_list[0].get("feature")
                        rank1_tally[top1_name] = rank1_tally.get(top1_name, 0) + 1

                if step_history:
                    df_steps = pd.DataFrame(step_history)
                    agg_records = []
                    for fname, grp in df_steps.groupby("feature"):
                        arr = grp["importance"].values
                        agg_records.append({
                            "Feature": str(fname),
                            "Mean Importance": float(np.mean(arr)),
                            "Median Importance": float(np.median(arr)),
                            "Std Importance": float(np.std(arr)),
                        })
                    df_agg = pd.DataFrame(agg_records).sort_values("Mean Importance", ascending=False)

                    if rank1_tally:
                        dominant_f = max(rank1_tally, key=rank1_tally.get)
                        d_cnt = rank1_tally[dominant_f]
                        d_pct = d_cnt / len(episode_logs) * 100.0
                        st.success(
                            f"🏆 **Most Frequently Ranked #1 Feature:** `{dominant_f}` "
                            f"— dominant in **{d_cnt}/{len(episode_logs)}** flight steps ({d_pct:.1f}% of mission)"
                        )

                    st.dataframe(df_agg, width="stretch", hide_index=True)

                    # Dynamic Trajectory line chart
                    fig_dyn = px.line(
                        df_steps,
                        x="step",
                        y="importance",
                        color="feature",
                        title="Feature Sensitivity Trajectory Across Mission Steps",
                    )
                    fig_dyn.update_layout(
                        xaxis_title="Flight Step",
                        yaxis_title="Estimated Feature Influence",
                        height=360,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_dyn, width="stretch")

    with tab_why_quantum:
        st.header("🔍 Why Did Quantum Underperform?")

        st.markdown("""
        ### Key Factors:

        1. **Simulation Overhead**
           - Qiskit statevector simulation: O(2^N) complexity
           - 4 qubits = 16-dimensional statevector operations per decision
           - Classical APF: direct analytical calculation (much faster)

        2. **Feature Compression Loss**
           - 12-dimensional features → 4 qubits
           - Information bottleneck: critical navigation data lost
           - Classical APF uses full 12D feature vector

        3. **Probability Dispersion**
           - Quantum model spreads probability across all 9 actions
           - Indecisive in dense obstacle chokepoints
           - Classical APF has focused, directional force vectors

        4. **Training Mismatch**
           - Teacher policy trained on synthetic random obstacles
           - Test scenarios have structured obstacle patterns
           - Poor generalization to benchmark maps

        ### What Would Help:

        - ✅ Hybrid architecture (classical + quantum fusion)
        - ✅ Feature selection (12D → 4D using domain knowledge)
        - ✅ Fine-tuning on test scenarios
        - ⏳ Better quantum ansatz (hardware-efficient with entanglement)
        - ⏳ Real quantum hardware (IBM Quantum) testing
        """)

        # Show comparison table
        st.dataframe(comparison_df, width="stretch", hide_index=True)

    with tab_train:
        st.subheader("Variational Quantum Circuit Training Dynamics")
        train_file = METRICS_DIR / "training_metrics.json"
        if train_file.exists():
            t_data = load_json(train_file)
            epochs = list(range(1, t_data.get("epochs", 15) + 1))
            t_loss = t_data.get("history", {}).get("loss", [])
            v_loss = t_data.get("history", {}).get("val_loss", [])
            t_acc = t_data.get("history", {}).get("accuracy", [])
            v_acc = t_data.get("history", {}).get("val_accuracy", [])

            fig_loss = go.Figure()
            if t_loss:
                fig_loss.add_trace(go.Scatter(x=epochs[:len(t_loss)], y=t_loss, mode='lines+markers', name='Train Loss', line=dict(color='#38BDF8', width=2)))
            if v_loss:
                fig_loss.add_trace(go.Scatter(x=epochs[:len(v_loss)], y=v_loss, mode='lines+markers', name='Val Loss', line=dict(color='#F97316', width=2, dash='dash')))

            fig_loss.update_layout(
                title="Cross-Entropy Loss vs Epochs",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                paper_bgcolor='#0F172A',
                plot_bgcolor='#1E293B',
                font=dict(color='#F8FAFC'),
            )
            st.plotly_chart(fig_loss, width="stretch")

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown(f"- **Final Training Loss:** `{t_loss[-1]:.4f}`" if t_loss else "")
                st.markdown(f"- **Final Validation Loss:** `{v_loss[-1]:.4f}`" if v_loss else "")
            with col_t2:
                st.markdown(f"- **Final Training Accuracy:** `{t_acc[-1]:.2%}`" if t_acc else "")
                st.markdown(f"- **Final Validation Accuracy:** `{v_acc[-1]:.2%}`" if v_acc else "")
        else:
            st.info("No training_metrics.json found. Run `python run.py --train-quantum` to generate VQC training curves.")

    with tab_circuit:
        st.subheader("Parameterized Variational Quantum Circuit (VQC)")
        circuit_summary_text = (
            f"- **Framework:** Qiskit 2.x\n"
            f"- **Qubits:** `{n_qubits}`\n"
            f"- **Ansatz Layers:** `{sim_config.quantum.n_layers}`\n"
            f"- **Feature Encoding:** RY + RZ rotation gates (Grouped / Linear)\n"
            f"- **Entanglement Topology:** Ring Circular CNOT ladder (q_i -> q_{{(i+1) % N}})\n"
            f"- **Observables:** Pauli-Z expectations <Z_i> on each wire\n"
            f"- **Action Projection:** Linear weight matrix W x <Z> + b -> Softmax Action Probabilities"
        )
        st.markdown(circuit_summary_text)

        st.markdown("#### 12-Dimensional Feature to Qubit Semantic Mapping:")
        st.markdown(
            """
            | Qubit | Subsystem | Features Encoded | Physical Rotation |
            | :---: | :--- | :--- | :--- |
            | **q0** | Target Guidance | Target distance & relative bearing (sin, cos) | $R_y(\\theta_{target}), R_z(\\lambda_{bearing})$ |
            | **q1** | Sector Clearance | Nearest obstacle & frontal sector clearance | $R_y(\\theta_{nearest}), R_z(\\theta_{front})$ |
            | **q2** | Flank Clearance | Left & Right lateral sector obstacle distances | $R_y(\\theta_{left}), R_z(\\theta_{right})$ |
            | **q3** | Kinematics & Boundary | Obstacle density, normalized speed & boundary margin | $R_y(\\theta_{boundary}), R_z(\\theta_{speed})$ |
            """
        )

        q_history = data["quantum"]["res"]["telemetry_history"]
        if q_history:
            st.markdown("#### Sample Step Quantum Action Probabilities:")
            sample_step = q_history[0]
            st.json(sample_step.get("action_probabilities", {}))

    with tab_telemetry:
        st.subheader("Step-by-Step Trajectory Telemetry")
        q_hist = data["quantum"]["res"]["telemetry_history"]
        if q_hist:
            telemetry_df = pd.DataFrame(q_hist)
            display_cols = [c for c in ["step", "drone_x", "drone_y", "action", "raw_action", "was_overridden", "confidence", "decision_time", "circuit_time", "distance_to_target", "nearest_obstacle_distance"] if c in telemetry_df.columns]
            st.dataframe(telemetry_df[display_cols], width="stretch")

            csv_download = telemetry_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Telemetry CSV",
                data=csv_download,
                file_name=f"telemetry_{selected_scn_id}.csv",
                mime="text/csv",
            )

    with tab_method:
        st.subheader("Methodology & Honest Scientific Disclosure")
        st.markdown(
            """
            ### Scientific Findings & Disclosure
            - **No Quantum Advantage:** Under current benchmark scenarios and depth-limited simulator execution, the classical APF baseline outperforms the variational quantum model in success rate, efficiency, and execution time.
            - **Honest Metrics:** All reported scores are measured directly from the simulation environment without artificial boosting.
            - **Shared Constraints:** Both models operate under strictly identical physics, obstacle maps, speed, maximum step budgets, and safety intervention rules.
            - **Local-Minimum Recovery:** Classical APF incorporates stagnation detection and tangential circulation forces ($F_{tan}$) to navigate around concave clusters.
            - **Classical Action Projection:** In the hybrid quantum architecture, the projection from Pauli-Z expectations to action logits is linear classical post-processing.
            """
        )


if __name__ == "__main__":
    main()
