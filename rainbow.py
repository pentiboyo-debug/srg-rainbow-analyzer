import streamlit as st
import numpy as np
import math
import pandas as pd
import plotly.graph_objects as go

# --- 1. Page Configuration & Compressed UI CSS ---
st.set_page_config(page_title="SRG OC Rainbow Noise Analyzer", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }
    [data-testid="stSidebar"] .st-emotion-cache-1wivap2, [data-testid="stSidebar"] .st-emotion-cache-1ob1npu { gap: 0.4rem !important; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] .stMarkdown p { font-size: 13px !important; margin-bottom: 0px !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] .stNumberInput input { font-size: 13px !important; height: 28px !important; padding: 0px 5px !important; }
    [data-testid="stSidebar"] .stSlider { margin-bottom: -15px !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌈 Out-Coupler Ambient Rainbow Artifact Analyzer (Fixed Gaze Mode)")
st.caption("This program predicts the reverse coordinates of ambient noise light sources penetrating the peripheral vision space and visualizes the actual rainbow patterns formed on the virtual retina while the eye fixes its gaze at the front center (0°).")

# --- 2. Dual Input Component for Synchronization ---
def dual_input(label, min_val, max_val, default_val, step, k, fmt=None):
    if f"{k}_slider" not in st.session_state: st.session_state[f"{k}_slider"] = float(default_val)
    if f"{k}_num" not in st.session_state: st.session_state[f"{k}_num"] = float(default_val)
    st.sidebar.markdown(f"<div style='font-size:11px; margin-top:5px;'>{label}</div>", unsafe_allow_html=True)
    col1, col2 = st.sidebar.columns([7, 3])
    with col1: st.slider(label, float(min_val), float(max_val), key=f"{k}_slider", step=float(step), on_change=lambda: st.session_state.update({f"{k}_num": float(st.session_state[f"{k}_slider"])}), label_visibility="collapsed", format=fmt)
    with col2: st.number_input(label, float(min_val), float(max_val), key=f"{k}_num", step=float(step), on_change=lambda: st.session_state.update({f"{k}_slider": float(st.session_state[f"{k}_num"])}), label_visibility="collapsed", format=fmt)
    return st.session_state[f"{k}_slider"]

def get_n(lam, n_d, V_d):
    B = (n_d - 1) / V_d * (486.1**2 * 656.3**2) / (486.1**2 - 656.3**2) * 1e-6
    return (n_d - B / (589.3/1000)**2) + B / (lam/1000)**2

# --- 3. Sidebar UI Input Panel ---
st.sidebar.markdown("### 🔍 Grating & Hardware Specifications")
n_d_in = dual_input("Substrate Refractive Index (n_d at 589nm)", 1.0, 3.0, 1.74, 0.01, "n_d", "%.2f")
u_d_in = dual_input("Substrate Abbe Number (V_d)", 10.0, 100.0, 32.0, 0.1, "v_d", "%.1f")

# [Modification] Refracted grating limit pitch calculation synced exactly at 555nm wavelength spec
target_lam_limit = 555.0
ref_index_555 = get_n(target_lam_limit, n_d_in, u_d_in)
limit_pitch_val = target_lam_limit / ref_index_555
st.sidebar.markdown(f"<div style='font-size:12px; color:#ff4b4b; font-weight:bold; margin-top:5px;'>📐 Limit Pitch: {limit_pitch_val:.2f} nm @555nm</div>", unsafe_allow_html=True)

lambda_oc = dual_input("OC Grating Period Λ_OC (nm)", 100.0, 1000.0, 350.0, 0.01, "lambda_oc", "%.2f")
angle_oc = dual_input("OC Grating Vector Angle (°)", 0.0, 360.0, 196.0, 0.01, "angle_oc", "%.2f")
m_order = st.sidebar.selectbox("Ambient Diffraction Order (m)", [1, -1, 2, -2], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**📐 Virtual Display FOV Settings**")
display_h_fov = dual_input("Display Horizontal FOV (Half, °)", 1.0, 45.0, 14.1, 0.1, "disp_h", "%.1f")
display_v_fov = dual_input("Display Vertical FOV (Half, °)", 1.0, 45.0, 9.2, 0.1, "disp_v", "%.1f")

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Ambient Source Analysis Mode**")
sim_mode = st.sidebar.radio("Simulation Mode", ["Full Hemisphere Space Scan Mode", "Target Ambient Source Coordinates Mode"], index=0)

if sim_mode == "Target Ambient Source Coordinates Mode":
    st.sidebar.markdown("🔴 **[Mode 2] Target Source Setup**")
    src_spec_x = dual_input("Source Horizontal Incident Angle (X, °)", -90.0, 90.0, -45.0, 0.1, "src_spec_x", "%.1f")
    src_spec_y = dual_input("Source Vertical Incident Angle (Y, °)", -90.0, 90.0, 15.0, 0.1, "src_spec_y", "%.1f")

# --- 4. CIE 1931 Wavelength-to-RGB Conversion Function ---
def wl_to_rgb(wl):
    if 380 <= wl < 440: R, G, B = -(wl - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= wl < 490: R, G, B = 0.0, (wl - 440) / (490 - 440), 1.0
    elif 490 <= wl < 510: R, G, B = 0.0, 1.0, -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580: R, G, B = (wl - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= wl < 645: R, G, B = 1.0, -(wl - 645) / (645 - 580), 0.0
    elif 645 <= wl <= 780: R, G, B = 1.0, 0.0, 0.0
    else: R, G, B = 0.0, 0.0, 0.0
    factor = 1.0 if wl >= 420 else 0.3 + 0.7 * (wl - 380) / (420 - 380)
    if wl > 700: factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    return f"rgb({int(R*factor*255)}, {int(G*factor*255)}, {int(B*factor*255)})"

# --- 5. Computational Engine & Dashboard Rendering ---
tab1, tab2 = st.tabs(["🌐 Rainbow Artifact Multi-Analysis View", "📈 Grating Pitch Margin Sweep"])

peripheral_limit_deg = 60.0
wavelengths = np.arange(400.0, 701.0, 5.0) 
p_angles = np.arange(-60.0, 61.0, 1.5) 

with tab1:
    col1, col2 = st.columns(2)
    fig_src = go.Figure() 
    fig_eye = go.Figure() 
    
    G_mag = 2 * math.pi / lambda_oc
    G_x = G_mag * math.cos(math.radians(angle_oc))
    G_y = G_mag * math.sin(math.radians(angle_oc))
    
    if sim_mode == "Full Hemisphere Space Scan Mode":
        for wl in wavelengths:
            k0 = 2 * math.pi / wl
            rgb_color = wl_to_rgb(wl)
            src_x, src_y, eye_x, eye_y = [], [], [], []
            
            for p_x in p_angles:
                for p_y in p_angles:
                    if p_x**2 + p_y**2 <= peripheral_limit_deg**2:
                        k_x_per = k0 * math.sin(math.radians(p_x))
                        k_y_per = k0 * math.sin(math.radians(p_y))
                        k_x_ext = k_x_per - m_order * G_x
                        k_y_ext = k_y_per - m_order * G_y
                        k_ext_mag = math.sqrt(k_x_ext**2 + k_y_ext**2)
                        
                        if k_ext_mag <= k0:
                            sin_theta_ext = k_ext_mag / k0
                            theta_ext_deg = np.degrees(math.asin(sin_theta_ext))
                            phi_ext = math.atan2(k_y_ext, k_x_ext)
                            
                            src_x.append(theta_ext_deg * math.cos(phi_ext))
                            src_y.append(theta_ext_deg * math.sin(phi_ext))
                            eye_x.append(p_x)
                            eye_y.append(p_y)
                            
            if src_x:
                fig_src.add_trace(go.Scatter(x=src_x, y=src_y, mode='markers', marker=dict(size=2.5, color=rgb_color), name=f"{wl:.0f}nm",
                                             customdata=np.stack((eye_x, eye_y), axis=-1),
                                             hovertemplate="<b>Source Position:</b> X:%{x:.1f}°, Y:%{y:.1f}°<br><b>Retinal Inflow Angle:</b> H:%{customdata[0]:.1f}°, V:%{customdata[1]:.1f}°<br><b>Matched WL:</b> %{text}<br><extra></extra>", text=[f"{wl:.0f} nm"]*len(src_x), showlegend=False))
                
                fig_eye.add_trace(go.Scatter(x=eye_x, y=eye_y, mode='markers', marker=dict(size=3.0, color=rgb_color), name=f"{wl:.0f}nm",
                                             customdata=np.stack((src_x, src_y), axis=-1),
                                             hovertemplate="<b>Retinal Inflow Position:</b> H:%{x:.1f}°, V:%{y:.1f}°<br><b>Causal Ambient Source:</b> X:%{customdata[0]:.1f}°, Y:%{customdata[1]:.1f}°<br><b>Artifact Color:</b> %{text}<br><extra></extra>", text=[f"{wl:.0f} nm"]*len(eye_x), showlegend=False))
    
    else:
        for wl in wavelengths:
            k0 = 2 * math.pi / wl
            rgb_color = wl_to_rgb(wl)
            
            k_x_ext = k0 * math.sin(math.radians(src_spec_x))
            k_y_ext = k0 * math.sin(math.radians(src_spec_y))
            
            k_x_per = k_x_ext + m_order * G_x
            k_y_per = k_y_ext + m_order * G_y
            k_per_mag = math.sqrt(k_x_per**2 + k_y_per**2)
            
            if k_per_mag <= k0:
                sin_theta_per = k_per_mag / k0
                theta_per_deg = np.degrees(math.asin(sin_theta_per))
                phi_per = math.atan2(k_y_per, k_x_per)
                
                p_x_res = theta_per_deg * math.cos(phi_per)
                p_y_res = theta_per_deg * math.sin(phi_per)
                
                if p_x_res**2 + p_y_res**2 <= peripheral_limit_deg**2:
                    fig_src.add_trace(go.Scatter(x=[src_spec_x], y=[src_spec_y], mode='markers', marker=dict(size=12, color=rgb_color, line=dict(color='black', width=1)), name=f"Target Source ({wl:.0f}nm)",
                                                 hovertemplate="<b>Target Source Position:</b> X:%{x:.1f}°, Y:%{y:.1f}°<br><b>Matched WL:</b> %{text}<br><extra></extra>", text=[f"{wl:.0f} nm"], showlegend=False))
                    
                    fig_eye.add_trace(go.Scatter(x=[p_x_res], y=[p_y_res], mode='markers', marker=dict(size=8, color=rgb_color, symbol='circle'), name=f"Retinal Image",
                                                 hovertemplate="<b>Retinal Inflow Position:</b> H:%{x:.1f}°, V:%{y:.1f}°<br><b>Wavelength Band:</b> %{text}<br><extra></extra>", text=[f"{wl:.0f} nm"], showlegend=False))

    # Layout Annotation & Guidelines Synchronization
    for f in [fig_src, fig_eye]:
        f.add_shape(type="rect", x0=-display_h_fov, y0=-display_v_fov, x1=display_h_fov, y1=display_v_fov, line=dict(color="red", width=2), fillcolor="rgba(255, 0, 0, 0.01)")
        for r in [30, 60, 90 if f==fig_src else 60]:
            f.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r, line=dict(color="lightgray", width=1, dash="dash" if r!=60 else "solid"))
            f.add_annotation(x=r*0.707, y=r*0.707, text=f"{r}° Peripheral" if r==60 else f"{r}°", showarrow=False, font=dict(color="gray" if r!=60 else "blue", size=9), xref="x", yref="y")

    # [Modification 1] Simplified and polished chart titles based on user requirements
    with col1:
        st.subheader("🌐 1. Ambient Source Map")
        st.caption("Displays the coordinates of external light sources in the hemisphere that trigger diffraction noise inside the peripheral vision.")
        fig_src.update_layout(xaxis=dict(title="Ambient Source Horizontal Azimuth Angle (X, °)", range=[-95, 95], scaleanchor="y", scaleratio=1), yaxis=dict(title="Ambient Source Vertical Elevation Angle (Y, °)", range=[-95, 95]), plot_bgcolor="white", hovermode='closest')
        st.plotly_chart(fig_src, use_container_width=True)

    with col2:
        st.subheader("👁️ 2. Eye-side Rainbow Map")
        st.caption("Visualizes where the rainbow flare artifacts physically land on the user's field of view under a fixed gaze condition.")
        fig_eye.update_layout(xaxis=dict(title="Human Eye Field of View Horizontal Angle (Horizontal FOV, °)", range=[-65, 65], scaleanchor="y", scaleratio=1), yaxis=dict(title="Human Eye Field of View Vertical Angle (Vertical FOV, °)", range=[-65, 65]), plot_bgcolor="white", hovermode='closest')
        st.plotly_chart(fig_eye, use_container_width=True)

# --- 6. Tab 2: Grating Pitch Margin Sweep Panel ---
with tab2:
    st.markdown("#### ⚙️ OC Grating Period (Pitch) Optimization Margin Sweep")
    pitch_arr = np.arange(250.0, 500.0, 2.0)
    sweep_results = []
    for p in pitch_arr:
        noise_count = 0
        G_m = 2 * math.pi / p
        G_x_m = G_m * math.cos(math.radians(angle_oc))
        G_y_m = G_m * math.sin(math.radians(angle_oc))
        for wl in wavelengths:
            k0_m = 2 * math.pi / wl
            has_noise = False
            for p_x in np.arange(-60.0, 61.0, 6.0):
                for p_y in np.arange(-60.0, 61.0, 6.0):
                    if p_x**2 + p_y**2 <= peripheral_limit_deg**2:
                        if math.sqrt((k0_m * math.sin(math.radians(p_x)) - m_order * G_x_m)**2 + (k0_m * math.sin(math.radians(p_y)) - m_order * G_y_m)**2) <= k0_m:
                            has_noise = True
                            break
                if has_noise: break
            if has_noise: noise_count += 1
        sweep_results.append({"Pitch (nm)": p, "Noise Color Density": noise_count})
        
    df_sweep = pd.DataFrame(sweep_results)
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=df_sweep["Pitch (nm)"], y=df_sweep["Noise Color Density"], mode="lines+markers", line=dict(color="purple", width=2)))
    fig_sweep.update_layout(title="Peripheral Noise Spectrum Density vs. Out-Coupler Grating Period", xaxis_title="OC Pitch (nm)", yaxis_title="Penetrated Wavelength Count", plot_bgcolor="white", height=400)
    st.plotly_chart(fig_sweep, use_container_width=True)