from processing.utils import APP

import streamlit as st
import sys
from loguru import logger

import processing.utils as utils
from processing.signal_processing import *

import streamlit.components.v1 as components
import plotly.express as px
import matplotlib.pyplot as plt



def main():
    if "logger_configured" not in st.session_state:
        logger.remove()
        logger.add(sys.stderr, format="{time:HH:mm:ss} | {level: <8} | {message}", level="INFO")
        st.session_state["logger_configured"] = True

        logger.info("Starting GOLEM Streamlit app")

    with st.sidebar.container(border=True):
        shot_no = st.number_input(
            label="Shot number",
            min_value=0,
            value=46697,
            step=1
            )

        ref_coil = st.number_input(
            label="Reference coil",
            min_value=1,
            max_value=16,
            value=11,
            step=1
            )
    

    with st.sidebar.expander(label="Mirnov coils - Raw", expanded=False):
        plot_raw = st.button(label="Plot raw Mirnov coils data")


    with st.sidebar.expander(label="Spectrogram", expanded=False):
        nperseg = st.number_input(
            label="NperSeg",
            min_value=APP["min_nperseg"],
            value=APP["nperseg"],
            key="nperseg"
            )

        noverlap = st.number_input(
            label="Noverlap",
            min_value=APP["min_noverlap"],
            value=APP["noverlap"],
            key="noverlap"
            )

        # TODO: add a default value
        vmax = st.slider(
            label="Vmax",
            min_value=0.0,
            max_value=st.session_state.get("spect_vmax", 50.0),
            value=st.session_state.get("spect_vmax_value", 40.0),
            step=0.1,
            key="vmax"
            )

        spect_display_mode = st.selectbox(
            label="Display mode",
            options=APP["2D_display_modes"],
            index=APP["spect_display_mode"],
            key="spect_display_mode"
            )

        plot_spect = st.button(label="Compute spectrogram", use_container_width=True)


    with st.sidebar.expander(label="Correlation", expanded=False):
        t_0 = st.number_input(
            label="$t_0$ (s)",
            min_value=APP["min_t_0"],
            value=APP["t_0"],
            key="t_0"
            )

        l_span = st.number_input(
            label="$L_\\mathrm{span}$ (s)",
            min_value=APP["min_l_span"],
            value=APP["l_span"],
            key="l_span"
            )

        l_interval = [t_0 - l_span/2, t_0 + l_span/2]

        corr_display_mode = st.selectbox(
            label="Display mode",
            options=APP["2D_display_modes"],
            index=APP["corr_display_mode"],
            key="corr_display_mode"
            )
        
        plot_corr = st.button(label="Compute correlation", use_container_width=True)


    if plot_raw:
        logger.info(f"Loading raw Mirnov coils data for shot {shot_no}")
        
        try:
            data = utils.load_MHDring_data(shot_no=shot_no)
        except RuntimeError as e:
            logger.error(str(e))
            st.error(str(e))
            st.stop()

        logger.info(f"Raw Mirnov coils data loaded successfully for shot {shot_no}")
        
        fig = px.line(
            x=data.time,
            y=data.sel(channel=ref_coil-1),
            labels={
                "x": r"$t$ (ms)",
                "y": f"Coil {ref_coil} signal (Au)"
            },

        )

        st.plotly_chart(fig, use_container_width=True)

        logger.info("Raw Mirnov coils data plot generated successfully")



    if plot_spect:
        logger.info(f"Computing spectrogram for shot {shot_no}, coil {ref_coil}, nperseg {nperseg}, noverlap {noverlap}")
        try:
            Sx = spectrogram(shot_no=shot_no, coil_id=ref_coil, nperseg=nperseg, hop=(nperseg-noverlap))
        except Exception as e:
            logger.error(f"Error computing spectrogram: {e}")
            st.error(f"Error computing spectrogram: {e}")
            st.stop()

        st.session_state["spect_vmax"] = float(np.max(np.abs(Sx)))

        logger.info(f"Spectrogram computed successfully for shot {shot_no}, coil {ref_coil}")

        fig, ax = plt.subplots(figsize=(8, 5))

        if spect_display_mode == "pcolormesh":
            im = ax.pcolormesh(Sx.time, Sx.frequency / 1000, np.abs(Sx), shading="auto", cmap="viridis", vmin=0, vmax=vmax)
        else:
            im = ax.contourf(Sx.time, Sx.frequency / 1000, np.abs(Sx), levels=50, cmap="viridis", vmin=0, vmax=vmax)

        cbar = fig.colorbar(mappable=im, ax=ax)

        ax.set_ylim(0, 200)

        ax.set_xlabel("$t$ (ms)", fontsize = 13)
        ax.set_ylabel("$f$ (kHz)", fontsize = 13)
        cbar.set_label(label="$S_{xx}(f,t)$ (Au)", fontsize=13)

        st.pyplot(fig)

        logger.info("Spectrogram plot generated successfully")
        # fig = px.imshow(
        #     np.abs(Sx),
        #     x=Sx.time,
        #     y=Sx.frequency / 1000,
        #     origin="lower",
        #     color_continuous_scale="Viridis",
        #     yaxis_range=[0, 200],
        #     zmin=0,
        #     zmax=vmax,
        #     labels={
        #         "x": r"$t \mathrm{ (ms)}$",
        #         "y": r"$f \mathrm{ (kHz)}$",
        #         "color": r"$|S_{x}(t)| \mathrm{ (Au)}$"
        #     }
        # )

        # html = fig.to_html(include_plotlyjs='cdn', include_mathjax='cdn')
        # components.html(html, height=1300)


    if plot_corr:
        logger.info(f"Computing correlation for shot {shot_no}, ref coil {ref_coil}, time interval {l_interval}")

        try:
            corr = correlation(shot_no=shot_no, l_interval=l_interval, ref_coil=ref_coil)
        except Exception as e:
            logger.error(f"Error computing correlation: {e}")
            st.error(f"Error computing correlation: {e}")
            st.stop()

        logger.info(f"Correlation computed successfully for shot {shot_no}, ref coil {ref_coil}")

        fig, ax = plt.subplots(figsize=(8, 5))
        
        if corr_display_mode == "contourf":
            im = ax.contourf(corr.lag, corr.channel, corr, levels=np.linspace(-1, 1, 32), cmap = "seismic", extend="both")
        else:
            im = ax.pcolormesh(corr.lag, corr.channel, corr, shading="auto", cmap = "seismic", vmin=-1, vmax=1)
    
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)

        cbar = fig.colorbar(mappable=im, ax=ax)

        ax.set_xticks(np.round(np.linspace(-l_span/2, l_span/2, 7), 3))
        cbar.set_ticks(np.linspace(-1, 1, 11))

        ax.set_title(f"$t = {t_0}$ s; Ref. coil number {ref_coil}")
        ax.set_xlabel("$L$ (ms)", fontsize = 13)
        ax.set_ylabel("coil number (-)", fontsize = 13)
        cbar.set_label(label="$P_{xy}(L)$ (Au)", fontsize=13)

        st.pyplot(fig)

        logger.info("Correlation plot generated successfully")



if __name__ == "__main__":
    main()