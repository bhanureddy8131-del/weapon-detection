from datetime import datetime, timedelta, date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sqlalchemy import desc

from server.database import SessionLocal
from server.models import Incident


# ============================================================
# SMARTVISION PRO - ANALYTICS DASHBOARD
# Reads actual historical incidents from existing SQL database
# ============================================================

st.set_page_config(
    page_title="SmartVision Pro Analytics",
    page_icon="📊",
    layout="wide",
)


# ------------------------------------------------------------
# THEME
# ------------------------------------------------------------

def analytics_theme():

    st.markdown(
        """
        <style>

        .main {
            background-color: #f5f8fc;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .analytics-title {
            font-size: 32px;
            font-weight: 800;
            color: #14213d;
        }

        .analytics-subtitle {
            color: #64748b;
            font-size: 14px;
        }

        .metric-card {
            background: white;
            padding: 22px;
            border-radius: 14px;
            border: 1px solid #e5eaf2;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
        }

        .metric-label {
            color: #64748b;
            font-size: 13px;
            font-weight: 600;
        }

        .metric-value {
            color: #14213d;
            font-size: 29px;
            font-weight: 800;
        }

        .section-title {
            color: #14213d;
            font-size: 20px;
            font-weight: 750;
            margin-top: 15px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------

def load_historical_incidents():

    db = SessionLocal()

    try:

        incidents = (
            db.query(Incident)
            .order_by(desc(Incident.created_at))
            .all()
        )

        records = []

        for incident in incidents:

            records.append({
                "id": incident.id,
                "created_at": incident.created_at,
                "camera_id": incident.camera_id or "Unknown",
                "threat_level": incident.threat_level or "Unknown",
                "label": incident.label or "Unknown",
                "detected_class": incident.detected_class or "Unknown",
                "confidence": incident.confidence,
                "tracking_id": incident.tracking_id,
                "notification_status": (
                    incident.notification_status or "Unknown"
                ),
                "notification_error": incident.notification_error,
                "image_path": incident.image_path,
            })

        return pd.DataFrame(records)

    finally:
        db.close()


# ------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------

def prepare_data(df):

    if df.empty:
        return df

    df = df.copy()

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True,
    )

    # Convert timestamps to local display time.
    df["created_at"] = df["created_at"].dt.tz_convert(
        "Asia/Kolkata"
    )

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce",
    )

    df["threat_level"] = (
        df["threat_level"]
        .fillna("Unknown")
        .astype(str)
        .str.upper()
    )

    df["label"] = (
        df["label"]
        .fillna("Unknown")
        .astype(str)
    )

    df["detected_class"] = (
        df["detected_class"]
        .fillna("Unknown")
        .astype(str)
    )

    df["notification_status"] = (
        df["notification_status"]
        .fillna("Unknown")
        .astype(str)
        .str.lower()
    )

    df["detection_category"] = (
        df["detected_class"]
        .replace("Unknown", pd.NA)
        .fillna(df["label"])
        .str.lower()
    )

    return df.dropna(subset=["created_at"])


# ------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------

def filter_incidents(df, selected_period):

    if df.empty:
        return df

    now = pd.Timestamp.now(tz="Asia/Kolkata")

    if selected_period == "Last 7 Days":
        start = now - pd.Timedelta(days=7)

    elif selected_period == "Last 30 Days":
        start = now - pd.Timedelta(days=30)

    elif selected_period == "Last 12 Months":
        start = now - pd.Timedelta(days=365)

    else:
        return df

    return df[df["created_at"] >= start]


# ------------------------------------------------------------
# METRIC CARD
# ------------------------------------------------------------

def metric_card(title, value, description=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div style="color:#94a3b8;font-size:12px;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# EMPTY CHART
# ------------------------------------------------------------

def empty_message():

    st.info(
        "No historical incidents are available for this selection."
    )


# ------------------------------------------------------------
# MAIN DASHBOARD
# ------------------------------------------------------------

def render_analytics_dashboard():

    analytics_theme()

    st.markdown(
        '<div class="analytics-title">'
        'SmartVision Pro Analytics'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="analytics-subtitle">'
        'Historical AI security monitoring and threat intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # Load actual database records.
    with st.spinner("Loading historical incident records..."):

        try:
            df = load_historical_incidents()
            df = prepare_data(df)

        except Exception as exc:

            st.error(
                f"Unable to load historical database records: {exc}"
            )

            return

    # --------------------------------------------------------
    # PERIOD FILTER
    # --------------------------------------------------------

    col1, col2 = st.columns([2, 5])

    with col1:

        period = st.selectbox(
            "Analytics period",
            [
                "Last 7 Days",
                "Last 30 Days",
                "Last 12 Months",
                "All Time",
            ],
            index=1,
        )

    with col2:

        st.markdown(
            f"""
            <div style="padding-top:32px;color:#64748b;">
                Database records available:
                <b>{len(df):,}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    filtered = filter_incidents(df, period)

    st.divider()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total = len(filtered)

    gun_count = filtered[
        filtered["detection_category"].str.contains(
            "gun", na=False
        )
    ].shape[0]

    knife_count = filtered[
        filtered["detection_category"].str.contains(
            "knife", na=False
        )
    ].shape[0]

    high_count = filtered[
        filtered["threat_level"].isin(
            ["HIGH", "CRITICAL", "SEVERE"]
        )
    ].shape[0]

    notification_sent = filtered[
        filtered["notification_status"].isin(
            ["sent", "success", "delivered"]
        )
    ].shape[0]

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("TOTAL INCIDENTS", f"{total:,}")

    with c2:
        metric_card("GUN DETECTIONS", f"{gun_count:,}")

    with c3:
        metric_card("KNIFE DETECTIONS", f"{knife_count:,}")

    with c4:
        metric_card("HIGH SEVERITY", f"{high_count:,}")

    with c5:
        metric_card("ALERTS SENT", f"{notification_sent:,}")

    st.write("")

    if filtered.empty:
        empty_message()
        return

    # --------------------------------------------------------
    # DAILY DETECTION TREND
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Detection Trends'
        '</div>',
        unsafe_allow_html=True,
    )

    daily = (
        filtered
        .set_index("created_at")
        .resample("D")
        .size()
        .reset_index(name="incidents")
    )

    fig_daily = px.line(
        daily,
        x="created_at",
        y="incidents",
        markers=True,
        title="Daily Incident Detection",
        labels={
            "created_at": "Date",
            "incidents": "Number of Incidents",
        },
    )

    fig_daily.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
    )

    fig_daily.update_layout(
        template="plotly_white",
        height=370,
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # THREAT DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Threat Distribution'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:

        threat_counts = (
            filtered["detection_category"]
            .value_counts()
            .reset_index()
        )

        threat_counts.columns = [
            "Threat",
            "Count",
        ]

        fig_pie = px.pie(
            threat_counts,
            names="Threat",
            values="Count",
            hole=0.55,
            title="Threat Category Distribution",
        )

        fig_pie.update_layout(
            template="plotly_white",
            height=400,
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True,
        )

    with right:

        severity_counts = (
            filtered["threat_level"]
            .value_counts()
            .reset_index()
        )

        severity_counts.columns = [
            "Severity",
            "Count",
        ]

        fig_severity = px.bar(
            severity_counts,
            x="Severity",
            y="Count",
            text="Count",
            title="Threat Severity Analysis",
        )

        fig_severity.update_layout(
            template="plotly_white",
            height=400,
        )

        st.plotly_chart(
            fig_severity,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # NOTIFICATION ANALYTICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Notification Analytics'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:

        notification_counts = (
            filtered["notification_status"]
            .value_counts()
            .reset_index()
        )

        notification_counts.columns = [
            "Status",
            "Count",
        ]

        fig_notifications = px.pie(
            notification_counts,
            names="Status",
            values="Count",
            hole=0.55,
            title="Notification Delivery Status",
        )

        fig_notifications.update_layout(
            template="plotly_white",
            height=380,
        )

        st.plotly_chart(
            fig_notifications,
            use_container_width=True,
        )

    with right:

        hourly = (
            filtered["created_at"]
            .dt.hour
            .value_counts()
            .sort_index()
            .reindex(range(24), fill_value=0)
            .reset_index()
        )

        hourly.columns = [
            "Hour",
            "Incidents",
        ]

        fig_hourly = px.bar(
            hourly,
            x="Hour",
            y="Incidents",
            title="Hourly Incident Distribution",
        )

        fig_hourly.update_layout(
            template="plotly_white",
            height=380,
            xaxis=dict(
                tickmode="linear",
                dtick=2,
            ),
        )

        st.plotly_chart(
            fig_hourly,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # CAMERA ANALYTICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Camera Analytics'
        '</div>',
        unsafe_allow_html=True,
    )

    camera_counts = (
        filtered["camera_id"]
        .value_counts()
        .reset_index()
    )

    camera_counts.columns = [
        "Camera",
        "Incidents",
    ]

    fig_camera = px.bar(
        camera_counts,
        x="Camera",
        y="Incidents",
        text="Incidents",
        title="Incidents by Camera",
    )

    fig_camera.update_layout(
        template="plotly_white",
        height=350,
    )

    st.plotly_chart(
        fig_camera,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # HISTORICAL INCIDENT TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Historical Incident Records'
        '</div>',
        unsafe_allow_html=True,
    )

    display_columns = [
        "id",
        "created_at",
        "camera_id",
        "detection_category",
        "threat_level",
        "confidence",
        "notification_status",
    ]

    history = filtered[display_columns].copy()

    history["created_at"] = (
        history["created_at"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # CSV EXPORT
    # --------------------------------------------------------

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Download Historical Incident Report",
        data=csv_data,
        file_name=(
            "smartvision_incident_history_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Analytics are generated from actual records in the "
        "existing SQLAlchemy incidents database. "
        "No demonstration records are generated."
    )