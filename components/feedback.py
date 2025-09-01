from __future__ import annotations

import datetime as dt
from email.message import EmailMessage
import smtplib
import ssl
from typing import Optional

import streamlit as st


def _send_feedback_email(message_text: str, contact: str = "") -> tuple[bool, Optional[str]]:
    """
    Sends feedback to the configured inbox using SMTP settings from Streamlit secrets.

    Returns (ok, error_message). On success, (True, None). On failure, (False, reason).
    """
    try:
        smtp_cfg = st.secrets.get("smtp", None)
        if not smtp_cfg:
            return False, "SMTP configuration not found in Streamlit secrets."

        host = smtp_cfg.get("host")
        port = int(smtp_cfg.get("port", 587))
        user = smtp_cfg.get("user")
        password = smtp_cfg.get("password")
        from_addr = smtp_cfg.get("from", user)
        to_addr = smtp_cfg.get("to")
        use_tls = bool(smtp_cfg.get("use_tls", True))

        if not all([host, port, user, password, to_addr]):
            return False, "Incomplete SMTP configuration."

        # Trim size to avoid excessively large payloads
        message_text = message_text.strip()
        if len(message_text) > 5000:
            message_text = message_text[:5000] + "\n...[truncated]"

        ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = f"App feedback ({ts})"
        body_lines = [
            "New feedback submitted:",
            "",
            message_text,
        ]
        if contact:
            body_lines += ["", f"Contact (optional): {contact.strip()}"]
        body_lines += ["", f"Page: {st.session_state.get('current_page', 'main')}"]

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.set_content("\n".join(body_lines))

        if use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, port) as server:
                server.login(user, password)
                server.send_message(msg)

        return True, None
    except Exception as e:
        return False, str(e)


def _supports_popover() -> bool:
    # Streamlit popover requires newer versions; fall back to expander otherwise.
    return hasattr(st, "popover")


def render_feedback_widget():
    """
    Renders a compact, clickable feedback popover with a form.
    Place this once in your app (e.g., top of the main page). Uses st.popover if available,
    otherwise falls back to st.expander.
    """
    st.session_state.setdefault("feedback_last_status", None)

    container_ctx = st.popover("💬 Feedback", use_container_width=False) if _supports_popover() else st.expander("💬 Feedback")
    with container_ctx:
        st.write("Tell us what you think. No email required.")
        with st.form("feedback_form", clear_on_submit=True):
            feedback = st.text_area(
                "Your feedback",
                placeholder="Type your feedback here...",
                height=160,
            )
            include_contact = st.checkbox("Include contact info (optional)")
            contact = st.text_input("Your email or handle", disabled=not include_contact)

            submitted = st.form_submit_button("Send")
            if submitted:
                if not feedback.strip():
                    st.warning("Please enter some feedback before sending.")
                else:
                    ok, err = _send_feedback_email(feedback, contact if include_contact else "")
                    if ok:
                        st.success("Thanks! Your feedback has been sent.")
                        st.session_state["feedback_last_status"] = "sent"
                    else:
                        st.error("Sorry, we couldn't send your feedback right now. Please try again later.")
                        # Log the error server-side for maintainers.
                        print(f"[feedback] send error: {err}")