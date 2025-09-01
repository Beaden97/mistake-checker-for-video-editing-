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
        try:
            smtp_cfg = st.secrets.get("smtp", None)
        except Exception:
            # No secrets file exists
            smtp_cfg = None
            
        if not smtp_cfg:
            return False, "SMTP configuration not found in Streamlit secrets."

        host = smtp_cfg.get("host")
        port = int(smtp_cfg.get("port", 587))
        user = smtp_cfg.get("user")
        password = smtp_cfg.get("password")
        from_addr = smtp_cfg.get("from", user)
        to_addr = smtp_cfg.get("to")
        use_tls = bool(smtp_cfg.get("use_tls", True))
        debug = bool(smtp_cfg.get("debug", False))

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
        
        # Add Reply-To header if contact looks like an email
        if contact and "@" in contact and "." in contact:
            msg["Reply-To"] = contact.strip()
        
        msg.set_content("\n".join(body_lines))

        if use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=20) as server:
                if debug:
                    server.set_debuglevel(1)
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()  # EHLO after STARTTLS for hardening
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, port, timeout=20) as server:
                if debug:
                    server.set_debuglevel(1)
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
                        st.session_state["feedback_last_error"] = None
                    else:
                        # Surface exact error to user with helpful hints
                        error_msg = f"Failed to send feedback: {err}"
                        
                        # Add helpful hints for common issues
                        err_lower = err.lower()
                        if "authentication" in err_lower or "auth" in err_lower:
                            error_msg += "\n\n💡 **Hint**: Check your SMTP username and password in the configuration."
                        elif "tls" in err_lower or "ssl" in err_lower:
                            error_msg += "\n\n💡 **Hint**: Try toggling the 'use_tls' setting or check the port number (587 for TLS, 465 for SSL, 25 for plain)."
                        elif "connection" in err_lower or "network" in err_lower:
                            error_msg += "\n\n💡 **Hint**: Check your SMTP server hostname and port. Ensure network connectivity."
                        elif "refused" in err_lower:
                            error_msg += "\n\n💡 **Hint**: The server refused the sender or recipient address. Check your 'from' and 'to' email addresses."
                        elif ("name" in err_lower and "resolve" in err_lower) or "hostname" in err_lower or "address associated" in err_lower:
                            error_msg += "\n\n💡 **Hint**: Check the SMTP server hostname for typos. Common hostnames: smtp.gmail.com, smtp-mail.outlook.com, smtp.sendgrid.net"
                        
                        st.error(error_msg)
                        st.session_state["feedback_last_status"] = "error"
                        st.session_state["feedback_last_error"] = err
                        # Log the error server-side for maintainers.
                        print(f"[feedback] send error: {err}")