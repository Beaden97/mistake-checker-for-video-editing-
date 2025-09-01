from __future__ import annotations

import datetime as dt
from email.message import EmailMessage
import smtplib
import ssl
from typing import Optional

import streamlit as st
from app_theme import apply_base_theme, apply_runtime_theme_controls


def test_smtp_configuration(debug: bool = False) -> tuple[bool, str]:
    """
    Test SMTP configuration by attempting to send a test email.
    
    Returns (success, message). Message contains either success confirmation
    or detailed error information with hints.
    """
    try:
        smtp_cfg = st.secrets.get("smtp", None)
        if not smtp_cfg:
            return False, "❌ **SMTP configuration not found in Streamlit secrets.**\n\nPlease add an [smtp] section to your secrets configuration."

        # Check all required fields
        host = smtp_cfg.get("host")
        port = smtp_cfg.get("port", 587)
        user = smtp_cfg.get("user")
        password = smtp_cfg.get("password")
        from_addr = smtp_cfg.get("from", user)
        to_addr = smtp_cfg.get("to")
        use_tls = bool(smtp_cfg.get("use_tls", True))

        # Validate configuration
        missing_fields = []
        if not host:
            missing_fields.append("host")
        if not port:
            missing_fields.append("port")
        if not user:
            missing_fields.append("user") 
        if not password:
            missing_fields.append("password")
        if not to_addr:
            missing_fields.append("to")
            
        if missing_fields:
            return False, f"❌ **Missing required SMTP fields:** {', '.join(missing_fields)}\n\nPlease check your secrets configuration."

        # Create test email
        ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = f"SMTP Test Email ({ts})"
        body_lines = [
            "This is a test email from the Video Editing Mistake Checker app.",
            "",
            f"Test sent at: {ts}",
            f"SMTP server: {host}:{port}",
            f"TLS enabled: {use_tls}",
            f"Debug mode: {debug}",
            "",
            "If you receive this email, your SMTP configuration is working correctly!"
        ]

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.set_content("\n".join(body_lines))

        # Attempt to send email
        if use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, int(port), timeout=20) as server:
                if debug:
                    server.set_debuglevel(1)
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()  # EHLO after STARTTLS for hardening
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, int(port), timeout=20) as server:
                if debug:
                    server.set_debuglevel(1)
                server.login(user, password)
                server.send_message(msg)

        return True, f"✅ **Test email sent successfully!**\n\nA test email has been sent to: {to_addr}\n\nYour SMTP configuration is working correctly."

    except Exception as e:
        error_str = str(e).lower()
        error_msg = f"❌ **SMTP Test Failed:** {str(e)}"
        
        # Add helpful hints for common issues
        if "authentication" in error_str or "auth" in error_str:
            error_msg += "\n\n💡 **Common Solutions:**\n- Verify your username and password\n- For Gmail: Use an App Password instead of your regular password\n- For Office 365: Ensure the account has SMTP AUTH enabled"
        elif "tls" in error_str or "ssl" in error_str:
            error_msg += "\n\n💡 **Common Solutions:**\n- Try toggling the 'use_tls' setting\n- Check port: 587 for STARTTLS, 465 for SSL, 25 for plain text\n- Gmail: use port 587 with use_tls=true\n- Office 365: use port 587 with use_tls=true"
        elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
            error_msg += "\n\n💡 **Common Solutions:**\n- Check your SMTP server hostname\n- Verify the port number\n- Ensure network connectivity\n- Check firewall settings"
        elif "refused" in error_str:
            error_msg += "\n\n💡 **Common Solutions:**\n- Verify the 'from' email address is authorized to send\n- Check that the 'to' email address is valid\n- Some providers require domain verification"
        elif "name" in error_str and "resolve" in error_str:
            error_msg += "\n\n💡 **Common Solutions:**\n- Check the SMTP server hostname for typos\n- Verify DNS resolution\n- Common hostnames: smtp.gmail.com, smtp-mail.outlook.com, smtp.sendgrid.net"
        
        return False, error_msg


def main():
    apply_base_theme()
    apply_runtime_theme_controls()
    
    st.title("📧 Email Diagnostics")
    
    st.markdown("""
    This page helps you test and troubleshoot your SMTP email configuration. 
    Use this tool to verify that your feedback emails will work correctly.
    """)
    
    # Show current configuration status (without revealing secrets)
    st.subheader("📋 Current Configuration")
    
    smtp_cfg = st.secrets.get("smtp", None)
    if not smtp_cfg:
        st.error("❌ No SMTP configuration found in Streamlit secrets.")
        st.markdown("""
        **To configure SMTP:**
        1. Add an `[smtp]` section to your `.streamlit/secrets.toml` file
        2. See `.streamlit/secrets.example.toml` for examples
        3. Deploy with your secrets configured
        """)
        return
    
    # Show configuration status without revealing actual values
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Required Fields:**")
        fields = ["host", "port", "user", "password", "to"]
        for field in fields:
            value = smtp_cfg.get(field)
            if value:
                st.success(f"✅ {field}: configured")
            else:
                st.error(f"❌ {field}: not set")
    
    with col2:
        st.markdown("**Optional Fields:**")
        from_addr = smtp_cfg.get("from", smtp_cfg.get("user", ""))
        use_tls = smtp_cfg.get("use_tls", True)
        debug = smtp_cfg.get("debug", False)
        
        st.info(f"📧 from: {'configured' if from_addr else 'using username'}")
        st.info(f"🔒 use_tls: {use_tls}")
        st.info(f"🐛 debug: {debug}")
    
    st.markdown("---")
    
    # Test email section
    st.subheader("🧪 Send Test Email")
    
    st.markdown("""
    Click the button below to send a test email using your current SMTP configuration.
    This will help identify any issues with your setup.
    """)
    
    # Debug option
    enable_debug = st.checkbox(
        "Enable SMTP transcript (debug mode)", 
        value=False,
        help="Shows detailed SMTP server communication. Only enable for troubleshooting."
    )
    
    if enable_debug:
        st.warning("⚠️ Debug mode enabled. SMTP transcript will be shown in the server logs.")
    
    # Test button
    if st.button("Send Test Email", type="primary"):
        with st.spinner("Sending test email..."):
            success, message = test_smtp_configuration(debug=enable_debug)
        
        if success:
            st.success(message)
        else:
            st.error(message)
    
    st.markdown("---")
    
    # Configuration help
    st.subheader("🔧 Configuration Help")
    
    with st.expander("Common SMTP Providers", expanded=False):
        st.markdown("""
        **Gmail:**
        ```toml
        [smtp]
        host = "smtp.gmail.com"
        port = 587
        user = "your-email@gmail.com"
        password = "your-app-password"  # Use App Password, not regular password
        from = "your-email@gmail.com"
        to = "feedback@yourcompany.com"
        use_tls = true
        ```
        
        **Microsoft 365:**
        ```toml
        [smtp]
        host = "smtp-mail.outlook.com"
        port = 587
        user = "your-email@yourcompany.com"
        password = "your-password"
        from = "your-email@yourcompany.com"
        to = "feedback@yourcompany.com"
        use_tls = true
        ```
        
        **SendGrid:**
        ```toml
        [smtp]
        host = "smtp.sendgrid.net"
        port = 587
        user = "apikey"
        password = "SG.your-api-key"
        from = "no-reply@yourcompany.com"
        to = "feedback@yourcompany.com"
        use_tls = true
        ```
        """)
    
    with st.expander("Troubleshooting Tips", expanded=False):
        st.markdown("""
        **Authentication Issues:**
        - Gmail: Use App Passwords, not your regular password
        - Office 365: Ensure SMTP AUTH is enabled for the account
        - Verify username/password are correct
        
        **Connection Issues:**
        - Check firewall settings
        - Verify server hostname and port
        - Test network connectivity
        
        **TLS/SSL Issues:**
        - Gmail/Office 365: Use port 587 with use_tls=true
        - For SSL: Use port 465 with use_tls=false
        - For plain text: Use port 25 with use_tls=false (not recommended)
        
        **Permission Issues:**
        - Verify 'from' address is authorized to send
        - Some providers require domain verification
        - Check recipient address restrictions
        """)


if __name__ == "__main__":
    main()