"""Interactive corrections checklist for video editing issues."""
import streamlit as st
from typing import List, Dict, Any


def render_corrections_checklist(issues: List[Dict[str, Any]]) -> bool:
    """
    Render an interactive checklist for video editing corrections.
    
    Args:
        issues: List of issues from analysis results
        
    Returns:
        bool: True if all issues are checked off, False otherwise
    """
    if not issues:
        return True
    
    # Initialize session state for checklist if not exists
    if 'checklist_states' not in st.session_state:
        st.session_state.checklist_states = {}
    
    # Clean up states for issues that no longer exist
    current_issue_keys = {f"{issue['timestamp']}_{issue['message']}" for issue in issues}
    st.session_state.checklist_states = {
        k: v for k, v in st.session_state.checklist_states.items() 
        if k in current_issue_keys
    }
    
    st.subheader("✅ Corrections Checklist")
    st.markdown("*Check off each issue as you address it in your video:*")
    
    all_checked = True
    
    # Group issues by severity for better organization
    severity_groups = {'error': [], 'warning': [], 'info': []}
    for issue in issues:
        severity = issue.get('severity', 'info')
        if severity in severity_groups:
            severity_groups[severity].append(issue)
        else:
            severity_groups['info'].append(issue)
    
    # Display issues by severity
    for severity, severity_issues in severity_groups.items():
        if not severity_issues:
            continue
            
        # Severity header
        severity_config = {
            'error': {'icon': '🔴', 'label': 'Critical Issues', 'color': 'red'},
            'warning': {'icon': '🟡', 'label': 'Warnings', 'color': 'orange'}, 
            'info': {'icon': '🔵', 'label': 'Info', 'color': 'blue'}
        }
        
        config = severity_config[severity]
        st.markdown(f"**{config['icon']} {config['label']}**")
        
        for issue in severity_issues:
            issue_key = f"{issue['timestamp']}_{issue['message']}"
            
            # Create checkbox for this issue
            is_checked = st.session_state.checklist_states.get(issue_key, False)
            
            # Use columns for better layout
            col1, col2 = st.columns([0.05, 0.95])
            
            with col1:
                checked = st.checkbox(
                    "", 
                    value=is_checked,
                    key=f"checkbox_{issue_key}",
                    label_visibility="collapsed"
                )
                st.session_state.checklist_states[issue_key] = checked
            
            with col2:
                # Style the text based on whether it's checked
                if checked:
                    st.markdown(f"~~**[{issue['timestamp']}]** {issue['message']}~~")
                else:
                    st.markdown(f"**[{issue['timestamp']}]** {issue['message']}")
                    all_checked = False
        
        st.markdown("")  # Add some spacing
    
    # Show progress
    total_issues = len(issues)
    checked_count = sum(1 for v in st.session_state.checklist_states.values() if v)
    
    # Progress bar
    progress = checked_count / total_issues if total_issues > 0 else 1.0
    st.progress(progress)
    st.caption(f"Progress: {checked_count}/{total_issues} issues addressed ({progress:.0%})")
    
    # Show completion message if all checked
    if all_checked and total_issues > 0:
        st.success("🎉 **All issues checked off!** Good to go or refresh the page to upload again for a second pass!")
        st.balloons()
        
        # Option to reset checklist
        if st.button("🔄 Reset Checklist", help="Clear all checkboxes to review again"):
            st.session_state.checklist_states = {}
            st.rerun()
    
    return all_checked


def clear_checklist():
    """Clear all checklist states."""
    if 'checklist_states' in st.session_state:
        st.session_state.checklist_states = {}