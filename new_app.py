import streamlit as st
import logging
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
import hashlib # Keeping the import in case you want to re-enable hashing later
from views import factory_manager_view, sales_manager_view, procurement_view, production_manager_view, quality_manager_view
from utils.data_loader import DataLoader
import config
# from streamlit_gsheets import GSheetsConnection


# --- Page Configuration ---
st.set_page_config(
    page_title="Dairy Production Plant Management",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- File path for user data ---
USERS_FILE = Path("users.csv")

# --- Role to Icon Mapping ---
ROLE_ICONS = {
    "Developer": "👑",
    "Factory Manager": "👷",
    "Production": "🏭",
    "Sales": "📊",
    "Logistics": "🚛",
    "Engineering": "⚙️"
}

# --- User Data Management Functions ---
def load_users():
    """Loads user data from the CSV file into a dictionary."""
    if not USERS_FILE.exists():
        # Create a default file with a dev user if it doesn't exist
        default_users = pd.DataFrame([
            {"username": "dev", "password": "dev_password", "role": "Developer"}
        ])
        default_users.to_csv(USERS_FILE, index=False)
    
    # Read the csv file
    df = pd.read_csv(USERS_FILE)
    users = df.to_dict('records')
    print(users)
    
    # Convert list of dicts to a nested dictionary for easy lookup
    user_dict = {}
    for user in users:
        user_dict[user['username']] = {
            "password": user['password'],
            "role": user['role'],
            "icon": ROLE_ICONS.get(user['role'], '👤') # Get icon from mapping
        }
    return user_dict

def save_users(user_dict):
    """Saves the user dictionary back to the CSV file."""
    user_list = []
    for username, data in user_dict.items():
        user_list.append({
            "username": username,
            "password": data['password'],
            "role": data['role']
        })
    df = pd.DataFrame(user_list)
    df.to_csv(USERS_FILE, index=False)


# --- Initialize user data into session state ---
if 'users' not in st.session_state:
    st.session_state.users = load_users()


# --- Authentication Functions ---
def check_login(username, password):
    """Verify username and plain-text password."""
    username = 'dev'
    password = 'dev_password'
    if username in st.session_state.users:
        # Using plain-text password check as requested
        if password == st.session_state.users[username]["password"]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = st.session_state.users[username]["role"]
            st.session_state.icon = st.session_state.users[username]["icon"]
            st.session_state.current_view = None
            print(True)
            return True
    print(False)
    return False

def render_login_screen():
    """Display the login interface."""
    st.markdown("<h1 style='text-align: center;'>Dairy Plant Login</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username").strip()
            password = st.text_input("Password", type="password", key="login_password").strip()
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

            if submitted:
                if check_login(username, password):
                    st.success(f"Welcome {username}! Loading dashboard...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

# --- UI Rendering Functions ---
def apply_custom_styling():
    """Apply enhanced custom CSS styling for a modern, cohesive dark-mode UI."""
    st.markdown("""
    <style>
        /* --- Import Google Fonts --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* --- Base & Body --- */
        .stApp {
            background: linear-gradient(135deg, #000000 0%, #000000 100%);
            font-family: 'Inter', sans-serif;
        }
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        /* --- Enhanced Header --- */
        .app-header {
            background: linear-gradient(135deg, #8E44AD 0%, #667EEA 100%);
            padding: 2.5rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 25px rgba(142, 68, 173, 0.2);
        }
        .app-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 600;
            text-align: center;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }
        .app-header p {
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            margin: 0.8rem 0 0 0;
            font-size: 1.1rem;
            font-weight: 400;
        }
    </style>
    """, unsafe_allow_html=True)

def render_app_header():
    current_time = datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
    st.markdown(f"""
    <div class="app-header fade-in-up">
        <h1>🏭 Dairy Production Plant Management</h1>
        <p>System Status: Online | {current_time}</p>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-header" style="background: linear-gradient(135deg, #8E44AD 0%, #667EEA 100%); border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; text-align: center;">
            <h2 style="margin: 0; color: white; font-size: 1.5rem;">{st.session_state.icon} {st.session_state.role}</h2>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8); font-size: 0.9rem;">User: {st.session_state.username}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Logout", use_container_width=True):
            # Keep essential state like 'users' but clear login state
            for key in ['logged_in', 'username', 'role', 'icon', 'current_view']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # --- DEV ONLY: User Management ---
        if st.session_state.role == "Developer":
            with st.expander("⚙️ User Management", expanded=False):
                st.subheader("Add New User")
                with st.form("add_user_form", clear_on_submit=True):
                    new_username = st.text_input("New Username")
                    new_password = st.text_input("New Password", type="password")
                    new_role = st.selectbox("Role", options=list(ROLE_ICONS.keys()))
                    
                    submitted = st.form_submit_button("Add User")
                    if submitted:
                        if new_username and new_password:
                            if new_username in st.session_state.users:
                                st.error(f"Username '{new_username}' already exists.")
                            else:
                                # Add user to session state
                                st.session_state.users[new_username] = {
                                    "password": new_password,
                                    "role": new_role,
                                    "icon": ROLE_ICONS[new_role]
                                }
                                # Save updated user list to CSV
                                save_users(st.session_state.users)
                                st.success(f"User '{new_username}' added successfully.")
                        else:
                            st.warning("Please provide both username and password.")
                
                st.subheader("Existing Users")
                for username in list(st.session_state.users.keys()):
                    if username == 'dev': # Prevent dev from deleting themselves
                        continue
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"- {username} ({st.session_state.users[username]['role']})")
                    with col2:
                        if st.button("❌", key=f"del_{username}", help=f"Delete {username}"):
                            # Remove user from session state
                            del st.session_state.users[username]
                            # Save updated list to CSV
                            save_users(st.session_state.users)
                            st.rerun()

        # --- Role and View Navigation ---
        if st.session_state.get('current_view'):
            # Only dev can change role from the sidebar
            if st.session_state.role == "Developer":
                if st.button("🔄 Change Role", use_container_width=True):
                    st.session_state.current_view = None
                    st.rerun()

def render_role_selection():
    st.markdown("## 👥 Select Your Role")
    roles = [
        {"name": "Factory Manager", "icon": "👷"},
        {"name": "Production", "icon": "🏭"},
        {"name": "Sales", "icon": "📊"},
        {"name": "Logistics", "icon": "🚛"},
        {"name": "Engineering", "icon": "⚙️"}
    ]
    
    cols = st.columns(len(roles))
    for i, role in enumerate(cols):
        with role:
            if st.button(f"{roles[i]['icon']} {roles[i]['name']}", use_container_width=True):
                st.session_state.current_view = roles[i]['name']
                st.session_state.current_icon = roles[i]['icon']
                st.rerun()

def render_view(view_name, view_icon):
    if view_name == "Factory Manager":
        factory_manager_view.render()
    if view_name == "Production":
        production_manager_view.render()
    if view_name == "Sales":
        sales_manager_view.render()
        
    # In a real app, you would call your view rendering functions here
    # e.g., if view_name == "Production": production_manager_view.render()
def initialize_app_state():
    if 'app_initialized' not in st.session_state:
        # Create a more engaging loading experience
        progress_container = st.container()
        with progress_container:
            st.info("🚀 Initializing Dairy Production Management System...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate initialization steps with progress
            steps = [
                ("Loading configuration files...", 20),
                ("Connecting to database...", 40),
                ("Initializing data loader...", 60),
                ("Loading production data...", 80),
                ("Finalizing setup...", 100)
            ]
            
            for step_text, progress in steps:
                status_text.text(step_text)
                progress_bar.progress(progress)
                time.sleep(0.3)  # Simulate loading time
            
            # Initialize actual components
            data_loader = DataLoader(data_dir=config.DATA_DIR)
            data_loader.load_all_data()
            
            # Store in session state
            st.session_state.data_loader = data_loader
            st.session_state.app_initialized = True
            st.session_state.last_schedule_result = None
            st.session_state.current_view = None
            st.session_state.current_icon = None
            
            logging.info("Application initialized. All configuration data has been loaded.")
            
            # Clear loading interface and show success
            progress_container.empty()
            st.success("✅ System initialized successfully! Welcome to the Dairy Production Management System.")
            time.sleep(1)
            st.rerun()

# --- Main App Router ---
def main():
    apply_custom_styling()
    if not st.session_state.get("logged_in"):
        render_login_screen()
        return

    if 'app_initialized' not in st.session_state:
        initialize_app_state()
        return

    render_app_header()
    render_enhanced_sidebar()
    
    user_role = st.session_state.role

    if user_role == "Developer":
        if not st.session_state.get('current_view'):
            render_role_selection()
        else:
            render_view(st.session_state.current_view, st.session_state.current_icon)
    else:
        # For other users, automatically set their view to their role
        if st.session_state.current_view is None:
            st.session_state.current_view = user_role
            st.session_state.current_icon = st.session_state.icon
            st.rerun()
        
        render_view(st.session_state.current_view, st.session_state.current_icon)

if __name__ == "__main__":
    main()