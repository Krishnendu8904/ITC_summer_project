import streamlit as st
import random
import networkx as nx
import pandas as pd

# ----- SUBJECT LIST -----
SUBJECTS = [
    "CHI-421", "AR-318", "AR-316", "MA-312", "MA-311", "PYM-545", "EP-361", "EP-511",
    "BC-312", "BC-311", "BM-201", "BM-302", "MCR-304", "CHE-573", "CHE-321", "CHE-331",
    "CE-353", "CE-422", "AI-112", "CSE-342", "AI-111", "CSE-574", "EC-449", "EC-423",
    "EO-201", "EE-319", "LLG-309", "HSL-301", "HPH-305", "ME-433", "ME-454", "MT-322",
    "MT-312", "MN-412", "MN-461", "PH-331", "PH-323", "MS-304", "MS-311"
]

# ----- SESSION STATE -----
if "graph" not in st.session_state:
    st.session_state.graph = nx.DiGraph()
    st.session_state.graph.add_nodes_from(SUBJECTS)

if "asked_pairs" not in st.session_state:
    st.session_state.asked_pairs = set()

if "current_pair" not in st.session_state:
    st.session_state.current_pair = None

# ----- FUNCTIONS -----
def is_comparison_needed(a, b):
    # If there's already a path from a to b or b to a, no need to compare
    return not (nx.has_path(st.session_state.graph, a, b) or nx.has_path(st.session_state.graph, b, a))

def get_next_pair():
    # Try all unordered pairs and find one that needs comparison
    for a in SUBJECTS:
        for b in SUBJECTS:
            if a != b and tuple(sorted([a, b])) not in st.session_state.asked_pairs:
                if is_comparison_needed(a, b):
                    return (a, b)
    return None

def record_vote(winner, loser):
    st.session_state.graph.add_edge(winner, loser)
    st.session_state.asked_pairs.add(tuple(sorted([winner, loser])))
    st.session_state.current_pair = get_next_pair()

def reset():
    st.session_state.graph = nx.DiGraph()
    st.session_state.graph.add_nodes_from(SUBJECTS)
    st.session_state.asked_pairs = set()
    st.session_state.current_pair = get_next_pair()

def get_live_ranking():
    try:
        ordered = list(nx.topological_sort(st.session_state.graph))
        return pd.DataFrame({"Rank": range(1, len(ordered) + 1), "Subject": ordered})
    except nx.NetworkXUnfeasible:
        return pd.DataFrame(columns=["Rank", "Subject"])

# ----- UI -----
st.title("⚖️ Rank Subjects by Pairwise Voting (Smart)")
st.caption("Click on the subject you prefer. The app will infer the rest using transitive logic.")

if st.button("🔄 Reset All Progress"):
    reset()

if st.session_state.current_pair is None:
    st.session_state.current_pair = get_next_pair()

pair = st.session_state.current_pair

if pair:
    col1, col2 = st.columns(2)
    col1.button(f"👍 {pair[0]}", on_click=record_vote, args=(pair[0], pair[1]))
    col2.button(f"👍 {pair[1]}", on_click=record_vote, args=(pair[1], pair[0]))

    st.subheader("📊 Live Subject Ranking (So Far)")
    st.dataframe(get_live_ranking(), use_container_width=True)

else:
    st.success("✅ All meaningful comparisons completed!")
    final_order = list(nx.topological_sort(st.session_state.graph))
    st.subheader("🏁 Final Subject Ranking")
    st.dataframe(pd.DataFrame({"Rank": range(1, len(final_order) + 1), "Subject": final_order}), use_container_width=True)
