"""
Streamlit app wrapper for NCERT RAG v2.0
Provides interactive UI for agentic RAG system
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Lazy imports to prevent startup crashes
try:
    from stage3_generation import StudyAssistantV2, AgenticStudyAssistant, build_llm
    from main import load_chunks_from_disk, rebuild_retriever
except ImportError as e:
    st.error(f"⚠️ Failed to import required modules: {e}")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="NCERT Agentic RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def initialize_assistant(agentic: bool, k: int, api_key: str) -> any:
    """Initialize and cache the RAG assistant"""
    if not api_key:
        return None
        
    try:
        chunks = load_chunks_from_disk()
        retriever = rebuild_retriever(chunks, k=k)
        llm = build_llm(api_key)
        
        cls = AgenticStudyAssistant if agentic else StudyAssistantV2
        assistant = cls(retriever, llm, k=k, use_strict_prompt=True)
        return assistant
    except Exception as e:
        st.error(f"Failed to initialize assistant: {str(e)}")
        return None


def main():
    # Header
    st.markdown("<h1 class='main-header'>📚 NCERT Agentic RAG System</h1>", unsafe_allow_html=True)
    st.markdown(
        "Ask questions about NCERT textbooks and get AI-powered responses with retrieval augmentation."
    )
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    agentic = st.sidebar.checkbox("Use Agentic Mode", value=True, 
                                   help="Enable advanced agentic reasoning for better answers")
    
    k = st.sidebar.slider("Retrieval Top-K", min_value=1, max_value=10, value=5,
                          help="Number of chunks to retrieve for context")
    
    api_key = st.sidebar.text_input(
        "Groq API Key (required)",
        type="password",
        placeholder="Enter your Groq API key",
        help="Get a free key at https://console.groq.com/keys"
    )
    
    # Check for env var as fallback
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    
    # Require API key before initializing
    if not api_key:
        st.warning("⚠️ **API Key Required**\n\nPlease enter your Groq API key in the sidebar to continue.\n\nGet a free key at: https://console.groq.com/keys")
        return
    
    # Initialize assistant
    assistant = initialize_assistant(agentic, k, api_key)
    
    if not assistant:
        st.error("❌ Failed to initialize the assistant. Check your configuration.")
        return
    
    # Main content area
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mode", "Agentic" if agentic else "Standard")
    with col2:
        st.metric("Top-K Chunks", k)
    with col3:
        st.metric("Status", "✅ Ready")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Question input
    question = st.text_area(
        "Ask a question:",
        placeholder="e.g., What is photosynthesis? or Explain the structure of an atom.",
        height=100,
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        ask_button = st.button("🔍 Ask", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    # Process question
    if ask_button and question.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                response = assistant.ask(question)
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "response": response
                })
                
                # Display response
                st.success("✅ Got a response!")
                
                if isinstance(response, dict):
                    st.markdown("### Answer")
                    st.markdown(response.get("answer", "No answer generated"))
                    
                    if "sources" in response:
                        st.markdown("### 📖 Sources")
                        for i, source in enumerate(response["sources"][:k], 1):
                            with st.expander(f"Source {i}"):
                                st.text(source)
                else:
                    st.write(response)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("## 📜 Chat History")
        
        for i, item in enumerate(st.session_state.chat_history, 1):
            with st.expander(f"Q{i}: {item['question'][:60]}..."):
                st.markdown("**Question:**")
                st.write(item['question'])
                st.markdown("**Response:**")
                if isinstance(item['response'], dict):
                    st.write(item['response'].get("answer", item['response']))
                else:
                    st.write(item['response'])


if __name__ == "__main__":
    main()
