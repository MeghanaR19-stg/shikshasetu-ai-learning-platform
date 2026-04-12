import streamlit as st
import os
import requests
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ShikshaSetu: Multilingual Learning Platform", page_icon="🎓", layout="wide")

# Custom CSS for better padding and font sizes
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stHeading {
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        color: #1E3A8A; /* Navy Blue for academic feel */
    }
    .stText, .stMarkdown {
        line-height: 1.8; /* Increased line-height for readability */
        font-size: 1.15rem;
        word-spacing: 0.05rem; /* Subtle gap between words */
        letter-spacing: 0.01rem;
    }
    .lesson-section {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .quiz-question {
        background-color: #ffffff;
        padding: 1rem;
        border-right: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "lesson_data" not in st.session_state:
    st.session_state.lesson_data = None
if "current_topic_name" not in st.session_state:
    st.session_state.current_topic_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/graduation-cap.png", width=100)
    st.title("🎓 ShikshaSetu")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.divider()
    st.info("Interactive AI learning platform for India.")
    if st.button("🔄 Reset Platform"):
        st.session_state.lesson_data = None
        st.session_state.chat_history = []
        st.rerun()

st.title("🎓 ShikshaSetu – Multilingual AI Learning Platform")
st.markdown("Unlock curriculum-aligned knowledge with an interactive AI tutor.")

# Input Layer in a container
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        subject = st.selectbox("Subject", ["Science", "Mathematics", "Social Science", "English"])
        grade = st.selectbox("Grade", ["Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12"])
    with col2:
        topic_input = st.text_input("Topic", placeholder="e.g. Magnetism, Fractions, Human Rights")
        language = st.selectbox("Preferred Language", ["English", "Hindi", "Kannada", "Tamil"])
    
    if st.button("✨ Generate Personalized Lesson", use_container_width=True):
        if not api_key:
            st.error("Please enter your API Key in the sidebar.")
        elif not topic_input:
            st.warning("Please enter a topic.")
        else:
            with st.spinner(f"Preparing your high-quality {language} lesson..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/generate_lesson",
                        json={
                            "topic": topic_input,
                            "subject": subject,
                            "grade": grade,
                            "language": language,
                            "api_key": api_key
                        }
                    )
                    
                    if response.status_code == 200:
                        st.session_state.lesson_data = response.json()
                        st.session_state.current_topic_name = topic_input
                        st.session_state.chat_history = []
                        st.balloons()
                    else:
                        st.error(f"Backend Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed. Ensure the FastAPI backend is running. Error: {str(e)}")

# Display the lesson with better spacing
if st.session_state.lesson_data:
    data = st.session_state.lesson_data
    topic_display = st.session_state.current_topic_name
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True) # Professional spacing
    
    # 1. Heading
    st.markdown(f"## 📖 Lesson: {topic_display}")
    
    # 2. Explanation in a dedicated block
    with st.container():
        st.markdown(f"**Explanation:**\n\n{data['lesson_explanation']}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Key Points & Examples in balanced columns
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 💡 Key Learning Points")
        for point in data["key_points"]:
            st.markdown(f"- {point}")
    with c2:
        st.markdown("### 🇮🇳 Real-world Examples (India)")
        for example in data["indian_examples"]:
            st.markdown(f"🔹 {example}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. Summary
    with st.container():
        st.markdown(f"**Summary:** {data['short_summary']}")
    
    # 5. Quiz Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📝 Test Your Knowledge")
    st.markdown("*Select the correct option for each question below:*")
    
    for i, q in enumerate(data["quiz"]):
        with st.container():
            st.markdown(f"**Q{i+1}: {q['question']}**")
            choice = st.radio(
                f"Options for Q{i+1}", 
                q["options"], 
                key=f"quiz_q_{i}_{topic_display[:5]}", 
                index=None,
                label_visibility="collapsed"
            )
            if choice:
                if choice == q["answer"]:
                    st.success("Correct! ✅")
                else:
                    st.error(f"Try again. The correct answer is: {q['answer']}")
            st.markdown("<br>", unsafe_allow_html=True)
    
    # 6. Interactive Chat Assistant
    st.markdown("---")
    st.markdown("### 🎓 ShikshaSetu Assistant")
    st.markdown("*Need a simpler explanation? Type 'Simplify this' below or ask any question.*")
    
    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            
    if followup := st.chat_input("Ask a follow-up question..."):
        st.session_state.chat_history.append({"role": "user", "content": followup})
        with st.chat_message("user"):
            st.markdown(followup)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    res = requests.post(
                        "http://localhost:8000/ask_question",
                        json={
                            "lesson_data": data,
                            "query": followup,
                            "language": language,
                            "api_key": api_key
                        }
                    )
                    if res.status_code == 200:
                        ans = res.json()["response"]
                        st.markdown(ans)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    else:
                        st.error("Could not get a response from the AI tutor.")
                except Exception as e:
                    st.error(f"Communication error: {str(e)}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📚 Educational Context"):
    st.write("Source: NCERT English Standard Textbooks.")
    st.caption("Content is AI-validated for factual accuracy and Indian cultural relevance.")
