import streamlit as st
import requests
import pandas as pd
import os
import time

# API URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Complaint Triage", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["User Dashboard", "Annotator Workspace (HITL)"])

# =========================================================
# PAGE 1: USER DASHBOARD (Your existing code)
# =========================================================
if page == "User Dashboard":
    st.title("🛡️ Anti-Fragile Complaint Analytics")
    st.markdown("### Tiered Architecture: RoBERTa (CPU) -> Mistral (GPU)")

    # --- BATCH UPLOAD ---
    with st.expander("Batch Processing (Upload CSV)", expanded=False):
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if st.button("Run Batch Processing") and uploaded_file:
            df = pd.read_csv(uploaded_file)
            results = []
            progress_bar = st.progress(0)
            
            for index, row in df.iterrows():
                payload = {"id": str(row["id"]), "text": row["text"]}
                try:
                    res = requests.post(f"{API_URL}/analyze", json=payload).json()
                    results.append(res)
                except:
                    st.error(f"Failed on ID {row['id']}")
                progress_bar.progress((index + 1) / len(df))
            st.success("Batch Complete!")
            st.json(results)

    # --- LIVE TEST AREA ---
    st.subheader("Live Test Area")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_text = st.text_area("Enter a complaint manually:", height=150)
        if st.button("Analyze Complaint"):
            if user_text:
                with st.spinner("Routing..."):
                    payload = {"id": "manual_1", "text": user_text}
                    try:
                        res = requests.post(f"{API_URL}/analyze", json=payload).json()
                        
                        # Display Routing Logic
                        decision = res['routing']['decision']
                        color = "green" if decision == "Simple" else "orange" if decision == "Complex" else "red"
                        
                        st.markdown(f"### Routing Decision: :{color}[{decision}]")
                        st.markdown(f"**Confidence:** {res['routing']['confidence']:.2f}")
                        st.markdown(f"**Reason:** {res['routing']['reason']}")
                        
                        if res.get('analysis'):
                            st.markdown("---")
                            st.markdown("### Tier 1b Analysis (GPU)")
                            st.json(res['analysis'])
                            
                        if decision == "Review_Queue":
                            st.warning("⚠️ Flagged for Human Review. Sent to Annotator Workspace.")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        st.info("💡 **Demo Guide**")
        st.markdown("- **Simple:** 'Reset my password'")
        st.markdown("- **Complex:** 'Screen flickers...'")
        st.markdown("- **Safety:** 'Great job, useless app!'")

    # --- EXECUTIVE REPORT ---
    st.markdown("---")
    st.header("📊 Executive Reporting")
    if st.button("Generate Weekly Strategy Report"):
        with st.spinner("Consulting Llama 3.3 Strategy Agent..."):
            try:
                response = requests.get(f"{API_URL}/generate-report")
                if response.status_code == 200:
                    st.success("Report Generated")
                    st.info(response.json()["report"])
            except Exception as e:
                st.error(f"Connection Error: {e}")

# =========================================================
# PAGE 2: ANNOTATOR WORKSPACE (The New HITL Part)
# =========================================================
elif page == "Annotator Workspace (HITL)":
    st.title("👷 Human-in-the-Loop Workspace")
    st.markdown("Review ambiguous cases and retrain the model to fix data drift.")
    
    # 1. LOAD QUEUE
    queue_path = "data/human_review_queue.csv"
    if os.path.exists(queue_path):
        df_queue = pd.read_csv(queue_path)
        
        if not df_queue.empty:
            st.markdown(f"### 🚨 Review Queue ({len(df_queue)} pending)")
            
            # Editable Dataframe
            # We add a 'Correct_Label' column for the human to fix it
            if "Correct_Label" not in df_queue.columns:
                df_queue["Correct_Label"] = "Negative" # Default
            
            edited_df = st.data_editor(
                df_queue,
                num_rows="dynamic",
                column_config={
                    "text": "Complaint Text",
                    "reason_for_flagging": "AI Flag Reason",
                    "Correct_Label": st.column_config.SelectboxColumn(
                        "Correct Label",
                        options=["Negative", "Neutral", "Positive", "Sarcasm"],
                        required=True
                    )
                },
                use_container_width=True
            )
            
            # 2. SAVE TO GOLDEN SET
            if st.button("✅ Approve & Save to Golden Set"):
                # Append to Golden Set (Mock DB)
                golden_path = "data/golden_set.csv"
                # Check if file exists to write header or not
                header = not os.path.exists(golden_path)
                edited_df.to_csv(golden_path, mode='a', header=header, index=False)
                
                # Clear the Review Queue
                # In a real app, we'd only remove the approved rows. 
                # For demo, we clear the file to show "Work Done".
                open(queue_path, 'w').close() 
                # Re-write header
                with open(queue_path, "w") as f:
                    f.write("text,reason_for_flagging\n")
                
                st.success(f"Saved {len(edited_df)} rows to Golden Set. Queue cleared!")
                st.rerun()
                
        else:
            st.success("🎉 Queue is empty! Good job.")
            
    else:
        st.info("No Review Queue file found yet.")

    st.markdown("---")
    
    # 3. MOCK JENKINS TRIGGER
    st.subheader("⚙️ MLOps Pipeline")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Trigger Jenkins Retraining"):
            with st.status("🚀 Starting Jenkins Pipeline...", expanded=True) as status:
                st.write("Checking Golden Set data...")
                time.sleep(1)
                st.write("Fine-tuning RoBERTa model...")
                time.sleep(2)
                st.write("Validating F1 Score...")
                time.sleep(1)
                st.write("Deploying v2.1 to Production...")
                time.sleep(1)
                status.update(label="✅ Model Retrained & Deployed!", state="complete", expanded=False)
                st.balloons()