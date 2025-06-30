import streamlit as st

st.title("MarketPulse — Hello World")
ticker = st.text_input("Ticker", "AAPL")
st.write(f"You entered **{ticker}**")
