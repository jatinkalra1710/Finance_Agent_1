import os
import logging
from typing import Dict, Optional, Tuple
@@ -20,9 +12,6 @@
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Centralized configuration management"""
@@ -58,15 +47,12 @@ class MarketIndices:
        "ITC Limited": "ITC.NS",
        "Larsen & Toubro (L&T)": "LT.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        
        # Mid/Small Cap & Tech
        "Mahindra & Mahindra": "M&M.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "Axis Bank": "AXISBANK.NS",
        "Zomato": "ZOMATO.NS",
        "Paytm": "PAYTM.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "Wipro": "WIPRO.NS",
    }

@@ -77,11 +63,6 @@ class MarketIndices:
        "Nifty IT": "^CNXIT",
    }


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class StockData:
    """Structured stock data model"""
@@ -939,14 +920,14 @@ def render_main_ui():
        RateLimiter.increment_usage(target_ticker)

        # Run analysis
        with st.status("🤖 7 AI Agents are analyzing the market...", expanded=True) as status:
            st.write("⚙️ **Agent 1:** Research Analyst gathering data...")
            st.write("🔢 **Agent 2:** Quantitative Analyst evaluating fundamentals...")
            st.write("📈 **Agent 3:** Technical Analyst studying charts...")
            st.write("📰 **Agent 4:** Sentiment Analyst reading news...")
            st.write("🏭 **Agent 5:** Sector Specialist analyzing industry...")
            st.write("⚠️ **Agent 6:** Risk Officer assessing threats...")
            st.write("💼 **Agent 7:** Investment Strategist synthesizing report...")

            report = run_analysis(target_ticker, company_name or target_ticker)

@@ -1037,11 +1018,6 @@ def main():
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>🤖 Powered by 7-Agent Multi-AI System</p>
        <p>Built with CrewAI, Streamlit & Google Gemini 2.0</p>
        <p>© 2026 AI Stock Analyst Pro | For Educational Use Only</p>
    </div>
    """, unsafe_allow_html=True)
