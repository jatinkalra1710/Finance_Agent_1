"""
AI-Powered Stock Analysis Platform
==================================
A production-grade financial analysis system using Multi-Agent AI Architecture
Author: [Your Name]
"""

import os
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum

import streamlit as st
import yfinance as yf
import pandas as pd
from tavily import TavilyClient
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Centralized configuration management"""
    MODEL = "gemini-2.0-flash"
    DAILY_LIMIT = 5  # Increased from 2
    MAX_NEWS_RESULTS = 10
    STOCK_HISTORY_PERIOD = "1mo"
    CACHE_TTL = 3600  # 1 hour cache
    
    @staticmethod
    def load_secrets():
        """Load API keys from Streamlit secrets"""
        try:
            os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
            os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
            return True
        except KeyError as e:
            st.error(f"❌ Missing API Key: {str(e)}. Please configure in Streamlit secrets.")
            return False


class MarketIndices:
    """Indian stock market indices and popular stocks"""
    POPULAR_STOCKS = {
        # Large Cap
        "Reliance Industries": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Tata Consultancy Services (TCS)": "TCS.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Infosys": "INFY.NS",
        "State Bank of India (SBI)": "SBIN.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "ITC Limited": "ITC.NS",
        "Larsen & Toubro (L&T)": "LT.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        
        # Mid/Small Cap & Tech
        "Mahindra & Mahindra": "M&M.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "Axis Bank": "AXISBANK.NS",
        "Zomato": "ZOMATO.NS",  # Fixed ticker
        "Paytm": "PAYTM.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "Wipro": "WIPRO.NS",
    }
    
    INDICES = {
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN",
        "Nifty Bank": "^NSEBANK",
        "Nifty IT": "^CNXIT",
    }


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class StockData:
    """Structured stock data model"""
    ticker: str
    current_price: float
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    day_change: Optional[float]
    week_52_high: Optional[float]
    week_52_low: Optional[float]
    volume: Optional[int]
    avg_volume: Optional[int]
    timestamp: str
    
    def to_report_string(self) -> str:
        """Format as professional report section"""
        market_cap_str = f"₹{self.market_cap:,.0f}" if self.market_cap else "N/A"
        pe_str = f"{self.pe_ratio:.2f}" if self.pe_ratio else "N/A"
        change_str = f"{self.day_change:+.2f}%" if self.day_change else "N/A"
        
        return f"""
**📊 Market Data Summary**
- **Ticker**: {self.ticker}
- **Current Price**: ₹{self.current_price:,.2f} ({change_str})
- **Market Capitalization**: {market_cap_str}
- **P/E Ratio**: {pe_str}
- **52-Week Range**: ₹{self.week_52_low:,.2f} - ₹{self.week_52_high:,.2f}
- **Volume**: {self.volume:,} (Avg: {self.avg_volume:,})
- **Data Retrieved**: {self.timestamp}
"""


class SentimentType(Enum):
    """Investment sentiment classification"""
    BULLISH = "Bullish 📈"
    BEARISH = "Bearish 📉"
    NEUTRAL = "Neutral ⚖️"
    MIXED = "Mixed Signals ⚡"


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class RateLimiter:
    """Advanced rate limiting with session state"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables"""
        if 'last_used_date' not in st.session_state:
            st.session_state['last_used_date'] = date.today()
            st.session_state['usage_count'] = 0
            st.session_state['analysis_history'] = []
    
    @staticmethod
    def reset_if_new_day():
        """Reset counter on new day"""
        today = date.today()
        if st.session_state['last_used_date'] != today:
            st.session_state['last_used_date'] = today
            st.session_state['usage_count'] = 0
    
    @staticmethod
    def can_analyze() -> Tuple[bool, int]:
        """Check if user can run analysis"""
        RateLimiter.initialize_session()
        RateLimiter.reset_if_new_day()
        
        remaining = Config.DAILY_LIMIT - st.session_state['usage_count']
        return remaining > 0, remaining
    
    @staticmethod
    def increment_usage(ticker: str):
        """Record usage"""
        st.session_state['usage_count'] += 1
        st.session_state['analysis_history'].append({
            'ticker': ticker,
            'timestamp': datetime.now().isoformat()
        })


class DataCache:
    """Simple caching mechanism"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_stock_data(ticker: str) -> Optional[StockData]:
        """Cached stock data retrieval"""
        return fetch_stock_data(ticker)


# ============================================================================
# CORE FINANCIAL DATA FUNCTIONS
# ============================================================================

def fetch_stock_data(ticker: str) -> Optional[StockData]:
    """
    Fetch comprehensive stock data with error handling
    
    Args:
        ticker: Yahoo Finance ticker symbol
        
    Returns:
        StockData object or None if fetch fails
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical data
        hist = stock.history(period=Config.STOCK_HISTORY_PERIOD)
        if hist.empty:
            logger.warning(f"No historical data for {ticker}")
            return None
        
        # Get current info
        info = stock.info
        
        # Calculate metrics
        current_price = round(hist['Close'].iloc[-1], 2)
        
        # Handle day change calculation
        if len(hist) > 1:
            prev_close = hist['Close'].iloc[-2]
            day_change = ((current_price - prev_close) / prev_close) * 100
        else:
            day_change = None
        
        return StockData(
            ticker=ticker,
            current_price=current_price,
            market_cap=info.get('marketCap'),
            pe_ratio=info.get('trailingPE'),
            day_change=day_change,
            week_52_high=info.get('fiftyTwoWeekHigh'),
            week_52_low=info.get('fiftyTwoWeekLow'),
            volume=int(hist['Volume'].iloc[-1]) if 'Volume' in hist else None,
            avg_volume=info.get('averageVolume'),
            timestamp=datetime.now().strftime("%B %d, %Y at %I:%M %p IST")
        )
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate ticker symbol format and existence
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not ticker:
        return False, "Ticker cannot be empty"
    
    # Basic format validation
    if not ticker.endswith(('.NS', '.BO')):
        return False, "Indian stocks should end with .NS (NSE) or .BO (BSE)"
    
    # Try to fetch data to verify existence
    try:
        test_stock = yf.Ticker(ticker)
        info = test_stock.info
        
        if not info or 'regularMarketPrice' not in info:
            return False, f"Ticker {ticker} not found or has no data"
        
        return True, "Valid ticker"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ============================================================================
# AI TOOLS
# ============================================================================

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))

@tool("advanced_web_search")
def advanced_web_search(query: str) -> str:
    """
    Advanced web search for financial news with context
    Searches multiple sources and returns structured results
    """
    try:
        today = datetime.now().strftime("%B %d, %Y")
        search_query = f"{query} stock market India news {today}"
        
        response = tavily_client.search(
            query=search_query,
            max_results=Config.MAX_NEWS_RESULTS,
            search_depth="advanced"
        )
        
        # Structure the results
        if 'results' in response:
            formatted_results = []
            for idx, result in enumerate(response['results'][:5], 1):
                formatted_results.append(
                    f"{idx}. **{result.get('title', 'N/A')}**\n"
                    f"   Source: {result.get('url', 'N/A')}\n"
                    f"   Summary: {result.get('content', 'N/A')[:200]}...\n"
                )
            return "\n".join(formatted_results)
        
        return str(response)
        
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return f"⚠️ Web search temporarily unavailable: {str(e)}"


@tool("comprehensive_yfinance_data")
def comprehensive_yfinance_data(ticker: str) -> str:
    """
    Fetches comprehensive real-time financial data with fallback mechanisms
    """
    try:
        stock_data = fetch_stock_data(ticker)
        
        if stock_data is None:
            return (
                f"⚠️ YFinance data unavailable for {ticker}.\n"
                f"FALLBACK REQUIRED: Use advanced_web_search for current price and metrics."
            )
        
        return stock_data.to_report_string()
        
    except Exception as e:
        logger.error(f"YFinance tool error: {str(e)}")
        return f"❌ Data fetch failed: {str(e)}. Use web search as backup."


# ============================================================================
# AI AGENT SYSTEM
# ============================================================================

def create_analysis_crew(ticker: str, company_name: str) -> Crew:
    """
    Creates a multi-agent crew for comprehensive stock analysis
    
    Args:
        ticker: Stock ticker symbol
        company_name: Human-readable company name
        
    Returns:
        Configured Crew instance
    """
    today = datetime.now().strftime("%B %d, %Y")
    
    # Agent 1: Senior Market Research Analyst
    research_agent = Agent(
        role="Senior Market Research Analyst",
        goal=f"Gather and validate comprehensive financial data and news for {company_name} ({ticker}) as of {today}",
        backstory="""You are a veteran market researcher with 15+ years at top investment banks.
        You have access to multiple data sources and always cross-verify information.
        You use web search as a backup when primary data sources fail.""",
        tools=[comprehensive_yfinance_data, advanced_web_search],
        llm=Config.MODEL,
        verbose=True
    )
    
    # Agent 2: Quantitative Analyst
    quant_agent = Agent(
        role="Quantitative Financial Analyst",
        goal="Perform technical and fundamental analysis on the gathered data",
        backstory="""You are a PhD in Financial Engineering specializing in equity valuation.
        You analyze price movements, valuation ratios, and financial health metrics.
        You provide data-driven insights without speculation.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # Agent 3: Market Sentiment & News Analyst
    sentiment_agent = Agent(
        role="Market Sentiment & News Analyst",
        goal="Determine investor sentiment (Bullish/Bearish/Neutral) based on news and market behavior",
        backstory="""You are a behavioral economist and former journalist who understands
        how news cycles impact stock prices. You analyze sentiment from multiple angles:
        news tone, analyst opinions, social sentiment, and market reaction.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # Agent 4: Chief Risk Officer
    risk_agent = Agent(
        role="Chief Risk Officer",
        goal=f"Identify top 3-5 material risks facing {company_name} in current market conditions",
        backstory="""You are a highly experienced risk manager who has navigated
        multiple market crashes. You focus on: regulatory risks, macro-economic factors,
        industry disruption, competitive threats, and company-specific vulnerabilities.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # Agent 5: Investment Strategist (Synthesizer)
    strategist_agent = Agent(
        role="Lead Investment Strategist",
        goal="Synthesize all findings into an executive investment memo with actionable insights",
        backstory="""You are a Managing Director at a top investment firm with $50B+ AUM.
        You write clear, professional reports for institutional investors.
        Your reports are balanced, data-driven, and include both bull and bear cases.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # Define Tasks
    tasks = [
        Task(
            description=f"""Gather comprehensive data for {ticker}:
            1. Current price, market cap, P/E ratio, 52-week range
            2. Latest 5-10 news articles
            3. Recent price movements and volume trends
            4. Any earnings announcements or corporate actions
            
            If YFinance fails, use web search as backup.""",
            expected_output="Detailed data dossier with all metrics and news sources",
            agent=research_agent
        ),
        
        Task(
            description=f"""Perform quantitative analysis on {ticker}:
            1. Valuation assessment (is it overvalued/undervalued?)
            2. Technical indicators and price trends
            3. Comparison with industry peers
            4. Financial health metrics
            
            Provide numerical analysis with context.""",
            expected_output="Quantitative analysis report with metrics and interpretations",
            agent=quant_agent
        ),
        
        Task(
            description=f"""Analyze market sentiment for {ticker}:
            1. Review news articles for tone (positive/negative/neutral)
            2. Identify key themes in recent coverage
            3. Assess institutional vs retail sentiment
            4. Classify overall sentiment: Bullish, Bearish, Neutral, or Mixed
            
            Support your classification with evidence.""",
            expected_output="Sentiment analysis report with clear classification and reasoning",
            agent=sentiment_agent
        ),
        
        Task(
            description=f"""Identify and explain the TOP 3-5 risks for {ticker}:
            1. Regulatory/Compliance risks
            2. Macroeconomic headwinds
            3. Industry/Competitive threats
            4. Company-specific vulnerabilities
            5. Market/Liquidity risks
            
            For each risk, explain the potential impact.""",
            expected_output="Comprehensive risk assessment with prioritized list",
            agent=risk_agent
        ),
        
        Task(
            description=f"""Create the final Executive Investment Memo for {company_name} ({ticker}).
            
            Structure:
            ## Executive Summary
            - One-paragraph overview with key recommendation
            
            ## Current Market Position
            - Price, valuation, and recent performance
            
            ## Investment Thesis
            - Bull case (reasons to buy)
            - Bear case (reasons to avoid/sell)
            
            ## Sentiment Analysis
            - Current market sentiment with supporting evidence
            
            ## Risk Factors
            - Top 3-5 material risks
            
            ## Conclusion
            - Balanced assessment
            - Target investor profile (who should consider this?)
            
            Use **Indian Rupees (₹)** for all currency values.
            Include today's date: {today}
            Be professional, balanced, and data-driven.""",
            expected_output="Professional markdown-formatted investment memo ready for institutional investors",
            agent=strategist_agent
        )
    ]
    
    # Create and return crew
    crew = Crew(
        agents=[research_agent, quant_agent, sentiment_agent, risk_agent, strategist_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    return crew


def run_analysis(ticker: str, company_name: str) -> str:
    """
    Execute the full AI-powered analysis workflow
    
    Args:
        ticker: Stock ticker symbol
        company_name: Human-readable company name
        
    Returns:
        Analysis report as markdown string
    """
    try:
        crew = create_analysis_crew(ticker, company_name)
        result = crew.kickoff(inputs={"company": company_name, "ticker": ticker})
        return str(result)
    except Exception as e:
        logger.error(f"Analysis execution error: {str(e)}")
        return f"❌ **Analysis Failed**\n\nError: {str(e)}\n\nPlease try again or contact support."


# ============================================================================
# STREAMLIT UI
# ============================================================================

def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="AI Stock Analyst Pro",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded"
    )


def render_sidebar():
    """Render sidebar with usage stats and info"""
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/stock-market.png", width=120)
        st.title("📊 AI Stock Analyst")
        st.markdown("---")
        
        # Usage metrics
        st.header("📈 Account Status")
        can_run, remaining = RateLimiter.can_analyze()
        usage = Config.DAILY_LIMIT - remaining
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Used Today", usage)
        with col2:
            st.metric("Remaining", remaining)
        
        progress = usage / Config.DAILY_LIMIT
        st.progress(progress)
        
        if not can_run:
            st.error("🚫 Daily limit reached!")
            st.info("Resets at midnight IST")
        
        st.markdown("---")
        
        # Analysis history
        if st.session_state.get('analysis_history'):
            st.subheader("📜 Recent Analyses")
            for entry in st.session_state['analysis_history'][-3:]:
                ts = datetime.fromisoformat(entry['timestamp'])
                st.caption(f"• {entry['ticker']} - {ts.strftime('%I:%M %p')}")
        
        st.markdown("---")
        
        # Info section
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            **Multi-Agent AI System**
            - 5 specialized AI agents
            - Real-time market data
            - News sentiment analysis
            - Comprehensive risk assessment
            
            **Tech Stack:**
            - CrewAI for orchestration
            - Google Gemini 2.0 Flash
            - yFinance for market data
            - Tavily for news search
            """)


def render_main_ui():
    """Render main application interface"""
    st.title("🤖 AI-Powered Stock Analysis Platform")
    st.markdown("""
    Generate **institutional-grade investment memos** using a multi-agent AI system.
    Powered by advanced LLMs and real-time market data.
    """)
    
    st.info("""
    💡 **How to find ticker symbols:**
    - Search on [Yahoo Finance India](https://in.finance.yahoo.com/)
    - NSE stocks end with `.NS` (e.g., `RELIANCE.NS`)
    - BSE stocks end with `.BO` (e.g., `RELIANCE.BO`)
    """)
    
    # Stock selection
    st.subheader("🎯 Select Stock for Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selection_mode = st.radio(
            "Choose input method:",
            ["📋 Popular Stocks", "🔍 Custom Ticker"],
            horizontal=True
        )
    
    target_ticker = None
    company_name = None
    
    if selection_mode == "📋 Popular Stocks":
        company_name = st.selectbox(
            "Select a company:",
            options=list(MarketIndices.POPULAR_STOCKS.keys())
        )
        target_ticker = MarketIndices.POPULAR_STOCKS[company_name]
        st.caption(f"Ticker: `{target_ticker}`")
        
    else:
        target_ticker = st.text_input(
            "Enter Yahoo Finance ticker:",
            placeholder="e.g., TATAPOWER.NS"
        ).upper().strip()
        
        if target_ticker:
            is_valid, message = validate_ticker(target_ticker)
            if is_valid:
                st.success(f"✅ {message}")
                company_name = target_ticker.split('.')[0]
            else:
                st.error(f"❌ {message}")
                target_ticker = None
    
    # Quick preview
    if target_ticker and st.checkbox("📊 Show Quick Market Preview"):
        with st.spinner("Fetching live data..."):
            stock_data = DataCache.get_stock_data(target_ticker)
            if stock_data:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"₹{stock_data.current_price:,.2f}")
                col2.metric("Day Change", f"{stock_data.day_change:+.2f}%" if stock_data.day_change else "N/A")
                col3.metric("P/E Ratio", f"{stock_data.pe_ratio:.2f}" if stock_data.pe_ratio else "N/A")
                col4.metric("Volume", f"{stock_data.volume:,}" if stock_data.volume else "N/A")
            else:
                st.warning("Unable to fetch preview data")
    
    st.markdown("---")
    
    # Analysis button
    can_run, remaining = RateLimiter.can_analyze()
    
    if st.button(
        "🚀 Generate Investment Memo",
        type="primary",
        disabled=not (can_run and target_ticker),
        use_container_width=True
    ):
        if not target_ticker:
            st.warning("⚠️ Please select or enter a valid ticker symbol")
            return
        
        # Increment usage
        RateLimiter.increment_usage(target_ticker)
        
        # Run analysis
        with st.status("🤖 AI Agents are analyzing the market...", expanded=True) as status:
            st.write("⚙️ Initializing multi-agent system...")
            st.write("📊 Fetching real-time market data...")
            st.write("📰 Scanning latest news and sentiment...")
            st.write("⚠️ Assessing risk factors...")
            st.write("📝 Synthesizing investment memo...")
            
            report = run_analysis(target_ticker, company_name or target_ticker)
            
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
        
        # Display report
        st.success("🎉 Investment Memo Generated Successfully!")
        st.markdown("---")
        
        # Report header
        st.markdown("### 📄 Executive Investment Memo")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**TO:**")
            st.markdown("**FROM:**")
            st.markdown("**DATE:**")
            st.markdown("**RE:**")
            st.markdown("**ASSET:**")
        with col2:
            st.markdown("Investment Committee")
            st.markdown("AI-Powered Analysis System")
            st.markdown(datetime.now().strftime("%B %d, %Y"))
            st.markdown("Strategic Outlook & Risk Assessment")
            st.markdown(f"**{company_name}** (`{target_ticker}`)")
        
        st.markdown("---")
        
        # Render the AI-generated report
        st.markdown(report)
        
        # Download button
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"{target_ticker}_investment_memo_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        
        st.markdown("---")
        
        # Compliance disclaimer
        render_disclaimer()


def render_disclaimer():
    """Render compliance and legal disclaimer"""
    st.warning("""
    ⚖️ **REGULATORY & COMPLIANCE DISCLAIMER**
    
    - **Not SEBI Registered:** This system is NOT registered with the Securities and Exchange Board of India (SEBI) 
      as an Investment Advisor under the SEBI (Investment Advisers) Regulations, 2013.
    
    - **No Financial Advice:** This document is for **educational and informational purposes only** and does not 
      constitute investment advice, a recommendation to buy or sell securities, or any form of solicitation.
    
    - **AI-Generated Content:** Reports are generated by Artificial Intelligence systems. Data may be delayed, 
      incomplete, or subject to errors and hallucinations. **Always independently verify all information.**
    
    - **Market Risks:** Equity investments are subject to market risks. Past performance is not indicative of 
      future results. Consult a **SEBI-registered financial advisor** before making investment decisions.
    
    - **No Liability:** The creators and operators of this tool assume no liability for any financial losses 
      incurred based on information provided herein.
    
    **By using this tool, you acknowledge that you have read and understood this disclaimer.**
    """, icon="⚖️")


def main():
    """Main application entry point"""
    # Load configuration
    if not Config.load_secrets():
        st.stop()
    
    # Initialize session state
    RateLimiter.initialize_session()
    
    # Setup page
    setup_page()
    
    # Render UI
    render_sidebar()
    render_main_ui()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>Built with ❤️ using CrewAI, Streamlit & Google Gemini</p>
        <p>© 2026 AI Stock Analyst Pro | For Educational Use Only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
