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


class Config:
    """Centralized configuration management"""
    MODEL = "gemini-2.5-flash"
    DAILY_LIMIT = 5
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
        "Mahindra & Mahindra": "M&M.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "Axis Bank": "AXISBANK.NS",
        "Zomato": "ETERNAL.NS",
        "Paytm": "PAYTM.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Wipro": "WIPRO.NS",
    }
    
    INDICES = {
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN",
        "Nifty Bank": "^NSEBANK",
        "Nifty IT": "^CNXIT",
    }

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
# AI AGENT SYSTEM - 7 AGENTS
# ============================================================================

def create_analysis_crew(ticker: str, company_name: str) -> Crew:
    """
    Creates a 7-agent crew for comprehensive stock analysis
    
    Args:
        ticker: Stock ticker symbol
        company_name: Human-readable company name
        
    Returns:
        Configured Crew instance with 7 specialized agents
    """
    today = datetime.now().strftime("%B %d, %Y")
    
    # ========================================================================
    # AGENT 1: Senior Market Research Analyst
    # ========================================================================
    research_agent = Agent(
        role="Senior Market Research Analyst",
        goal=f"Gather and validate comprehensive financial data and news for {company_name} ({ticker}) as of {today}",
        backstory="""You are a veteran market researcher with 15+ years at top investment banks 
        like Goldman Sachs and JP Morgan. You have access to multiple data sources and always 
        cross-verify information. You use web search as a backup when primary data sources fail. 
        You're known for your meticulous attention to detail and accuracy.""",
        tools=[comprehensive_yfinance_data, advanced_web_search],
        llm=Config.MODEL,
        verbose=True
    )
    
    # ========================================================================
    # AGENT 2: Quantitative Financial Analyst
    # ========================================================================
    quant_agent = Agent(
        role="Quantitative Financial Analyst",
        goal="Perform fundamental analysis on financial metrics and company valuation",
        backstory="""You are a PhD in Financial Engineering from MIT, specializing in equity 
        valuation and financial modeling. You analyze P/E ratios, market cap, revenue growth, 
        profit margins, and other fundamental metrics. You compare companies against their 
        industry peers and historical performance. You provide data-driven insights without 
        speculation, always backing your analysis with numbers.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # ========================================================================
    # AGENT 3: Technical Analyst
    # ========================================================================
    technical_agent = Agent(
        role="Senior Technical Analyst",
        goal="Analyze price trends, chart patterns, and technical indicators",
        backstory="""You are a Chartered Market Technician (CMT) with 12+ years of experience 
        in technical analysis. You analyze 52-week price ranges, volume trends, support and 
        resistance levels, and price momentum. You identify chart patterns like head and shoulders, 
        double tops/bottoms, and trend channels. You calculate relative strength and identify 
        whether stocks are overbought or oversold. Your insights help time market entry and exit.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # ========================================================================
    # AGENT 4: Market Sentiment & News Analyst
    # ========================================================================
    sentiment_agent = Agent(
        role="Market Sentiment & News Analyst",
        goal="Determine investor sentiment (Bullish/Bearish/Neutral) based on news and market behavior",
        backstory="""You are a behavioral economist and former Bloomberg journalist who understands 
        how news cycles, social media, and analyst opinions impact stock prices. You analyze 
        sentiment from multiple angles: news tone, analyst ratings, social media buzz, and market 
        reaction to events. You can detect fear, greed, optimism, and pessimism in market behavior. 
        You classify sentiment as Bullish, Bearish, Neutral, or Mixed with supporting evidence.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # ========================================================================
    # AGENT 5: Sector & Industry Specialist
    # ========================================================================
    sector_agent = Agent(
        role="Sector & Industry Specialist",
        goal=f"Analyze {company_name}'s position within its industry and identify competitive dynamics",
        backstory="""You are an industry analyst who has covered multiple sectors for major 
        research firms. You understand industry trends, competitive landscapes, market share 
        dynamics, and sector-specific risks. You compare companies against their direct competitors, 
        identify market leaders and laggards, and spot emerging threats and opportunities. You 
        analyze how macroeconomic factors, regulatory changes, and technological disruptions 
        impact different sectors.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # ========================================================================
    # AGENT 6: Chief Risk Officer
    # ========================================================================
    risk_agent = Agent(
        role="Chief Risk Officer",
        goal=f"Identify top 3-5 material risks facing {company_name} in current market conditions",
        backstory="""You are a highly experienced risk manager who has navigated multiple market 
        crashes including the 2008 financial crisis and 2020 COVID crash. You have a keen eye 
        for spotting risks before they materialize. You focus on: regulatory and compliance risks, 
        macroeconomic headwinds, industry disruption, competitive threats, management quality issues, 
        debt and liquidity concerns, and company-specific vulnerabilities. You quantify risk impact 
        and probability whenever possible.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # ========================================================================
    # AGENT 7: Lead Investment Strategist (Synthesizer)
    # ========================================================================
    strategist_agent = Agent(
        role="Lead Investment Strategist & Portfolio Manager",
        goal="Synthesize all findings into an executive investment memo with actionable insights",
        backstory="""You are a Managing Director and Portfolio Manager at a top investment firm 
        managing $50B+ in assets. You have an MBA from Harvard Business School and 20+ years of 
        experience. You write clear, professional reports for institutional investors, pension funds, 
        and ultra-high-net-worth individuals. Your reports are balanced, data-driven, and include 
        both bull and bear cases. You're known for your ability to synthesize complex information 
        into actionable investment recommendations. You always consider risk-adjusted returns and 
        investor suitability.""",
        llm=Config.MODEL,
        verbose=True
    )
    
    # ========================================================================
    # DEFINE TASKS FOR ALL 7 AGENTS
    # ========================================================================
    
    tasks = [
        # Task 1: Data Gathering
        Task(
            description=f"""Gather comprehensive data for {company_name} ({ticker}):
            
            1. **Financial Metrics:**
               - Current stock price
               - Market capitalization
               - P/E ratio and other valuation metrics
               - 52-week high and low prices
               - Trading volume (current vs average)
            
            2. **News & Events:**
               - Latest 5-10 news articles
               - Recent earnings announcements
               - Corporate actions (dividends, splits, buybacks)
               - Management changes or strategic initiatives
            
            3. **Market Context:**
               - Recent price movements (1 week, 1 month, 3 months)
               - Comparison with major indices (Nifty 50, Sensex)
            
            **Important:** If YFinance fails for any data point, use advanced_web_search as backup.
            Verify all critical numbers from multiple sources when possible.""",
            expected_output="Comprehensive data dossier with all metrics, news sources, and market context. Include exact numbers with sources.",
            agent=research_agent
        ),
        
        # Task 2: Fundamental Analysis
        Task(
            description=f"""Perform quantitative fundamental analysis on {company_name} ({ticker}):
            
            1. **Valuation Analysis:**
               - Is the P/E ratio high or low compared to industry average?
               - Market cap analysis - is it fairly valued?
               - Price-to-book ratio assessment if available
               - Any other valuation metrics you can derive
            
            2. **Financial Health:**
               - Revenue and profit trends (if available in news/reports)
               - Debt levels and financial stability mentions
               - Cash flow and liquidity indicators
            
            3. **Growth Prospects:**
               - Historical price performance
               - Growth trajectory based on available data
               - Expansion plans or new business initiatives
            
            4. **Peer Comparison:**
               - How does it compare to competitors in the same sector?
               - Is it a market leader or follower?
            
            Provide numerical analysis with clear context and interpretations.""",
            expected_output="Detailed fundamental analysis report with valuations, financial health assessment, and peer comparisons. Use specific numbers.",
            agent=quant_agent
        ),
        
        # Task 3: Technical Analysis
        Task(
            description=f"""Perform technical analysis on {company_name} ({ticker}):
            
            1. **Price Trend Analysis:**
               - Current price vs 52-week high/low - is it near support or resistance?
               - Price momentum - uptrend, downtrend, or sideways?
               - Distance from 52-week high/low (in percentage terms)
            
            2. **Volume Analysis:**
               - Current volume vs average volume - is there unusual activity?
               - Volume trends - increasing or decreasing?
               - What does volume tell us about buying/selling pressure?
            
            3. **Key Levels:**
               - Support levels (52-week low and intermediate supports)
               - Resistance levels (52-week high and intermediate resistances)
               - Psychological price levels
            
            4. **Technical Outlook:**
               - Short-term outlook (1-4 weeks)
               - Medium-term outlook (1-3 months)
               - Are there any chart patterns visible?
               - Is the stock overbought, oversold, or fairly priced technically?
            
            Base your analysis on the price data, 52-week range, and volume information available.""",
            expected_output="Technical analysis report with price trends, volume analysis, support/resistance levels, and technical outlook with specific price targets if possible.",
            agent=technical_agent
        ),
        
        # Task 4: Sentiment Analysis
        Task(
            description=f"""Analyze market sentiment for {company_name} ({ticker}):
            
            1. **News Sentiment:**
               - Review all news articles for overall tone (positive/negative/neutral)
               - Identify key themes in recent coverage
               - Are there any major controversies or positive developments?
            
            2. **Market Behavior:**
               - How has the market reacted to recent news?
               - Is there evidence of institutional buying or selling?
               - What does price action tell us about sentiment?
            
            3. **Analyst Sentiment:**
               - Any analyst upgrades/downgrades mentioned in news?
               - What are professional analysts saying?
            
            4. **Overall Classification:**
               - Classify sentiment as: **Bullish 📈**, **Bearish 📉**, **Neutral ⚖️**, or **Mixed Signals ⚡**
               - Provide strong evidence for your classification
               - Rate sentiment intensity (mildly/moderately/strongly bullish or bearish)
            
            Support all conclusions with specific examples from news and market data.""",
            expected_output="Comprehensive sentiment analysis with clear classification (Bullish/Bearish/Neutral/Mixed), evidence from news, and market behavior insights.",
            agent=sentiment_agent
        ),
        
        # Task 5: Sector & Competitive Analysis
        Task(
            description=f"""Analyze {company_name} ({ticker}) within its industry context:
            
            1. **Industry Identification:**
               - What sector/industry does this company operate in?
               - What are the key characteristics of this industry?
            
            2. **Competitive Position:**
               - Who are the main competitors?
               - What is the company's market share or ranking?
               - Is it a market leader, challenger, or follower?
            
            3. **Industry Trends:**
               - What are the major trends affecting this industry?
               - Is the industry growing, stable, or declining?
               - Any regulatory changes impacting the sector?
            
            4. **Competitive Advantages/Disadvantages:**
               - What are the company's strengths vs competitors?
               - What are its weaknesses or vulnerabilities?
               - Does it have any moats (brand, technology, network effects)?
            
            5. **Sector Outlook:**
               - What's the outlook for this sector in India?
               - How do macroeconomic factors affect this industry?
            
            Use information from news and general industry knowledge.""",
            expected_output="Industry and competitive analysis report covering sector trends, competitive positioning, market share, and outlook.",
            agent=sector_agent
        ),
        
        # Task 6: Risk Assessment
        Task(
            description=f"""Identify and explain the TOP 3-5 material risks for {company_name} ({ticker}):
            
            Analyze and prioritize risks in these categories:
            
            1. **Regulatory/Compliance Risks:**
               - Government policy changes
               - Regulatory investigations or penalties
               - Compliance issues
            
            2. **Macroeconomic Risks:**
               - Interest rate sensitivity
               - Currency risks
               - Inflation impact
               - Economic slowdown effects
            
            3. **Industry/Competitive Risks:**
               - New entrants or disruptive competitors
               - Technology disruption
               - Market share loss
               - Pricing pressure
            
            4. **Company-Specific Risks:**
               - Management quality concerns
               - Debt and leverage issues
               - Operational challenges
               - Customer concentration
            
            5. **Market/Liquidity Risks:**
               - Stock volatility
               - Low liquidity issues
               - Market sentiment shifts
            
            For each identified risk:
            - **Explain** what the risk is
            - **Assess impact:** High/Medium/Low
            - **Assess probability:** Likely/Possible/Unlikely
            - **Provide mitigation:** What could reduce this risk?
            
            Prioritize by severity (probability × impact).""",
            expected_output="Comprehensive risk assessment with TOP 3-5 prioritized risks, each with impact level, probability, and mitigation strategies.",
            agent=risk_agent
        ),
        
        # Task 7: Final Investment Memo
        Task(
            description=f"""Create the final Executive Investment Memo for {company_name} ({ticker}).
            
            Synthesize insights from all 6 previous analysts into a cohesive, professional report.
            
            **Required Structure:**
            
            ## Executive Summary
            - 3-4 sentence overview capturing the investment opportunity
            - Clear statement of recommendation context (not financial advice)
            
            ## Current Market Position
            - Stock price and recent performance
            - Valuation metrics (P/E, market cap)
            - Technical position (trend, key levels)
            
            ## Investment Thesis
            
            ### Bull Case 💚
            - 3-5 compelling reasons to consider this stock
            - Use insights from fundamental, technical, and sector analysis
            - Include specific data points
            
            ### Bear Case 🔴
            - 3-5 reasons for caution or to avoid
            - Include competitive threats and valuation concerns
            - Reference the risk assessment
            
            ## Market Sentiment Analysis
            - Overall sentiment classification (Bullish/Bearish/Neutral/Mixed)
            - Evidence from news and market behavior
            - 2-3 sentences explaining the sentiment
            
            ## Industry & Competitive Context
            - Sector overview and trends
            - Competitive positioning
            - Industry outlook
            
            ## Technical Outlook
            - Short-term and medium-term technical view
            - Key support and resistance levels
            - Volume and momentum analysis
            
            ## Key Risk Factors
            List the TOP 3-5 risks with:
            1. **[Risk Name]:** Description, Impact, Probability
            2. **[Risk Name]:** Description, Impact, Probability
            [Continue for all major risks]
            
            ## Financial Snapshot
            - Table or bullets with key metrics
            - Current Price, Market Cap, P/E, 52-Week Range, Volume
            
            ## Conclusion & Suitability
            - Balanced final assessment
            - **Target Investor Profile:** Who should consider this? (Growth investors, value investors, income seekers, aggressive traders, conservative long-term holders)
            - Investment horizon recommendation (short-term swing, medium-term hold, long-term investment)
            
            **Formatting Requirements:**
            - Use **Indian Rupees (₹)** for all currency values
            - Include today's date: {today}
            - Use markdown formatting with headers, bold, bullets
            - Be professional, balanced, and data-driven
            - Length: 1000-1500 words
            - Include specific numbers and percentages throughout
            
            **Tone:**
            - Professional and institutional-grade
            - Objective and balanced (show both sides)
            - Data-driven with evidence
            - Clear and actionable
            - Educational (explain why, not just what)""",
            expected_output="Complete professional investment memo in markdown format, 1000-1500 words, covering all required sections with data-driven insights and balanced perspective.",
            agent=strategist_agent
        )
    ]
    
    # ========================================================================
    # CREATE AND RETURN CREW
    # ========================================================================
    
    crew = Crew(
        agents=[
            research_agent,
            quant_agent,
            technical_agent,
            sentiment_agent,
            sector_agent,
            risk_agent,
            strategist_agent
        ],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    return crew


def run_analysis(ticker: str, company_name: str) -> str:
    """
    Execute the full 7-agent AI-powered analysis workflow
    
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
        page_title="AI Stock Analyst Pro - 7 Agent System",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded"
    )


def render_sidebar():
    """Render sidebar with usage stats and info"""
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/stock-market.png", width=120)
        st.title("📊 AI Stock Analyst")
        st.caption("7-Agent Multi-AI System")
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
        
        # Agent info section
        with st.expander("🤖 7-Agent AI System"):
            st.markdown("""
            **Specialized AI Agents:**
            1. 📊 Market Research Analyst
            2. 🔢 Quantitative Analyst
            3. 📈 Technical Analyst
            4. 📰 Sentiment Analyst
            5. 🏭 Sector Specialist
            6. ⚠️ Chief Risk Officer
            7. 💼 Investment Strategist
            
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
    Generate **institutional-grade investment memos** using a **7-agent multi-AI system**.
    Powered by advanced LLMs and real-time market data.
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🤖 AI Agents", "7", help="Specialized agents for comprehensive analysis")
    with col2:
        st.metric("📊 Data Sources", "3+", help="yFinance, Tavily News, Web Search")
    with col3:
        st.metric("⚡ Avg Time", "45-60s", help="Complete analysis time")
    
    st.markdown("---")
    
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
        "🚀 Generate Investment Memo (7 AI Agents)",
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
        with st.status("7 AI Agents are analyzing the market...", expanded=True) as status:
            st.write("**Agent 1:** Research Analyst gathering data...")
            st.write("**Agent 2:** Quantitative Analyst evaluating fundamentals...")
            st.write("**Agent 3:** Technical Analyst studying charts...")
            st.write("**Agent 4:** Sentiment Analyst reading news...")
            st.write("**Agent 5:** Sector Specialist analyzing industry...")
            st.write("**Agent 6:** Risk Officer assessing threats...")
            st.write("**Agent 7:** Investment Strategist synthesizing report...")
            
            report = run_analysis(target_ticker, company_name or target_ticker)
            
            status.update(label="✅ Analysis Complete! 7 Agents Collaborated Successfully", state="complete", expanded=False)
        
        # Display report
        st.success("🎉 Investment Memo Generated by 7-Agent AI System!")
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
            st.markdown("**SYSTEM:**")
        with col2:
            st.markdown("Investment Committee")
            st.markdown("7-Agent AI Analysis System")
            st.markdown(datetime.now().strftime("%B %d, %Y"))
            st.markdown("Comprehensive Strategic Outlook & Risk Assessment")
            st.markdown(f"**{company_name}** (`{target_ticker}`)")
            st.markdown("Multi-Agent AI Collaboration")
        
        st.markdown("---")
        
        # Render the AI-generated report
        st.markdown(report)
        
        # Download button
        st.download_button(
            label="📥 Download Investment Memo",
            data=report,
            file_name=f"{target_ticker}_7agent_investment_memo_{datetime.now().strftime('%Y%m%d')}.md",
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
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
