import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from streamlit_option_menu import option_menu
import feedparser
import matplotlib.pyplot as plt

# Top ribbon with logo and title
st.markdown("""
<div style="background-color:#0074D9;padding:3px;border-radius:10px">
<h1 style="color:white;text-align:center;">📈 Stock Forecast</h1>
</div>
""", unsafe_allow_html=True)
# Display logo
st.sidebar.image('finvestigator-logo.png', width=250)  # Replace with your logo URL

# Navigation menu
menu_options = {
    "Home": "🏠",
    "Latest Market News": "📰",
    "Disclaimer": "⚠️",
}

def streamlit_menu():
    with st.sidebar:
        st.sidebar.markdown("---")
        selected = option_menu(
            menu_title="Main Menu",  # required
            options=['Home','Latest Market News','Disclaimer'],  # required
            icons=['house','newspaper','exclamation-diamond-fill'],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="vertical",
            styles={"menu": {"font-size": "20px"}},
        )
    return selected

@st.cache_resource
def load_data(ticker_symbol, start, end):
    try:
        data = yf.download(ticker_symbol, start, end)
        if data.empty:
            st.error("No data available for the specified company within the specified date range.")
            st.stop()
        return data
    except Exception as e:
        st.error(f"Error occurred while loading data: {e}")

@st.cache_resource
def train_prophet_model(data):
    df_train = data[["Close"]].reset_index()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    return m
selected_page = streamlit_menu()

if selected_page == "Home":
    # Home page content
    
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    
    # Add vertical space before the ticker_symbol input box
    st.write("<br>", unsafe_allow_html=True)
    ticker_symbol = st.text_input('Enter the company ticker symbol (e.g., TCS.NS, RELIANCE.NS)')
    st.markdown("<style>input[type='text'] {font-size: 18px;}</style>", unsafe_allow_html=True)

    n_years = st.slider('Years of prediction:', 1, 6)
    period = n_years * 365

    if not ticker_symbol:
        st.warning('Please visit https://finance.yahoo.com/ for ticker symbols.')
        st.stop()


    # Search for ticker symbol based on company name
    try:
        ticker_info = yf.Ticker(ticker_symbol).info
    except Exception as e:
        st.error(f"Error occurred while retrieving company information: {e}")
        st.stop()

    # Display company information in a box
    st.sidebar.subheader('Company Description')
    st.sidebar.info(f"**Company Name:** {ticker_info['shortName']}")
    st.sidebar.info(f"**Industry:** {ticker_info['industry']}")
    st.sidebar.info(f"**Sector:** {ticker_info['sector']}")
    st.sidebar.info(f"**Country:** {ticker_info['country']}")
    st.sidebar.info(f"**Short Summary:** {ticker_info['longBusinessSummary']}")

    # Load data
    try:
        data = yf.download(ticker_symbol, START, TODAY)
    except Exception as e:
        st.error(f"Error occurred while loading data: {e}")
        st.stop()

    # Check if data is empty
    if data.empty:
        st.warning("No data available for the specified company within the specified date range.")
        st.stop()

    # Display raw data
    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="stock_close"))
        fig.update_layout(
    title={
        'text': 'Time series data with Rangeslider',
       # 'y': 0.9,  # Adjust y-position if needed
       # 'x': 0.5,  # Adjust x-position if needed
        'font': {
            'family': 'Arial',  # Or another bold font
            'size': 25,
        },
    },
    xaxis_rangeslider_visible=True,
)
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Close']].reset_index()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)  # Change periods as per requirement
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast Data')
    st.write(forecast.tail())

    st.subheader('Forecast Plot')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1, use_container_width=True, height=400)

    st.subheader("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

elif selected_page == "Latest Market News":
    # RSS Feed page content
    def display_rss_feed(feed_url):
        feed = feedparser.parse(feed_url)
        feed_content = ""
        for entry in feed.entries:
            feed_content += f"<b>{entry.title}</b><br>"
            feed_content += f"{entry.summary}<br>"
            feed_content += f"<a href='{entry.link}'>Read more</a><br><br>"
        return feed_content

    # Display RSS feed in the sidebar with adjusted layout and background color
    rss_feed_url = "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
    if rss_feed_url:
        st.subheader('Latest Market News 📰')
        rss_feed_content = display_rss_feed(rss_feed_url)
        st.markdown(f"<div style='background-color: lightblue; padding: 10px; border-radius: 10px; overflow-y: auto;'>{rss_feed_content}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please provide an RSS feed URL.")

elif selected_page == "Disclaimer":
    # Disclaimer page content
    st.write("<br>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color: lightblue; padding: 10px; border-radius: 10px;'>
            <h2 style='color: #0074D9;'>Disclaimer ⚠️</h2>
            <p style='font-size: 16px; font-family: Arial, sans-serif;'>
                The information provided here is for educational and informational purposes only. It should not be considered as financial advice. Invest in the stock market at your own risk. Always do your own research or consult with a qualified financial advisor before making any investment decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)
def main():
    # Write your Streamlit UI code here
    st.title("My Streamlit Web App")

    # Embedding the Google AdSense HTML code
    st.write("""
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2656334449772852"
     crossorigin="anonymous"></script>
    <!-- Your AdSense Ad Slot Code -->
    <ins class="adsbygoogle"
         style="display:block"
         data-ad-client="ca-pub-2656334449772852"
         data-ad-format="auto"
         data-full-width-responsive="true"></ins>
    <script>
         (adsbygoogle = window.adsbygoogle || []).push({});
    </script>
    """)

if __name__ == "__main__":
    main()


