"""
DISCLAIMER: 

• This is NOT investment advice - for educational purposes only
• Always verify data and consult a financial advisor
• Market conditions change rapidly - data may be stale
• Slippage, commissions, and assignment risk not factored in
• Past volatility does not guarantee future results

"""

import customtkinter as ctk
from tkinter import messagebox
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import threading

# ============= DEBUGGING FLAG =============
DEBUG = False  # Set to False to disable debugging output (DISABLED FOR SPEED)

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")  # Options: "dark", "light", "system"
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

def debug_print(label, value):
    """Helper function to print debug information"""
    if DEBUG:
        print(f"\n{'='*60}")
        print(f"DEBUG - {label}")
        print(f"{'='*60}")
        if isinstance(value, (np.ndarray, list)):
            print(f"Type: {type(value)}")
            print(f"Length: {len(value)}")
            print(f"Value: {value}")
        elif hasattr(value, 'shape'):  # DataFrame or Series
            print(f"Type: {type(value)}")
            print(f"Shape: {value.shape}")
            print(f"Value:\n{value}")
        else:
            print(f"Type: {type(value)}")
            print(f"Value: {value}")
        print(f"{'='*60}\n")

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    
    debug_print("Input dates to filter_dates", dates)
    debug_print("Today's date", today)
    debug_print("Cutoff date (today + 45 days)", cutoff_date)
    
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    debug_print("Sorted dates", sorted_dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    
    debug_print("Filtered dates array", arr)
    
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            result = arr[1:]
            debug_print("Filtered dates (removed today)", result)
            return result
        return arr

    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    debug_print("Price data input to yang_zhang", price_data)
    debug_print("Yang-Zhang parameters", f"window={window}, trading_periods={trading_periods}")
    
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    debug_print("log_ho (High/Open)", log_ho)
    debug_print("log_lo (Low/Open)", log_lo)
    debug_print("log_co (Close/Open)", log_co)
    
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    debug_print("log_oc (Open/Close_shifted)", log_oc)
    debug_print("log_oc_sq", log_oc_sq)
    
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    debug_print("log_cc (Close/Close_shifted)", log_cc)
    debug_print("log_cc_sq", log_cc_sq)
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    debug_print("rs (Rogers-Satchell)", rs)
    
    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    debug_print("close_vol", close_vol)

    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    debug_print("open_vol", open_vol)

    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    debug_print("window_rs", window_rs)

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)) )
    debug_print("k coefficient", k)
    
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)
    debug_print("Yang-Zhang volatility (full series)", result)

    if return_last_only:
        final_result = result.iloc[-1]
        debug_print("Yang-Zhang volatility (last value only)", final_result)
        return final_result
    else:
        return result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    
    debug_print("Days to expiry (DTEs)", days)
    debug_print("Implied Volatilities (IVs)", ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]
    
    debug_print("Sorted DTEs", days)
    debug_print("Sorted IVs", ivs)

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:  
            result = ivs[0]
        elif dte > days[-1]:
            result = ivs[-1]
        else:  
            result = float(spline(dte))
        
        if DEBUG:
            print(f"Term spline at DTE={dte}: {result}")
        return result

    return term_spline

def get_current_price(ticker):
    todays_data = ticker.history(period='1d')
    debug_print("Today's price data", todays_data)
    
    current_price = todays_data['Close'][0]
    debug_print("Current stock price", current_price)
    return current_price

def compute_recommendation(ticker):
    try:
        ticker = ticker.strip().upper()
        debug_print("Ticker symbol", ticker)
        
        if not ticker:
            return "No stock symbol provided."
        
        try:
            stock = yf.Ticker(ticker)
            options_dates = stock.options
            debug_print("Available option expiration dates", options_dates)
            
            if len(options_dates) == 0:
                raise KeyError()
        except KeyError:
            return f"Error: No options found for stock symbol '{ticker}'."
        
        exp_dates = list(stock.options)
        try:
            exp_dates = filter_dates(exp_dates)
            debug_print("Filtered expiration dates", exp_dates)
        except:
            return "Error: Not enough option data."
        
        options_chains = {}
        for exp_date in exp_dates:
            chain = stock.option_chain(exp_date)
            options_chains[exp_date] = chain
            debug_print(f"Options chain for {exp_date} - Calls", chain.calls)
            debug_print(f"Options chain for {exp_date} - Puts", chain.puts)
        
        try:
            underlying_price = get_current_price(stock)
            if underlying_price is None:
                raise ValueError("No market price found.")
        except Exception:
            return "Error: Unable to retrieve underlying stock price."
        
        atm_iv = {}
        straddle = None 
        i = 0
        
        for exp_date, chain in options_chains.items():
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                debug_print(f"Empty chain for {exp_date}", "Skipping")
                continue

            call_diffs = (calls['strike'] - underlying_price).abs()
            call_idx = call_diffs.idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']
            debug_print(f"ATM Call for {exp_date}", {
                'strike': calls.loc[call_idx, 'strike'],
                'IV': call_iv
            })

            put_diffs = (puts['strike'] - underlying_price).abs()
            put_idx = put_diffs.idxmin()
            put_iv = puts.loc[put_idx, 'impliedVolatility']
            debug_print(f"ATM Put for {exp_date}", {
                'strike': puts.loc[put_idx, 'strike'],
                'IV': put_iv
            })

            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = atm_iv_value
            debug_print(f"Average ATM IV for {exp_date}", atm_iv_value)

            if i == 0:
                call_bid = calls.loc[call_idx, 'bid']
                call_ask = calls.loc[call_idx, 'ask']
                put_bid = puts.loc[put_idx, 'bid']
                put_ask = puts.loc[put_idx, 'ask']
                
                debug_print("First expiry call bid/ask", f"bid={call_bid}, ask={call_ask}")
                debug_print("First expiry put bid/ask", f"bid={put_bid}, ask={put_ask}")
                
                if call_bid is not None and call_ask is not None:
                    call_mid = (call_bid + call_ask) / 2.0
                else:
                    call_mid = None

                if put_bid is not None and put_ask is not None:
                    put_mid = (put_bid + put_ask) / 2.0
                else:
                    put_mid = None

                if call_mid is not None and put_mid is not None:
                    straddle = (call_mid + put_mid)
                    debug_print("Straddle price", straddle)

            i += 1
        
        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."
        
        today = datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(iv)
        
        debug_print("DTEs for term structure", dtes)
        debug_print("IVs for term structure", ivs)
        
        term_spline = build_term_structure(dtes, ivs)
        
        ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45-dtes[0])
        debug_print("Term structure slope (0 to 45)", {
            'IV at first DTE': term_spline(dtes[0]),
            'IV at 45 days': term_spline(45),
            'Slope': ts_slope_0_45
        })
        
        price_history = stock.history(period='3mo')
        debug_print("3-month price history", price_history)
        
        rv30 = yang_zhang(price_history)
        iv30 = term_spline(30)
        iv30_rv30 = iv30 / rv30
        debug_print("IV30 / RV30 calculation", {
            'IV30': iv30,
            'RV30': rv30,
            'Ratio': iv30_rv30
        })

        avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]
        debug_print("30-day average volume", avg_volume)

        expected_move = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else None
        debug_print("Expected move", expected_move)

        # Final checks
        result = {
            'avg_volume': avg_volume >= 1500000,
            'iv30_rv30': iv30_rv30 >= 1.25,
            'ts_slope_0_45': ts_slope_0_45 <= -0.00406,
            'expected_move': expected_move
        }
        
        debug_print("Final recommendation result", result)
        debug_print("Threshold checks", {
            'avg_volume >= 1,500,000': f"{avg_volume} >= 1500000 = {result['avg_volume']}",
            'iv30_rv30 >= 1.25': f"{iv30_rv30} >= 1.25 = {result['iv30_rv30']}",
            'ts_slope_0_45 <= -0.00406': f"{ts_slope_0_45} <= -0.00406 = {result['ts_slope_0_45']}"
        })
        
        return result
        
    except Exception as e:
        debug_print("Exception occurred", str(e))
        import traceback
        if DEBUG:
            traceback.print_exc()
        raise Exception(f'Error occurred processing')


class EarningsPositionChecker:
    def __init__(self, root):
        self.root = root
        self.root.title("Earnings Position Checker")
        self.root.geometry("1000x750")
        self.root.minsize(800, 600)
        
        # Create main container
        main_frame = ctk.CTkFrame(root)
        main_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # ============ HEADER FRAME ============
        header_frame = ctk.CTkFrame(main_frame, fg_color=("gray90", "gray15"), height=100)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="📊 Earnings Position Checker",
            font=("Helvetica", 28, "bold"),
            text_color=("black", "white")
        )
        title_label.pack(pady=10)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Analyze options data to find trading opportunities",
            font=("Helvetica", 12),
            text_color=("gray40", "gray70")
        )
        subtitle_label.pack()
        
        # ============ INPUT FRAME ============
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.pack(fill="x", padx=20, pady=20)
        
        label_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        label_frame.pack(fill="x", side="left", padx=(0, 20))
        
        ctk.CTkLabel(
            label_frame,
            text="Stock Symbol:",
            font=("Helvetica", 13, "bold"),
            text_color=("black", "white")
        ).pack(side="left", padx=(0, 10))
        
        self.stock_entry = ctk.CTkEntry(
            label_frame,
            font=("Helvetica", 13),
            width=120,
            placeholder_text="e.g., AAPL"
        )
        self.stock_entry.pack(side="left")
        self.stock_entry.bind('<Return>', lambda e: self.submit())
        
        # Button frame
        button_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        button_frame.pack(fill="x", side="right", padx=0)
        
        self.submit_button = ctk.CTkButton(
            button_frame,
            text="Analyze",
            command=self.submit,
            font=("Helvetica", 13, "bold"),
            width=120,
            height=40,
            fg_color="#0066CC",
            hover_color="#0052A3"
        )
        self.submit_button.pack(side="right", padx=(10, 0))
        
        info_button = ctk.CTkButton(
            button_frame,
            text="ℹ️ How It Works",
            command=self.show_description,
            font=("Helvetica", 13, "bold"),
            width=140,
            height=40,
            fg_color="#666666",
            hover_color="#555555"
        )
        info_button.pack(side="right", padx=(0, 10))
        
        # ============ DESCRIPTION/INFO FRAME ============
        desc_label_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        desc_label_frame.pack(fill="x", padx=20, pady=(10, 5))
        
        desc_label = ctk.CTkLabel(
            desc_label_frame,
            text="📋 Description & How It Works:",
            font=("Helvetica", 13, "bold"),
            text_color=("black", "white")
        )
        desc_label.pack(side="left")
        
        # Description text area
        self.description_text = ctk.CTkTextbox(
            main_frame,
            wrap="word",
            font=("Helvetica", 10),
            fg_color=("gray95", "gray20"),
            text_color=("black", "white"),
            border_width=2,
            border_color=("gray70", "gray50")
        )
        self.description_text.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        
        # Load initial description
        self.load_description()
        
        # ============ STATUS FRAME ============
        status_frame = ctk.CTkFrame(main_frame, fg_color=("gray90", "gray15"), height=50)
        status_frame.pack(fill="x", padx=0, pady=0)
        status_frame.pack_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready to analyze",
            font=("Helvetica", 11),
            text_color=("gray60", "gray70")
        )
        self.status_label.pack(pady=12)
        
        # Focus on entry
        self.stock_entry.focus()
    
    def load_description(self):
        """Load the description of what the code does"""
        description = """╔═══════════════════════════════════════════════════════════════════════════════════╗
║                    EARNINGS POSITION CHECKER - DOCUMENTATION                      ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

📋 WHAT THIS CODE DOES:
━━━━━━━━━━━━━━━━━━━━━━━━
This application analyzes stock options data to determine whether a stock is a good candidate
for an earnings trade. It evaluates three key criteria and provides a recommendation:
RECOMMENDED, CONSIDER, or AVOID.


🎯 THE THREE CRITERIA EVALUATED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. AVERAGE VOLUME (avg_volume):
   • Checks if the 30-day average trading volume >= 1,500,000 shares
   • High volume ensures liquidity and tighter bid-ask spreads
   • Calculation: Rolling 30-day mean of daily volume from yfinance

2. IMPLIED VS REALIZED VOLATILITY RATIO (iv30_rv30):
   • Compares 30-day implied volatility to 30-day realized (historical) volatility
   • Threshold: IV/RV ratio >= 1.25
   • When IV > RV, options may be overpriced, favorable for sellers
   • IV30: Interpolated from ATM option implied volatilities
   • RV30: Calculated using Yang-Zhang volatility estimator (more accurate than simple std dev)

3. TERM STRUCTURE SLOPE (ts_slope_0_45):
   • Measures the slope of the volatility term structure
   • Threshold: Slope <= -0.00406 (negative/downward sloping)
   • Negative slope indicates near-term volatility is elevated (earnings event priced in)
   • Calculation: (IV at 45 days - IV at first expiry) / (45 - days to first expiry)


⚙️ HOW IT WORKS - STEP BY STEP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: DATA COLLECTION
   ├─ Fetches option expiration dates from yfinance
   ├─ Filters dates to include only those 45+ days out
   └─ Retrieves full options chains (calls & puts) for each expiration

STEP 2: ATM IMPLIED VOLATILITY EXTRACTION
   ├─ For each expiration date:
   │  ├─ Finds ATM (at-the-money) call strike closest to current stock price
   │  ├─ Finds ATM put strike closest to current stock price
   │  └─ Averages the two IVs to get ATM IV for that expiration
   └─ Builds arrays of [days_to_expiry, implied_volatility] pairs

STEP 3: TERM STRUCTURE CONSTRUCTION
   ├─ Uses scipy.interpolate.interp1d to create a spline function
   ├─ Allows querying IV at any DTE (e.g., IV at 30 days, 45 days)
   └─ Handles extrapolation for dates outside available range

STEP 4: REALIZED VOLATILITY CALCULATION (Yang-Zhang Estimator)
   ├─ Downloads 3 months of OHLC price data
   ├─ Computes logarithmic returns across multiple price points:
   │  ├─ log(High/Open), log(Low/Open), log(Close/Open)
   │  ├─ log(Open/Close_previous), log(Close/Close_previous)
   │  └─ Rogers-Satchell component: combines high-low-close ranges
   ├─ Rolling 30-day window calculations for:
   │  ├─ Close-to-close volatility
   │  ├─ Open-to-close volatility
   │  └─ Rogers-Satchell volatility
   ├─ Combines components with optimal weighting (k = 0.34/formula)
   └─ Annualizes result by multiplying by sqrt(252 trading days)

STEP 5: EXPECTED MOVE CALCULATION
   ├─ Uses the FIRST expiration date (nearest term)
   ├─ Calculates ATM straddle price:
   │  └─ Straddle = (Call_mid + Put_mid) where mid = (bid + ask) / 2
   └─ Expected Move = (Straddle / Stock Price) × 100%

STEP 6: DECISION LOGIC
   ├─ ALL 3 criteria PASS → "RECOMMENDED" (Green)
   ├─ Term structure PASS + 1 other PASS → "CONSIDER" (Orange)
   └─ Otherwise → "AVOID" (Red)


🔬 TECHNICAL DETAILS:
━━━━━━━━━━━━━━━━━━━━━━

Yang-Zhang Volatility Formula:
   YZ = sqrt(open_vol + k×close_vol + (1-k)×window_rs) × sqrt(252)
   where k = 0.34 / (1.34 + (window+1)/(window-1))

Term Structure Slope:
   slope = (IV_45 - IV_first) / (45 - DTE_first)

Interpolation:
   Linear interpolation between known IVs at different DTEs
   Flat extrapolation beyond available range (uses edge values)


📊 DATA SOURCES:
━━━━━━━━━━━━━━━━
• yfinance library fetches data from Yahoo Finance
• Historical OHLC prices (3-month lookback)
• Real-time option chains with bid/ask/IV data
• Current stock price from most recent trading day


🎨 OUTPUT INTERPRETATION:
━━━━━━━━━━━━━━━━━━━━━━━━━
✅ PASS (Green) - Criterion met, favorable for trade
❌ FAIL (Red) - Criterion not met
📈 Expected Move - Market's implied price movement by first expiration


⚠️ IMPORTANT NOTES:
━━━━━━━━━━━━━━━━━━
• This is NOT investment advice - for educational purposes only
• Always verify data and consult a financial advisor
• Market conditions change rapidly - data may be stale
• Slippage, commissions, and assignment risk not factored in
• Past volatility does not guarantee future results


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enter a stock symbol above and click Analyze to get started!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.description_text.insert("1.0", description)
        self.description_text.configure(state="disabled")
    
    def show_description(self):
        """Show description in a popup window"""
        desc_window = ctk.CTkToplevel(self.root)
        desc_window.title("How It Works - Detailed Explanation")
        desc_window.geometry("1000x800")
        
        # Bring window to front
        desc_window.lift()
        desc_window.attributes('-topmost', True)
        desc_window.after_idle(desc_window.attributes, '-topmost', False)
        desc_window.focus()
        
        text_widget = ctk.CTkTextbox(
            desc_window,
            wrap="word",
            font=("Helvetica", 10),
            fg_color=("gray95", "gray20"),
            text_color=("black", "white"),
            border_width=2,
            border_color=("gray70", "gray50")
        )
        text_widget.pack(expand=True, fill='both', padx=15, pady=15)
        
        content = self.description_text.get("1.0", "end")
        text_widget.insert("1.0", content)
        text_widget.configure(state="disabled")
        
        # Close button
        close_button = ctk.CTkButton(
            desc_window,
            text="Close",
            command=desc_window.destroy,
            font=("Helvetica", 12, "bold"),
            width=100,
            height=35,
            fg_color="#666666",
            hover_color="#555555"
        )
        close_button.pack(pady=10)
    
    def submit(self):
        """Handle submit button click"""
        stock = self.stock_entry.get().strip()
        if not stock:
            messagebox.showwarning("Input Required", "Please enter a stock symbol.")
            return
        
        self.submit_button.configure(state="disabled")
        self.status_label.configure(text="⏳ Loading... Please wait.")
        self.root.update()
        
        # Run in thread to prevent GUI freezing
        result_holder = {}
        
        def worker():
            try:
                result = compute_recommendation(stock)
                result_holder['result'] = result
            except Exception as e:
                result_holder['error'] = str(e)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join()
        
        self.submit_button.configure(state="normal")
        
        if 'error' in result_holder:
            self.status_label.configure(text="❌ Error during analysis")
            messagebox.showerror("Error", result_holder['error'])
        elif 'result' in result_holder:
            result = result_holder['result']
            if isinstance(result, str):
                self.status_label.configure(text="❌ Error during analysis")
                messagebox.showerror("Error", result)
            else:
                self.status_label.configure(text="✅ Analysis complete!")
                self.show_results(result)
    
    def show_results(self, result):
        """Display results in a popup window"""
        avg_volume_bool = result['avg_volume']
        iv30_rv30_bool = result['iv30_rv30']
        ts_slope_bool = result['ts_slope_0_45']
        expected_move = result['expected_move']
        
        # Determine recommendation
        if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
            title = "✅ RECOMMENDED"
            title_color = "#00DD00"
            bg_color = "#1a3a1a"
        elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
            title = "⚠️ CONSIDER"
            title_color = "#FFB800"
            bg_color = "#3a2a1a"
        else:
            title = "❌ AVOID"
            title_color = "#FF6B6B"
            bg_color = "#3a1a1a"
        
        # Create result window
        result_window = ctk.CTkToplevel(self.root)
        result_window.title("Analysis Results")
        result_window.geometry("500x400")
        result_window.minsize(400, 350)
        
        # Bring window to front
        result_window.lift()
        result_window.attributes('-topmost', True)
        result_window.after_idle(result_window.attributes, '-topmost', False)
        result_window.focus()
        
        # Main container
        main_container = ctk.CTkFrame(result_window, fg_color=("gray95", "gray20"))
        main_container.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Title section
        title_frame = ctk.CTkFrame(main_container, fg_color=bg_color, height=80)
        title_frame.pack(fill="x", padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = ctk.CTkLabel(
            title_frame,
            text=title,
            font=("Helvetica", 28, "bold"),
            text_color=title_color
        )
        title_label.pack(pady=20)
        
        # Results frame
        results_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        results_frame.pack(expand=True, fill='both', padx=30, pady=20)
        
        # Individual criteria
        criteria = [
            ("🔹 Average Volume", "≥ 1.5M shares", avg_volume_bool),
            ("🔹 IV30/RV30 Ratio", "≥ 1.25", iv30_rv30_bool),
            ("🔹 Term Structure Slope", "≤ -0.00406", ts_slope_bool)
        ]
        
        for i, (criterion, threshold, passed) in enumerate(criteria):
            # Criterion name
            criterion_label = ctk.CTkLabel(
                results_frame,
                text=criterion,
                font=("Helvetica", 12, "bold"),
                text_color=("black", "white")
            )
            criterion_label.grid(row=i, column=0, sticky="w", pady=8)
            
            # Threshold
            threshold_label = ctk.CTkLabel(
                results_frame,
                text=threshold,
                font=("Helvetica", 11),
                text_color=("gray50", "gray60")
            )
            threshold_label.grid(row=i, column=1, sticky="w", padx=20, pady=8)
            
            # Status
            status = "✅ PASS" if passed else "❌ FAIL"
            color = "#00DD00" if passed else "#FF6B6B"
            
            status_label = ctk.CTkLabel(
                results_frame,
                text=status,
                font=("Helvetica", 12, "bold"),
                text_color=color
            )
            status_label.grid(row=i, column=2, sticky="e", pady=8)
        
        # Expected move
        if expected_move:
            move_label = ctk.CTkLabel(
                results_frame,
                text=f"\n📊 Expected Move: {expected_move}",
                font=("Helvetica", 13, "bold"),
                text_color="#0066CC"
            )
            move_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=15)
        
        # Button frame
        button_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        button_frame.pack(fill="x", padx=30, pady=20)
        
        close_button = ctk.CTkButton(
            button_frame,
            text="Close",
            command=result_window.destroy,
            font=("Helvetica", 12, "bold"),
            width=200,
            height=40,
            fg_color="#666666",
            hover_color="#555555"
        )
        close_button.pack()


def main():
    root = ctk.CTk()
    app = EarningsPositionChecker(root)
    root.mainloop()


if __name__ == "__main__":
    main()
