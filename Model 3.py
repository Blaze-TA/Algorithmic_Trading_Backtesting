from AlgorithmImports import *
import numpy as np
from collections import deque
from datetime import timedelta
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

class DistanceMethodPairsTrading(QCAlgorithm):

    # Core Strategy Parameters
    FORM_DAYS   = 252
    TRADE_DAYS  = 126
    ENTRY_Z     = 2.0 

    # VIX-BASED POSITION SIZING (This is the key innovation!)
    VIX_LOW_LEVEL = 12.0           # Below this = very calm markets
    VIX_MEDIUM_LEVEL = 20.0        # Normal volatility
    VIX_HIGH_LEVEL = 35.0          # Elevated volatility
    VIX_EXTREME_LEVEL = 50.0       # Extreme volatility (cap here)
    
    # Position sizes based on VIX regime
    MIN_GROSS_CAP = 0.20           # 20% in very calm markets (VIX < 12)
    BASE_GROSS_CAP = 0.40          # 40% in normal markets (VIX ~20)
    HIGH_GROSS_CAP = 0.70          # 70% in elevated vol (VIX ~35)
    MAX_GROSS_CAP = 0.90           # 90% maximum in extreme vol (VIX 50+)
    
    # Emergency exit only at catastrophic levels
    VIX_CATASTROPHIC = 75.0        # Only exit if VIX > 75 (2008/2020 peak levels)
    
    VIX_LOOKBACK = 20              # Number of VIX observations to track

    # ETF A is treated as Y, B as X.
    ETF_A = "VOO"
    ETF_B = "IVV"
    
    # Resolution is the duration of time used for sampling the data source.
    TRADE_FREQUENCY = Resolution.DAILY
    
    # Starting date for our back-testing strategy.
    START_YEAR = 2006
    START_MONTH = 1
    START_DAY = 1
    
    # Starting cash for our back-testing strategy.
    START_CASH = 10000

    def Initialize(self):

        
        #1) Pick a start date for our strategy to back-test from and set our initial starting cash.
        self.set_start_date(self.START_YEAR, self.START_MONTH, self.START_DAY)
        self.set_cash(self.START_CASH)

        #2) Add both of the ETFs to the algorithm. ETF_A is mapped to 'Y', and ETF_B is mapped to 'X'.
        # This line ensures that Open, High, Low, Close, Volume data at the given trade frequency (daily in our case)
        # is obtained and added to our algorithm such that we can begin trading and holding positions.
        self.y = self.add_equity(self.ETF_A, self.TRADE_FREQUENCY).symbol
        self.x = self.add_equity(self.ETF_B, self.TRADE_FREQUENCY).symbol
        
        #3) Add VIX data to the algorithm. VIX is the CBOE Volatility Index, often called the "fear gauge".
        # We use VIX to dynamically adjust our position sizing based on market volatility conditions.
        self.vix = self.add_data(CBOE, "VIX", self.TRADE_FREQUENCY).symbol
        
        #4) Initialize a rolling window to store the most recent VIX values.
        # This window stores the last VIX_LOOKBACK (20) VIX observations for calculating position sizes.
        # RollingWindow[float] is a QuantConnect data structure that automatically maintains the most recent N values.
        self.vix_window = RollingWindow[float](self.VIX_LOOKBACK)
        
        #5) Initialize historical data storage for cointegration testing.
        # This will store the complete price history from start_date to current time,
        # which is required for the Johansen cointegration test and ADF stationarity test.
        self.historical_data = None

        #6) Initialize the state tracking variables. These ensure that we are either only in formation mode or trading mode,
        # but not both at the same time.
        self.in_trade_window = False
        self.trade_window_end = None
        self.form_window_start = None
        
        #7) The variables for our algorithm are initialised as none for each ETF.
        # These will store the baseline prices from the first day of the formation window.
        self.y0 = None
        self.x0 = None
        
        #8) The formation statistics for the mean and standard deviation are initialised as none.
        self.form_mean = None
        self.form_std = None
        
        #9) The tracking variable which tracks if prices cross is initialised to none.
        # This variable uses the previous spread of prices to compare against the current spread for price crossing.
        self.prev_spread = None

        #10) This ensures that QuantConnect calls our Trading Logic at a specific time, in this case every TRADING day
        # (not calendar day) when y (ETF_A) is trading, 1 minute after the market opens.
        self.schedule.on(
            self.date_rules.every_day(self.y),
            self.time_rules.after_market_open(self.y, 1),
            self.RunTradingLogic
        )

        #11) Print debug messages to show initialization is complete and display the VIX-based position sizing levels.
        self.debug(f"Initialized VIX-BASED POSITION SIZING strategy with cointegration testing")
        self.debug(f"VIX Levels: <{self.VIX_LOW_LEVEL}={self.MIN_GROSS_CAP*100:.0f}% | ~{self.VIX_MEDIUM_LEVEL}={self.BASE_GROSS_CAP*100:.0f}% | ~{self.VIX_HIGH_LEVEL}={self.HIGH_GROSS_CAP*100:.0f}% | >{self.VIX_EXTREME_LEVEL}={self.MAX_GROSS_CAP*100:.0f}%")

    def OnData(self, data):
        """
        This method is called whenever new data arrives for any subscribed security.
        We use it to update our VIX rolling window with the latest VIX values.
        
        Parameters:
        - data: A dictionary-like object containing the latest market data for all securities
        """
        
        #1) Check if VIX data is present in the current data slice.
        # data.contains_key() checks if the VIX symbol has data in this particular update.
        if data.contains_key(self.vix):
            #2) Extract the VIX value and convert it to a float.
            vix_value = float(data[self.vix].value)
            
            #3) Only add valid (positive) VIX values to our rolling window.
            # VIX should always be positive, so this is a data quality check.
            if vix_value > 0:
                # Add the VIX value to our rolling window. The window automatically
                # maintains only the most recent VIX_LOOKBACK values.
                self.vix_window.add(vix_value)

    def CalculateVIXBasedPositionSize(self):
        """
        Calculate position size based on current VIX level.
        Higher VIX = Larger position (more opportunity from bigger divergences).
        
        This method implements piecewise linear interpolation between VIX levels
        to smoothly scale position sizes based on market volatility.
        
        Returns: 
        - tuple: (gross_cap, vix_regime_name)
          - gross_cap: float between MIN_GROSS_CAP and MAX_GROSS_CAP representing total position size
          - vix_regime_name: string describing the current VIX regime
        """
        
        #1) Check if we have enough VIX data to make a decision.
        # is_ready returns True only when the window has been filled with VIX_LOOKBACK values.
        # If not ready, return the baseline position size.
        if not self.vix_window.is_ready:
            return self.BASE_GROSS_CAP, "Unknown"
        
        #2) Get the most recent VIX value from our rolling window.
        # Index [0] gives us the most recently added value.
        current_vix = self.vix_window[0]
        
        #3) Calculate position size based on VIX level using piecewise linear interpolation.
        # We divide the VIX range into several regimes and interpolate position sizes within each regime.
        
        if current_vix <= self.VIX_LOW_LEVEL:
            #3a) Very calm markets (VIX <= 12): Use minimum position size.
            # When VIX is very low, spreads tend to be tight with less mispricing opportunity.
            gross_cap = self.MIN_GROSS_CAP
            regime = "VERY_CALM"
            
        elif current_vix <= self.VIX_MEDIUM_LEVEL:
            #3b) Calm to normal markets (12 < VIX <= 20): Interpolate between MIN and BASE.
            # Calculate how far we are between VIX_LOW_LEVEL and VIX_MEDIUM_LEVEL (0 to 1).
            ratio = (current_vix - self.VIX_LOW_LEVEL) / (self.VIX_MEDIUM_LEVEL - self.VIX_LOW_LEVEL)
            # Linear interpolation: start at MIN_GROSS_CAP and move toward BASE_GROSS_CAP.
            gross_cap = self.MIN_GROSS_CAP + (self.BASE_GROSS_CAP - self.MIN_GROSS_CAP) * ratio
            regime = "CALM"
            
        elif current_vix <= self.VIX_HIGH_LEVEL:
            #3c) Normal to elevated markets (20 < VIX <= 35): Interpolate between BASE and HIGH.
            # As volatility increases, we increase position sizes to capture larger mispricings.
            ratio = (current_vix - self.VIX_MEDIUM_LEVEL) / (self.VIX_HIGH_LEVEL - self.VIX_MEDIUM_LEVEL)
            gross_cap = self.BASE_GROSS_CAP + (self.HIGH_GROSS_CAP - self.BASE_GROSS_CAP) * ratio
            regime = "ELEVATED"
            
        elif current_vix <= self.VIX_EXTREME_LEVEL:
            #3d) Elevated to extreme markets (35 < VIX <= 50): Interpolate between HIGH and MAX.
            # High volatility creates the largest mispricings, so we use our largest position sizes.
            ratio = (current_vix - self.VIX_HIGH_LEVEL) / (self.VIX_EXTREME_LEVEL - self.VIX_HIGH_LEVEL)
            gross_cap = self.HIGH_GROSS_CAP + (self.MAX_GROSS_CAP - self.HIGH_GROSS_CAP) * ratio
            regime = "HIGH"
            
        else:
            #3e) Extreme markets (VIX > 50): Cap at maximum position size.
            # Beyond VIX of 50, we don't increase position size further to maintain some risk control.
            gross_cap = self.MAX_GROSS_CAP
            regime = "EXTREME"
        
        #4) Return the calculated position size and the regime name for logging purposes.
        return gross_cap, regime

    def ShouldCatastrophicExit(self):
        """
        Only exit on truly catastrophic VIX levels (>75).
        This is 2008 Lehman collapse / 2020 COVID crash peak territory.
        
        At these extreme levels, pair relationships can break down temporarily
        and capital preservation becomes more important than staying in positions.
        
        Returns:
        - tuple: (should_exit, current_vix)
          - should_exit: Boolean indicating if we should exit all positions
          - current_vix: Current VIX value (or None if VIX data not ready)
        """
        
        #1) Check if we have VIX data available.
        if not self.vix_window.is_ready:
            return False, None
        
        #2) Get the current VIX value.
        current_vix = self.vix_window[0]
        
        #3) Check if VIX exceeds our catastrophic threshold.
        # VIX > 75 has only occurred during the most severe market dislocations:
        # - 2008 Financial Crisis (peaked around 80-89)
        # - 2020 COVID Crash (peaked around 82)
        # At these levels, market structure can break down and pairs may not behave normally.
        if current_vix > self.VIX_CATASTROPHIC:
            return True, current_vix
        
        #4) Normal conditions - no catastrophic exit needed.
        return False, current_vix

    def RunTradingLogic(self):
        """
        This method runs the trading logic for the pairs strategy by:
        - Validating that both ETFs have data for the day
        - Checking for catastrophic VIX levels and exiting if necessary
        - Switching between formation and trading modes
        - Starting and ending trade windows, with liquidations when required
        - Computing normalised prices, spreads, and z-scores
        - Detecting spread crossings to trigger exits
        - Checking entry conditions and opening long/short positions with VIX-based position sizing
        - Updating the previous spread each day for future crossing detection
        """
        
        #1) Run a data validation check to ensure both securities have valid data.
        # self.securities is a dictionary of all the ETFs initialised using add_equity in the Initialize method.
        # has_data ensures that there are valid data points for each ETF given the resolution of trading.
        if not (self.securities[self.y].has_data and self.securities[self.x].has_data):
            return

        #2) Check for catastrophic VIX exit conditions before doing anything else.
        # This is our emergency "circuit breaker" for extreme market conditions.
        should_exit_catastrophic, current_vix = self.ShouldCatastrophicExit()
        
        #3) If VIX is at catastrophic levels AND we have open positions, liquidate immediately.
        if should_exit_catastrophic and (self.portfolio[self.y].invested or self.portfolio[self.x].invested):
            # Liquidate both legs of the pair trade.
            self.liquidate(self.y, "CATASTROPHIC VIX level - emergency exit")
            self.liquidate(self.x, "CATASTROPHIC VIX level - emergency exit")
            # Log the emergency exit with VIX level.
            self.debug(f"Date: {self.time.date()} | CATASTROPHIC EXIT | VIX: {current_vix:.2f}")
            # Reset the previous spread since we've exited all positions.
            self.prev_spread = None
            return

        """
        4) Checks whether we are currently not in a trading window. If true, we are in formation mode
        and should recompute the formation stats.
        """
        if not self.in_trade_window:
            #4a) Calls the formation method to build the sample of ETFs, normalise prices, compute spread mean and standard deviation,
            # check for liquidity issues, test for cointegration, and returns true if successful or otherwise false.
            formation = self.BuildFormationWindow()
            
            if formation:
                self.in_trade_window = True
                # timedelta turns the integer into calendar days and adds it to the current time to obtain the trade_window_end.
                self.trade_window_end = self.time + timedelta(days=int(self.TRADE_DAYS * 1.5))
                self.prev_spread = None
                
                # Show VIX regime at start of trade window.
                # This helps us understand what position sizing we're starting with.
                gross_cap, regime = self.CalculateVIXBasedPositionSize()
                vix_status = f"VIX: {current_vix:.2f} | Regime: {regime} | Position Size: {gross_cap*100:.0f}%" if current_vix else "VIX: N/A"
                self.debug(f"New Trade Window | {self.time.date()} | Mean: {self.form_mean:.4f} SD: {self.form_std:.4f} | {vix_status}")
            return

        #5) Check if a trade window end date was set and that the current time is at or past that date.
        # If the date has elapsed, then we want to exit all trades.
        if self.trade_window_end and self.time >= self.trade_window_end:
            #5a) Check our portfolio to see if we have anything invested in either of the two ETFs
            # and liquidate the position, adding a tag explaining why it was liquidated.
            
            # self.portfolio keeps a record of every security owned and related information such as quantity, average price, total cost, unrealised gains.
            # self.invested returns True if a position (non-zero) is held in a specific security, and false otherwise.
            # self.liquidate closes all open positions by entering opposing trades. For example, if currently long it will place a sell order, if short it will place a buy order.
            if self.portfolio[self.y].invested or self.portfolio[self.x].invested:
                self.liquidate(self.y, "Trade window ended")
                self.liquidate(self.x, "Trade window ended")
            
            #5b) Resets our in_trade_window tracking variable to be false once we have left the trade window,
            # and prints a debug to the console giving the date of the end of the trade window with a message.
            self.in_trade_window = False
            self.debug(f"Date: {self.time.date()} | Trade window ended")
            return

        #6) Reads the most recent close prices (in decimal form) and updates the ETF variables
        # to reflect these close prices. If our trading algorithm runs on open, it would thus use yesterday's close prices!
        y_price = float(self.securities[self.y].close)
        x_price = float(self.securities[self.x].close)

        #7) If our algorithm has not been initialised for whatever reason, then we have no
        # formation prices to obtain normalised prices. Early return in this case, and if the prices are invalid (negative).
        if self.y0 is None or self.x0 is None or self.y0 <= 0 or self.x0 <= 0:
            return

        #8) Obtain the current day's normalised prices for each ETF, along with the normalised price spread.
        # Normalisation is done by dividing current price by the first day's price from the formation window.
        y_norm = y_price / self.y0
        x_norm = x_price / self.x0
        spread = y_norm - x_norm

        #9) If the formation standard deviation has not been initialised, or is 0 (invalid), early return.
        if self.form_std is None or self.form_std == 0:
            return

        #10) Obtain the current day's z-score.
        # Z-score tells us how many standard deviations the current spread is from the formation mean.
        z = (spread - self.form_mean) / self.form_std

        #11) Check if we already hold a position in EITHER of the ETF Pairs.
        invested = self.portfolio[self.y].invested or self.portfolio[self.x].invested

        #12) If we currently hold a position in either ETF and we already have a previous spread 
        # recorded (i.e. not the very first day of the trading window), then check for a price crossing.
        # A crossing is detected if the sign of the spread changes.
        # 
        # The product of the two spreads, (prev_spread * spread) will be negative (< 0) in the event of
        # a sign change. Thus, this is our condition to check for a crossing of prices.
        
        # In the case that all 3 conditions are met, we liquidate both positions and update prev_spread
        # to the current spread. Early return after closing the positions given a crossing of prices.
        if invested and self.prev_spread is not None and (self.prev_spread * spread < 0):
            self.liquidate(self.y, "Prices crossed")
            self.liquidate(self.x, "Prices crossed")
            vix_status = f"VIX: {current_vix:.2f}" if current_vix else "VIX: N/A"
            self.debug(f"Date: {self.time} | Exit on crossing | Z: {z:.2f} | {vix_status}")
            self.prev_spread = spread
            return

        #13) If we don't hold a position in either ETF, and the magnitude of the current Z-Score >= Entry Z-Score,
        # we consider opening a new trade. abs(z) ensures that the sign of the Z-Score is irrelevant here, as we
        # only care about magnitude.
        if not invested and abs(z) >= self.ENTRY_Z:
            #13a) Calculate position size based on current VIX level.
            # Higher VIX = larger position (more mispricing opportunity from bigger divergences).
            # This is the core innovation: we INCREASE size when volatility is high, not decrease it.
            gross_cap, regime = self.CalculateVIXBasedPositionSize()
            
            #13b) Compute the per-leg weight: each ETF (long and short) gets half of the VIX-adjusted gross cap.
            # Example: if VIX gives us gross_cap = 0.70, each ETF leg is 0.35 (35% long, 35% short).
            # The min(..., 0.49) is a safeguard so no single ETF leg ever exceeds 49% of portfolio value,
            # even if gross_cap is set too high. This ensures balanced exposure and prevents over-allocation.
            per_leg_weight = min(gross_cap / 2.0, 0.49)

            #13c) The type of position we would like to enter is based on which normalised price is higher.
            # A positive spread (> 0) means that Y_Norm (ETF_A) > X_Norm (ETF_B), thus Y > X (relatively).
            # Hence, we go long on X (ETF_B), short on Y (ETF_A). Opposite logic occurs if the spread is negative (< 0).
            # We enter into the trades with 'per_leg_weight' of the portfolio value per ETF. set_holdings calculates
            # the number of asset units to purchase based on the aforementioned weighting!
            if spread > 0:
                self.set_holdings(self.x, +per_leg_weight)
                self.set_holdings(self.y, -per_leg_weight)
                side = f"Long {self.ETF_B} / Short {self.ETF_A}"
            else:
                self.set_holdings(self.y, +per_leg_weight)
                self.set_holdings(self.x, -per_leg_weight)
                side = f"Long {self.ETF_A} / Short {self.ETF_B}"

            #13d) Print to the console our trade with VIX regime and position size information, then update our previous spread.
            # Return, as we have traded for the day.
            vix_status = f"VIX: {current_vix:.2f} | Regime: {regime}" if current_vix else "VIX: N/A"
            self.debug(f"Date: {self.time} | {side} | Z: {z:.2f} | Size: {gross_cap*100:.0f}% | {vix_status}")
            self.prev_spread = spread
            return

        #14) Always update the previous spread for the next day's crossing detection.
        self.prev_spread = spread

    def BuildFormationWindow(self) -> bool:

        #1) Request historical data for Open, Close, Low, High, Volume information for both ETFs over the formation window.
        # self.history() asks QuantConnect for past data (in this case, FORM_DAYS worth at TRADE_FREQUENCY resolution).
        # The result is a  DataFrame with multi-index (time + symbol) information, containing price and volume data.
        bars = self.history([self.y, self.x], self.FORM_DAYS, self.TRADE_FREQUENCY).copy()

        #2) If the dataframe is empty, return false.
        if bars.empty:
            return False

        #3) Restructure the historical data to get a cleand up time series of the close prices and volume per ETF.
        # If the restructuring fails, return False.
        try:
            # Restructures the data frame to have only time, close prices, per ETF.
            close = bars['close'].unstack(level=0)
            vol   = bars['volume'].unstack(level=0)
        except Exception:
            return False

        #4) Returns false if either ETF doesn't exist in the historical data.
        if self.y not in close.columns or self.x not in close.columns:
            return False

        #5) Checks for zero-volume days for either ETF. If any zero-volume days exist, we cannot form data (validity check).
        # Missing values are replaced with 0, and sets 0 values to equal True. If a True exists, we skip formation and return False.
        if vol[self.y].fillna(0).eq(0).any() or vol[self.x].fillna(0).eq(0).any():
            self.debug("Formation skipped due to zero-volume day in history")
            return False

        #6) Removes any rows where one or both of the ETFs have missing values.
        # Such as, if only one ETF has data on one day and the other doesn't.
        close = close.dropna()

        #7) Counts how many rows of data exist. If the number of rows is less than the
        # formation days we expect in a period, then the formation history is incomplete.
        # Returns false when incomplete.
        if len(close) < self.FORM_DAYS:
            return False

        #8) Creates a series for each ETF indexed by data, with close price values, as a float.
        y_series = close[self.y].astype(float)
        x_series = close[self.x].astype(float)

        #9) Obtains the very first day of the formation window to get the starting price for both ETFs.
        y0 = float(y_series.iloc[0])
        x0 = float(x_series.iloc[0])

        #10) If the starting price is negative, then it is incorrect and we return False.
        if y0 <= 0 or x0 <= 0:
            return False

        #11) Obtain the normalised prices and the spread of these normalised prices for the formation period.
        y_norm = y_series / y0
        x_norm = x_series / x0
        spread = y_norm - x_norm

        #12) Print debug information showing current time and start date for cointegration testing context.
        self.debug(f"Current time = {self.time}; start_date = {self.start_date}")

        hist = self.history([self.y, self.x], self.start_date, self.time, self.TRADE_FREQUENCY).copy()
        
        # If no historical data is available, we cannot perform cointegration testing, so return False.
        if hist.empty:
            self.debug("No historical data available for cointegration testing.")
            return False


        try:
            # Unstack converts the multi-index DataFrame to have symbols as columns.
            close_hist = hist['close'].unstack(level=0)

            close_hist = close_hist[[self.y, self.x]].dropna()
        except Exception as e:
            self.debug(f"Error processing historical closes for cointegration: {e}")
            return False

        self.historical_data = close_hist.copy()
        closes_pair = close_hist[[self.y, self.x]]

        raw_prices = np.column_stack((closes_pair[self.y].values, closes_pair[self.x].values))

        adf_y_levels = adfuller(raw_prices[:, 0])
        adf_x_levels = adfuller(raw_prices[:, 1])
        
        y_diff = np.diff(raw_prices[:, 0])
        x_diff = np.diff(raw_prices[:, 1])
        adf_y_diff = adfuller(y_diff)
        adf_x_diff = adfuller(x_diff)
        

        y_levels_nonstationary = abs(adf_y_levels[0]) < abs(adf_y_levels[4]['5%'])
        x_levels_nonstationary = abs(adf_x_levels[0]) < abs(adf_x_levels[4]['5%'])
        y_diff_stationary = abs(adf_y_diff[0]) >= abs(adf_y_diff[4]['5%'])
        x_diff_stationary = abs(adf_x_diff[0]) >= abs(adf_x_diff[4]['5%'])
        
        # Both series must be I(1) for Johansen test to be valid
        if not (y_levels_nonstationary and x_levels_nonstationary and y_diff_stationary and x_diff_stationary):
            self.debug(f"I(1) assumption violated:")
            self.debug(f"  {self.ETF_A} levels non-stationary: {y_levels_nonstationary} (tau={adf_y_levels[0]:.4f})")
            self.debug(f"  {self.ETF_B} levels non-stationary: {x_levels_nonstationary} (tau={adf_x_levels[0]:.4f})")
            self.debug(f"  {self.ETF_A} diff stationary: {y_diff_stationary} (tau={adf_y_diff[0]:.4f})")
            self.debug(f"  {self.ETF_B} diff stationary: {x_diff_stationary} (tau={adf_x_diff[0]:.4f})")
            return False

        coint_result = coint_johansen(self.historical_data, det_order=0, k_ar_diff=1)
        

        w1 = coint_result.evec[0, 0]
        w2 = coint_result.evec[1, 0]

        epsilon = w1 * raw_prices[:, 0] + w2 * raw_prices[:, 1]

        adf_result = adfuller(epsilon)

        if not abs(adf_result[0]) >= abs(adf_result[4]['10%']):
            self.debug(f"Pair not statistically significant: tau = {abs(adf_result[0]):.4f}, 10% S.L = {abs(adf_result[4]['10%']):.4f}")
            self.debug(f"Total data points per ticker = {len(self.historical_data)}")
            return False

        self.form_mean = float(spread.mean())
        self.form_std  = float(spread.std(ddof=1))
        self.y0 = y0
        self.x0 = x0
        self.form_window_start = close.index[0]

        return self.form_std > 0